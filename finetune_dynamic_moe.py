#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA finetuning entrypoint for the dynamic-router MoE model.

This script keeps only the pieces that are specific to the dynamic-router
variant:
- model/config imports
- legacy dense-router -> cross-attention-router initialization
- k-bit loading policy for the customized router modules

Datasets, collator, and the main training loop are reused from `finetune.py`
to avoid duplicating the entire training stack.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from finetune import (
    LMDataCollator,
    build_optimizer,
    cleanup_distributed,
    evaluate,
    is_main_process,
    is_quantized_model,
    load_and_pack_arc_challenge_ppl_opencompass,
    load_and_pack_arc_easy_ppl_opencompass,
    load_and_pack_bbh_ppl_opencompass,
    load_and_pack_commonsenseqa_ppl_opencompass,
    load_and_pack_hellaswag_ppl_opencompass,
    load_and_pack_mmlu_ppl_opencompass,
    load_and_pack_openbookqa_ppl_opencompass,
    load_and_pack_piqa_ppl_opencompass,
    load_and_pack_siqa_ppl_opencompass,
    load_and_pack_winogrande_ppl_opencompass,
    set_seed,
)


def setup_distributed_safe():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return -1, 1, False

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed launch detected, but CUDA is unavailable.")

    n_visible = torch.cuda.device_count()
    if local_rank < 0 or local_rank >= n_visible:
        raise RuntimeError(
            f"Invalid LOCAL_RANK={local_rank} for visible CUDA device count={n_visible}. "
            f"Set --nproc_per_node <= {n_visible}."
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    dist.barrier(device_ids=[local_rank])
    return local_rank, world_size, True


def import_moe_classes():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in os.sys.path:
        os.sys.path.insert(0, here)

    tried = []
    for model_mod, cfg_mod in [
        ("modeling_moe_dm", "configuration_moe_dm"),
        ("qwen_moe.modeling.modeling_moe_dm", "qwen_moe.modeling.configuration_moe_dm"),
    ]:
        try:
            mod_model = __import__(model_mod, fromlist=["MoEForCausalLM"])
            mod_cfg = __import__(cfg_mod, fromlist=["MoEConfig"])
            return mod_model.MoEForCausalLM, mod_cfg.MoEConfig
        except Exception as exc:
            tried.append((model_mod, cfg_mod, repr(exc)))

    msg = "Failed to import MoEForCausalLM/MoEConfig. Tried:\n"
    msg += "\n".join([f"- {m} / {c}: {e}" for m, c, e in tried])
    raise ImportError(msg)


def _load_local_checkpoint_state_dict(model_path: str) -> Dict[str, torch.Tensor]:
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"model_path not found: {model_path}")

    def _load_safetensors_file(fp: Path) -> Dict[str, torch.Tensor]:
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError(f"Found safetensors checkpoint ({fp}) but safetensors is unavailable.") from exc
        return load_file(str(fp))

    st_index = model_dir / "model.safetensors.index.json"
    if st_index.exists():
        with st_index.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        state_dict: Dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(meta["weight_map"].values())):
            state_dict.update(_load_safetensors_file(model_dir / shard_name))
        return state_dict

    st_single = model_dir / "model.safetensors"
    if st_single.exists():
        return _load_safetensors_file(st_single)

    pt_index = model_dir / "pytorch_model.bin.index.json"
    if pt_index.exists():
        with pt_index.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        state_dict = {}
        for shard_name in sorted(set(meta["weight_map"].values())):
            state_dict.update(torch.load(model_dir / shard_name, map_location="cpu"))
        return state_dict

    pt_single = model_dir / "pytorch_model.bin"
    if pt_single.exists():
        return torch.load(pt_single, map_location="cpu")

    raise FileNotFoundError(f"No recognized checkpoint file under {model_path}")


def _init_cross_attention_router_from_legacy_dense(
    model: nn.Module,
    legacy_sd: Dict[str, torch.Tensor],
    config,
) -> int:
    """
    Initialize cross-attention router from legacy dense router weight.

    Rules:
    - query: partial copy from legacy dense router weight
    - expert_embed/shared_expert_embed: standalone random init
    - key/value: small-variance random init
    """
    num_inited = 0
    base_std = float(getattr(config, "initializer_range", 0.02))
    kv_std = base_std * 0.1

    with torch.no_grad():
        shared_expert_embed = getattr(model.model, "shared_expert_embed", None)
        if shared_expert_embed is not None:
            shared_expert_embed.normal_(mean=0.0, std=base_std)

        for layer_idx, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            if not getattr(mlp, "use_switch", False):
                continue
            if not getattr(mlp, "use_cross_attention_router", False):
                continue
            if not hasattr(mlp, "router"):
                continue

            dense_w = legacy_sd.get(f"model.layers.{layer_idx}.mlp.router.weight")
            if dense_w is None:
                continue

            router = mlp.router
            if not all(hasattr(router, name) for name in ("query", "key", "value")):
                # Legacy path (q_proj/expert_keys/...) is no longer used by current simplified router.
                continue

            dense_w = dense_w.float()

            query = getattr(router, "query", None)
            if query is not None and hasattr(query, "weight"):
                query.weight.zero_()
                rows = min(query.weight.shape[0], dense_w.shape[0])
                cols = min(query.weight.shape[1], dense_w.shape[1])
                query.weight[:rows, :cols].copy_(
                    dense_w[:rows, :cols].to(dtype=query.weight.dtype, device=query.weight.device)
                )

            for proj_name in ("key", "value"):
                proj = getattr(router, proj_name, None)
                if proj is None or not hasattr(proj, "weight"):
                    continue
                proj.weight.normal_(mean=0.0, std=kv_std)

            if getattr(router, "_shared_expert_embed_ref", None) is None and getattr(router, "expert_embed", None) is not None:
                router.expert_embed.normal_(mean=0.0, std=base_std)

            num_inited += 1

    return num_inited


def _enable_new_router_params_trainable(model: nn.Module) -> int:
    keys = (
        "router.query",
        "router.key",
        "router.value",
        "router.expert_embed",
        "shared_expert_embed",
        # Legacy (unused in current simplified router):
        # "router.q_proj", "expert_keys", "expert_values", "log_router_temperature",
        # "router_value_proj", "router_context_gate_proj",
    )
    n_params = 0
    for name, param in model.named_parameters():
        if any(key in name for key in keys):
            param.requires_grad = True
            n_params += param.numel()
    return n_params


def _cast_selected_trainable_params_to_fp32(model: nn.Module) -> int:
    """
    Cast all floating-point trainable parameters to fp32.

    This keeps base frozen/quantized weights untouched (requires_grad=False),
    while ensuring every trainable parameter (LoRA/router/custom) stays in fp32.
    """
    n_params = 0
    with torch.no_grad():
        for _name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if not torch.is_floating_point(param):
                continue
            if param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
                n_params += param.numel()
    return n_params

def guess_lora_targets(model: nn.Module) -> List[str]:
    candidates = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        # Legacy (unused in current simplified router):
        # "router_value_proj",
        # "router_context_gate_proj",
    ]
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            short_name = name.split(".")[-1]
            if short_name in candidates:
                names.add(short_name)
    return sorted(names)


def apply_lora(
    model: nn.Module,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Optional[List[str]] = None,
):
    target_modules = target_modules or guess_lora_targets(model)
    if not target_modules:
        raise RuntimeError("Could not infer LoRA target modules; please pass --lora_target_modules")

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    return get_peft_model(model, lora_cfg)


def train(
    model: nn.Module,
    train_dl: DataLoader,
    eval_dl: Optional[DataLoader],
    run_args: argparse.Namespace,
    device: torch.device,
    output_dir: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    grad_accum: int,
    max_grad_norm: float,
    fp16: bool,
    bf16: bool,
    use_bnb_8bit: bool,
    log_every: int,
    eval_every: int,
    save_every: int,
):
    os.makedirs(output_dir, exist_ok=True)

    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay, use_bnb_8bit=use_bnb_8bit)

    steps_per_epoch = math.ceil(len(train_dl) / max(1, grad_accum))
    total_optim_steps = steps_per_epoch * epochs
    warmup_steps = int(total_optim_steps * warmup_ratio)

    pbar = tqdm(
        total=total_optim_steps,
        disable=not is_main_process(),
        dynamic_ncols=True,
        desc="train",
    )

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device.type == "cuda"))

    model.train()

    global_step = 0
    optim_step = 0
    t0 = time.time()

    metrics_f = None
    ma_win = 50
    ma_loss_buf = deque(maxlen=ma_win)
    ma_loss_sum = 0.0
    ema_loss = None
    ema_momentum = 0.98

    if is_main_process():
        records_dir = Path(__file__).resolve().parent / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        run_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        metrics_path = records_dir / f"{run_ts}.tsv"
        metrics_f = open(metrics_path, "w", encoding="utf-8")
        metrics_f.write(f"# run_started_at\t{run_ts}\n")
        for key, value in sorted(vars(run_args).items()):
            metrics_f.write("\t".join([
                "# arg",
                key,
                json.dumps(value, ensure_ascii=False, default=str),
            ]) + "\n")
        metrics_f.write("\t".join([
            "time", "epoch", "global_step", "optim_step", "lr",
            "loss", f"loss_ma{ma_win}", "loss_ema",
        ]) + "\n")
        metrics_f.flush()

    amp_dtype = torch.float16 if fp16 else (torch.bfloat16 if bf16 else None)

    for epoch in range(epochs):
        if isinstance(train_dl.sampler, DistributedSampler):
            train_dl.sampler.set_epoch(epoch)

        it = enumerate(train_dl)
        if is_main_process():
            it = tqdm(it, total=len(train_dl), desc=f"epoch {epoch+1}/{epochs}", dynamic_ncols=True)

        for step, batch in it:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            if amp_dtype is not None and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(**batch)
                    loss = out.loss / max(1, grad_accum)
            else:
                out = model(**batch)
                loss = out.loss / max(1, grad_accum)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            global_step += 1

            if global_step % grad_accum == 0:
                if max_grad_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optim_step += 1

                loss_real = float(loss.detach().float().item() * grad_accum)

                if len(ma_loss_buf) == ma_loss_buf.maxlen:
                    ma_loss_sum -= ma_loss_buf[0]
                ma_loss_buf.append(loss_real)
                ma_loss_sum += loss_real
                loss_ma = ma_loss_sum / max(1, len(ma_loss_buf))

                if ema_loss is None:
                    ema_loss = loss_real
                else:
                    ema_loss = ema_momentum * ema_loss + (1.0 - ema_momentum) * loss_real

                if is_main_process() and (optim_step % log_every == 0):
                    cur_lr = scheduler.get_last_lr()[0]

                    if hasattr(it, "set_postfix"):
                        it.set_postfix({
                            "loss": f"{loss_real:.4f}",
                            f"ma{ma_win}": f"{loss_ma:.4f}",
                            "ema": f"{ema_loss:.4f}",
                            "lr": f"{cur_lr:.2e}",
                        }, refresh=False)

                    if metrics_f is not None:
                        metrics_f.write("\t".join([
                            f"{time.time():.3f}",
                            str(epoch),
                            str(global_step),
                            str(optim_step),
                            f"{cur_lr:.6e}",
                            f"{loss_real:.6f}",
                            f"{loss_ma:.6f}",
                            f"{ema_loss:.6f}",
                        ]) + "\n")
                        metrics_f.flush()

                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{(loss.detach().float().item() * grad_accum):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.3e}",
                }, refresh=False)

                if is_main_process() and (optim_step % log_every == 0):
                    cur_lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    print(
                        f"[train] epoch={epoch+1}/{epochs} step={optim_step}/{total_optim_steps} "
                        f"loss={loss.detach().float().item() * grad_accum:.4f} lr={cur_lr:.3e} "
                        f"elapsed={elapsed/60:.1f}m"
                    )

                if eval_dl is not None and (optim_step % eval_every == 0):
                    ev = evaluate(model, eval_dl, device, fp16=fp16, bf16=bf16)
                    if is_main_process():
                        print(f"[eval] step={optim_step} loss={ev:.4f} ppl={math.exp(min(20, ev)):.2f}")

                if False and save_every > 0 and (optim_step % save_every == 0) and is_main_process():
                    save_dir = os.path.join(output_dir, f"checkpoint-{optim_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    model.save_pretrained(save_dir)
                    print(f"[save] {save_dir}")

    pbar.close()
    if is_main_process() and metrics_f is not None:
        metrics_f.close()

    if is_main_process():
        model_to_save = model.module if hasattr(model, "module") else model
        was_quantized = is_quantized_model(model_to_save)
        try:
            merged = model_to_save.merge_and_unload()
        except Exception as e:
            raise RuntimeError(
                "Failed to merge LoRA adapters into the base model. "
                "Your PEFT model may not support merge_and_unload()."
            ) from e

        if was_quantized and hasattr(merged, "dequantize"):
            print("[save] trying to dequantize merged model before export")
            maybe_dequantized = merged.dequantize()
            if maybe_dequantized is not None:
                merged = maybe_dequantized

        state_dict = merged.state_dict()
        for k, v in list(state_dict.items()):
            if torch.is_floating_point(v) and v.dtype != torch.float32:
                state_dict[k] = v.to(torch.float32)

        if was_quantized:
            for attr_name in ("is_loaded_in_8bit", "is_loaded_in_4bit", "quantization_method"):
                if hasattr(merged, attr_name):
                    setattr(merged, attr_name, False if attr_name != "quantization_method" else None)
            print(f"[save] merged + dequantized + fp32 full model -> {output_dir}")
        else:
            print(f"[save] merged + fp32 full model -> {output_dir}")

        merged.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=True)


def _patch_safe_initialize_missing_keys(model_cls):
    if getattr(model_cls, "_bnb_safe_initialize_missing_keys_patched", False):
        return

    def _safe_initialize_missing_keys(self, missing_keys: List[str], is_quantized: bool) -> None:
        state_dict = self.state_dict()
        for key in state_dict:
            if key in missing_keys:
                continue
            try:
                param_or_buffer = self.get_parameter_or_buffer(key)
            except AttributeError:
                continue
            param_or_buffer._is_hf_initialized = True

        def set_is_initialized_for_modules(module):
            if (
                all(getattr(child, "_is_hf_initialized", False) for child in module.children())
                and all(getattr(param, "_is_hf_initialized", False) for param in module.parameters(recurse=False))
                and all(
                    getattr(buffer, "_is_hf_initialized", False)
                    for buffer in module.buffers(recurse=False)
                    if buffer not in module._non_persistent_buffers_set
                )
            ):
                module._is_hf_initialized = True

        self.apply(set_is_initialized_for_modules)

        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            not_initialized_parameters = list(
                {v for v in state_dict.values() if not getattr(v, "_is_hf_initialized", False)}
            )
            with deepspeed.zero.GatheredParameters(not_initialized_parameters, modifier_rank=0):
                self.initialize_weights()
        else:
            self.initialize_weights()

    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

    model_cls._initialize_missing_keys = _safe_initialize_missing_keys
    model_cls._bnb_safe_initialize_missing_keys_patched = True


def build_quantization_config(args):
    if bool(args.load_in_4bit):
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bool(args.bnb_4bit_use_double_quant),
            bnb_4bit_compute_dtype=dtype_map[args.bnb_4bit_compute_dtype],
            llm_int8_skip_modules=[
                "router",
                "query",
                "key",
                "value",
                # Legacy (unused in current simplified router):
                # "router_value_proj",
                # "router_context_gate_proj",
            ],
        )

    if bool(args.load_in_8bit):
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=float(args.llm_int8_threshold),
            llm_int8_skip_modules=[
                "router",
                "query",
                "key",
                "value",
                # Legacy (unused in current simplified router):
                # "router_value_proj",
                # "router_context_gate_proj",
            ],
        )

    return None


def build_model_with_router_compat(
    args,
    model_cls,
    config,
    device: torch.device,
    quantization_config,
    local_rank: int,
    is_distributed: bool,
):
    is_kbit = bool(args.load_in_4bit) or bool(args.load_in_8bit)
    if is_kbit and device.type != "cuda":
        raise RuntimeError("bitsandbytes k-bit loading requires CUDA.")

    if bool(args.load_in_8bit):
        _patch_safe_initialize_missing_keys(model_cls)
        for attr_name in ["_keys_to_ignore_on_load_missing", "_keys_to_ignore_on_load_unexpected"]:
            patterns = list(getattr(model_cls, attr_name, []) or [])
            for pat in [r".*\.SCB$", r".*\.weight_format$"]:
                if pat not in patterns:
                    patterns.append(pat)
            setattr(model_cls, attr_name, patterns)

    if not is_kbit:
        load_dtype = torch.float16 if bool(args.load_in_fp16) else torch.float32
        model = model_cls(config)
        model = model.to(dtype=load_dtype)
        legacy_sd = _load_local_checkpoint_state_dict(args.model_path)
        if load_dtype != torch.float32:
            legacy_sd = {
                name: tensor.to(dtype=load_dtype) if torch.is_floating_point(tensor) else tensor
                for name, tensor in legacy_sd.items()
            }
        missing_keys, unexpected_keys = model.load_state_dict(legacy_sd, strict=False)
        if is_main_process():
            print(
                f"[load] checkpoint loaded with strict=False: missing={len(missing_keys)} "
                f"unexpected={len(unexpected_keys)} dtype={load_dtype}"
            )
        if bool(args.init_new_router_from_legacy):
            inited = _init_cross_attention_router_from_legacy_dense(model=model, legacy_sd=legacy_sd, config=config)
            if is_main_process():
                print(f"[init] initialized new router modules from legacy dense router for {inited} layers")
        model.to(device=device)
        return model

    device_map = {"": local_rank} if is_distributed else {"": 0}
    torch_dtype = (
        quantization_config.bnb_4bit_compute_dtype
        if bool(args.load_in_4bit)
        else torch.float16
    )
    model = model_cls.from_pretrained(
        args.model_path,
        config=config,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )

    if bool(args.init_new_router_from_legacy):
        legacy_sd = _load_local_checkpoint_state_dict(args.model_path)
        inited = _init_cross_attention_router_from_legacy_dense(model=model, legacy_sd=legacy_sd, config=config)
        if is_main_process():
            print(f"[init] initialized new router modules from legacy dense router for {inited} layers")

    return model


def make_dataset(name: str, tokenizer, args, split: str, max_samples: Optional[int]):
    use_label = bool(args.use_label)

    if name == "piqa":
        return load_and_pack_piqa_ppl_opencompass(
            tokenizer=tokenizer,
            block_size=args.block_size,
            split=split,
            num_proc=args.num_proc,
            bos=True,
            eos=False,
            max_samples=max_samples,
            use_label=use_label,
        )
    if name == "siqa":
        return load_and_pack_siqa_ppl_opencompass(
            tokenizer=tokenizer,
            block_size=args.block_size,
            split=split,
            num_proc=args.num_proc,
            bos=True,
            eos=False,
            max_samples=max_samples,
            use_label=use_label,
        )
    if name == "hellaswag":
        return load_and_pack_hellaswag_ppl_opencompass(
            tokenizer=tokenizer,
            block_size=args.block_size,
            split=split,
            num_proc=args.num_proc,
            bos=True,
            eos=False,
            max_samples=max_samples,
            use_label=use_label,
        )
    if name == "arc-e":
        return load_and_pack_arc_easy_ppl_opencompass(
            tokenizer=tokenizer,
            block_size=args.block_size,
            split=split,
            num_proc=args.num_proc,
            bos=True,
            eos=False,
            max_samples=max_samples,
            use_label=use_label,
        )
    if name == "csqa":
        return load_and_pack_commonsenseqa_ppl_opencompass(
            tokenizer=tokenizer,
            block_size=args.block_size,
            split=split,
            num_proc=args.num_proc,
            bos=True,
            eos=False,
            max_samples=max_samples,
            use_label=use_label,
        )
    if name == "bbh":
        return load_and_pack_bbh_ppl_opencompass(
            tokenizer=tokenizer,
            block_size=args.block_size,
            split=split,
            num_proc=args.num_proc,
            bos=True,
            eos=False,
            max_samples=max_samples,
            task=args.bbh_task,
        )
    if name == "winogrande":
        return load_and_pack_winogrande_ppl_opencompass(
            tokenizer=tokenizer,
            block_size=args.block_size,
            split=split,
            num_proc=args.num_proc,
            bos=True,
            eos=False,
            max_samples=max_samples,
            config_name=args.winogrande_config,
            use_label=use_label,
        )
    if name == "mmlu":
        subjects = None
        if args.mmlu_subjects.strip().lower() != "all":
            subjects = [x.strip() for x in args.mmlu_subjects.split(",") if x.strip()]
        return load_and_pack_mmlu_ppl_opencompass(
            tokenizer=tokenizer,
            block_size=args.block_size,
            split=split,
            num_proc=args.num_proc,
            bos=True,
            eos=False,
            max_samples=max_samples,
            subjects=subjects,
            answer_mode=args.mmlu_answer_mode,
        )
    if name == "arc-c":
        return load_and_pack_arc_challenge_ppl_opencompass(
            tokenizer=tokenizer,
            block_size=args.block_size,
            split=split,
            num_proc=args.num_proc,
            bos=True,
            eos=False,
            max_samples=max_samples,
            use_label=use_label,
        )
    if name == "openbookqa":
        return load_and_pack_openbookqa_ppl_opencompass(
            tokenizer=tokenizer,
            block_size=args.block_size,
            split=split,
            num_proc=args.num_proc,
            bos=True,
            eos=False,
            max_samples=max_samples,
            use_label=use_label,
        )
    raise ValueError(f"Unknown dataset: {name}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument(
        "--dataset",
        type=str,
        default="piqa",
        choices=["piqa", "siqa", "hellaswag", "arc-e", "csqa", "bbh", "winogrande", "mmlu", "mix", "arc-c", "openbookqa"],
    )
    ap.add_argument(
        "--eval_dataset",
        type=str,
        default="piqa",
        choices=["piqa", "siqa", "hellaswag", "arc-e", "csqa", "bbh", "winogrande", "mmlu", "arc-c", "openbookqa"],
    )
    ap.add_argument("--mix_datasets", type=str, default="piqa,siqa")
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--eval_split", type=str, default="validation")
    ap.add_argument("--block_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--num_proc", type=int, default=8)
    ap.add_argument("--train_max_samples", type=int, default=None)
    ap.add_argument("--eval_max_samples", type=int, default=20)
    ap.add_argument("--use_label", type=int, default=1)

    ap.add_argument("--fp16", type=int, default=0)
    ap.add_argument("--bf16", type=int, default=0)
    ap.add_argument("--load_in_fp16", type=int, default=0)

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", type=str, default="")
    ap.add_argument("--train_new_router_params", type=int, default=1)

    ap.add_argument("--use_bnb_8bit", type=int, default=0)
    ap.add_argument("--load_in_4bit", type=int, default=0)
    ap.add_argument("--load_in_8bit", type=int, default=0)
    ap.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--bnb_4bit_use_double_quant", type=int, default=1)
    ap.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--llm_int8_threshold", type=float, default=6.0)
    ap.add_argument("--gradient_checkpointing", type=int, default=0)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=200)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--router_top_k", type=int, default=0)
    ap.add_argument("--router_topk", type=int, default=0)
    ap.add_argument("--init_new_router_from_legacy", type=int, default=1)
    ap.add_argument(
        "--router_use_entmax",
        type=int,
        default=0,
        choices=[-1, 0, 1],
        help="Set -1 to keep config value, 0 to disable, 1 to enable entmax routing.",
    )
    ap.add_argument(
        "--router_entmax_alpha",
        type=float,
        default=None,
        help="Override config.router_entmax_alpha when provided.",
    )
    ap.add_argument(
        "--share_router_expert_embedding",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help="Set -1 to keep config value, 0 for per-layer expert embeddings, 1 for one global expert embedding shared across layers.",
    )

    ap.add_argument("--bbh_task", type=str, default="boolean_expressions")
    ap.add_argument("--winogrande_config", type=str, default="winogrande_xl")
    ap.add_argument("--mmlu_subjects", type=str, default="all")
    ap.add_argument("--mmlu_answer_mode", type=str, default="text", choices=["text", "letter"])

    ap.add_argument(
        "--train_extra_params_in_fp32",
        type=int,
        default=1,
        help="Cast LoRA/new-router floating trainable params to fp32. Recommended for k-bit training.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    if bool(args.load_in_4bit) and bool(args.load_in_8bit):
        raise ValueError("Only one of --load_in_4bit and --load_in_8bit can be enabled.")

    local_rank, world_size, is_distributed = setup_distributed_safe()
    set_seed(args.seed + (local_rank if is_distributed else 0))

    device = torch.device(f"cuda:{local_rank}") if is_distributed else torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    use_fp16 = bool(args.fp16) and device.type == "cuda" and not bool(args.bf16)
    use_bf16 = bool(args.bf16) and device.type == "cuda"

    if is_main_process():
        print("[load] tokenizer from model_path:", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    model_cls, config_cls = import_moe_classes()
    config = config_cls.from_pretrained(args.model_path)
    if hasattr(config, "ensure_model_attributes"):
        config.ensure_model_attributes()

    effective_top_k = int(args.router_top_k) if int(args.router_top_k) > 0 else int(args.router_topk)
    if effective_top_k > 0:
        config.router_top_k = effective_top_k
    if int(args.router_use_entmax) >= 0:
        config.router_use_entmax = bool(args.router_use_entmax)
    if args.router_entmax_alpha is not None:
        config.router_entmax_alpha = float(args.router_entmax_alpha)
    if int(args.share_router_expert_embedding) >= 0:
        config.share_router_expert_embedding = bool(args.share_router_expert_embedding)
    if hasattr(config, "ensure_model_attributes"):
        config.ensure_model_attributes()
    if int(getattr(config, "router_top_k", 0)) <= 0:
        raise ValueError(f"config.router_top_k must be >= 1, got {getattr(config, 'router_top_k', None)}")
    if int(getattr(config, "num_experts", 0)) > 0 and int(config.router_top_k) > int(config.num_experts):
        raise ValueError(f"router_top_k ({config.router_top_k}) cannot exceed num_experts ({config.num_experts})")

    quantization_config = build_quantization_config(args)
    if is_main_process():
        print(f"[load] distributed={is_distributed} world_size={world_size} 4bit={bool(args.load_in_4bit)} 8bit={bool(args.load_in_8bit)}")
        if bool(args.load_in_4bit):
            print(f"[load] 4bit compute_dtype={quantization_config.bnb_4bit_compute_dtype}")
        print(f"[train] mixed precision: fp16={use_fp16} bf16={use_bf16}")

    model = build_model_with_router_compat(
        args=args,
        model_cls=model_cls,
        config=config,
        device=device,
        quantization_config=quantization_config,
        local_rank=local_rank,
        is_distributed=is_distributed,
    )

    if quantization_config is not None:
        if bool(args.gradient_checkpointing):
            model.gradient_checkpointing_enable()
        prep_sig = inspect.signature(prepare_model_for_kbit_training)
        if "use_gradient_checkpointing" in prep_sig.parameters:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=bool(args.gradient_checkpointing),
            )
        else:
            model = prepare_model_for_kbit_training(model)

    targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()] if args.lora_target_modules else None
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, target_modules=targets)

    if bool(args.train_new_router_params):
        n_router = _enable_new_router_params_trainable(model)
        if is_main_process():
            print(f"[train] extra trainable new-router params: {n_router}")

    n_fp32 = 0
    if bool(args.train_extra_params_in_fp32):
        n_fp32 = _cast_selected_trainable_params_to_fp32(model)
    if is_main_process():
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[train] cast trainable params to fp32: {n_fp32}")
        print(f"[lora] trainable params: {trainable}/{total} ({trainable / total * 100:.2f}%)")
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    if args.dataset == "mix":
        names = [x.strip() for x in args.mix_datasets.split(",") if x.strip()]
        train_ds = concatenate_datasets([
            make_dataset(name, tokenizer, args, args.train_split, args.train_max_samples) for name in names
        ])
    else:
        train_ds = make_dataset(args.dataset, tokenizer, args, args.train_split, args.train_max_samples)
    eval_ds = make_dataset(args.eval_dataset, tokenizer, args, args.eval_split, args.eval_max_samples)

    if is_main_process():
        print(f"[data] train={len(train_ds)} eval={len(eval_ds)} block_size={args.block_size}")

    collator = LMDataCollator(pad_id=0)
    if is_distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=dist.get_rank(), shuffle=True, drop_last=True)
        eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size, rank=dist.get_rank(), shuffle=False, drop_last=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        sampler=eval_sampler,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
    )

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    train(
        model=model,
        train_dl=train_dl,
        eval_dl=eval_dl,
        run_args=args,
        device=device,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_accum=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        fp16=use_fp16,
        bf16=use_bf16,
        use_bnb_8bit=bool(args.use_bnb_8bit),
        log_every=max(1, args.log_every),
        eval_every=max(1, args.eval_every),
        save_every=max(0, args.save_every),
    )

    if is_main_process():
        try:
            tokenizer.save_pretrained(args.output_dir)
        except Exception:
            pass

    cleanup_distributed()


if __name__ == "__main__":
    main()
