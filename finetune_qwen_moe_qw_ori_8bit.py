#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8-bit LoRA finetuning entrypoint for the original Qwen MoE model.

This script keeps the data processing aligned with the existing local training
utilities while switching the model/config import to the `qw_ori` variant.
It supports:
- `torchrun` multi-GPU training
- bitsandbytes 8-bit model loading
- LoRA on attention/MLP plus MoE gate modules
- merged checkpoint export, preferring a directly usable 8-bit full model
"""
from __future__ import annotations

import argparse
import gc
import inspect
import math
import os
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

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
from finetune_qwen_dynamic_moe import load_and_pack_piqa_local, setup_distributed_safe
from qwen_moe.modeling.configuration_moe_qw_ori import Qwen2MoeConfig
from qwen_moe.modeling.modeling_moe_qw_ori import Qwen2MoeForCausalLM


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def maybe_prepare_kbit_model_for_training(model: nn.Module, args) -> nn.Module:
    if not bool(args.load_in_8bit):
        return model

    if bool(args.gradient_checkpointing):
        model.gradient_checkpointing_enable()

    prep_sig = inspect.signature(prepare_model_for_kbit_training)
    if "use_gradient_checkpointing" in prep_sig.parameters:
        return prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(args.gradient_checkpointing),
        )
    return prepare_model_for_kbit_training(model)


def build_quantization_config(args) -> Optional[BitsAndBytesConfig]:
    if not bool(args.load_in_8bit):
        return None
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=float(args.llm_int8_threshold),
    )


def _guess_lora_targets(model: nn.Module) -> List[str]:
    candidates = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate",
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
) -> nn.Module:
    target_modules = target_modules or _guess_lora_targets(model)
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


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, use_bnb_8bit: bool):
    params = [param for param in model.parameters() if param.requires_grad]
    if use_bnb_8bit:
        try:
            import bitsandbytes as bnb
        except Exception as exc:
            raise ImportError("bitsandbytes is not available but --use_bnb_8bit=1 was set") from exc
        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def make_dataset(name: str, tokenizer, args, split: str, max_samples: Optional[int]):
    use_label = bool(args.use_label)

    if name == "piqa":
        if args.piqa_local_dir:
            return load_and_pack_piqa_local(
                tokenizer=tokenizer,
                block_size=args.block_size,
                piqa_local_dir=args.piqa_local_dir,
                split=split,
                max_samples=max_samples,
                use_label=use_label,
            )
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


def build_train_eval_datasets(tokenizer, args):
    if args.dataset == "mix":
        names = [x.strip() for x in args.mix_datasets.split(",") if x.strip()]
        train_ds = concatenate_datasets(
            [make_dataset(name, tokenizer, args, args.train_split, args.train_max_samples) for name in names]
        )
    else:
        train_ds = make_dataset(args.dataset, tokenizer, args, args.train_split, args.train_max_samples)

    eval_ds = make_dataset(args.eval_dataset, tokenizer, args, args.eval_split, args.eval_max_samples)
    return train_ds, eval_ds


def build_dataloaders(train_ds, eval_ds, args, world_size: int, is_distributed: bool):
    collator = LMDataCollator(pad_id=0)
    num_workers = max(0, int(args.dataloader_num_workers))
    if is_distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
        )
        eval_sampler = DistributedSampler(
            eval_ds,
            num_replicas=world_size,
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False,
        )
    else:
        train_sampler = None
        eval_sampler = None

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        sampler=eval_sampler,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    return train_dl, eval_dl


def _save_quantized_model(model: nn.Module, output_dir: str, tokenizer, config) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    model_config = getattr(model, "config", None)
    if model_config is not None:
        model_config.save_pretrained(output_dir)
    else:
        config.save_pretrained(output_dir)


def save_merged_model_for_inference(
    model: nn.Module,
    output_dir: str,
    tokenizer,
    config,
    args,
    device: torch.device,
) -> None:
    model_to_save = model.module if hasattr(model, "module") else model
    was_quantized = is_quantized_model(model_to_save)

    try:
        merged = model_to_save.merge_and_unload()
    except Exception as exc:
        raise RuntimeError(
            "Failed to merge LoRA adapters into the base model. "
            "Your PEFT model may not support merge_and_unload()."
        ) from exc

    if was_quantized and is_quantized_model(merged):
        print(f"[save] merged full model keeps quantized weights -> {output_dir}", flush=True)
        _save_quantized_model(merged, output_dir, tokenizer, config)
        return

    if not was_quantized:
        print(f"[save] merged full model is dense -> {output_dir}", flush=True)
        os.makedirs(output_dir, exist_ok=True)
        merged.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        merged.config.save_pretrained(output_dir)
        return

    if device.type != "cuda":
        raise RuntimeError("Re-quantizing the merged model for export requires CUDA.")

    print("[save] merged model is no longer quantized after merge; re-quantizing for export", flush=True)
    tmp_dir = tempfile.mkdtemp(prefix="qw-ori-merged-", dir=str(Path(output_dir).resolve().parent))
    try:
        merged = merged.to(device="cpu", dtype=torch.float16)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        merged.save_pretrained(tmp_dir, safe_serialization=True)
        tokenizer.save_pretrained(tmp_dir)
        merged.config.save_pretrained(tmp_dir)

        del merged
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        quant_config = build_quantization_config(args)
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        requantized = Qwen2MoeForCausalLM.from_pretrained(
            tmp_dir,
            config=Qwen2MoeConfig.from_pretrained(tmp_dir),
            quantization_config=quant_config,
            device_map={"": device_index},
            low_cpu_mem_usage=True,
        )
        print(f"[save] requantized merged full model -> {output_dir}", flush=True)
        _save_quantized_model(requantized, output_dir, tokenizer, config)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def train(
    model: nn.Module,
    train_dl: DataLoader,
    eval_dl: Optional[DataLoader],
    device: torch.device,
    output_dir: str,
    tokenizer,
    config,
    args,
    fp16: bool,
    bf16: bool,
):
    os.makedirs(output_dir, exist_ok=True)

    optimizer = build_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_bnb_8bit=bool(args.use_bnb_8bit),
    )

    steps_per_epoch = math.ceil(len(train_dl) / max(1, args.grad_accum))
    total_optim_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_optim_steps * args.warmup_ratio)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_optim_steps - warmup_steps))
        return max(args.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device.type == "cuda"))
    amp_dtype = torch.float16 if fp16 else (torch.bfloat16 if bf16 else None)

    ma_win = 50
    loss_hist = deque(maxlen=ma_win)
    ema_loss = None
    ema_momentum = 0.98

    global_step = 0
    optim_step = 0
    t0 = time.time()

    pbar = tqdm(
        total=total_optim_steps,
        disable=not is_main_process(),
        dynamic_ncols=True,
        desc="train",
    )
    model.train()

    for epoch in range(args.epochs):
        if isinstance(train_dl.sampler, DistributedSampler):
            train_dl.sampler.set_epoch(epoch)

        iterator = enumerate(train_dl)
        if is_main_process():
            iterator = tqdm(
                iterator,
                total=len(train_dl),
                desc=f"epoch {epoch + 1}/{args.epochs}",
                dynamic_ncols=True,
            )

        for _step, batch in iterator:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            if amp_dtype is not None and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(**batch)
                    loss = out.loss / max(1, args.grad_accum)
            else:
                out = model(**batch)
                loss = out.loss / max(1, args.grad_accum)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            global_step += 1
            if global_step % args.grad_accum != 0:
                continue

            if args.max_grad_norm > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optim_step += 1

            loss_real = float(loss.detach().float().item() * args.grad_accum)
            loss_hist.append(loss_real)
            loss_ma = sum(loss_hist) / max(1, len(loss_hist))
            if ema_loss is None:
                ema_loss = loss_real
            else:
                ema_loss = ema_momentum * ema_loss + (1.0 - ema_momentum) * loss_real

            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": f"{loss_real:.4f}",
                    "ma50": f"{loss_ma:.4f}",
                    "ema": f"{ema_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.3e}",
                },
                refresh=False,
            )

            if is_main_process() and (optim_step % max(1, args.log_every) == 0):
                elapsed = time.time() - t0
                aux_loss = getattr(out, "aux_loss", None)
                aux_cell = ""
                if aux_loss is not None:
                    aux_cell = f" aux={float(aux_loss.detach().float().item()):.4f}"
                print(
                    f"[train] epoch={epoch + 1}/{args.epochs} step={optim_step}/{total_optim_steps} "
                    f"loss={loss_real:.4f} ma50={loss_ma:.4f} ema={ema_loss:.4f}{aux_cell} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} elapsed={elapsed / 60:.1f}m",
                    flush=True,
                )

            if eval_dl is not None and (optim_step % max(1, args.eval_every) == 0):
                eval_loss = evaluate(model, eval_dl, device, fp16=fp16, bf16=bf16)
                if is_main_process():
                    print(
                        f"[eval] step={optim_step} loss={eval_loss:.4f} "
                        f"ppl={math.exp(min(20.0, eval_loss)):.2f}",
                        flush=True,
                    )

    pbar.close()

    optimizer.zero_grad(set_to_none=True)
    del optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if is_main_process():
        save_merged_model_for_inference(
            model=model,
            output_dir=output_dir,
            tokenizer=tokenizer,
            config=config,
            args=args,
            device=device,
        )


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
    ap.add_argument("--piqa_local_dir", type=str, default="")
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--eval_split", type=str, default="validation")
    ap.add_argument("--block_size", type=int, default=192)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--min_lr_ratio", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--num_proc", type=int, default=8)
    ap.add_argument("--dataloader_num_workers", type=int, default=0)
    ap.add_argument("--train_max_samples", type=int, default=None)
    ap.add_argument("--eval_max_samples", type=int, default=20)
    ap.add_argument("--use_label", type=int, default=1)

    ap.add_argument("--load_in_8bit", type=int, default=1)
    ap.add_argument("--llm_int8_threshold", type=float, default=6.0)
    ap.add_argument("--gradient_checkpointing", type=int, default=0)
    ap.add_argument("--fp16", type=int, default=0)
    ap.add_argument("--bf16", type=int, default=0)
    ap.add_argument("--use_bnb_8bit", type=int, default=0)

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,gate",
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--router_aux_loss_coef", type=float, default=0.001)
    ap.add_argument("--output_router_logits", type=int, default=1)
    ap.add_argument(
        "--router_topk",
        type=int,
        default=0,
        help="If > 0, override config.num_experts_per_tok for qw_ori MoE.",
    )

    ap.add_argument("--bbh_task", type=str, default="boolean_expressions")
    ap.add_argument("--winogrande_config", type=str, default="winogrande_xl")
    ap.add_argument("--mmlu_subjects", type=str, default="all")
    ap.add_argument("--mmlu_answer_mode", type=str, default="text", choices=["text", "letter"])
    return ap.parse_args()


def main():
    args = parse_args()
    local_rank, world_size, is_distributed = setup_distributed_safe()
    set_seed(args.seed + (local_rank if is_distributed else 0))

    device = torch.device(f"cuda:{local_rank}") if is_distributed else torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    use_fp16 = bool(args.fp16) and device.type == "cuda" and not bool(args.bf16)
    use_bf16 = bool(args.bf16) and device.type == "cuda"

    if is_main_process():
        print("[load] tokenizer from model_path:", args.model_path, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    config = Qwen2MoeConfig.from_pretrained(args.model_path)
    config.output_router_logits = bool(args.output_router_logits)
    config.router_aux_loss_coef = float(args.router_aux_loss_coef)
    if int(args.router_topk) > 0:
        config.num_experts_per_tok = int(args.router_topk)

    quantization_config = build_quantization_config(args)
    model_kwargs = {
        "config": config,
        "low_cpu_mem_usage": True,
    }
    if quantization_config is not None:
        if device.type != "cuda":
            raise RuntimeError("bitsandbytes 8-bit loading requires CUDA.")
        device_index = local_rank if is_distributed else 0
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = {"": device_index}
    else:
        model_kwargs["torch_dtype"] = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else torch.float32)

    if is_main_process():
        print(
            f"[load] distributed={is_distributed} world_size={world_size} "
            f"8bit={bool(args.load_in_8bit)} fp16={use_fp16} bf16={use_bf16}",
            flush=True,
        )

    model = Qwen2MoeForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    if quantization_config is None:
        model.to(device)

    model = maybe_prepare_kbit_model_for_training(model, args)
    targets = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    model = apply_lora(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=targets,
    )

    if is_main_process():
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total = sum(param.numel() for param in model.parameters())
        print(f"[lora] trainable params: {trainable}/{total} ({trainable / total * 100:.2f}%)", flush=True)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    train_ds, eval_ds = build_train_eval_datasets(tokenizer, args)
    if is_main_process():
        print(f"[data] train={len(train_ds)} eval={len(eval_ds)} block_size={args.block_size}", flush=True)
    train_dl, eval_dl = build_dataloaders(train_ds, eval_ds, args, world_size, is_distributed)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    train(
        model=model,
        train_dl=train_dl,
        eval_dl=eval_dl,
        device=device,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        config=config,
        args=args,
        fp16=use_fp16,
        bf16=use_bf16,
    )

    cleanup_distributed()


if __name__ == "__main__":
    main()
