#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA fine-tuning entrypoint for qwen_moe.modeling.modeling_moe_yuan.

- Reuses the OpenCompass-style dataset packing and training loop from
  `fintune_baseline.py`.
- Loads old checkpoints with `strict=False`-style behavior via Hugging Face
  `from_pretrained`, so missing attention-router weights are randomly
  initialized.
- Forces Yuan MoE routing to fixed top-2 attention gating.
- Optional two-stage workflow:
  stage1 trains randomly initialized Yuan attention-router query/key/value
  weights against legacy dense-router logits, then saves ckpt_after_stage1;
  stage2 loads that checkpoint and runs LoRA fine-tuning.
- Saves only the merged full model for standalone inference.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer

_BASELINE_IMPORTED = False
_baseline_candidates = [
    os.path.dirname(os.path.abspath(__file__)),
    "/home/cyx/Predict_MoE",
]
for _candidate in _baseline_candidates:
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)
    try:
        from fintune_baseline import (
            LMDataCollator,
            apply_lora,
            cleanup_distributed,
            guess_lora_targets,
            is_main_process,
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
            setup_distributed,
            train,
        )
        _BASELINE_IMPORTED = True
        break
    except ImportError:
        continue

if not _BASELINE_IMPORTED:
    raise ImportError(
        "Failed to import fintune_baseline. Checked the current directory and /home/cyx/Predict_MoE."
    )


DATASET_CHOICES = [
    "piqa",
    "siqa",
    "hellaswag",
    "arc-e",
    "arc-c",
    "csqa",
    "bbh",
    "winogrande",
    "mmlu",
    "openbookqa",
]


def import_yuan_classes() -> Tuple[type, type]:
    tried = []
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []

    env_root = os.environ.get("QWEN_MOE_REPO")
    if env_root:
        candidates.append(env_root)
    candidates.extend(
        [
            os.path.abspath(os.path.join(here, "..", "qwen_moe")),
            os.path.abspath(os.path.join(os.getcwd(), "qwen_moe")),
            os.getcwd(),
        ]
    )

    seen = set()
    for root in candidates:
        if not root or root in seen:
            continue
        seen.add(root)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from qwen_moe.modeling.configuration_moe_yuan import MoEConfig
            from qwen_moe.modeling.modeling_moe_yuan import MoEForCausalLM

            return MoEForCausalLM, MoEConfig
        except Exception as exc:
            tried.append((root, repr(exc)))

    msg = "Failed to import Yuan MoE classes. Tried repo roots:\n"
    msg += "\n".join([f"- {root}: {err}" for root, err in tried])
    raise ImportError(msg)


def build_lora_targets(model: nn.Module, cli_targets: str) -> List[str]:
    if cli_targets.strip():
        return sorted({t.strip() for t in cli_targets.split(",") if t.strip()})

    targets = set(guess_lora_targets(model))
    leaf_linear_names = {
        name.split(".")[-1]
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }
    for extra in ("query", "key", "value"):
        if extra in leaf_linear_names:
            targets.add(extra)
    return sorted(targets)


def unfreeze_router_qkv_weights(model: nn.Module) -> List[str]:
    changed = []
    patterns = (
        ".router.query.weight",
        ".router.key.weight",
        ".router.value.weight",
        ".router.query.base_layer.weight",
        ".router.key.base_layer.weight",
        ".router.value.base_layer.weight",
    )
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            param.requires_grad = True
            changed.append(name)
    return changed


def cast_extra_trainable_params_to_fp32(model: nn.Module) -> List[str]:
    changed = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            continue
        if param.dtype in (torch.float16, torch.bfloat16):
            param.data = param.data.float()
            changed.append(name)
    return changed


def summarize_loading_info(loading_info: Optional[dict]):
    if not loading_info or not is_main_process():
        return

    missing = loading_info.get("missing_keys", [])
    unexpected = loading_info.get("unexpected_keys", [])
    mismatched = loading_info.get("mismatched_keys", [])
    router_missing = [k for k in missing if ".router." in k]
    router_unexpected = [k for k in unexpected if ".router" in k]

    print(
        f"[load] missing={len(missing)} unexpected={len(unexpected)} mismatched={len(mismatched)}"
    )
    if router_missing:
        print("[load] router missing keys (expected with old checkpoint):")
        for key in router_missing[:12]:
            print(f"  - {key}")
    if router_unexpected:
        print("[load] old router keys ignored from checkpoint:")
        for key in router_unexpected[:12]:
            print(f"  - {key}")


def load_yuan_model(MoEForCausalLM, model_path: str, config, dtype):
    attempts = [
        {"output_loading_info": True, "ignore_mismatched_sizes": True},
        {"ignore_mismatched_sizes": True},
        {},
    ]
    last_error = None

    for extra_kwargs in attempts:
        try:
            loaded = MoEForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                **extra_kwargs,
            )
            if isinstance(loaded, tuple):
                model, loading_info = loaded
            else:
                model, loading_info = loaded, None
            return model, loading_info
        except TypeError as exc:
            last_error = exc
            continue

    raise last_error


def _unwrap_base_model(model: nn.Module) -> nn.Module:
    model_for_ops = model.module if hasattr(model, "module") else model
    return model_for_ops.get_base_model() if hasattr(model_for_ops, "get_base_model") else model_for_ops


def configure_yuan_config(config) -> None:
    config.router_topk = 2
    config.experts_topk = 2
    config.norm_topk_prob = True


def get_switch_layers(model: nn.Module) -> List[tuple[int, nn.Module]]:
    base_model = _unwrap_base_model(model)
    moe_model = getattr(base_model, "model", None)
    layers = getattr(moe_model, "layers", None)
    if layers is None:
        raise AttributeError("Cannot locate decoder layers on current Yuan model.")

    out = []
    for layer_idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and getattr(mlp, "use_switch", False):
            out.append((layer_idx, mlp))
    return out


def freeze_all_params(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def enable_yuan_router_only(model: nn.Module) -> int:
    n_params = 0
    for _layer_idx, mlp in get_switch_layers(model):
        router = getattr(mlp, "router", None)
        if router is None:
            continue
        for name, param in router.named_parameters():
            if not any(key in name for key in ("query", "key", "value")):
                continue
            param.requires_grad = True
            n_params += param.numel()
    return n_params


def prepare_stage1_router_trainables(model: nn.Module) -> int:
    freeze_all_params(model)
    n_router_params = enable_yuan_router_only(model)
    if n_router_params == 0:
        raise RuntimeError("Stage1 found no trainable Yuan attention-router parameters.")
    return n_router_params


def build_stage1_subset_dataset(ds, stage1_ratio: float, seed: int):
    if not (0.0 < float(stage1_ratio) < 1.0):
        raise ValueError(f"stage1_ratio must be in (0, 1), got {stage1_ratio}")

    n_total = len(ds)
    if n_total <= 1:
        return ds

    ds_shuf = ds.shuffle(seed=int(seed))
    n_stage1 = int(n_total * float(stage1_ratio))
    n_stage1 = max(1, min(n_stage1, n_total - 1))
    return ds_shuf.select(range(0, n_stage1))


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


def _align_legacy_router_weight(weight: torch.Tensor, hidden_size: int, num_experts: int) -> torch.Tensor:
    w = weight.detach().float()
    if w.shape == (num_experts, hidden_size):
        return w
    if w.shape == (hidden_size, num_experts):
        return w.transpose(0, 1).contiguous()

    candidates = (w, w.transpose(0, 1))

    def _score(candidate: torch.Tensor) -> tuple[int, int, int]:
        return (
            int(candidate.shape == (num_experts, hidden_size)),
            int(candidate.shape[0] == num_experts) + int(candidate.shape[1] == hidden_size),
            min(candidate.shape[0], num_experts) * min(candidate.shape[1], hidden_size),
        )

    best = max(candidates, key=_score)
    aligned = best.new_zeros((num_experts, hidden_size))
    rows = min(best.shape[0], num_experts)
    cols = min(best.shape[1], hidden_size)
    aligned[:rows, :cols] = best[:rows, :cols]
    return aligned


def load_legacy_router_teacher_weights(
    teacher_model_path: str,
    switch_layers: List[tuple[int, nn.Module]],
) -> Dict[int, torch.Tensor]:
    legacy_sd = _load_local_checkpoint_state_dict(teacher_model_path)
    teacher_weights: Dict[int, torch.Tensor] = {}

    for layer_idx, mlp in switch_layers:
        key = f"model.layers.{layer_idx}.mlp.router.weight"
        weight = legacy_sd.get(key)
        if weight is None:
            continue
        router = getattr(mlp, "router", None)
        if router is None:
            continue
        teacher_weights[layer_idx] = _align_legacy_router_weight(
            weight=weight,
            hidden_size=int(getattr(router, "hidden_size")),
            num_experts=int(getattr(mlp, "num_experts")),
        ).cpu()

    del legacy_sd
    gc.collect()
    return teacher_weights


def save_full_model_checkpoint(model: nn.Module, output_dir: str, tokenizer, config) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = _unwrap_base_model(model)

    state_dict = {}
    for name, tensor in model_to_save.state_dict().items():
        value = tensor.detach().cpu()
        if torch.is_floating_point(value) and value.dtype != torch.float32:
            value = value.to(torch.float32)
        state_dict[name] = value

    model_to_save.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)


def make_dataset(args, tokenizer, dataset_name: str, split: str, max_samples: Optional[int]):
    common_kwargs = dict(
        tokenizer=tokenizer,
        block_size=args.block_size,
        split=split,
        num_proc=args.num_proc,
        bos=True,
        eos=False,
        max_samples=max_samples,
    )

    if dataset_name == "piqa":
        return load_and_pack_piqa_ppl_opencompass(
            **common_kwargs,
            use_label=bool(args.use_label),
        )
    if dataset_name == "siqa":
        return load_and_pack_siqa_ppl_opencompass(
            **common_kwargs,
            use_label=bool(args.use_label),
        )
    if dataset_name == "hellaswag":
        return load_and_pack_hellaswag_ppl_opencompass(
            **common_kwargs,
            use_label=bool(args.use_label),
        )
    if dataset_name == "arc-e":
        return load_and_pack_arc_easy_ppl_opencompass(
            **common_kwargs,
            use_label=bool(args.use_label),
        )
    if dataset_name == "arc-c":
        return load_and_pack_arc_challenge_ppl_opencompass(
            **common_kwargs,
            use_label=bool(args.use_label),
        )
    if dataset_name == "csqa":
        return load_and_pack_commonsenseqa_ppl_opencompass(
            **common_kwargs,
            use_label=bool(args.use_label),
        )
    if dataset_name == "bbh":
        return load_and_pack_bbh_ppl_opencompass(
            **common_kwargs,
            task=args.bbh_task,
        )
    if dataset_name == "winogrande":
        return load_and_pack_winogrande_ppl_opencompass(
            **common_kwargs,
            config_name=args.winogrande_config,
            use_label=bool(args.use_label),
        )
    if dataset_name == "mmlu":
        subjects = None
        if args.mmlu_subjects and args.mmlu_subjects.strip().lower() != "all":
            subjects = [item.strip() for item in args.mmlu_subjects.split(",") if item.strip()]
        return load_and_pack_mmlu_ppl_opencompass(
            **common_kwargs,
            subjects=subjects,
            answer_mode=args.mmlu_answer_mode,
        )
    if dataset_name == "openbookqa":
        return load_and_pack_openbookqa_ppl_opencompass(
            **common_kwargs,
            use_label=bool(args.use_label),
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")


def build_train_eval_datasets(tokenizer, args, *, stage: str):
    train_ds = make_dataset(args, tokenizer, args.dataset, args.train_split, args.train_max_samples)
    if stage == "stage1":
        train_ds = build_stage1_subset_dataset(
            train_ds,
            stage1_ratio=float(args.stage1_data_ratio),
            seed=int(args.stage_split_seed),
        )
        return train_ds, None

    eval_ds = make_dataset(args, tokenizer, args.eval_dataset, args.eval_split, args.eval_max_samples)
    return train_ds, eval_ds


def build_dataloaders(train_ds, eval_ds, args, world_size: int, is_distributed: bool):
    collator = LMDataCollator(pad_id=0)

    if is_distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
        )
        eval_sampler = (
            DistributedSampler(
                eval_ds,
                num_replicas=world_size,
                rank=dist.get_rank(),
                shuffle=False,
                drop_last=False,
            )
            if eval_ds is not None
            else None
        )
    else:
        train_sampler = None
        eval_sampler = None

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
    )
    eval_dl = None
    if eval_ds is not None:
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
    return train_dl, eval_dl


def stage1_train_yuan_router(
    model: nn.Module,
    train_dl: DataLoader,
    run_args: argparse.Namespace,
    device: torch.device,
    teacher_router_weights: Dict[int, torch.Tensor],
):
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("Stage1 found no parameters with requires_grad=True before optimizer creation.")
    if not teacher_router_weights:
        raise RuntimeError(
            "Stage1 found no legacy dense router weights. "
            "Use --stage2_init_path or --stage 2 if the source checkpoint is already Yuan-style."
        )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(run_args.stage1_lr),
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    switch_layers = get_switch_layers(model)
    if not switch_layers:
        raise RuntimeError("Stage1 found no switch layers.")

    captured: List[tuple[int, nn.Module, torch.Tensor]] = []
    hook_handles = []

    def _make_pre_hook(layer_idx: int):
        def _hook(_module: nn.Module, inputs):
            captured.append((layer_idx, _module, inputs[0].detach()))
        return _hook

    model.eval()
    for layer_idx, mlp in switch_layers:
        if layer_idx not in teacher_router_weights:
            continue
        hook_handles.append(mlp.register_forward_pre_hook(_make_pre_hook(layer_idx)))
        mlp.router.train(True)

    global_step = 0
    optim_step = 0
    t0 = time.time()
    stage1_metrics_f = None

    def _normalize_logits(logits: torch.Tensor) -> torch.Tensor:
        mean = logits.mean(dim=-1, keepdim=True)
        std = logits.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (logits - mean) / std

    if is_main_process():
        records_dir = Path(__file__).resolve().parent / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        run_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        stage1_metrics_path = records_dir / f"{run_ts}_yuan_stage1_router.tsv"
        stage1_metrics_f = open(stage1_metrics_path, "w", encoding="utf-8")
        stage1_metrics_f.write(f"# run_started_at\t{run_ts}\n")
        for key, value in sorted(vars(run_args).items()):
            stage1_metrics_f.write("\t".join([
                "# arg",
                key,
                json.dumps(value, ensure_ascii=False, default=str),
            ]) + "\n")
        stage1_metrics_f.write("\t".join([
            "time", "epoch", "global_step", "optim_step", "layer_idx",
            "layer_loss", "kl_loss", "logit_loss",
            "teacher_entropy", "student_entropy",
            "top1_agreement", "topk_overlap", "logits_cosine",
        ]) + "\n")
        stage1_metrics_f.flush()

    try:
        for epoch in range(int(run_args.stage1_epochs)):
            if isinstance(train_dl.sampler, DistributedSampler):
                train_dl.sampler.set_epoch(epoch)

            iterator = enumerate(train_dl)
            if is_main_process():
                iterator = tqdm(
                    iterator,
                    total=len(train_dl),
                    desc=f"stage1 {epoch + 1}/{int(run_args.stage1_epochs)}",
                    dynamic_ncols=True,
                )

            optimizer.zero_grad(set_to_none=True)

            for _step, batch in iterator:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                input_ids = batch["input_ids"]
                valid_mask = input_ids.ne(0)

                captured.clear()
                with torch.no_grad():
                    _ = model(**batch)

                total_loss = input_ids.new_zeros((), dtype=torch.float32)
                kl_acc = input_ids.new_zeros((), dtype=torch.float32)
                logit_acc = input_ids.new_zeros((), dtype=torch.float32)
                teacher_entropy_acc = input_ids.new_zeros((), dtype=torch.float32)
                student_entropy_acc = input_ids.new_zeros((), dtype=torch.float32)
                top1_agreement_acc = input_ids.new_zeros((), dtype=torch.float32)
                topk_overlap_acc = input_ids.new_zeros((), dtype=torch.float32)
                logits_cosine_acc = input_ids.new_zeros((), dtype=torch.float32)
                layer_metric_rows = []
                layer_count = 0

                for layer_idx, mlp, hidden_states in captured:
                    teacher_weight_cpu = teacher_router_weights.get(layer_idx)
                    if teacher_weight_cpu is None:
                        continue

                    teacher_weight = teacher_weight_cpu.to(device=device, dtype=hidden_states.dtype)
                    teacher_logits = F.linear(hidden_states.to(dtype=teacher_weight.dtype), teacher_weight).float()
                    student_logits = mlp.router(hidden_states).view(
                        hidden_states.shape[0],
                        hidden_states.shape[1],
                        -1,
                    ).float()

                    mask = valid_mask
                    if teacher_logits.shape[:2] != mask.shape:
                        raise RuntimeError(
                            f"Stage1 mask shape mismatch at layer {layer_idx}: "
                            f"logits={tuple(teacher_logits.shape)} mask={tuple(mask.shape)}"
                        )

                    temp = float(max(run_args.stage1_distill_temperature, 1e-6))
                    teacher_probs = torch.softmax(teacher_logits / temp, dim=-1)
                    student_log_probs = torch.log_softmax(student_logits / temp, dim=-1)
                    kl_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1) * (temp * temp)

                    if bool(run_args.stage1_use_logit_loss):
                        teacher_norm = _normalize_logits(teacher_logits)
                        student_norm = _normalize_logits(student_logits)
                        if str(run_args.stage1_logit_loss_type).lower() == "mse":
                            logit_token = (student_norm - teacher_norm).pow(2).mean(dim=-1)
                        else:
                            logit_token = F.smooth_l1_loss(student_norm, teacher_norm, reduction="none").mean(dim=-1)
                    else:
                        logit_token = torch.zeros_like(kl_token)

                    mask_f = mask.to(dtype=kl_token.dtype)
                    denom = mask_f.sum().clamp_min(1.0)
                    kl_loss = (kl_token * mask_f).sum() / denom
                    logit_loss = (logit_token * mask_f).sum() / denom
                    teacher_probs_raw = torch.softmax(teacher_logits, dim=-1)
                    student_probs_raw = torch.softmax(student_logits, dim=-1)
                    teacher_entropy_token = -(
                        teacher_probs_raw.clamp_min(1e-9) * teacher_probs_raw.clamp_min(1e-9).log()
                    ).sum(dim=-1)
                    student_entropy_token = -(
                        student_probs_raw.clamp_min(1e-9) * student_probs_raw.clamp_min(1e-9).log()
                    ).sum(dim=-1)
                    teacher_top1 = teacher_logits.argmax(dim=-1)
                    student_top1 = student_logits.argmax(dim=-1)
                    top1_agreement_token = teacher_top1.eq(student_top1).to(mask_f.dtype)
                    topk_k = min(max(1, int(getattr(run_args, "router_topk", 2))), teacher_logits.size(-1))
                    teacher_topk = torch.topk(teacher_logits, k=topk_k, dim=-1).indices
                    student_topk = torch.topk(student_logits, k=topk_k, dim=-1).indices
                    topk_overlap_token = (
                        teacher_topk.unsqueeze(-1) == student_topk.unsqueeze(-2)
                    ).any(dim=-1).to(mask_f.dtype).mean(dim=-1)
                    logits_cosine_token = F.cosine_similarity(teacher_logits, student_logits, dim=-1)
                    teacher_entropy = (teacher_entropy_token * mask_f).sum() / denom
                    student_entropy = (student_entropy_token * mask_f).sum() / denom
                    top1_agreement = (top1_agreement_token * mask_f).sum() / denom
                    topk_overlap = (topk_overlap_token * mask_f).sum() / denom
                    logits_cosine = (logits_cosine_token * mask_f).sum() / denom
                    layer_loss = (
                        float(run_args.stage1_kl_coef) * kl_loss
                        + float(run_args.stage1_logit_coef) * logit_loss
                    )

                    total_loss = total_loss + layer_loss
                    kl_acc = kl_acc + kl_loss.detach()
                    logit_acc = logit_acc + logit_loss.detach()
                    teacher_entropy_acc = teacher_entropy_acc + teacher_entropy.detach()
                    student_entropy_acc = student_entropy_acc + student_entropy.detach()
                    top1_agreement_acc = top1_agreement_acc + top1_agreement.detach()
                    topk_overlap_acc = topk_overlap_acc + topk_overlap.detach()
                    logits_cosine_acc = logits_cosine_acc + logits_cosine.detach()
                    layer_metric_rows.append({
                        "layer_idx": layer_idx,
                        "layer_loss": float(layer_loss.detach().item()),
                        "kl_loss": float(kl_loss.detach().item()),
                        "logit_loss": float(logit_loss.detach().item()),
                        "teacher_entropy": float(teacher_entropy.detach().item()),
                        "student_entropy": float(student_entropy.detach().item()),
                        "top1_agreement": float(top1_agreement.detach().item()),
                        "topk_overlap": float(topk_overlap.detach().item()),
                        "logits_cosine": float(logits_cosine.detach().item()),
                    })
                    layer_count += 1

                if layer_count == 0:
                    continue

                total_loss = total_loss / float(layer_count)
                total_loss_to_backprop = total_loss / max(1, int(run_args.stage1_grad_accum))
                total_loss_to_backprop.backward()
                global_step += 1

                if global_step % int(run_args.stage1_grad_accum) != 0:
                    continue

                if float(run_args.stage1_max_grad_norm) > 0:
                    nn.utils.clip_grad_norm_(trainable, float(run_args.stage1_max_grad_norm))

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optim_step += 1

                mean_kl = kl_acc / float(layer_count)
                mean_logit = logit_acc / float(layer_count)
                mean_teacher_entropy = teacher_entropy_acc / float(layer_count)
                mean_student_entropy = student_entropy_acc / float(layer_count)
                mean_top1_agreement = top1_agreement_acc / float(layer_count)
                mean_topk_overlap = topk_overlap_acc / float(layer_count)
                mean_logits_cosine = logits_cosine_acc / float(layer_count)

                if dist.is_initialized():
                    for tensor in (
                        total_loss,
                        mean_kl,
                        mean_logit,
                        mean_teacher_entropy,
                        mean_student_entropy,
                        mean_top1_agreement,
                        mean_topk_overlap,
                        mean_logits_cosine,
                    ):
                        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)

                if is_main_process() and (optim_step % max(1, int(run_args.stage1_log_every)) == 0):
                    elapsed = time.time() - t0
                    print(
                        f"[stage1] epoch={epoch + 1}/{int(run_args.stage1_epochs)} "
                        f"step={optim_step} loss={float(total_loss.item()):.6f} "
                        f"kl={float(mean_kl.item()):.6f} "
                        f"logit={float(mean_logit.item()):.6f} "
                        f"t_ent={float(mean_teacher_entropy.item()):.4f} "
                        f"s_ent={float(mean_student_entropy.item()):.4f} "
                        f"top1={float(mean_top1_agreement.item()):.4f} "
                        f"topk={float(mean_topk_overlap.item()):.4f} "
                        f"cos={float(mean_logits_cosine.item()):.4f} "
                        f"elapsed={elapsed / 60:.1f}m"
                    )
                    if stage1_metrics_f is not None:
                        for row in layer_metric_rows:
                            stage1_metrics_f.write("\t".join([
                                f"{time.time():.3f}",
                                str(epoch),
                                str(global_step),
                                str(optim_step),
                                str(row["layer_idx"]),
                                f"{row['layer_loss']:.6f}",
                                f"{row['kl_loss']:.6f}",
                                f"{row['logit_loss']:.6f}",
                                f"{row['teacher_entropy']:.6f}",
                                f"{row['student_entropy']:.6f}",
                                f"{row['top1_agreement']:.6f}",
                                f"{row['topk_overlap']:.6f}",
                                f"{row['logits_cosine']:.6f}",
                            ]) + "\n")
                        stage1_metrics_f.flush()

            if global_step % int(run_args.stage1_grad_accum) != 0:
                if float(run_args.stage1_max_grad_norm) > 0:
                    nn.utils.clip_grad_norm_(trainable, float(run_args.stage1_max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optim_step += 1
    finally:
        for handle in hook_handles:
            handle.remove()
        if stage1_metrics_f is not None:
            stage1_metrics_f.close()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument(
        "--stage",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="0: run stage1+stage2, 1: run stage1 only, 2: run stage2 only",
    )
    ap.add_argument(
        "--stage2_init_path",
        type=str,
        default="",
        help="Optional initialization path for stage2. Defaults to output_dir/ckpt_after_stage1 when available.",
    )
    ap.add_argument(
        "--stage1_teacher_path",
        type=str,
        default="",
        help="Checkpoint path that provides legacy dense router weights for stage1 teacher. Defaults to model_path.",
    )

    ap.add_argument("--dataset", type=str, default="piqa", choices=DATASET_CHOICES)
    ap.add_argument("--eval_dataset", type=str, default="piqa", choices=DATASET_CHOICES)

    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--eval_split", type=str, default="validation")
    ap.add_argument("--train_max_samples", type=int, default=None)
    ap.add_argument("--eval_max_samples", type=int, default=20)
    ap.add_argument("--use_label", type=int, default=1)
    ap.add_argument("--num_proc", type=int, default=8)

    ap.add_argument("--block_size", type=int, default=192)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--use_bnb_8bit", type=int, default=0)
    ap.add_argument("--fp16", type=int, default=1)
    ap.add_argument("--bf16", type=int, default=0)

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="",
        help="Comma-separated module names. Empty means auto-detect + router query/key/value.",
    )
    ap.add_argument(
        "--train_router_qkv",
        type=int,
        default=1,
        help="Keep new attention router query/key/value base weights trainable.",
    )

    ap.add_argument("--bbh_task", type=str, default="boolean_expressions")
    ap.add_argument("--winogrande_config", type=str, default="winogrande_xl")
    ap.add_argument("--mmlu_subjects", type=str, default="all")
    ap.add_argument("--mmlu_answer_mode", type=str, default="text", choices=["text", "letter"])

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage_split_seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--stage1_epochs", type=int, default=1)
    ap.add_argument("--stage1_lr", type=float, default=2e-4)
    ap.add_argument("--stage1_grad_accum", type=int, default=1)
    ap.add_argument("--stage1_log_every", type=int, default=50)
    ap.add_argument("--stage1_max_grad_norm", type=float, default=1.0)
    ap.add_argument("--stage1_data_ratio", type=float, default=0.2)
    ap.add_argument("--stage1_distill_temperature", type=float, default=1.0)
    ap.add_argument("--stage1_kl_coef", type=float, default=1.0)
    ap.add_argument("--stage1_logit_coef", type=float, default=1.0)
    ap.add_argument("--stage1_use_logit_loss", type=int, default=1)
    ap.add_argument("--stage1_logit_loss_type", type=str, default="huber", choices=["huber", "mse"])
    return ap.parse_args()


def resolve_train_dtype(args: argparse.Namespace, device: torch.device) -> tuple[torch.dtype, bool, bool]:
    if device.type != "cuda":
        return torch.float32, False, False

    use_bf16 = bool(args.bf16)
    use_fp16 = bool(args.fp16)
    if use_bf16 and not torch.cuda.is_bf16_supported():
        if is_main_process():
            print("[precision] bf16 requested but unsupported on this GPU; falling back to fp16")
        use_bf16 = False
        use_fp16 = True
    if use_bf16:
        return torch.bfloat16, False, True
    if use_fp16:
        return torch.float16, True, False
    return torch.float32, False, False


def main():
    args = parse_args()
    if not (0.0 < float(args.stage1_data_ratio) < 1.0):
        raise ValueError(f"--stage1_data_ratio must be in (0, 1), got {args.stage1_data_ratio}")

    local_rank, world_size, is_distributed = setup_distributed()
    set_seed(args.seed + (local_rank if is_distributed else 0))

    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    MoEForCausalLM, MoEConfig = import_yuan_classes()

    dtype, use_fp16, use_bf16 = resolve_train_dtype(args, device)
    if is_main_process():
        print(f"[load] model_path={args.model_path}")
        print(f"[load] dtype={dtype} fp16={use_fp16} bf16={use_bf16} distributed={is_distributed} world_size={world_size}")

    stage1_ckpt_dir = os.path.join(args.output_dir, "ckpt_after_stage1")

    if args.stage in (0, 1):
        if is_main_process():
            print("[stage1] loading base Yuan model with randomly initialized new router params")

        stage1_config = MoEConfig.from_pretrained(args.model_path)
        configure_yuan_config(stage1_config)
        stage1_model, loading_info = load_yuan_model(
            MoEForCausalLM,
            args.model_path,
            stage1_config,
            dtype,
        )
        summarize_loading_info(loading_info)
        stage1_model.to(device)

        n_stage1_router_params = prepare_stage1_router_trainables(stage1_model)
        if is_main_process():
            print(f"[stage1] trainable Yuan attention-router params: {n_stage1_router_params}")

        stage1_train_ds, _ = build_train_eval_datasets(tokenizer, args, stage="stage1")
        stage1_train_dl, _ = build_dataloaders(stage1_train_ds, None, args, world_size, is_distributed)
        if is_main_process():
            print(
                f"[stage1][data] train={len(stage1_train_ds)} "
                f"(ratio={float(args.stage1_data_ratio):.3f}, seed={int(args.stage_split_seed)})"
            )

        stage1_switch_layers = get_switch_layers(stage1_model)
        teacher_path = args.stage1_teacher_path or args.model_path
        teacher_router_weights = load_legacy_router_teacher_weights(teacher_path, stage1_switch_layers)
        if is_main_process():
            print(f"[stage1] loaded teacher router weights for {len(teacher_router_weights)} layers from {teacher_path}")

        if is_distributed:
            stage1_model = torch.nn.parallel.DistributedDataParallel(
                stage1_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )

        stage1_train_yuan_router(
            model=stage1_model,
            train_dl=stage1_train_dl,
            run_args=args,
            device=device,
            teacher_router_weights=teacher_router_weights,
        )

        if is_main_process():
            print(f"[stage1] saving full checkpoint to {stage1_ckpt_dir}")
            save_full_model_checkpoint(stage1_model, stage1_ckpt_dir, tokenizer, stage1_config)
        if is_distributed:
            dist.barrier()

        del stage1_model, stage1_train_dl, stage1_train_ds, teacher_router_weights, stage1_switch_layers
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if args.stage == 1:
            cleanup_distributed()
            return

    if args.stage2_init_path:
        stage2_model_path = args.stage2_init_path
    elif os.path.isdir(stage1_ckpt_dir):
        stage2_model_path = stage1_ckpt_dir
    else:
        stage2_model_path = args.model_path

    if is_main_process():
        print(f"[stage2] loading model from {stage2_model_path}")

    stage2_config = MoEConfig.from_pretrained(stage2_model_path)
    configure_yuan_config(stage2_config)
    stage2_model, loading_info = load_yuan_model(
        MoEForCausalLM,
        stage2_model_path,
        stage2_config,
        dtype,
    )
    summarize_loading_info(loading_info)
    stage2_model.to(device)

    targets = build_lora_targets(stage2_model, args.lora_target_modules)
    if is_main_process():
        print(f"[stage2][lora] target_modules={targets}")
    stage2_model = apply_lora(
        stage2_model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=targets,
    )

    unfrozen = []
    if bool(args.train_router_qkv):
        unfrozen = unfreeze_router_qkv_weights(stage2_model)
        if is_main_process():
            print(f"[stage2][router] unfroze {len(unfrozen)} router base weights")
            for name in unfrozen[:12]:
                print(f"  - {name}")

    fp32_extra = cast_extra_trainable_params_to_fp32(stage2_model)
    if is_main_process():
        print(f"[stage2][dtype] cast {len(fp32_extra)} non-LoRA trainable params to fp32")
        for name in fp32_extra[:12]:
            print(f"  - {name}")

        trainable = 0
        total = 0
        for param in stage2_model.parameters():
            count = param.numel()
            total += count
            if param.requires_grad:
                trainable += count
        print(f"[stage2][trainable] {trainable}/{total} ({100.0 * trainable / max(1, total):.2f}%)")
        try:
            stage2_model.print_trainable_parameters()
        except Exception:
            pass

    stage2_train_ds, eval_ds = build_train_eval_datasets(tokenizer, args, stage="stage2")
    if is_main_process():
        print(f"[stage2][data] train={len(stage2_train_ds)} eval={len(eval_ds)} block_size={args.block_size}")
    train_dl, eval_dl = build_dataloaders(stage2_train_ds, eval_ds, args, world_size, is_distributed)

    if is_distributed:
        stage2_model = torch.nn.parallel.DistributedDataParallel(
            stage2_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    train(
        model=stage2_model,
        train_dl=train_dl,
        eval_dl=eval_dl,
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
        save_every=0,
    )

    if is_main_process():
        try:
            tokenizer.save_pretrained(args.output_dir)
        except Exception:
            pass

    cleanup_distributed()


if __name__ == "__main__":
    main()
