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
- Saves only the merged full model for standalone inference.
"""
# TODO 加两阶段训练
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

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
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()

    local_rank, world_size, is_distributed = setup_distributed()
    set_seed(args.seed + (local_rank if is_distributed else 0))

    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    MoEForCausalLM, MoEConfig = import_yuan_classes()

    config = MoEConfig.from_pretrained(args.model_path)
    config.router_topk = 2
    config.experts_topk = 2
    config.norm_topk_prob = True

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if is_main_process():
        print(f"[load] tokenizer/model from {args.model_path}")
        print(f"[load] dtype={dtype} distributed={is_distributed} world_size={world_size}")

    model, loading_info = load_yuan_model(
        MoEForCausalLM,
        args.model_path,
        config,
        dtype,
    )
    summarize_loading_info(loading_info)
    model.to(device)

    targets = build_lora_targets(model, args.lora_target_modules)
    if is_main_process():
        print(f"[lora] target_modules={targets}")
    model = apply_lora(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=targets,
    )

    unfrozen = []
    if bool(args.train_router_qkv):
        unfrozen = unfreeze_router_qkv_weights(model)
        if is_main_process():
            print(f"[router] unfroze {len(unfrozen)} router base weights")
            for name in unfrozen[:12]:
                print(f"  - {name}")

    fp32_extra = cast_extra_trainable_params_to_fp32(model)
    if is_main_process():
        print(f"[dtype] cast {len(fp32_extra)} non-LoRA trainable params to fp32")
        for name in fp32_extra[:12]:
            print(f"  - {name}")

    if is_main_process():
        trainable = 0
        total = 0
        for param in model.parameters():
            count = param.numel()
            total += count
            if param.requires_grad:
                trainable += count
        print(f"[trainable] {trainable}/{total} ({100.0 * trainable / max(1, total):.2f}%)")
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    train_ds = make_dataset(args, tokenizer, args.dataset, args.train_split, args.train_max_samples)
    eval_ds = make_dataset(args, tokenizer, args.eval_dataset, args.eval_split, args.eval_max_samples)

    if is_main_process():
        print(f"[data] train={len(train_ds)} eval={len(eval_ds)} block_size={args.block_size}")

    collator = LMDataCollator(pad_id=0)

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
        shuffle=(train_sampler is None),
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
        device=device,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_accum=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        fp16=(device.type == "cuda"),
        bf16=False,
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
