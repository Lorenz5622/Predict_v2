#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen1.5-MoE-2.7B LoRA finetuning script with end-to-end 4-bit workflow.

Compared to finetune_qwen1_5_moe_2_7b_4bit.py:
- Model is loaded in 4-bit via BitsAndBytesConfig.
- Training loop runs in full precision context (no fp16/bf16 autocast).
- Optimizer uses bitsandbytes 4-bit optimizer (AdamW4bit / PagedAdamW4bit).

Note:
- In QLoRA-style training, base weights are quantized (4-bit), while trainable LoRA
  parameters are still real-valued tensors. Here we additionally use a 4-bit optimizer
  path to keep the finetune pipeline in a strict "4-bit" setup.
"""
from __future__ import annotations

import argparse
import inspect
import math
import os
import time
from collections import deque
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import concatenate_datasets
from peft import prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from finetune_baseline_example import (
    LMDataCollator,
    apply_lora,
    cleanup_distributed,
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
            f"Invalid LOCAL_RANK={local_rank}; visible CUDA devices={n_visible}. "
            f"Set --nproc_per_node <= {n_visible}."
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    dist.barrier(device_ids=[local_rank])
    return local_rank, world_size, True


def import_qwen_moe_classes():
    from qwen_moe.modeling.configuration_moe import Qwen2MoeConfig
    from qwen_moe.modeling.modeling_moe import Qwen2MoeForCausalLM

    return Qwen2MoeForCausalLM, Qwen2MoeConfig


def build_optimizer_4bit(model: nn.Module, lr: float, weight_decay: float, optim_name: str):
    try:
        import bitsandbytes as bnb
    except Exception as e:
        raise ImportError("bitsandbytes is required for 4-bit optimizer training.") from e

    name = optim_name.lower()
    params = [p for p in model.parameters() if p.requires_grad]

    if name == "adamw4bit":
        if hasattr(bnb.optim, "AdamW4bit"):
            return bnb.optim.AdamW4bit(params, lr=lr, weight_decay=weight_decay)
        else:
            print("Warning: AdamW4bit not available, falling back to torch.optim.AdamW")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    if name == "paged_adamw4bit":
        if hasattr(bnb.optim, "PagedAdamW4bit"):
            return bnb.optim.PagedAdamW4bit(params, lr=lr, weight_decay=weight_decay)
        else:
            print("Warning: PagedAdamW4bit not available, falling back to torch.optim.AdamW")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    if name == "adamw8bit":
        if hasattr(bnb.optim, "AdamW8bit"):
            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
        else:
            print("Warning: AdamW8bit not available, falling back to torch.optim.AdamW")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unknown 4-bit optimizer: {optim_name}")


@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []

    for batch in dl:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        losses.append(out.loss.detach().float())

    if len(losses) == 0:
        return float("nan")

    loss = torch.stack(losses).mean()
    if dist.is_initialized():
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss = loss / dist.get_world_size()
    return float(loss.item())


def train_4bit(
    model: nn.Module,
    train_dl: DataLoader,
    eval_dl: Optional[DataLoader],
    device: torch.device,
    output_dir: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    grad_accum: int,
    max_grad_norm: float,
    log_every: int,
    eval_every: int,
    optim_4bit: str,
):
    os.makedirs(output_dir, exist_ok=True)
    optimizer = build_optimizer_4bit(model, lr=lr, weight_decay=weight_decay, optim_name=optim_4bit)

    steps_per_epoch = math.ceil(len(train_dl) / max(1, grad_accum))
    total_optim_steps = steps_per_epoch * epochs
    warmup_steps = int(total_optim_steps * warmup_ratio)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_optim_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    pbar = tqdm(total=total_optim_steps, disable=not is_main_process(), dynamic_ncols=True, desc="train")
    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    optim_step = 0
    t0 = time.time()

    ma_win = 50
    ma_loss_buf = deque(maxlen=ma_win)
    ma_loss_sum = 0.0

    for epoch in range(epochs):
        if isinstance(train_dl.sampler, DistributedSampler):
            train_dl.sampler.set_epoch(epoch)

        it = enumerate(train_dl)
        if is_main_process():
            it = tqdm(it, total=len(train_dl), desc=f"epoch {epoch + 1}/{epochs}", dynamic_ncols=True)

        for _, batch in it:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / max(1, grad_accum)
            loss.backward()
            global_step += 1

            if global_step % grad_accum != 0:
                continue

            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

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

            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss_real:.4f}", "ma50": f"{loss_ma:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"}, refresh=False)

            if is_main_process() and (optim_step % log_every == 0):
                elapsed = (time.time() - t0) / 60
                print(
                    f"[train-4bit] epoch={epoch + 1}/{epochs} step={optim_step}/{total_optim_steps} "
                    f"loss={loss_real:.4f} ma50={loss_ma:.4f} lr={scheduler.get_last_lr()[0]:.3e} elapsed={elapsed:.1f}m"
                )

            if eval_dl is not None and (optim_step % eval_every == 0):
                ev = evaluate(model, eval_dl, device)
                if is_main_process():
                    print(f"[eval] step={optim_step} loss={ev:.4f} ppl={math.exp(min(20, ev)):.2f}")

    pbar.close()

    if is_main_process():
        model_to_save = model.module if hasattr(model, "module") else model
        merged = model_to_save.merge_and_unload()
        merged.save_pretrained(output_dir)
        print(f"[save] merged full model -> {output_dir}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--dataset", type=str, default="piqa",
                    choices=["piqa", "siqa", "hellaswag", "arc-e", "csqa", "bbh", "winogrande", "mmlu", "arc-c", "openbookqa", "mix"])
    ap.add_argument("--eval_dataset", type=str, default="piqa",
                    choices=["piqa", "siqa", "hellaswag", "arc-e", "csqa", "bbh", "winogrande", "mmlu", "arc-c", "openbookqa"])
    ap.add_argument("--mix_datasets", type=str, default="piqa,siqa")

    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--eval_split", type=str, default="validation")

    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--num_proc", type=int, default=8)
    ap.add_argument("--train_max_samples", type=int, default=None)
    ap.add_argument("--eval_max_samples", type=int, default=200)
    ap.add_argument("--use_label", type=int, default=1)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", type=str, default="")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=1000)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_experts_per_tok", type=int, default=0)
    ap.add_argument("--router_topk", type=int, default=0)

    ap.add_argument("--bbh_task", type=str, default="boolean_expressions")
    ap.add_argument("--winogrande_config", type=str, default="winogrande_xl")
    ap.add_argument("--mmlu_subjects", type=str, default="all")
    ap.add_argument("--mmlu_answer_mode", type=str, default="text", choices=["text", "letter"])

    # 4-bit options
    ap.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--bnb_4bit_use_double_quant", type=int, default=1)
    ap.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--gradient_checkpointing", type=int, default=1)
    ap.add_argument("--optim_4bit", type=str, default="adamw8bit", choices=["adamw4bit", "paged_adamw4bit", "adamw8bit"])

    return ap.parse_args()


def main():
    args = parse_args()

    local_rank, world_size, is_distributed = setup_distributed_safe()
    set_seed(args.seed + (local_rank if is_distributed else 0))

    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if is_main_process():
        print("[load] tokenizer from model_path:", args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    Qwen2MoeForCausalLM, Qwen2MoeConfig = import_qwen_moe_classes()
    config = Qwen2MoeConfig.from_pretrained(args.model_path)
    requested_topk = int(args.num_experts_per_tok) if int(args.num_experts_per_tok) > 0 else int(args.router_topk)
    if requested_topk > 0:
        if hasattr(config, "num_experts") and requested_topk > int(config.num_experts):
            raise ValueError(f"requested top-k ({requested_topk}) cannot exceed config.num_experts ({config.num_experts}).")
        config.num_experts_per_tok = requested_topk

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    compute_dtype = dtype_map[args.bnb_4bit_compute_dtype]

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bool(args.bnb_4bit_use_double_quant),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    if is_main_process():
        print(f"[load] model 4bit compute_dtype={compute_dtype} distributed={is_distributed} world_size={world_size}")
        print(f"[train] strict 4bit mode with optimizer={args.optim_4bit} (no fp16/bf16 autocast).")

    device_map = {"": local_rank} if is_distributed else ({"": 0} if device.type == "cuda" else None)

    model = Qwen2MoeForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )

    if bool(args.gradient_checkpointing):
        model.gradient_checkpointing_enable()

    prep_sig = inspect.signature(prepare_model_for_kbit_training)
    if "use_gradient_checkpointing" in prep_sig.parameters:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(args.gradient_checkpointing))
    else:
        model = prepare_model_for_kbit_training(model)

    targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()] if args.lora_target_modules else None
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, target_modules=targets)

    if is_main_process():
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[lora] trainable params: {trainable}/{total} ({trainable / total * 100:.2f}%)")

    use_label = bool(args.use_label)

    def make_ds(name: str, split: str, max_samples: Optional[int]):
        if name == "piqa":
            return load_and_pack_piqa_ppl_opencompass(tokenizer=tokenizer, block_size=args.block_size, split=split, num_proc=args.num_proc, bos=True, eos=False, max_samples=max_samples, use_label=use_label)
        if name == "siqa":
            return load_and_pack_siqa_ppl_opencompass(tokenizer=tokenizer, block_size=args.block_size, split=split, num_proc=args.num_proc, bos=True, eos=False, max_samples=max_samples, use_label=use_label)
        if name == "hellaswag":
            return load_and_pack_hellaswag_ppl_opencompass(tokenizer=tokenizer, block_size=args.block_size, split=split, num_proc=args.num_proc, bos=True, eos=False, max_samples=max_samples, use_label=use_label)
        if name == "arc-e":
            return load_and_pack_arc_easy_ppl_opencompass(tokenizer=tokenizer, block_size=args.block_size, split=split, num_proc=args.num_proc, bos=True, eos=False, max_samples=max_samples, use_label=use_label)
        if name == "csqa":
            return load_and_pack_commonsenseqa_ppl_opencompass(tokenizer=tokenizer, block_size=args.block_size, split=split, num_proc=args.num_proc, bos=True, eos=False, max_samples=max_samples, use_label=use_label)
        if name == "bbh":
            return load_and_pack_bbh_ppl_opencompass(tokenizer=tokenizer, block_size=args.block_size, split=split, num_proc=args.num_proc, bos=True, eos=False, max_samples=max_samples, task=args.bbh_task)
        if name == "winogrande":
            return load_and_pack_winogrande_ppl_opencompass(tokenizer=tokenizer, block_size=args.block_size, split=split, num_proc=args.num_proc, bos=True, eos=False, max_samples=max_samples, config_name=args.winogrande_config, use_label=use_label)
        if name == "mmlu":
            subs = None
            if args.mmlu_subjects and args.mmlu_subjects.strip().lower() != "all":
                subs = [x.strip() for x in args.mmlu_subjects.split(",") if x.strip()]
            return load_and_pack_mmlu_ppl_opencompass(tokenizer=tokenizer, block_size=args.block_size, split=split, num_proc=args.num_proc, bos=True, eos=False, max_samples=max_samples, subjects=subs, answer_mode=args.mmlu_answer_mode)
        if name == "arc-c":
            return load_and_pack_arc_challenge_ppl_opencompass(tokenizer=tokenizer, block_size=args.block_size, split=split, num_proc=args.num_proc, bos=True, eos=False, max_samples=max_samples, use_label=use_label)
        if name == "openbookqa":
            return load_and_pack_openbookqa_ppl_opencompass(tokenizer=tokenizer, block_size=args.block_size, split=split, num_proc=args.num_proc, bos=True, eos=False, max_samples=max_samples, use_label=use_label)
        raise ValueError(f"Unknown dataset: {name}")

    if args.dataset == "mix":
        parts = [x.strip() for x in args.mix_datasets.split(",") if x.strip()]
        train_ds = concatenate_datasets([make_ds(p, args.train_split, args.train_max_samples) for p in parts])
    else:
        train_ds = make_ds(args.dataset, args.train_split, args.train_max_samples)

    eval_ds = make_ds(args.eval_dataset, args.eval_split, args.eval_max_samples)

    if is_main_process():
        print(f"[data] train={len(train_ds)} eval={len(eval_ds)} block_size={args.block_size}")

    collator = LMDataCollator(pad_id=0)
    if is_distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=dist.get_rank(), shuffle=True, drop_last=True)
        eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size, rank=dist.get_rank(), shuffle=False, drop_last=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), drop_last=True, num_workers=2, pin_memory=True, collate_fn=collator)
    eval_dl = DataLoader(eval_ds, batch_size=args.batch_size, sampler=eval_sampler, shuffle=False, drop_last=False, num_workers=2, pin_memory=True, collate_fn=collator)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    train_4bit(
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
        log_every=max(1, args.log_every),
        eval_every=max(1, args.eval_every),
        optim_4bit=args.optim_4bit,
    )

    if is_main_process():
        try:
            tokenizer.save_pretrained(args.output_dir)
        except Exception:
            pass

    cleanup_distributed()


if __name__ == "__main__":
    main()