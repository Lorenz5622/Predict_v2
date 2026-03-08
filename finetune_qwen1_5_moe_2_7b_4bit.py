#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen1.5-MoE-2.7B LoRA finetuning script with 4-bit quantization.

Design goals:
- Keep dataset organization exactly aligned with finetune_baseline_example.py
- Swap model implementation to qwen_moe/modeling/modeling_moe.py
- Enable 4bit loading/training via bitsandbytes + PEFT prepare_model_for_kbit_training
"""
from __future__ import annotations

import argparse
from typing import Optional

import torch
from datasets import concatenate_datasets
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

from finetune_baseline_example import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    set_seed,
    LMDataCollator,
    apply_lora,
    train,
    load_and_pack_piqa_ppl_opencompass,
    load_and_pack_siqa_ppl_opencompass,
    load_and_pack_hellaswag_ppl_opencompass,
    load_and_pack_arc_easy_ppl_opencompass,
    load_and_pack_commonsenseqa_ppl_opencompass,
    load_and_pack_bbh_ppl_opencompass,
    load_and_pack_winogrande_ppl_opencompass,
    load_and_pack_mmlu_ppl_opencompass,
    load_and_pack_arc_challenge_ppl_opencompass,
    load_and_pack_openbookqa_ppl_opencompass,
)


def import_qwen_moe_classes():
    from qwen_moe.modeling.configuration_moe import Qwen2MoeConfig
    from qwen_moe.modeling.modeling_moe import Qwen2MoeForCausalLM

    return Qwen2MoeForCausalLM, Qwen2MoeConfig


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

    ap.add_argument("--fp16", type=int, default=0)
    ap.add_argument("--bf16", type=int, default=1)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", type=str, default="")

    ap.add_argument("--use_bnb_8bit", type=int, default=0)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--metrics_every", type=int, default=10)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--router_topk", type=int, default=0)

    ap.add_argument("--bbh_task", type=str, default="boolean_expressions")
    ap.add_argument("--winogrande_config", type=str, default="winogrande_xl")
    ap.add_argument("--mmlu_subjects", type=str, default="all")
    ap.add_argument("--mmlu_answer_mode", type=str, default="text", choices=["text", "letter"])

    # 4bit options
    ap.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--bnb_4bit_use_double_quant", type=int, default=1)
    ap.add_argument("--gradient_checkpointing", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()

    local_rank, world_size, is_distributed = setup_distributed()
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
    if hasattr(args, "router_topk"):
        config.num_experts_per_tok = int(args.router_topk)

    use_fp16 = bool(args.fp16) and (device.type == "cuda") and (not bool(args.bf16))
    use_bf16 = bool(args.bf16) and (device.type == "cuda")
    compute_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else torch.float32)

    if is_main_process():
        print(f"[load] model 4bit compute_dtype={compute_dtype} distributed={is_distributed} world_size={world_size}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bool(args.bnb_4bit_use_double_quant),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    if is_distributed:
        device_map = {"": local_rank}
    else:
        device_map = {"": 0} if device.type == "cuda" else None

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

    model = prepare_model_for_kbit_training(model)

    targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()] if args.lora_target_modules else None
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, target_modules=targets)

    if is_main_process():
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[lora] trainable params: {trainable}/{total} ({trainable / total * 100:.2f}%)")
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

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

    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    import torch.distributed as dist

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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

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
