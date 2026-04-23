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
import gc
import inspect
import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from finetune import (
    LMDataCollator,
    PairwiseDataCollator,
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
    load_and_pack_piqa_pairwise_opencompass,
    load_and_pack_piqa_ppl_opencompass,
    load_and_pack_siqa_ppl_opencompass,
    load_and_pack_winogrande_ppl_opencompass,
    set_seed,
)


PIQA_PAIRWISE_COEF = 0.0


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
    - expert_key: initialized from the matching legacy dense router weight
    - expert_anchor: copied from expert_key after initialization
    """
    num_inited = 0
    per_layer_init_messages = []

    with torch.no_grad():
        for layer_idx, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            if not getattr(mlp, "use_switch", False):
                continue
            if not getattr(mlp, "use_cross_attention_router", False):
                continue
            if not hasattr(mlp, "router"):
                continue

            router = mlp.router
            dense_w = legacy_sd.get(f"model.layers.{layer_idx}.mlp.router.weight")
            if dense_w is None:
                if getattr(router, "expert_key", None) is not None:
                    per_layer_init_messages.append(
                        f"[init] layer {layer_idx} expert_key kept random init (legacy router.weight not found)"
                    )
                continue

            if not hasattr(router, "query"):
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

            if hasattr(router, "_reshape_legacy_router_weight"):
                if getattr(router, "expert_key", None) is not None:
                    router.initialize_expert_key_from_legacy_router(dense_w)
                    per_layer_init_messages.append(
                        f"[init] layer {layer_idx} expert_key/expert_anchor initialized from legacy router.weight"
                    )

            num_inited += 1

        for msg in per_layer_init_messages:
            print(msg)

    return num_inited


def _enable_new_router_params_trainable(model: nn.Module) -> int:
    keys = (
        "router.query",
        "router.expert_key",
    )
    n_params = 0
    for name, param in model.named_parameters():
        if any(key in name for key in keys):
            param.requires_grad = True
            n_params += param.numel()
    return n_params


def _cast_selected_trainable_params_to_fp32(model: nn.Module) -> int:
    """
    Cast only the numerically sensitive router parameters to fp32.

    This keeps base frozen/quantized weights untouched (requires_grad=False),
    leaves LoRA weights in the model/autocast dtype, and promotes only:
    - router.query
    - router.expert_key
    """
    fp32_keys = (
        "router.query",
        "router.expert_key",
    )
    n_params = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if not any(key in name for key in fp32_keys):
                continue
            if not torch.is_floating_point(param):
                continue
            if param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
                n_params += param.numel()
    return n_params


def _unwrap_base_model(model: nn.Module) -> nn.Module:
    model_for_ops = model.module if hasattr(model, "module") else model
    return model_for_ops.get_base_model() if hasattr(model_for_ops, "get_base_model") else model_for_ops


def _get_moe_model(model: nn.Module):
    base_model = _unwrap_base_model(model)
    return getattr(base_model, "model", None)


def _get_moe_stat_dict(model: nn.Module, attr_name: str) -> Dict[str, Any]:
    moe_model = _get_moe_model(model)
    if moe_model is None:
        return {}
    stats = getattr(moe_model, attr_name, None)
    return stats if isinstance(stats, dict) else {}


def _metric_cell(value: Any) -> str:
    if value is None:
        return ""
    if torch.is_tensor(value):
        value = value.detach().float().cpu()
        if value.numel() == 1:
            return f"{float(value.item()):.6f}"
        return json.dumps([round(float(x), 6) for x in value.view(-1).tolist()], ensure_ascii=False)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    if isinstance(value, int):
        return str(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def get_switch_layers(model: nn.Module) -> List[tuple[int, nn.Module]]:
    base_model = _unwrap_base_model(model)
    moe_model = getattr(base_model, "model", None)
    layers = getattr(moe_model, "layers", None)
    if layers is None:
        raise AttributeError("Cannot locate decoder layers on current model.")

    out = []
    for layer_idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and getattr(mlp, "use_switch", False):
            out.append((layer_idx, mlp))
    return out


def freeze_all_params(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def enable_cross_attention_router_only(model: nn.Module) -> int:
    n_params = 0
    for _layer_idx, mlp in get_switch_layers(model):
        if not getattr(mlp, "use_cross_attention_router", False):
            continue
        router = getattr(mlp, "router", None)
        if router is None:
            continue
        for name, param in router.named_parameters():
            if not any(
                key in name
                for key in (
                    "query",
                    "expert_key",
                )
            ):
                continue
            param.requires_grad = True
            n_params += param.numel()
    return n_params


def prepare_stage1_router_trainables(model: nn.Module) -> int:
    freeze_all_params(model)
    n_router_params = enable_cross_attention_router_only(model)
    if n_router_params == 0:
        raise RuntimeError("Stage1 found no trainable CrossAttentionRouter parameters.")
    _cast_selected_trainable_params_to_fp32(model)
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
        teacher_weights[layer_idx] = _align_legacy_router_weight(
            weight=weight,
            hidden_size=int(getattr(mlp, "router").hidden_size),
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


def _apply_pending_router_anchor_updates(model: nn.Module) -> None:
    model_for_updates = model.module if hasattr(model, "module") else model
    base_model = model_for_updates.get_base_model() if hasattr(model_for_updates, "get_base_model") else model_for_updates
    moe_model = getattr(base_model, "model", None)
    if moe_model is None:
        return

    for layer in getattr(moe_model, "layers", []):
        mlp = getattr(layer, "mlp", None)
        apply_fn = getattr(mlp, "apply_pending_router_anchor_update", None)
        if callable(apply_fn):
            apply_fn()


def _set_runtime_router_anchor_collection(model: nn.Module, enabled: bool) -> None:
    for _layer_idx, mlp in get_switch_layers(model):
        setattr(mlp, "enable_router_anchor_collection", bool(enabled))


def _set_runtime_router_anchor_batch_mask(model: nn.Module, batch_mask: Optional[torch.Tensor]) -> None:
    for _layer_idx, mlp in get_switch_layers(model):
        setattr(mlp, "router_anchor_batch_mask", batch_mask)


def _is_pairwise_dataset(ds) -> bool:
    if ds is None:
        return False
    column_names = getattr(ds, "column_names", None)
    return isinstance(column_names, list) and "chosen_input_ids" in column_names


def _make_collator_for_dataset(ds):
    if _is_pairwise_dataset(ds):
        return PairwiseDataCollator(pad_id=0)
    return LMDataCollator(pad_id=0)


def _mean_answer_logprob(logits: torch.Tensor, labels: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    shift_answer_mask = answer_mask[:, 1:].to(dtype=torch.bool)
    valid_mask = shift_answer_mask & (shift_labels != -100)

    token_log_probs = torch.log_softmax(shift_logits, dim=-1)
    gathered = token_log_probs.gather(dim=-1, index=shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    gathered = gathered * valid_mask.to(gathered.dtype)
    token_counts = valid_mask.sum(dim=-1).clamp_min(1)
    return gathered.sum(dim=-1) / token_counts.to(gathered.dtype)


def _causal_lm_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def _forward_stage2_batch(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    amp_dtype,
    device: torch.device,
):
    is_pairwise = "chosen_input_ids" in batch

    def _run_model(inputs: Dict[str, torch.Tensor]):
        if amp_dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                return model(**inputs)
        return model(**inputs)

    if not is_pairwise:
        out = _run_model(batch)
        raw_loss = out.loss
        anchor_loss = getattr(out, "router_anchor_loss", None)
        if anchor_loss is None:
            anchor_loss = raw_loss.new_zeros(())
        else:
            anchor_loss = anchor_loss.to(device=raw_loss.device, dtype=raw_loss.dtype)
        budget_loss = getattr(out, "router_budget_loss", None)
        if budget_loss is None:
            budget_loss = raw_loss.new_zeros(())
        else:
            budget_loss = budget_loss.to(device=raw_loss.device, dtype=raw_loss.dtype)
        return {
            "ce_loss": raw_loss,
            "pairwise_loss": raw_loss.new_zeros(()),
            "anchor_loss": anchor_loss,
            "budget_loss": budget_loss,
            "pairwise_acc": raw_loss.new_zeros(()),
            "pairwise_margin": raw_loss.new_zeros(()),
            "chosen_score": raw_loss.new_zeros(()),
            "rejected_score": raw_loss.new_zeros(()),
        }

    chosen_input_ids = batch["chosen_input_ids"]
    chosen_labels = batch["chosen_labels"]
    chosen_attention_mask = batch["chosen_attention_mask"]
    rejected_input_ids = batch["rejected_input_ids"]
    rejected_labels = batch["rejected_labels"]
    rejected_attention_mask = batch["rejected_attention_mask"]

    combined_inputs = {
        "input_ids": torch.cat([chosen_input_ids, rejected_input_ids], dim=0),
        "labels": torch.cat([chosen_labels, rejected_labels], dim=0),
        "attention_mask": torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0),
    }
    chosen_batch = chosen_input_ids.size(0)
    anchor_batch_mask = torch.cat([
        torch.ones(chosen_batch, device=chosen_input_ids.device, dtype=torch.bool),
        torch.zeros(rejected_input_ids.size(0), device=chosen_input_ids.device, dtype=torch.bool),
    ], dim=0)

    _set_runtime_router_anchor_collection(model, True)
    _set_runtime_router_anchor_batch_mask(model, anchor_batch_mask)
    try:
        out = _run_model(combined_inputs)
    finally:
        _set_runtime_router_anchor_batch_mask(model, None)

    chosen_logits, rejected_logits = out.logits.split([chosen_batch, rejected_input_ids.size(0)], dim=0)
    ce_loss = _causal_lm_ce_loss(chosen_logits, chosen_labels)
    anchor_loss = getattr(out, "router_anchor_loss", None)
    if anchor_loss is None:
        anchor_loss = ce_loss.new_zeros(())
    else:
        anchor_loss = anchor_loss.to(device=ce_loss.device, dtype=ce_loss.dtype)
    budget_loss = getattr(out, "router_budget_loss", None)
    if budget_loss is None:
        budget_loss = ce_loss.new_zeros(())
    else:
        budget_loss = budget_loss.to(device=ce_loss.device, dtype=ce_loss.dtype)

    chosen_score = _mean_answer_logprob(
        chosen_logits,
        chosen_labels,
        batch["chosen_answer_mask"],
    )
    rejected_score = _mean_answer_logprob(
        rejected_logits,
        rejected_labels,
        batch["rejected_answer_mask"],
    )
    margin = chosen_score - rejected_score
    pairwise_loss = -F.logsigmoid(margin).mean()
    pairwise_acc = (margin > 0).to(chosen_score.dtype).mean()

    return {
        "ce_loss": ce_loss,
        "pairwise_loss": pairwise_loss,
        "anchor_loss": anchor_loss,
        "budget_loss": budget_loss,
        "pairwise_acc": pairwise_acc,
        "pairwise_margin": margin.mean(),
        "chosen_score": chosen_score.mean(),
        "rejected_score": rejected_score.mean(),
    }


@torch.no_grad()
def evaluate_stage2(model: nn.Module, dl: DataLoader, device: torch.device, fp16: bool, bf16: bool) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    amp_dtype = torch.float16 if fp16 else (torch.bfloat16 if bf16 else None)

    totals = {
        "ce_loss": 0.0,
        "pairwise_loss": 0.0,
        "total_loss": 0.0,
        "pairwise_acc": 0.0,
        "pairwise_margin": 0.0,
        "chosen_score": 0.0,
        "rejected_score": 0.0,
        "count": 0.0,
    }

    try:
        for batch in dl:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            stats = _forward_stage2_batch(model, batch, amp_dtype=amp_dtype, device=device)
            batch_size = float(next(iter(batch.values())).shape[0])
            ce_loss = float(stats["ce_loss"].detach().float().item())
            pairwise_loss = float(stats["pairwise_loss"].detach().float().item())
            pairwise_acc = float(stats["pairwise_acc"].detach().float().item())
            pairwise_margin = float(stats["pairwise_margin"].detach().float().item())
            chosen_score = float(stats["chosen_score"].detach().float().item())
            rejected_score = float(stats["rejected_score"].detach().float().item())
            total_loss = ce_loss + PIQA_PAIRWISE_COEF * pairwise_loss

            totals["ce_loss"] += ce_loss * batch_size
            totals["pairwise_loss"] += pairwise_loss * batch_size
            totals["total_loss"] += total_loss * batch_size
            totals["pairwise_acc"] += pairwise_acc * batch_size
            totals["pairwise_margin"] += pairwise_margin * batch_size
            totals["chosen_score"] += chosen_score * batch_size
            totals["rejected_score"] += rejected_score * batch_size
            totals["count"] += batch_size
    finally:
        _set_runtime_router_anchor_collection(model, True)
        if was_training:
            model.train()

    packed = torch.tensor(
        [
            totals["ce_loss"],
            totals["pairwise_loss"],
            totals["total_loss"],
            totals["pairwise_acc"],
            totals["pairwise_margin"],
            totals["chosen_score"],
            totals["rejected_score"],
            totals["count"],
        ],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)

    count = max(float(packed[-1].item()), 1.0)
    ce_loss = float((packed[0] / count).item())
    pairwise_loss = float((packed[1] / count).item())
    total_loss = float((packed[2] / count).item())
    return {
        "loss": total_loss,
        "ce_loss": ce_loss,
        "pairwise_loss": pairwise_loss,
        "pairwise_acc": float((packed[3] / count).item()),
        "pairwise_margin": float((packed[4] / count).item()),
        "chosen_score": float((packed[5] / count).item()),
        "rejected_score": float((packed[6] / count).item()),
        "ppl": math.exp(min(20.0, ce_loss)),
    }


def _get_runtime_router_top_k(model: nn.Module) -> Optional[int]:
    switch_layers = get_switch_layers(model)
    if not switch_layers:
        return None
    return int(getattr(switch_layers[0][1], "router_top_k"))


def _set_runtime_router_top_k(model: nn.Module, top_k: int) -> None:
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError(f"router top-k must be >= 1, got {top_k}")

    base_model = _unwrap_base_model(model)
    if hasattr(base_model, "config"):
        setattr(base_model.config, "router_top_k", top_k)

    moe_model = getattr(base_model, "model", None)
    if moe_model is not None and hasattr(moe_model, "config"):
        setattr(moe_model.config, "router_top_k", top_k)

    for _layer_idx, mlp in get_switch_layers(model):
        setattr(mlp, "router_top_k", top_k)


def _warn_ignored_legacy_router_settings(args) -> None:
    ignored = []
    if getattr(args, "top_p_threshold", None) is not None:
        ignored.append("top_p_threshold")
    if getattr(args, "router_top_p_final", None) is not None:
        ignored.append("router_top_p_final")
    if float(getattr(args, "router_top_p_schedule_start_ratio", 0.5)) != 0.5:
        ignored.append("router_top_p_schedule_start_ratio")
    if float(getattr(args, "router_top_p_schedule_end_ratio", 1.0)) != 1.0:
        ignored.append("router_top_p_schedule_end_ratio")
    if float(getattr(args, "router_budget_loss_coef", 0.0)) > 0.0:
        ignored.append("router_budget_loss_coef")
    if float(getattr(args, "router_budget_target_count", 0.0)) > 0.0:
        ignored.append("router_budget_target_count")
    if float(getattr(args, "router_budget_tau", 0.05)) != 0.05:
        ignored.append("router_budget_tau")
    if float(getattr(args, "router_budget_start_ratio", 0.5)) != 0.5:
        ignored.append("router_budget_start_ratio")
    if float(getattr(args, "router_budget_end_ratio", 1.0)) != 1.0:
        ignored.append("router_budget_end_ratio")
    if int(getattr(args, "use_router_context", 0)) not in (-1, 0):
        ignored.append("use_router_context")
    if float(getattr(args, "router_context_scale", 0.0) or 0.0) != 0.0:
        ignored.append("router_context_scale")
    if int(getattr(args, "share_router_expert_embedding", 0)) not in (-1, 0):
        ignored.append("share_router_expert_embedding")
    if float(getattr(args, "router_aux_loss_coef", 0.0)) > 0.0:
        ignored.append("router_aux_loss_coef")
    if float(getattr(args, "router_z_loss_coef", 0.0)) > 0.0:
        ignored.append("router_z_loss_coef")
    if float(getattr(args, "router_pull_loss_coef", 0.0)) > 0.0:
        ignored.append("router_pull_loss_coef")

    if ignored and is_main_process():
        print(
            "[router] anchor-router trunk ignores legacy router settings: "
            + ", ".join(sorted(set(ignored)))
        )


def _get_runtime_router_pull_loss_type(model: nn.Module) -> Optional[str]:
    switch_layers = get_switch_layers(model)
    if not switch_layers:
        return None
    return str(getattr(switch_layers[0][1], "router_pull_loss_type"))


def _set_runtime_router_pull_loss_type(model: nn.Module, pull_loss_type: str) -> None:
    pull_loss_type = str(pull_loss_type)
    if pull_loss_type not in {"soft", "hard_ce"}:
        raise ValueError(
            f"router pull loss type must be one of {{'soft', 'hard_ce'}}, got {pull_loss_type!r}"
        )

    base_model = _unwrap_base_model(model)
    if hasattr(base_model, "config"):
        setattr(base_model.config, "router_pull_loss_type", pull_loss_type)

    moe_model = getattr(base_model, "model", None)
    if moe_model is not None and hasattr(moe_model, "config"):
        setattr(moe_model.config, "router_pull_loss_type", pull_loss_type)

    for _layer_idx, mlp in get_switch_layers(model):
        setattr(mlp, "router_pull_loss_type", pull_loss_type)


def _schedule_alpha(progress: float, start_ratio: float, end_ratio: float) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    start_ratio = min(max(float(start_ratio), 0.0), 1.0)
    end_ratio = min(max(float(end_ratio), 0.0), 1.0)
    if end_ratio < start_ratio:
        start_ratio, end_ratio = end_ratio, start_ratio
    if progress <= start_ratio:
        return 0.0
    if progress >= end_ratio:
        return 1.0
    if end_ratio == start_ratio:
        return 1.0
    return (progress - start_ratio) / (end_ratio - start_ratio)


def _scheduled_float(
    progress: float,
    initial_value: float,
    final_value: Optional[float],
    start_ratio: float,
    end_ratio: float,
) -> float:
    initial_value = float(initial_value)
    if final_value is None:
        return initial_value
    alpha = _schedule_alpha(progress, start_ratio, end_ratio)
    return (1.0 - alpha) * initial_value + alpha * float(final_value)


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


def _is_stage2_router_param(name: str) -> bool:
    router_keys = (
        "router.query",
        "router.expert_key",
    )
    return any(key in name for key in router_keys)


def build_stage2_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    use_bnb_8bit: bool,
    router_lr_mult: float,
):
    router_params: List[torch.nn.Parameter] = []
    other_params: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_stage2_router_param(name):
            router_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": lr})
    if router_params:
        param_groups.append({"params": router_params, "lr": lr * router_lr_mult})
    if not param_groups:
        raise RuntimeError("No trainable parameters found for stage2 optimizer.")

    if is_main_process():
        print(
            "[stage2][opt] param groups: "
            f"other={len(other_params)} router={len(router_params)}"
        )
        print(
            "[stage2][opt] lrs: "
            f"other={lr:.3e} router={lr * router_lr_mult:.3e}"
        )

    if use_bnb_8bit:
        try:
            import bitsandbytes as bnb
        except Exception as exc:
            raise ImportError("bitsandbytes is not available but --use_bnb_8bit=1 was set") from exc
        return bnb.optim.AdamW8bit(param_groups, lr=lr, weight_decay=weight_decay)

    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


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
    router_lr_mult: float,
    min_lr_ratio: float,
    log_every: int,
    eval_every: int,
    save_every: int,
):
    os.makedirs(output_dir, exist_ok=True)

    optimizer = build_stage2_optimizer(
        model,
        lr=lr,
        weight_decay=weight_decay,
        use_bnb_8bit=use_bnb_8bit,
        router_lr_mult=router_lr_mult,
    )

    steps_per_epoch = math.ceil(len(train_dl) / max(1, grad_accum))
    total_optim_steps = steps_per_epoch * epochs
    warmup_steps = int(total_optim_steps * warmup_ratio)
    min_lr_ratio = float(min(max(min_lr_ratio, 0.0), 1.0))

    pbar = tqdm(
        total=total_optim_steps,
        disable=not is_main_process(),
        dynamic_ncols=True,
        desc="train",
    )

    current_router_top_k = 2
    _set_runtime_router_top_k(model, current_router_top_k)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_optim_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

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
    ema_ce_loss = None
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
            "loss", "loss_ce", "loss_pairwise", "loss_anchor", "loss_budget",
            f"loss_ma{ma_win}", "loss_ema", "loss_ce_ema", "eval_loss", "eval_ce_loss", "eval_pairwise_loss", "eval_pairwise_acc", "eval_pairwise_margin", "eval_ppl",
            "pairwise_coef", "router_top_k", "router_anchor_loss_coef",
            "train_pairwise_acc", "train_pairwise_margin", "train_chosen_score", "train_rejected_score",
            "router_score_mean", "router_score_std", "router_score_min", "router_score_max",
            "router_weight_row_sum_mean", "router_weight_row_sum_abs_err",
            "router_weight_entropy", "router_weight_top1_mass",
            "route_prob_min", "route_prob_has_neg", "route_prob_row_sum_mean", "route_prob_row_sum_abs_err",
            "expert_key_pairwise_cos_mean", "expert_key_pairwise_cos_max",
            "expert_anchor_pairwise_cos_mean", "expert_anchor_pairwise_cos_max",
            "token_q_norm_mean", "token_q_norm_std",
            "dispatch_avg_selected_count", "dispatch_soft_selected_count", "dispatch_dead_expert_ratio", "dispatch_top1_top2_margin",
            "dispatch_topk_pre_mass_mean", "dispatch_topk_post_sum_mean", "dispatch_topk_post_sum_abs_err", "expert_token_count_cv",
            "dispatch_soft_load", "dispatch_hard_load",
            "anchor_active_expert_count", "anchor_proto_count_mean", "anchor_proto_count_min", "anchor_proto_count_max",
            "anchor_key_cosine_mean", "anchor_proto_counts",
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
            step_progress = float(optim_step) / float(max(1, total_optim_steps - 1))

            budget_loss_scale = 0.0

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            loss_stats = _forward_stage2_batch(model, batch, amp_dtype=amp_dtype, device=device)
            ce_loss = loss_stats["ce_loss"]
            pairwise_loss = loss_stats["pairwise_loss"].to(device=ce_loss.device, dtype=ce_loss.dtype)
            anchor_loss = loss_stats["anchor_loss"].to(device=ce_loss.device, dtype=ce_loss.dtype)
            budget_loss = loss_stats["budget_loss"].to(device=ce_loss.device, dtype=ce_loss.dtype)
            total_loss = (
                ce_loss
                + PIQA_PAIRWISE_COEF * pairwise_loss
                + float(run_args.router_anchor_loss_coef) * anchor_loss
                + float(getattr(run_args, "router_budget_loss_coef", 0.0)) * budget_loss_scale * budget_loss
            )
            loss = total_loss / max(1, grad_accum)

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

                step_succeeded = True
                if scaler.is_enabled():
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    step_succeeded = scale_after >= scale_before
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                if not step_succeeded:
                    if is_main_process():
                        print(
                            "[train] scaler dropped "
                            f"{scale_before} -> {scale_after}; skip scheduler/optim_step"
                        )
                    continue

                _apply_pending_router_anchor_updates(model)
                optim_step += 1
                scheduler.step()

                loss_real = float(total_loss.detach().float().item())
                ce_loss_real = float(ce_loss.detach().float().item())
                pairwise_loss_real = float(pairwise_loss.detach().float().item())
                anchor_loss_real = float(anchor_loss.detach().float().item())
                budget_loss_real = float(budget_loss.detach().float().item())
                pairwise_acc_real = float(loss_stats["pairwise_acc"].detach().float().item())
                pairwise_margin_real = float(loss_stats["pairwise_margin"].detach().float().item())
                chosen_score_real = float(loss_stats["chosen_score"].detach().float().item())
                rejected_score_real = float(loss_stats["rejected_score"].detach().float().item())

                if len(ma_loss_buf) == ma_loss_buf.maxlen:
                    ma_loss_sum -= ma_loss_buf[0]
                ma_loss_buf.append(loss_real)
                ma_loss_sum += loss_real
                loss_ma = ma_loss_sum / max(1, len(ma_loss_buf))

                if ema_loss is None:
                    ema_loss = loss_real
                else:
                    ema_loss = ema_momentum * ema_loss + (1.0 - ema_momentum) * loss_real
                if ema_ce_loss is None:
                    ema_ce_loss = ce_loss_real
                else:
                    ema_ce_loss = ema_momentum * ema_ce_loss + (1.0 - ema_momentum) * ce_loss_real

                cur_lr = scheduler.get_last_lr()[0]
                router_forward_stats = _get_moe_stat_dict(model, "last_router_forward_stats")
                router_dispatch_stats = _get_moe_stat_dict(model, "last_router_dispatch_stats")
                router_anchor_stats = _get_moe_stat_dict(model, "last_router_anchor_stats")
                should_eval = eval_dl is not None and (optim_step % eval_every == 0)
                eval_loss_real = None
                eval_ce_loss_real = None
                eval_pairwise_loss_real = None
                eval_pairwise_acc_real = None
                eval_pairwise_margin_real = None
                eval_ppl_real = None
                if should_eval:
                    eval_stats = evaluate_stage2(model, eval_dl, device, fp16=fp16, bf16=bf16)
                    eval_loss_real = eval_stats["loss"]
                    eval_ce_loss_real = eval_stats["ce_loss"]
                    eval_pairwise_loss_real = eval_stats["pairwise_loss"]
                    eval_pairwise_acc_real = eval_stats["pairwise_acc"]
                    eval_pairwise_margin_real = eval_stats["pairwise_margin"]
                    eval_ppl_real = eval_stats["ppl"]

                if is_main_process() and (optim_step % log_every == 0):
                    if hasattr(it, "set_postfix"):
                        it.set_postfix({
                            "loss": f"{loss_real:.4f}",
                            "ce": f"{ce_loss_real:.4f}",
                            "pw": f"{pairwise_loss_real:.4f}",
                            "anchor": f"{anchor_loss_real:.4f}",
                            "budget": f"{budget_loss_real:.4f}",
                            f"ma{ma_win}": f"{loss_ma:.4f}",
                            "ema": f"{ema_loss:.4f}",
                            "ce_ema": f"{ema_ce_loss:.4f}",
                            "sel": f"{float(router_dispatch_stats.get('avg_selected_expert_count', 0.0)):.2f}",
                            "top_k": str(int(current_router_top_k)),
                            "ak": f"{float(router_anchor_stats.get('anchor_key_cosine_mean', 0.0)):.3f}",
                            "dead": f"{float(router_dispatch_stats.get('dead_expert_ratio', 0.0)):.2f}",
                            "neg": f"{float(router_forward_stats.get('route_prob_has_neg', 0.0)):.0f}",
                            "lr": f"{cur_lr:.2e}",
                        }, refresh=False)

                if is_main_process() and metrics_f is not None and ((optim_step % log_every == 0) or should_eval):
                        metrics_f.write("\t".join([
                            f"{time.time():.3f}",
                            str(epoch),
                            str(global_step),
                            str(optim_step),
                            f"{cur_lr:.6e}",
                            f"{loss_real:.6f}",
                            f"{ce_loss_real:.6f}",
                            f"{pairwise_loss_real:.6f}",
                            f"{anchor_loss_real:.6f}",
                            f"{budget_loss_real:.6f}",
                            f"{loss_ma:.6f}",
                            f"{ema_loss:.6f}",
                            f"{ema_ce_loss:.6f}",
                            "" if eval_loss_real is None else f"{eval_loss_real:.6f}",
                            "" if eval_ce_loss_real is None else f"{eval_ce_loss_real:.6f}",
                            "" if eval_pairwise_loss_real is None else f"{eval_pairwise_loss_real:.6f}",
                            "" if eval_pairwise_acc_real is None else f"{eval_pairwise_acc_real:.6f}",
                            "" if eval_pairwise_margin_real is None else f"{eval_pairwise_margin_real:.6f}",
                            "" if eval_ppl_real is None else f"{eval_ppl_real:.6f}",
                            f"{PIQA_PAIRWISE_COEF:.6f}",
                            str(int(current_router_top_k)),
                            f"{float(run_args.router_anchor_loss_coef):.6f}",
                            f"{pairwise_acc_real:.6f}",
                            f"{pairwise_margin_real:.6f}",
                            f"{chosen_score_real:.6f}",
                            f"{rejected_score_real:.6f}",
                            _metric_cell(router_forward_stats.get("attn_scores_mean")),
                            _metric_cell(router_forward_stats.get("attn_scores_std")),
                            _metric_cell(router_forward_stats.get("attn_scores_min")),
                            _metric_cell(router_forward_stats.get("attn_scores_max")),
                            _metric_cell(router_forward_stats.get("attn_weights_row_sum_mean")),
                            _metric_cell(router_forward_stats.get("attn_weights_row_sum_abs_err")),
                            _metric_cell(router_forward_stats.get("attn_weights_entropy")),
                            _metric_cell(router_forward_stats.get("attn_weights_top1_mass")),
                            # _metric_cell(router_forward_stats.get("attn_output_mean")),
                            # _metric_cell(router_forward_stats.get("attn_output_std")),
                            # _metric_cell(router_forward_stats.get("attn_output_min")),
                            # _metric_cell(router_forward_stats.get("attn_output_max")),
                            _metric_cell(router_forward_stats.get("route_prob_min")),
                            _metric_cell(router_forward_stats.get("route_prob_has_neg")),
                            _metric_cell(router_forward_stats.get("route_prob_row_sum_mean")),
                            _metric_cell(router_forward_stats.get("route_prob_row_sum_abs_err")),
                            _metric_cell(router_forward_stats.get("expert_key_pairwise_cos_mean")),
                            _metric_cell(router_forward_stats.get("expert_key_pairwise_cos_max")),
                            _metric_cell(router_forward_stats.get("expert_anchor_pairwise_cos_mean")),
                            _metric_cell(router_forward_stats.get("expert_anchor_pairwise_cos_max")),
                            _metric_cell(router_forward_stats.get("token_q_norm_mean")),
                            _metric_cell(router_forward_stats.get("token_q_norm_std")),
                            _metric_cell(router_dispatch_stats.get("avg_selected_expert_count")),
                            _metric_cell(router_dispatch_stats.get("soft_selected_expert_count")),
                            _metric_cell(router_dispatch_stats.get("dead_expert_ratio")),
                            _metric_cell(router_dispatch_stats.get("top1_top2_margin")),
                            _metric_cell(router_dispatch_stats.get("topk_pre_mass_mean")),
                            _metric_cell(router_dispatch_stats.get("topk_post_sum_mean")),
                            _metric_cell(router_dispatch_stats.get("topk_post_sum_abs_err")),
                            _metric_cell(router_dispatch_stats.get("expert_token_count_cv")),
                            _metric_cell(router_dispatch_stats.get("soft_load")),
                            _metric_cell(router_dispatch_stats.get("hard_load")),
                            _metric_cell(router_anchor_stats.get("active_expert_count")),
                            _metric_cell(router_anchor_stats.get("proto_count_mean")),
                            _metric_cell(router_anchor_stats.get("proto_count_min")),
                            _metric_cell(router_anchor_stats.get("proto_count_max")),
                            _metric_cell(router_anchor_stats.get("anchor_key_cosine_mean")),
                            _metric_cell(router_anchor_stats.get("proto_counts")),
                        ]) + "\n")
                        metrics_f.flush()

                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{loss_real:.4f}",
                    "ce": f"{ce_loss_real:.4f}",
                    "pw": f"{pairwise_loss_real:.4f}",
                    "anchor": f"{anchor_loss_real:.4f}",
                    "budget": f"{budget_loss_real:.4f}",
                    "ce_ema": f"{ema_ce_loss:.4f}",
                    "top_k": str(int(current_router_top_k)),
                    "lr": f"{scheduler.get_last_lr()[0]:.3e}",
                }, refresh=False)

                if is_main_process() and (optim_step % log_every == 0):
                    elapsed = time.time() - t0
                    print(
                        f"[train] epoch={epoch+1}/{epochs} step={optim_step}/{total_optim_steps} "
                        f"loss={loss_real:.4f} ce={ce_loss_real:.4f} ce_ema={ema_ce_loss:.4f} "
                        f"pairwise={pairwise_loss_real:.4f} pairwise_acc={pairwise_acc_real:.4f} "
                        f"anchor={anchor_loss_real:.4f} budget={budget_loss_real:.4f} "
                        f"top_k={int(current_router_top_k)} "
                        f"anchor_key_cos={float(router_anchor_stats.get('anchor_key_cosine_mean', 0.0)):.4f} "
                        f"lr={cur_lr:.3e} elapsed={elapsed/60:.1f}m"
                    )

                if should_eval and is_main_process():
                    assert eval_loss_real is not None and eval_ppl_real is not None
                    if is_main_process():
                        print(
                            f"[eval] step={optim_step} loss={eval_loss_real:.4f} "
                            f"ce={float(eval_ce_loss_real or 0.0):.4f} "
                            f"pairwise={float(eval_pairwise_loss_real or 0.0):.4f} "
                            f"pairwise_acc={float(eval_pairwise_acc_real or 0.0):.4f} "
                            f"ppl={eval_ppl_real:.2f}"
                        )

                if False and save_every > 0 and (optim_step % save_every == 0) and is_main_process():
                    save_dir = os.path.join(output_dir, f"checkpoint-{optim_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    model.save_pretrained(save_dir)
                    print(f"[save] {save_dir}")

    pbar.close()
    if is_main_process() and metrics_f is not None:
        metrics_f.close()

    optimizer.zero_grad(set_to_none=True)
    del optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if is_main_process():
        model_to_save = model.module if hasattr(model, "module") else model
        final_router_top_k = _get_runtime_router_top_k(model_to_save)
        if final_router_top_k is not None:
            _set_runtime_router_top_k(model_to_save, final_router_top_k)
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

        print("[save] moving merged model to cpu before export")
        merged = merged.to("cpu")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        state_dict = {}
        for k, v in merged.state_dict().items():
            tensor = v.detach().cpu()
            if torch.is_floating_point(tensor) and tensor.dtype != torch.float32:
                tensor = tensor.to(torch.float32)
            state_dict[k] = tensor

        if was_quantized:
            for attr_name in ("is_loaded_in_8bit", "is_loaded_in_4bit", "quantization_method"):
                if hasattr(merged, attr_name):
                    setattr(merged, attr_name, False if attr_name != "quantization_method" else None)
            print(f"[save] merged + dequantized + fp32 full model -> {output_dir}")
        else:
            print(f"[save] merged + fp32 full model -> {output_dir}")

        if final_router_top_k is not None:
            _set_runtime_router_top_k(merged, final_router_top_k)
            print(f"[save] final router top_k -> {int(final_router_top_k)}")

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
            ],
        )

    if bool(args.load_in_8bit):
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=float(args.llm_int8_threshold),
            llm_int8_skip_modules=[
                "router",
                "query",
            ],
        )

    return None


def build_model_with_router_compat(
    args,
    model_cls,
    config,
    model_path: str,
    device: torch.device,
    quantization_config,
    local_rank: int,
    is_distributed: bool,
    init_from_legacy: bool = True,
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
        legacy_sd = _load_local_checkpoint_state_dict(model_path)
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
        if bool(args.init_new_router_from_legacy) and bool(init_from_legacy):
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

    if bool(args.init_new_router_from_legacy) and bool(init_from_legacy):
        legacy_sd = _load_local_checkpoint_state_dict(model_path)
        inited = _init_cross_attention_router_from_legacy_dense(model=model, legacy_sd=legacy_sd, config=config)
        if is_main_process():
            print(f"[init] initialized new router modules from legacy dense router for {inited} layers")

    return model


def configure_model_config(config, args) -> None:
    if hasattr(config, "ensure_model_attributes"):
        config.ensure_model_attributes()

    config.router_top_k = 2
    if int(args.router_use_entmax) >= 0:
        config.router_use_entmax = bool(args.router_use_entmax)
    if args.router_entmax_alpha is not None:
        config.router_entmax_alpha = float(args.router_entmax_alpha)
    if int(args.router_use_softmax_temperature) >= 0:
        config.router_use_softmax_temperature = bool(args.router_use_softmax_temperature)
    if args.router_softmax_temperature is not None:
        config.router_softmax_temperature = float(args.router_softmax_temperature)
    config.use_router_context = False
    config.router_context_scale = 0.0
    config.share_router_expert_embedding = False
    config.router_budget_target_count = 0.0

    if hasattr(config, "ensure_model_attributes"):
        config.ensure_model_attributes()
    if int(getattr(config, "router_top_k", 0)) <= 0:
        raise ValueError(f"config.router_top_k must be >= 1, got {getattr(config, 'router_top_k', None)}")
    if int(getattr(config, "num_experts", 0)) > 0 and int(config.router_top_k) > int(config.num_experts):
        raise ValueError(f"router_top_k ({config.router_top_k}) cannot exceed num_experts ({config.num_experts})")

    anchor_momentum = getattr(args, "router_anchor_momentum", None)
    if anchor_momentum is None:
        anchor_momentum = float(args.router_ema_momentum)
    config.router_anchor_momentum = float(anchor_momentum)


def make_dataset(name: str, tokenizer, args, split: str, max_samples: Optional[int], *, stage: str):
    use_label = bool(args.use_label)

    if name == "piqa":
        # if stage == "stage2" and getattr(args, "dataset", "") == "piqa":
            # return load_and_pack_piqa_pairwise_opencompass(
            #     tokenizer=tokenizer,
            #     block_size=args.block_size,
            #     split=split,
            #     num_proc=args.num_proc,
            #     bos=True,
            #     eos=False,
            #     max_samples=max_samples,
            # )
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


DATASETS_WITH_VALIDATION = {
    "piqa",
    "siqa",
    "hellaswag",
    "arc-e",
    "csqa",
    "winogrande",
    "mmlu",
    "arc-c",
    "openbookqa",
}


def split_dataset_by_ratio(ds, ratio: float, seed: int):
    if not 0.0 <= float(ratio) <= 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")
    n_total = len(ds)
    if n_total == 0:
        return ds, ds
    n_first = int(round(n_total * float(ratio)))
    n_first = max(0, min(n_total, n_first))
    shuffled = ds.shuffle(seed=int(seed))
    first = shuffled.select(range(n_first))
    second = shuffled.select(range(n_first, n_total))
    return first, second


def build_train_eval_datasets(tokenizer, args, *, stage: str):
    if args.dataset == "mix":
        names = [x.strip() for x in args.mix_datasets.split(",") if x.strip()]
        train_ds_full = concatenate_datasets([
            make_dataset(name, tokenizer, args, args.train_split, args.train_max_samples, stage=stage) for name in names
        ])
    else:
        train_ds_full = make_dataset(args.dataset, tokenizer, args, args.train_split, args.train_max_samples, stage=stage)

    if stage == "stage1":
        train_ds = build_stage1_subset_dataset(
            train_ds_full,
            stage1_ratio=float(args.stage1_data_ratio),
            seed=int(args.stage_split_seed),
        )
    elif stage == "stage2":
        train_ds = train_ds_full
        borrowed_eval_remainders = {}
        if bool(args.stage2_include_validation_in_train):
            val_source_names = names if args.dataset == "mix" else [args.dataset]
            borrowed_parts = []
            for name in val_source_names:
                if name not in DATASETS_WITH_VALIDATION:
                    if args.dataset == "mix":
                        print(f"[stage2][data] skip validation augmentation for {name}: no validation split configured")
                        continue
                    raise ValueError(
                        f"Dataset '{name}' does not have a validation split configured for "
                        "--stage2_include_validation_in_train."
                    )
                val_ds = make_dataset(
                    name,
                    tokenizer,
                    args,
                    args.stage2_validation_train_split,
                    None,
                    stage=stage,
                )
                val_train_part, val_eval_part = split_dataset_by_ratio(
                    val_ds,
                    ratio=float(args.stage2_validation_train_ratio),
                    seed=int(args.stage_split_seed),
                )
                if len(val_train_part) > 0:
                    borrowed_parts.append(val_train_part)
                borrowed_eval_remainders[name] = val_eval_part
                print(
                    f"[stage2][data] {name}: borrowed {len(val_train_part)}/{len(val_ds)} "
                    f"from {args.stage2_validation_train_split} into train"
                )
            if borrowed_parts:
                train_ds = concatenate_datasets([train_ds] + borrowed_parts)
    else:
        raise ValueError(f"Unknown stage: {stage}")

    eval_ds = None
    if stage == "stage2":
        if (
            bool(args.stage2_include_validation_in_train)
            and args.eval_split == args.stage2_validation_train_split
            and args.eval_dataset in borrowed_eval_remainders
        ):
            eval_ds = borrowed_eval_remainders[args.eval_dataset]
            if args.eval_max_samples is not None:
                eval_ds = eval_ds.select(range(min(args.eval_max_samples, len(eval_ds))))
        else:
            eval_ds = make_dataset(args.eval_dataset, tokenizer, args, args.eval_split, args.eval_max_samples, stage=stage)
    return train_ds, eval_ds


def stage1_train_cross_attention_router(
    model: nn.Module,
    train_dl: DataLoader,
    run_args: argparse.Namespace,
    device: torch.device,
    teacher_router_weights: Dict[int, torch.Tensor],
):
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("Stage1 found no parameters with requires_grad=True before optimizer creation.")
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
        if not getattr(mlp, "use_cross_attention_router", False):
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
        stage1_metrics_path = records_dir / f"{run_ts}_stage1_router.tsv"
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
                    if not getattr(mlp, "use_cross_attention_router", False):
                        continue

                    teacher_weight_cpu = teacher_router_weights.get(layer_idx)
                    if teacher_weight_cpu is None:
                        continue

                    teacher_weight = teacher_weight_cpu.to(device=device, dtype=hidden_states.dtype)
                    teacher_logits = F.linear(hidden_states.to(dtype=teacher_weight.dtype), teacher_weight).float()
                    student_logits, _, _ = mlp.router(hidden_states)
                    student_logits = student_logits.float()

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
                    teacher_entropy_token = -(teacher_probs_raw.clamp_min(1e-9) * teacher_probs_raw.clamp_min(1e-9).log()).sum(dim=-1)
                    student_entropy_token = -(student_probs_raw.clamp_min(1e-9) * student_probs_raw.clamp_min(1e-9).log()).sum(dim=-1)
                    teacher_top1 = teacher_logits.argmax(dim=-1)
                    student_top1 = student_logits.argmax(dim=-1)
                    top1_agreement_token = teacher_top1.eq(student_top1).to(mask_f.dtype)
                    topk_k = min(max(1, int(getattr(run_args, "router_top_k", 1))), teacher_logits.size(-1))
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


def build_dataloaders(train_ds, eval_ds, args, world_size: int, is_distributed: bool):
    train_collator = _make_collator_for_dataset(train_ds)
    eval_collator = _make_collator_for_dataset(eval_ds)
    if is_distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=True,
        )
        eval_sampler = None
        if eval_ds is not None:
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
        num_workers=2,
        pin_memory=True,
        collate_fn=train_collator,
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
            collate_fn=eval_collator,
        )
    return train_dl, eval_dl


def maybe_prepare_kbit_model_for_training(model: nn.Module, args, quantization_config) -> nn.Module:
    if quantization_config is None:
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
    ap.add_argument("--min_lr_ratio", type=float, default=0.7)
    ap.add_argument("--router_lr_mult", type=float, default=1.0)
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
    ap.add_argument(
        "--router_anchor_loss_coef",
        type=float,
        default=0.01,
        help="Coefficient for expert-key to anchor alignment loss.",
    )
    ap.add_argument(
        "--router_anchor_momentum",
        type=float,
        default=None,
        help="EMA momentum for expert anchors. Unset falls back to --router_ema_momentum for config compatibility.",
    )
    ap.add_argument(
        "--router_aux_loss_coef",
        type=float,
        default=0.0,
        help="Deprecated. Kept only so older configs still parse.",
    )
    ap.add_argument(
        "--router_z_loss_coef",
        type=float,
        default=0.0,
        help="Deprecated. Kept only so older configs still parse.",
    )
    ap.add_argument(
        "--router_pull_loss_coef",
        type=float,
        default=0.0,
        help="Deprecated. Kept only so older configs still parse.",
    )
    ap.add_argument(
        "--router_pull_loss_coef_final",
        type=float,
        default=None,
        help="Deprecated. Kept only so older configs still parse.",
    )
    ap.add_argument(
        "--router_pull_loss_schedule_start_ratio",
        type=float,
        default=0.5,
        help="Deprecated. Kept only so older configs still parse.",
    )
    ap.add_argument(
        "--router_ema_momentum",
        type=float,
        default=0.99,
        help="Deprecated fallback for anchor EMA momentum when --router_anchor_momentum is unset.",
    )
    ap.add_argument(
        "--router_pull_loss_schedule_end_ratio",
        type=float,
        default=1.0,
        help="Deprecated. Kept only so older configs still parse.",
    )
    ap.add_argument(
        "--router_use_ema_update",
        type=int,
        default=1,
        help="Deprecated. Kept only so older configs still parse.",
    )
    ap.add_argument(
        "--router_pull_temperature",
        type=float,
        default=1.0,
        help="Deprecated. Kept only so older configs still parse.",
    )
    ap.add_argument(
        "--router_pull_loss_type",
        type=str,
        default="soft",
        choices=["soft", "hard_ce"],
        help="Deprecated. Kept only so older configs still parse.",
    )
    ap.add_argument(
        "--router_pull_loss_type_final",
        type=str,
        default="",
        help="Deprecated. Kept only so older configs still parse.",
    )
    ap.add_argument(
        "--router_pull_loss_type_switch_ratio",
        type=float,
        default=0.5,
        help="Deprecated. Kept only so older configs still parse.",
    )
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
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--stage1_epochs", type=int, default=1)
    ap.add_argument("--stage1_lr", type=float, default=2e-4)
    ap.add_argument("--stage1_grad_accum", type=int, default=1)
    ap.add_argument("--stage1_log_every", type=int, default=50)
    ap.add_argument("--stage1_max_grad_norm", type=float, default=1.0)
    ap.add_argument("--stage1_data_ratio", type=float, default=0.2)
    ap.add_argument("--stage_split_seed", type=int, default=42)
    ap.add_argument(
        "--stage2_include_validation_in_train",
        type=int,
        default=0,
        help="If set, append a slice of the training dataset validation split into the stage2 training set.",
    )
    ap.add_argument(
        "--stage2_validation_train_ratio",
        type=float,
        default=0.2,
        help="Fraction of validation data to borrow into the stage2 training set when enabled.",
    )
    ap.add_argument(
        "--stage2_validation_train_split",
        type=str,
        default="validation",
        help="Split name borrowed into the stage2 training set when validation augmentation is enabled.",
    )
    ap.add_argument("--stage1_distill_temperature", type=float, default=1.0)
    ap.add_argument("--stage1_kl_coef", type=float, default=1.0)
    ap.add_argument("--stage1_logit_coef", type=float, default=1.0)
    ap.add_argument("--stage1_use_logit_loss", type=int, default=1)
    ap.add_argument("--stage1_logit_loss_type", type=str, default="huber", choices=["huber", "mse"])

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--router_top_k", type=int, default=0)
    ap.add_argument("--router_topk", type=int, default=0)
    ap.add_argument(
        "--top_p_threshold",
        type=float,
        default=None,
        help="Deprecated under fixed top-k routing; accepted for backward compatibility but ignored.",
    )
    ap.add_argument(
        "--router_top_p_final",
        type=float,
        default=None,
        help="Deprecated under fixed top-k routing; accepted for backward compatibility but ignored.",
    )
    ap.add_argument(
        "--router_top_p_schedule_start_ratio",
        type=float,
        default=0.5,
        help="Deprecated under fixed top-k routing; accepted for backward compatibility but ignored.",
    )
    ap.add_argument(
        "--router_top_p_schedule_end_ratio",
        type=float,
        default=1.0,
        help="Deprecated under fixed top-k routing; accepted for backward compatibility but ignored.",
    )
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
        "--router_use_softmax_temperature",
        type=int,
        default=-1,
        help="Enable temperature scaling for CrossAttentionRouter softmax when set to 0/1.",
    )
    ap.add_argument(
        "--router_softmax_temperature",
        type=float,
        default=None,
        help="Softmax temperature for CrossAttentionRouter; smaller is sharper.",
    )
    ap.add_argument(
        "--use_router_context",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help="Set -1 to keep config value, 0 to disable adding router context back to hidden states, 1 to enable.",
    )
    ap.add_argument(
        "--router_context_scale",
        type=float,
        default=None,
        help="Scale factor applied to projected router context before adding it back to hidden states.",
    )
    ap.add_argument(
        "--share_router_expert_embedding",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help="Set -1 to keep config value, 0 for per-layer expert embeddings, 1 for one global expert embedding shared across layers.",
    )
    ap.add_argument(
        "--router_budget_loss_coef",
        type=float,
        default=0.0,
        help="Deprecated under fixed top-k routing; accepted for backward compatibility but ignored.",
    )
    ap.add_argument(
        "--router_budget_target_count",
        type=float,
        default=0.0,
        help="Deprecated under fixed top-k routing; accepted for backward compatibility but ignored.",
    )
    ap.add_argument(
        "--router_budget_tau",
        type=float,
        default=0.05,
        help="Deprecated under fixed top-k routing; accepted for backward compatibility but ignored.",
    )
    ap.add_argument(
        "--router_budget_start_ratio",
        type=float,
        default=0.5,
        help="Deprecated under fixed top-k routing; accepted for backward compatibility but ignored.",
    )
    ap.add_argument(
        "--router_budget_end_ratio",
        type=float,
        default=1.0,
        help="Deprecated under fixed top-k routing; accepted for backward compatibility but ignored.",
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
    if not (0.0 < float(args.stage1_data_ratio) < 1.0):
        raise ValueError(f"--stage1_data_ratio must be in (0, 1), got {args.stage1_data_ratio}")
    if not (0.0 <= float(args.stage2_validation_train_ratio) <= 1.0):
        raise ValueError(
            f"--stage2_validation_train_ratio must be in [0, 1], got {args.stage2_validation_train_ratio}"
        )
    if float(args.router_anchor_loss_coef) < 0.0:
        raise ValueError(
            f"--router_anchor_loss_coef must be >= 0, got {args.router_anchor_loss_coef}"
        )
    anchor_momentum = args.router_anchor_momentum
    if anchor_momentum is None:
        anchor_momentum = args.router_ema_momentum
    if not (0.0 <= float(anchor_momentum) < 1.0):
        raise ValueError(
            f"--router_anchor_momentum/--router_ema_momentum must be in [0, 1), got {anchor_momentum}"
        )
    for name in (
        "router_pull_loss_schedule_start_ratio",
        "router_pull_loss_schedule_end_ratio",
        "router_pull_loss_type_switch_ratio",
    ):
        value = float(getattr(args, name))
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"--{name} must be in [0, 1], got {value}")
    if float(args.router_pull_loss_coef) < 0.0:
        raise ValueError(
            f"--router_pull_loss_coef must be >= 0, got {args.router_pull_loss_coef}"
        )
    if args.router_pull_loss_coef_final is not None and float(args.router_pull_loss_coef_final) < 0.0:
        raise ValueError(
            f"--router_pull_loss_coef_final must be >= 0, got {args.router_pull_loss_coef_final}"
        )
    if args.router_pull_loss_type_final:
        args.router_pull_loss_type_final = str(args.router_pull_loss_type_final).strip().lower()
    if args.router_pull_loss_type_final and args.router_pull_loss_type_final not in {"soft", "hard_ce"}:
        raise ValueError(
            "--router_pull_loss_type_final must be one of {'soft', 'hard_ce'} when provided, "
            f"got {args.router_pull_loss_type_final!r}"
        )
    local_rank, world_size, is_distributed = setup_distributed_safe()
    set_seed(args.seed + (local_rank if is_distributed else 0))

    device = torch.device(f"cuda:{local_rank}") if is_distributed else torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    use_fp16 = bool(args.fp16) and device.type == "cuda" and not bool(args.bf16)
    use_bf16 = bool(args.bf16) and device.type == "cuda"
    _warn_ignored_legacy_router_settings(args)

    if is_main_process():
        print("[load] tokenizer from model_path:", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    model_cls, config_cls = import_moe_classes()
    quantization_config = build_quantization_config(args)
    if is_main_process():
        print(f"[load] distributed={is_distributed} world_size={world_size} 4bit={bool(args.load_in_4bit)} 8bit={bool(args.load_in_8bit)}")
        if bool(args.load_in_4bit):
            print(f"[load] 4bit compute_dtype={quantization_config.bnb_4bit_compute_dtype}")
        print(f"[train] mixed precision: fp16={use_fp16} bf16={use_bf16}")

    stage1_ckpt_dir = os.path.join(args.output_dir, "ckpt_after_stage1")

    if args.stage in (0, 1):
        if is_main_process():
            print("[stage1] loading base model for router distillation")

        stage1_config = config_cls.from_pretrained(args.model_path)
        configure_model_config(stage1_config, args)
        stage1_model = build_model_with_router_compat(
            args=args,
            model_cls=model_cls,
            config=stage1_config,
            model_path=args.model_path,
            device=device,
            quantization_config=quantization_config,
            local_rank=local_rank,
            is_distributed=is_distributed,
            init_from_legacy=True,
        )
        stage1_model = maybe_prepare_kbit_model_for_training(stage1_model, args, quantization_config)
        n_stage1_router_params = prepare_stage1_router_trainables(stage1_model)
        if is_main_process():
            print(f"[stage1] trainable cross-attention router params: {n_stage1_router_params}")

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

        if is_distributed:
            stage1_model = torch.nn.parallel.DistributedDataParallel(
                stage1_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )

        stage1_train_cross_attention_router(
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

    stage2_config = config_cls.from_pretrained(stage2_model_path)
    configure_model_config(stage2_config, args)
    stage2_model = build_model_with_router_compat(
        args=args,
        model_cls=model_cls,
        config=stage2_config,
        model_path=stage2_model_path,
        device=device,
        quantization_config=quantization_config,
        local_rank=local_rank,
        is_distributed=is_distributed,
        init_from_legacy=os.path.realpath(stage2_model_path) == os.path.realpath(args.model_path),
    )
    stage2_model = maybe_prepare_kbit_model_for_training(stage2_model, args, quantization_config)

    targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()] if args.lora_target_modules else None
    stage2_model = apply_lora(
        stage2_model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=targets,
    )

    if bool(args.train_new_router_params):
        n_router = _enable_new_router_params_trainable(stage2_model)
        if is_main_process():
            print(f"[stage2] extra trainable new-router params: {n_router}")

    n_fp32 = 0
    if bool(args.train_extra_params_in_fp32):
        n_fp32 = _cast_selected_trainable_params_to_fp32(stage2_model)
    if is_main_process():
        trainable = sum(p.numel() for p in stage2_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in stage2_model.parameters())
        print(f"[stage2] cast selected router params to fp32: {n_fp32}")
        print(f"[stage2] trainable params: {trainable}/{total} ({trainable / total * 100:.2f}%)")
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
        router_lr_mult=args.router_lr_mult,
        min_lr_ratio=args.min_lr_ratio,
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
