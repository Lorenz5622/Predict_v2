#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finetune baseline MoE CausalLM (local) on PIQA / SocialIQA (SIQA) with LoRA.

- Dataset format follows the "OpenCompass-style" prompt+answer with prompt masked in labels
  (same idea as load_and_pack_piqa_ppl_opencompass in train_p_then_lora.py).
- Works on single GPU (default) and supports torchrun DDP (optional).

Example (single GPU):
  python finetune_piqa_siqa_lora.py \
    --model_path /path/to/baseline_model \
    --output_dir ./out_piqa_lora \
    --dataset piqa --eval_dataset piqa \
    --block_size 512 --batch_size 2 --grad_accum 8 --epochs 1 \
    --lr 2e-4 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05

Example (DDP):
  torchrun --nproc_per_node 8 finetune_piqa_siqa_lora.py \
    --model_path /path/to/baseline_model --output_dir ./out_mix \
    --dataset mix --mix_datasets piqa,siqa --eval_dataset siqa

Notes:
- Assumes your baseline model is saved in HuggingFace format under --model_path, i.e. contains
  config.json + (pytorch_model.bin or safetensors) etc.
- If your codebase has MoEConfig/MoEForCausalLM under different module paths, this script tries
  a few common import fallbacks.
"""
from __future__ import annotations

import os
import math
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import deque
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

from peft import LoraConfig, get_peft_model, TaskType


# -----------------------------
# Distributed utils
# -----------------------------
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        return local_rank, world_size, True
    return -1, 1, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Model imports (baseline = modeling_moe_ori)
# -----------------------------
def import_moe_classes():
    """
    Try multiple import paths so this script can be dropped into different repos.

    Priority:
      1) Local directory: modeling_moe_ori.py + configuration_moe.py in the same folder.
      2) Package-style: Predict_MoE.modeling.(...)
      3) Anything already installed in PYTHONPATH.

    Returns:
      (MoEForCausalLM, MoEConfig)
    """
    # 1) try local files (same directory as this script)
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in os.sys.path:
        os.sys.path.insert(0, here)

    tried = []

    for (m_model, m_cfg) in [
        ("modeling_moe_ori", "configuration_moe"),
        ("Predict_MoE.modeling.modeling_moe_ori", "Predict_MoE.modeling.configuration_moe"),
    ]:
        try:
            mod_model = __import__(m_model, fromlist=["MoEForCausalLM"])
            mod_cfg = __import__(m_cfg, fromlist=["MoEConfig"])
            return mod_model.MoEForCausalLM, mod_cfg.MoEConfig
        except Exception as e:
            tried.append((m_model, m_cfg, repr(e)))

    # Last resort: raise with hints
    msg = "Failed to import MoEForCausalLM/MoEConfig. Tried:\n"
    msg += "\n".join([f"- {a} / {b}: {err}" for a, b, err in tried])
    raise ImportError(msg)


# -----------------------------
# Dataset packing (OpenCompass style)
# -----------------------------
def _build_prompt_mcq(question: str, choice_texts: List[str], choice_labels: Optional[List[str]] = None) -> str:
    """Deterministic MCQ prompt."""
    if choice_labels is None:
        choice_labels = [chr(ord("A") + i) for i in range(len(choice_texts))]
    lines = ["Question:", question.strip(), "", "Choices:"]
    for lab, txt in zip(choice_labels, choice_texts):
        lines.append(f"{lab}. {str(txt).strip()}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def _answerkey_to_index(answer_key: str, choice_labels: List[str]) -> Optional[int]:
    """answer_key usually like 'A'/'B'/... ; sometimes numeric string."""
    ak = str(answer_key).strip()
    if ak in choice_labels:
        return choice_labels.index(ak)
    try:
        k = int(ak)
        if 0 <= k < len(choice_labels):
            return k
    except Exception:
        pass
    return None

def _pad_to_block(input_ids: List[int], labels: List[int], *, block_size: int, pad_id: int = 0) -> Dict[str, List[int]]:
    input_ids = input_ids[:block_size]
    labels = labels[:block_size]
    pad_len = block_size - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len
    return {"input_ids": input_ids, "labels": labels}

def shuffle_select(ds, max_samples, seed: int):
    if max_samples is None:
        return ds
    ds = ds.shuffle(seed=int(seed))
    return ds.select(range(min(int(max_samples), len(ds))))

def load_and_pack_piqa_ppl_opencompass(
    tokenizer,
    block_size: int,
    split: str = "train",
    num_proc: int = 1,
    bos: bool = True,
    eos: bool = False,
    max_samples: Optional[int] = None,
    use_label: bool = True,
):
    """
    PIQA fields:
      goal, sol1, sol2, label (0->sol1, 1->sol2).
    """
    ds = load_dataset("ybisk/piqa", split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = 0

    def build(ex):
        goal = (ex.get("goal") or "").strip()
        sol1 = (ex.get("sol1") or "").strip()
        sol2 = (ex.get("sol2") or "").strip()
        label = ex.get("label", None)

        if label is None or (not use_label):
            sol = sol1
        else:
            lab = int(label)
            sol = sol1 if lab == 0 else sol2

        prompt = (
            "The following makes sense:\n"
            f"Q: {goal}\n"
            "A:"
        )

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ans_ids = tokenizer(" " + sol, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + ans_ids
        labels = ([-100] * len(prompt_ids)) + ans_ids

        if bos and bos_id is not None:
            input_ids = [bos_id] + input_ids
            labels = [-100] + labels
        if eos and eos_id is not None and eos:
            input_ids = input_ids + [eos_id]
            labels = labels + [eos_id]

        return _pad_to_block(input_ids, labels, block_size=block_size, pad_id=pad_id)

    ds = ds.map(build, num_proc=num_proc, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "labels"])
    return ds


def load_and_pack_siqa_ppl_opencompass(
    tokenizer,
    block_size: int,
    split: str = "train",
    num_proc: int = 1,
    bos: bool = True,
    eos: bool = False,
    max_samples: Optional[int] = None,
    use_label: bool = True,
    seed: int = 42,
):
    """
    Social IQa (allenai/social_i_qa) fields:
      context, question, answerA/B/C, label ("1".."3")
    """
    ds = load_dataset("allenai/social_i_qa", split=split, trust_remote_code=True)
    ds = shuffle_select(ds, max_samples=max_samples, seed=seed)
    # if max_samples is not None:
    #     ds = ds.select(range(min(max_samples, len(ds))))

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = 0

    def build(ex):
        ctx = (ex.get("context") or "").strip()
        q = (ex.get("question") or "").strip()
        aA = (ex.get("answerA") or "").strip()
        aB = (ex.get("answerB") or "").strip()
        aC = (ex.get("answerC") or "").strip()
        lab = ex.get("label", None)

        if (lab is None) or (not use_label):
            # fallback: just use answerA
            ans = aA
        else:
            # dataset uses "1","2","3"
            try:
                k = int(str(lab).strip())
            except Exception:
                k = 1
            ans = {1: aA, 2: aB, 3: aC}.get(k, aA)

        prompt = (
            "Social commonsense question:\n"
            f"Context: {ctx}\n"
            f"Question: {q}\n"
            "Answer:"
        )

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ans_ids = tokenizer(" " + ans, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + ans_ids
        labels = ([-100] * len(prompt_ids)) + ans_ids

        if bos and bos_id is not None:
            input_ids = [bos_id] + input_ids
            labels = [-100] + labels
        if eos and eos_id is not None and eos:
            input_ids = input_ids + [eos_id]
            labels = labels + [eos_id]

        return _pad_to_block(input_ids, labels, block_size=block_size, pad_id=pad_id)

    ds = ds.map(build, num_proc=num_proc, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "labels"])
    return ds

def load_and_pack_hellaswag_ppl_opencompass(
    tokenizer,
    block_size: int,
    split: str = "train",
    num_proc: int = 1,
    bos: bool = True,
    eos: bool = False,
    max_samples: Optional[int] = None,
    use_label: bool = True,
    seed: int = 42,
):
    """
    HellaSwag (Rowan/hellaswag) fields:
      ctx_a, ctx_b, endings(list[str]), label(str/int index)
    """

    ds = load_dataset("Rowan/hellaswag", split=split, trust_remote_code=True)
    if max_samples is not None:
        ds = shuffle_select(ds, max_samples=max_samples, seed=seed)
    # if max_samples is not None:
    #     ds = ds.select(range(min(max_samples, len(ds))))

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = 0

    def build(ex):
        ctx_a = (ex.get("ctx_a") or "").strip()
        ctx_b = (ex.get("ctx_b") or "").strip()
        endings = ex.get("endings") or []
        lab = ex.get("label", None)

        # label: correct ending index (often stored as string)
        gold = 0
        if use_label and lab is not None:
            try:
                gold = int(str(lab).strip())
            except Exception:
                gold = 0
        gold = max(0, min(gold, len(endings) - 1)) if endings else 0

        prompt = "Complete the following:\n" + f"{ctx_a} {ctx_b}".strip() + "\nAnswer:"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ans_text = str(endings[gold]).strip() if endings else ""
        ans_ids = tokenizer(" " + ans_text, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + ans_ids
        labels = ([-100] * len(prompt_ids)) + ans_ids

        if bos and bos_id is not None:
            input_ids = [bos_id] + input_ids
            labels = [-100] + labels
        if eos and eos_id is not None and eos:
            input_ids = input_ids + [eos_id]
            labels = labels + [eos_id]

        return _pad_to_block(input_ids, labels, block_size=block_size, pad_id=pad_id)

    ds = ds.map(build, num_proc=num_proc, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "labels"])
    return ds


def load_and_pack_arc_easy_ppl_opencompass(
    tokenizer,
    block_size: int,
    split: str = "train",
    num_proc: int = 1,
    bos: bool = True,
    eos: bool = False,
    max_samples: Optional[int] = None,
    use_label: bool = True,
):
    """
    ARC-Easy (allenai/ai2_arc, config=ARC-Easy) fields:
      question, choices{text,label}, answerKey
    """
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = 0

    def build(ex):
        q = (ex.get("question") or "").strip()
        ch = ex.get("choices") or {}
        texts = list(ch.get("text") or [])
        labels_ = list(ch.get("label") or [])
        ak = ex.get("answerKey", None)

        gold = 0
        if use_label and ak is not None:
            idx = _answerkey_to_index(ak, labels_)
            gold = 0 if idx is None else idx
        gold = max(0, min(gold, len(texts) - 1)) if texts else 0

        prompt = _build_prompt_mcq(q, texts, labels_)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ans_text = str(texts[gold]).strip() if texts else ""
        ans_ids = tokenizer(" " + ans_text, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + ans_ids
        out_labels = ([-100] * len(prompt_ids)) + ans_ids

        if bos and bos_id is not None:
            input_ids = [bos_id] + input_ids
            out_labels = [-100] + out_labels
        if eos and eos_id is not None and eos:
            input_ids = input_ids + [eos_id]
            out_labels = out_labels + [eos_id]

        return _pad_to_block(input_ids, out_labels, block_size=block_size, pad_id=pad_id)

    ds = ds.map(build, num_proc=num_proc, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "labels"])
    return ds


def load_and_pack_commonsenseqa_ppl_opencompass(
    tokenizer,
    block_size: int,
    split: str = "train",
    num_proc: int = 1,
    bos: bool = True,
    eos: bool = False,
    max_samples: Optional[int] = None,
    use_label: bool = True,
):
    """
    CommonsenseQA (tau/commonsense_qa) fields:
      question, choices{label,text}, answerKey
    """
    ds = load_dataset("tau/commonsense_qa", split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = 0

    def build(ex):
        q = (ex.get("question") or "").strip()
        ch = ex.get("choices") or {}
        labels_ = list(ch.get("label") or [])
        texts = list(ch.get("text") or [])
        ak = ex.get("answerKey", None)

        gold = 0
        if use_label and ak is not None:
            idx = _answerkey_to_index(ak, labels_)
            gold = 0 if idx is None else idx
        gold = max(0, min(gold, len(texts) - 1)) if texts else 0

        prompt = _build_prompt_mcq(q, texts, labels_)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ans_text = str(texts[gold]).strip() if texts else ""
        ans_ids = tokenizer(" " + ans_text, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + ans_ids
        out_labels = ([-100] * len(prompt_ids)) + ans_ids

        if bos and bos_id is not None:
            input_ids = [bos_id] + input_ids
            out_labels = [-100] + out_labels
        if eos and eos_id is not None and eos:
            input_ids = input_ids + [eos_id]
            out_labels = out_labels + [eos_id]

        return _pad_to_block(input_ids, out_labels, block_size=block_size, pad_id=pad_id)

    ds = ds.map(build, num_proc=num_proc, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "labels"])
    return ds


def load_and_pack_bbh_ppl_opencompass(
    tokenizer,
    block_size: int,
    split: str = "test",
    num_proc: int = 1,
    bos: bool = True,
    eos: bool = False,
    max_samples: Optional[int] = None,
    task: str = "boolean_expressions",
):
    """
    BBH (lukaemon/bbh): each task is a config, fields: input, target.
    Most tasks only provide split 'test'.
    """
    ds = load_dataset("lukaemon/bbh", task, split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = 0

    def build(ex):
        inp = (ex.get("input") or "").strip()
        tgt = (ex.get("target") or "").strip()

        prompt = inp + "\nAnswer:"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ans_ids = tokenizer(" " + tgt, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + ans_ids
        labels = ([-100] * len(prompt_ids)) + ans_ids

        if bos and bos_id is not None:
            input_ids = [bos_id] + input_ids
            labels = [-100] + labels
        if eos and eos_id is not None and eos:
            input_ids = input_ids + [eos_id]
            labels = labels + [eos_id]

        return _pad_to_block(input_ids, labels, block_size=block_size, pad_id=pad_id)

    ds = ds.map(build, num_proc=num_proc, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "labels"])
    return ds


def load_and_pack_winogrande_ppl_opencompass(
    tokenizer,
    block_size: int,
    split: str = "train",
    num_proc: int = 1,
    bos: bool = True,
    eos: bool = False,
    max_samples: Optional[int] = None,
    config_name: str = "winogrande_xl",
    use_label: bool = True,
):
    """
    WinoGrande (allenai/winogrande) fields:
      sentence, option1, option2, answer ("1" or "2")
    HF dataset card lists configs like winogrande_xl/winogrande_l/...
    """
    ds = load_dataset("allenai/winogrande", config_name, split=split, trust_remote_code=True)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    def build(ex):
        sent = (ex.get("sentence") or "").strip()
        o1 = (ex.get("option1") or "").strip()
        o2 = (ex.get("option2") or "").strip()
        ans = ex.get("answer", None)

        # answer is "1"/"2" in HF dataset
        if (ans is None) or (not use_label):
            gold_idx = 0
        else:
            try:
                gold_idx = int(str(ans).strip()) - 1
            except Exception:
                gold_idx = 0
            gold_idx = 0 if gold_idx not in (0, 1) else gold_idx

        # OpenCompass-style: prompt + correct answer text continuation
        prompt = (
            "Fill in the blank:\n"
            f"{sent}\n"
            "Options:\n"
            f"A. {o1}\n"
            f"B. {o2}\n"
            "Answer:"
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

        gold_text = o1 if gold_idx == 0 else o2
        ans_ids = tokenizer(" " + gold_text, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + ans_ids
        labels = ([-100] * len(prompt_ids)) + ans_ids

        if bos and bos_id is not None:
            input_ids = [bos_id] + input_ids
            labels = [-100] + labels
        if eos and eos_id is not None and eos:
            input_ids = input_ids + [eos_id]
            labels = labels + [eos_id]

        packed = _pad_to_block(input_ids, labels, block_size=block_size, pad_id=pad_id)
        return {"input_ids": packed["input_ids"], "labels": packed["labels"]}

    ds = ds.map(build, num_proc=num_proc, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "labels"])
    return ds

def load_and_pack_mmlu_ppl_opencompass(
    tokenizer,
    block_size: int,
    split: str = "dev",
    num_proc: int = 1,
    bos: bool = True,
    eos: bool = False,
    max_samples: Optional[int] = None,
    subjects: Optional[List[str]] = None,   # e.g. ["abstract_algebra", "anatomy"] or None for all
    answer_mode: str = "text",              # "text" (default) or "letter"
):
    """
    MMLU (cais/mmlu): each subject is a config. Fields:
      question: str
      choices: list[str] length 4
      answer: ClassLabel with names ["A","B","C","D"] (or int-like)
      subject: str
    Splits include dev/validation/test. (HF dataset_infos.json) :contentReference[oaicite:2]{index=2}
    """
    from datasets import get_dataset_config_names

    if subjects is None or len(subjects) == 0:
        subjects = get_dataset_config_names("cais/mmlu")  # all subjects
    # 过滤掉非学科 config
    skip = {"all", "auxiliary_train", "auxillary_train"}  # 兼容常见拼写
    subjects = [s for s in subjects if s not in skip]
    # load and concatenate subjects
    parts = []
    for subj in subjects:
        d = load_dataset("cais/mmlu", subj, split=split, trust_remote_code=True)
        parts.append(d)
    ds = concatenate_datasets(parts)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    letters = ["A", "B", "C", "D"]

    def build(ex):
        q = (ex.get("question") or "").strip()
        choices = ex.get("choices") or []
        # normalize to 4
        if len(choices) < 4:
            choices = list(choices) + [""] * (4 - len(choices))
        choices = [str(x).strip() for x in choices[:4]]

        ans = ex.get("answer", None)  # often int-like classlabel index
        if ans is None:
            gold_idx = 0
        else:
            try:
                gold_idx = int(ans)
            except Exception:
                # sometimes it's "A"/"B"/...
                a = str(ans).strip()
                gold_idx = letters.index(a) if a in letters else 0
        gold_idx = 0 if gold_idx not in (0, 1, 2, 3) else gold_idx

        prompt = (
            "Multiple choice question:\n"
            f"Question: {q}\n"
            "Choices:\n"
            f"A. {choices[0]}\n"
            f"B. {choices[1]}\n"
            f"C. {choices[2]}\n"
            f"D. {choices[3]}\n"
            "Answer:"
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

        if answer_mode == "letter":
            gold = letters[gold_idx]
        else:
            gold = choices[gold_idx]
        ans_ids = tokenizer(" " + str(gold).strip(), add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + ans_ids
        labels = ([-100] * len(prompt_ids)) + ans_ids

        if bos and bos_id is not None:
            input_ids = [bos_id] + input_ids
            labels = [-100] + labels
        if eos and eos_id is not None and eos:
            input_ids = input_ids + [eos_id]
            labels = labels + [eos_id]

        packed = _pad_to_block(input_ids, labels, block_size=block_size, pad_id=pad_id)
        return {"input_ids": packed["input_ids"], "labels": packed["labels"]}

    ds = ds.map(build, num_proc=num_proc, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "labels"])
    return ds

def load_and_pack_arc_challenge_ppl_opencompass(
    tokenizer,
    block_size: int,
    split: str = "train",
    num_proc: int = 1,
    bos: bool = True,
    eos: bool = False,
    max_samples: Optional[int] = None,
    use_label: bool = True,
):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = 0

    def build(ex):
        q = (ex.get("question") or "").strip()
        ch = ex.get("choices") or {}
        texts = list(ch.get("text") or [])
        labels_ = list(ch.get("label") or [])
        ak = ex.get("answerKey", None)

        gold = 0
        if use_label and ak is not None:
            idx = _answerkey_to_index(ak, labels_)
            gold = 0 if idx is None else idx
        if not texts:
            texts = [""]
            labels_ = ["A"]
            gold = 0
        gold = max(0, min(gold, len(texts) - 1))

        prompt = _build_prompt_mcq(q, texts, labels_)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ans_text = str(texts[gold]).strip()
        ans_ids = tokenizer(" " + ans_text, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + ans_ids
        out_labels = ([-100] * len(prompt_ids)) + ans_ids

        if bos and bos_id is not None:
            input_ids = [bos_id] + input_ids
            out_labels = [-100] + out_labels
        if eos and eos_id is not None and eos:
            input_ids = input_ids + [eos_id]
            out_labels = out_labels + [eos_id]

        return _pad_to_block(input_ids, out_labels, block_size=block_size, pad_id=pad_id)

    ds = ds.map(build, num_proc=num_proc, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "labels"])
    return ds


def load_and_pack_openbookqa_ppl_opencompass(
    tokenizer,
    block_size: int,
    split: str = "train",
    num_proc: int = 1,
    bos: bool = True,
    eos: bool = False,
    max_samples: Optional[int] = None,
    use_label: bool = True,
):
    ds = load_dataset("allenai/openbookqa", "main", split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = 0

    def build(ex):
        q = (ex.get("question_stem") or "").strip()
        ch = ex.get("choices") or {}
        texts = list(ch.get("text") or [])
        labels_ = list(ch.get("label") or [])
        ak = ex.get("answerKey", None)

        gold = 0
        if use_label and ak is not None:
            idx = _answerkey_to_index(ak, labels_)
            gold = 0 if idx is None else idx
        if not texts:
            texts = [""]
            labels_ = ["A"]
            gold = 0
        gold = max(0, min(gold, len(texts) - 1))

        prompt = _build_prompt_mcq(q, texts, labels_)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ans_text = str(texts[gold]).strip()
        ans_ids = tokenizer(" " + ans_text, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + ans_ids
        out_labels = ([-100] * len(prompt_ids)) + ans_ids

        if bos and bos_id is not None:
            input_ids = [bos_id] + input_ids
            out_labels = [-100] + out_labels
        if eos and eos_id is not None and eos:
            input_ids = input_ids + [eos_id]
            out_labels = out_labels + [eos_id]

        return _pad_to_block(input_ids, out_labels, block_size=block_size, pad_id=pad_id)

    ds = ds.map(build, num_proc=num_proc, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "labels"])
    return ds
# def load_and_pack_winogrande_ppl_opencompass(
#     tokenizer,
#     block_size: int,
#     split: str = "train",
#     num_proc: int = 1,
#     bos: bool = True,
#     eos: bool = False,
#     max_samples: Optional[int] = None,
#     config_name: str = "winogrande_xl",
#     use_label: bool = True,
# ):
#     """
#     WinoGrande (allenai/winogrande) fields:
#       sentence, option1, option2, answer ("1" or "2")
#     HF dataset card lists configs like winogrande_xl/winogrande_l/...
#     """
#     ds = load_dataset("allenai/winogrande", config_name, split=split, trust_remote_code=True)
#     if max_samples is not None:
#         ds = ds.select(range(min(max_samples, len(ds))))

#     pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
#     bos_id = tokenizer.bos_token_id
#     eos_id = tokenizer.eos_token_id

#     def build(ex):
#         sent = (ex.get("sentence") or "").strip()
#         o1 = (ex.get("option1") or "").strip()
#         o2 = (ex.get("option2") or "").strip()
#         ans = ex.get("answer", None)

#         # answer is "1"/"2" in HF dataset
#         if (ans is None) or (not use_label):
#             gold_idx = 0
#         else:
#             try:
#                 gold_idx = int(str(ans).strip()) - 1
#             except Exception:
#                 gold_idx = 0
#             gold_idx = 0 if gold_idx not in (0, 1) else gold_idx

#         # OpenCompass-style: prompt + correct answer text continuation
#         prompt = (
#             "Fill in the blank:\n"
#             f"{sent}\n"
#             "Options:\n"
#             f"A. {o1}\n"
#             f"B. {o2}\n"
#             "Answer:"
#         )
#         prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

#         gold_text = o1 if gold_idx == 0 else o2
#         ans_ids = tokenizer(" " + gold_text, add_special_tokens=False)["input_ids"]

#         input_ids = prompt_ids + ans_ids
#         labels = ([-100] * len(prompt_ids)) + ans_ids

#         if bos and bos_id is not None:
#             input_ids = [bos_id] + input_ids
#             labels = [-100] + labels
#         if eos and eos_id is not None and eos:
#             input_ids = input_ids + [eos_id]
#             labels = labels + [eos_id]

#         return _pad_to_block(input_ids, labels, block_size=block_size, pad_id=pad_id)

#     ds = ds.map(build, num_proc=num_proc, remove_columns=ds.column_names)
#     ds.set_format(type="torch", columns=["input_ids", "labels"])
#     return ds

# def load_and_pack_mmlu_ppl_opencompass(
#     tokenizer,
#     block_size: int,
#     split: str = "dev",
#     num_proc: int = 1,
#     bos: bool = True,
#     eos: bool = False,
#     max_samples: Optional[int] = None,
#     subjects: Optional[List[str]] = None,   # e.g. ["abstract_algebra", "anatomy"] or None for all
#     answer_mode: str = "text",              # "text" (default) or "letter"
# ):
#     """
#     MMLU (cais/mmlu): each subject is a config. Fields:
#       question: str
#       choices: list[str] length 4
#       answer: ClassLabel with names ["A","B","C","D"] (or int-like)
#       subject: str
#     Splits include dev/validation/test. (HF dataset_infos.json) :contentReference[oaicite:2]{index=2}
#     """
#     from datasets import get_dataset_config_names

#     if subjects is None or len(subjects) == 0:
#         subjects = get_dataset_config_names("cais/mmlu")  # all subjects
#     # 过滤掉非学科 config
#     skip = {"all", "auxiliary_train", "auxillary_train"}  # 兼容常见拼写
#     subjects = [s for s in subjects if s not in skip]
#     # load and concatenate subjects
#     parts = []
#     for subj in subjects:
#         d = load_dataset("cais/mmlu", subj, split=split, trust_remote_code=True)
#         parts.append(d)
#     ds = concatenate_datasets(parts)

#     if max_samples is not None:
#         ds = ds.select(range(min(max_samples, len(ds))))

#     pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
#     bos_id = tokenizer.bos_token_id
#     eos_id = tokenizer.eos_token_id
#     letters = ["A", "B", "C", "D"]

#     def build(ex):
#         q = (ex.get("question") or "").strip()
#         choices = ex.get("choices") or []
#         # normalize to 4
#         if len(choices) < 4:
#             choices = list(choices) + [""] * (4 - len(choices))
#         choices = [str(x).strip() for x in choices[:4]]

#         ans = ex.get("answer", None)  # often int-like classlabel index
#         if ans is None:
#             gold_idx = 0
#         else:
#             try:
#                 gold_idx = int(ans)
#             except Exception:
#                 # sometimes it's "A"/"B"/...
#                 a = str(ans).strip()
#                 gold_idx = letters.index(a) if a in letters else 0
#         gold_idx = 0 if gold_idx not in (0, 1, 2, 3) else gold_idx

#         prompt = (
#             "Multiple choice question:\n"
#             f"Question: {q}\n"
#             "Choices:\n"
#             f"A. {choices[0]}\n"
#             f"B. {choices[1]}\n"
#             f"C. {choices[2]}\n"
#             f"D. {choices[3]}\n"
#             "Answer:"
#         )
#         prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

#         if answer_mode == "letter":
#             gold = letters[gold_idx]
#         else:
#             gold = choices[gold_idx]
#         ans_ids = tokenizer(" " + str(gold).strip(), add_special_tokens=False)["input_ids"]

#         input_ids = prompt_ids + ans_ids
#         labels = ([-100] * len(prompt_ids)) + ans_ids

#         if bos and bos_id is not None:
#             input_ids = [bos_id] + input_ids
#             labels = [-100] + labels
#         if eos and eos_id is not None and eos:
#             input_ids = input_ids + [eos_id]
#             labels = labels + [eos_id]

#         return _pad_to_block(input_ids, labels, block_size=block_size, pad_id=pad_id)

#     ds = ds.map(build, num_proc=num_proc, remove_columns=ds.column_names)
#     ds.set_format(type="torch", columns=["input_ids", "labels"])
#     return ds



# -----------------------------
# Data collator
# -----------------------------
@dataclass
class LMDataCollator:
    pad_id: int = 0

    def __call__(self, features):
        input_ids = torch.stack([f["input_ids"] for f in features], dim=0)
        labels = torch.stack([f["labels"] for f in features], dim=0)
        # attention_mask is optional; model can infer from pad_id if needed, but providing helps some impls
        attention_mask = (input_ids != self.pad_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# -----------------------------
# LoRA helpers
# -----------------------------
def guess_lora_targets(model: nn.Module) -> List[str]:
    """
    Best-effort guesses for common target modules.
    If you want to be strict, pass --lora_target_modules manually.
    """
    # Common LLaMA-ish names
    candidates = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    # Keep only those present
    names = set()
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            short = n.split(".")[-1]
            if short in candidates:
                names.add(short)
    # For PEFT, target_modules expects module *names*, not full paths, unless you configured otherwise.
    return sorted(names)


def apply_lora(
    model: nn.Module,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Optional[List[str]] = None,
):
    if target_modules is None or len(target_modules) == 0:
        target_modules = guess_lora_targets(model)
        if len(target_modules) == 0:
            raise RuntimeError("Could not infer LoRA target modules; please pass --lora_target_modules")

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    return model


def is_quantized_model(model: nn.Module) -> bool:
    return any([
        bool(getattr(model, "is_loaded_in_8bit", False)),
        bool(getattr(model, "is_loaded_in_4bit", False)),
        getattr(model, "quantization_method", None) is not None,
    ])


# -----------------------------
# Train / eval
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, device: torch.device, fp16: bool, bf16: bool) -> float:
    model.eval()
    losses = []
    amp_dtype = torch.float16 if fp16 else (torch.bfloat16 if bf16 else None)

    for batch in dl:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if amp_dtype is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(**batch)
                loss = out.loss
        else:
            out = model(**batch)
            loss = out.loss
        losses.append(loss.detach().float())

    if len(losses) == 0:
        return float("nan")
    loss = torch.stack(losses).mean()

    # DDP reduce
    if dist.is_initialized():
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss = loss / dist.get_world_size()

    return float(loss.item())


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, use_bnb_8bit: bool):
    if use_bnb_8bit:
        try:
            import bitsandbytes as bnb
        except Exception as e:
            raise ImportError("bitsandbytes is not available but --use_bnb_8bit=1 was set") from e
        return bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train(
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
    fp16: bool,
    bf16: bool,
    use_bnb_8bit: bool,
    log_every: int,
    eval_every: int,
    save_every: int,
):
    os.makedirs(output_dir, exist_ok=True)

    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay, use_bnb_8bit=use_bnb_8bit)

    # total steps
    steps_per_epoch = math.ceil(len(train_dl) / max(1, grad_accum))
    total_optim_steps = steps_per_epoch * epochs
    warmup_steps = int(total_optim_steps * warmup_ratio)

    pbar = tqdm(
        total=total_optim_steps,
        disable=not is_main_process(),   # DDP 只让 rank0 打印，避免一堆条
        dynamic_ncols=True,
        desc="train"
    )

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine
        progress = (step - warmup_steps) / float(max(1, total_optim_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device.type == "cuda"))

    model.train()

    global_step = 0
    optim_step = 0
    t0 = time.time()


        # -----------------------------
    # Logging like train_p_then_lora: loss_ma + step
    # -----------------------------
    metrics_f = None
    ma_win = 50                       # 对齐你那边常用的 ma50
    ma_loss_buf = deque(maxlen=ma_win)
    ma_loss_sum = 0.0
    ema_loss = None
    ema_momentum = 0.98               # 你也可以改成 0.99 / 0.95

    if is_main_process():
        metrics_path = os.path.join("./records", "metrics_baseline_ft.tsv")
        metrics_f = open(metrics_path, "a", encoding="utf-8")
        if metrics_f.tell() == 0:
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


                # 每个 optim_step 都更新统计（不落盘）
                loss_real = float(loss.detach().float().item() * grad_accum)

                # moving average (每步更新，保证 loss_ma 精准)
                if len(ma_loss_buf) == ma_loss_buf.maxlen:
                    ma_loss_sum -= ma_loss_buf[0]
                ma_loss_buf.append(loss_real)
                ma_loss_sum += loss_real
                loss_ma = ma_loss_sum / max(1, len(ma_loss_buf))

                # EMA (每步更新)
                if ema_loss is None:
                    ema_loss = loss_real
                else:
                    ema_loss = ema_momentum * ema_loss + (1.0 - ema_momentum) * loss_real

                # 仅每 N 步才：tqdm postfix + 写 metrics.tsv
                if is_main_process() and (optim_step % log_every == 0):
                    cur_lr = scheduler.get_last_lr()[0]

                    if hasattr(it, "set_postfix"):
                        it.set_postfix({
                            "loss": f"{loss_real:.4f}",
                            f"ma{ma_win}": f"{loss_ma:.4f}",
                            "ema": f"{ema_loss:.4f}",
                            "lr": f"{cur_lr:.2e}",
                        }, refresh=False)  # 避免强制刷新，交给 tqdm 节流 :contentReference[oaicite:2]{index=2}

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
                    "loss": f"{(loss.detach().float().item()*grad_accum):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.3e}"
                }, refresh=False)

                # logging
                if is_main_process() and (optim_step % log_every == 0):
                    cur_lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    print(f"[train] epoch={epoch+1}/{epochs} step={optim_step}/{total_optim_steps} "
                          f"loss={loss.detach().float().item()*grad_accum:.4f} lr={cur_lr:.3e} "
                          f"elapsed={elapsed/60:.1f}m")

                # eval
                if eval_dl is not None and (optim_step % eval_every == 0):
                    ev = evaluate(model, eval_dl, device, fp16=fp16, bf16=bf16)
                    if is_main_process():
                        print(f"[eval] step={optim_step} loss={ev:.4f} ppl={math.exp(min(20, ev)):.2f}")

                # NOTE: Saving a *merged full model* requires merge_and_unload(), which must only
                # happen after training ends (merging removes the adapters and would break training).
                # Therefore we disable intermediate checkpoint saving by default.
                if False and save_every > 0 and (optim_step % save_every == 0) and is_main_process():
                    save_dir = os.path.join(output_dir, f"checkpoint-{optim_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    model.save_pretrained(save_dir)
                    print(f"[save] {save_dir}")


    pbar.close()
    if is_main_process() and metrics_f is not None:
        metrics_f.close()
    # final save (MERGED full model)
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


# -----------------------------
# Main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="Local HF model dir (baseline).")
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--mix_datasets", type=str, default="piqa,siqa", help="When --dataset=mix, comma-separated.")

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
        choices=["piqa", "siqa", "hellaswag", "arc-e", "csqa", "bbh", "winogrande", "mmlu", "mix", "arc-c", "openbookqa"],
    )
    ap.add_argument("--winogrande_config", type=str, default="winogrande_xl",
                help="allenai/winogrande config name, e.g. winogrande_xl/winogrande_l/...")

    ap.add_argument("--mmlu_subjects", type=str, default="all",
                    help="Comma-separated MMLU subjects, or 'all' for all configs.")

    ap.add_argument("--mmlu_answer_mode", type=str, default="text", choices=["text", "letter"],
                    help="MMLU supervision target: correct choice 'text' (default) or 'letter'.")
    # BBH 选择具体 task（= HF config_name）
    ap.add_argument("--bbh_task", type=str, default="boolean_expressions")

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
    ap.add_argument("--use_label", type=int, default=1, help="1 uses dataset label to pick correct answer")

    # precision
    ap.add_argument("--fp16", type=int, default=1)
    ap.add_argument("--bf16", type=int, default=0)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", type=str, default="", help="Comma-separated module names, e.g. q_proj,v_proj")

    # optim
    ap.add_argument("--use_bnb_8bit", type=int, default=0, help="Use bitsandbytes AdamW8bit")

    # runtime
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=200)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--metrics_every", type=int, default=10,
               help="Write metrics.tsv / tqdm postfix every N optimizer steps")

    ap.add_argument("--router_topk", type=int, default=0,
                    help="Fixed top-k routing for MoE. 0 disables and falls back to original top-p routing. सुझाव: 1 or 2.")
    return ap.parse_args()
    

def main():
    args = parse_args()

    local_rank, world_size, is_distributed = setup_distributed()
    set_seed(args.seed + (local_rank if is_distributed else 0))

    # device
    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if is_main_process():
        print("[load] tokenizer from model_path:", args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    # training uses pad_id=0; don't force add pad token if missing
    if tokenizer.pad_token_id is None:
        # keep None; our attention_mask uses (input_ids != 0)
        pass

    MoEForCausalLM, MoEConfig = import_moe_classes()
    config = MoEConfig.from_pretrained(args.model_path)
    # [ADD] pass fixed top-k routing into config (works even if MoEConfig doesn't define it explicitly)
    if hasattr(args, "router_topk"):
        config.router_topk = int(args.router_topk)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # dtype
    use_fp16 = bool(args.fp16) and (device.type == "cuda") and (not bool(args.bf16))
    use_bf16 = bool(args.bf16) and (device.type == "cuda")

    dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else torch.float32)

    if is_main_process():
        print(f"[load] model dtype={dtype} distributed={is_distributed} world_size={world_size}")

    model = MoEForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    # apply LoRA (train only adapters)
    targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()] if args.lora_target_modules else None
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, target_modules=targets)

    # IMPORTANT: print trainable params
    if is_main_process():
        trainable = 0
        total = 0
        for p in model.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
        print(f"[lora] trainable params: {trainable}/{total} ({trainable/total*100:.2f}%)")
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    # datasets
    use_label = bool(args.use_label)

    def make_ds(name: str, split: str, max_samples: Optional[int], seed: int = 42):
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
            # 注意：多数 BBH task 只有 test split（HF dataset card 也是这样列的）
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
            subs = None
            if args.mmlu_subjects and args.mmlu_subjects.strip().lower() != "all":
                subs = [x.strip() for x in args.mmlu_subjects.split(",") if x.strip()]
            return load_and_pack_mmlu_ppl_opencompass(
                tokenizer=tokenizer,
                block_size=args.block_size,
                split=split,                # 推荐 train 用 dev
                num_proc=args.num_proc,
                bos=True,
                eos=False,
                max_samples=max_samples,
                subjects=subs,
                answer_mode=args.mmlu_answer_mode,
            )
        if name == "arc-c":
            return load_and_pack_arc_challenge_ppl_opencompass(
                tokenizer=tokenizer,
                block_size=argsargs.block_size,
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

    if args.dataset == "mix":
        parts = [x.strip() for x in args.mix_datasets.split(",") if x.strip()]
        train_parts = [make_ds(p, args.train_split, args.train_max_samples) for p in parts]
        train_ds = concatenate_datasets(train_parts)
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

    # DDP wrap (after LoRA injection)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    # train
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

    # Save tokenizer alongside the merged model for standalone inference
    if is_main_process():
        try:
            tokenizer.save_pretrained(args.output_dir)
        except Exception:
            pass

    cleanup_distributed()


if __name__ == "__main__":
    main()
