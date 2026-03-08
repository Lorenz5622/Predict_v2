#!/usr/bin/env bash
set -Eeuo pipefail

# ====== 你可以改的统一配置 ======
MODEL_PATH="/data/cyx/models/Qwen1.5-MoE-A2.7B"
OUT_ROOT="/data/cyx/models"
LOG_PATH="/home/cyx/qwen_moe"
BLOCK_SIZE=256
BATCH_SIZE=5
GRAD_ACCUM=4
EPOCHS=3
LR=2e-4
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.05
NUM_PROC=16

# 可选：固定 GPU（没有就注释掉）

mkdir -p "$OUT_ROOT/logs"

run_one () {
  local name="$1"; shift
  local log="$LOG_PATH/logs/${name}_$(date +%Y%m%d_%H%M%S).log"

  echo "==== [$name] START $(date) ====" | tee -a "$log"
  echo "CMD: $*" | tee -a "$log"

  # 记录运行耗时：/usr/bin/time 输出到 log（GNU time 常见）
  /usr/bin/time -v "$@" 2>&1 | tee -a "$log"

  echo "==== [$name] DONE  $(date) ====" | tee -a "$log"
  echo
}

run_one "piqa_4bit" torchrun --nproc_per_node "$NUM_PROC" finetune_qwen1_5_moe_2_7b_4bit.py \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUT_ROOT/out_piqa_lora_4bit" \
  --dataset piqa --eval_dataset piqa \
  --train_split train --eval_split validation \
  --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs "$EPOCHS" \
  --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --num_proc "$NUM_PROC" \
  --num_experts_per_tok 4 \
  --eval_max_samples 200 \
  --bf16 1 --fp16 0 \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_use_double_quant 1 \
  --gradient_checkpointing 1

echo "ALL DONE ✅  $(date)"