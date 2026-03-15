#!/usr/bin/env bash
set -Eeuo pipefail

# ====== 你可以改的统一配置 ======
MODEL_PATH="/data/cyx/models/Qwen1.5-MoE-A2.7B"
OUT_ROOT="/data/cyx/models"
LOG_PATH="/home/cyx/qwen_moe"
BLOCK_SIZE=192
BATCH_SIZE=3
GRAD_ACCUM=5
EPOCHS=2
LR=2e-4
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
NUM_PROC=8
NPROC_PER_NODE=2

# low-rank router / 4bit
NUM_EXPERTS_PER_TOK=4
USE_LOW_RANK_ROUTER=1
ROUTER_RANK=128
USE_SHARP_ROUTER=1
ROUTER_TEMPERATURE_INIT=10.0
ROUTER_NORMALIZE_Q=1
ROUTER_NORMALIZE_K=1
ROUTER_EPS=1e-6

mkdir -p "$LOG_PATH/logs"

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

run_one "piqa_qw_4bit" torchrun --nproc_per_node "$NPROC_PER_NODE" finetune_qwen1_5_moe_2_7b_4bit.py \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUT_ROOT/out_piqa_qw_lowrank_4bit_v1" \
  --dataset piqa --eval_dataset piqa \
  --train_split train --eval_split validation \
  --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs "$EPOCHS" \
  --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --num_proc "$NUM_PROC" \
  --num_experts_per_tok "$NUM_EXPERTS_PER_TOK" \
  --use_low_rank_router "$USE_LOW_RANK_ROUTER" \
  --router_rank "$ROUTER_RANK" \
  --use_sharp_router "$USE_SHARP_ROUTER" \
  --router_temperature_init "$ROUTER_TEMPERATURE_INIT" \
  --router_normalize_q "$ROUTER_NORMALIZE_Q" \
  --router_normalize_k "$ROUTER_NORMALIZE_K" \
  --router_eps "$ROUTER_EPS" \
  --eval_max_samples 50 \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_use_double_quant 1 \
  --bnb_4bit_compute_dtype bfloat16 \
  --gradient_checkpointing 0

echo "ALL DONE ✅  $(date)"
