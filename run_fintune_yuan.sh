#!/usr/bin/env bash
set -uo pipefail

# Batch launcher for /home/cyx/Predict_MoE/fintune_yuan.py
#
# Current template includes only one PIQA experiment, but the structure is
# intentionally flat so you can add more run_one ... blocks later.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="/home/cyx/qwen_moe/finetune_yuan.py"

# ====== Shared settings you are expected to edit ======
MODEL_PATH="/data/cyx/models/Dynamic_MoE"
LOG_ROOT="$SCRIPT_DIR/logs/fintune_yuan"

BLOCK_SIZE=128
BATCH_SIZE=16
GRAD_ACCUM=1
EPOCHS=2
LR=1e-4
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.05
NUM_PROC=16
NPROC_PER_NODE=2

mkdir -p "$LOG_ROOT"

FAILED_EXPERIMENTS=()

run_one() {
  local name="$1"
  local output_dir="$2"
  shift 2

  local log_file="$LOG_ROOT/${name}.log"

  echo "==== [$name] START $(date) ====" | tee "$log_file"
  echo "OUTPUT_DIR: $output_dir" | tee -a "$log_file"
  echo "CMD: torchrun --nproc_per_node ${NPROC_PER_NODE} ${TRAIN_SCRIPT} $*" | tee -a "$log_file"

  if /usr/bin/time -v \
    torchrun --nproc_per_node "${NPROC_PER_NODE}" "${TRAIN_SCRIPT}" "$@" \
    2>&1 | tee -a "$log_file"; then
    echo "==== [$name] DONE $(date) ====" | tee -a "$log_file"
  else
    local status=${PIPESTATUS[0]}
    echo "==== [$name] FAILED (exit=${status}) $(date) ====" | tee -a "$log_file"
    FAILED_EXPERIMENTS+=("${name}")
  fi

  echo | tee -a "$log_file"
}

# -----------------------------------------------------------------------------
# PIQA example
# Edit OUTPUT_DIR below to wherever you want the merged inference model saved.
# Duplicate this block and change dataset / eval_dataset / splits for more runs.
# -----------------------------------------------------------------------------
run_one "piqa" "/data/cyx/models/out_piqa_yuan_lora" \
  --model_path "$MODEL_PATH" \
  --output_dir "/data/cyx/models/out_piqa_yuan_lora" \
  --dataset piqa \
  --eval_dataset piqa \
  --train_split train \
  --eval_split validation \
  --block_size "$BLOCK_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --num_proc "$NUM_PROC"

echo "==== Sweep Summary $(date) ===="
if ((${#FAILED_EXPERIMENTS[@]} == 0)); then
  echo "All experiments finished successfully."
else
  echo "Failed experiments:"
  for name in "${FAILED_EXPERIMENTS[@]}"; do
    echo "  - $name"
  done
  exit 1
fi
