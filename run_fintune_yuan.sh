#!/usr/bin/env bash
set -Eeuo pipefail

# Batch launcher for finetune_yuan.py.
# Style follows /home/cyx/Predict_MoE/run_fintune.sh:
# shared defaults above, one run_one command per dataset below.

# ====== Shared settings you are expected to edit ======
MODEL_PATH="/data/cyx/models/Dynamic_MoE"
OUT_ROOT="/data/cyx/models"
LOG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$LOG_PATH/finetune_yuan.py"
NPROC_PER_NODE=2

BLOCK_SIZE=128
BATCH_SIZE=16
GRAD_ACCUM=1
EPOCHS=2
LR=2e-4
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
NUM_PROC=16
EVAL_MAX_SAMPLES=20
STAGE=0
STAGE1_EPOCHS=1
STAGE1_LR=5e-5
STAGE1_GRAD_ACCUM=2
STAGE1_DATA_RATIO=0.2
TRAIN_ROUTER_QKV=0
FP16=0
BF16=1

mkdir -p "$OUT_ROOT" "$LOG_PATH/logs/fintune_yuan"

run_one () {
  local name="$1"; shift
  local log="$LOG_PATH/logs/fintune_yuan/${name}_$(date +%Y%m%d_%H%M%S).log"

  echo "==== [$name] START $(date) ====" | tee -a "$log"
  echo "CMD: $*" | tee -a "$log"

  /usr/bin/time -v "$@" 2>&1 | tee -a "$log"

  echo "==== [$name] DONE  $(date) ====" | tee -a "$log"
  echo
}

# ====== Run datasets sequentially ======

# run_one "piqa_yuan" torchrun --nproc_per_node "$NPROC_PER_NODE" "$TRAIN_SCRIPT" \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_piqa_yuan_lora" \
#   --dataset piqa --eval_dataset piqa \
#   --train_split train --eval_split validation \
#   --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs "$EPOCHS" \
#   --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
#   --num_proc "$NUM_PROC" --eval_max_samples "$EVAL_MAX_SAMPLES"

run_one "arc_easy_yuan" torchrun --nproc_per_node "$NPROC_PER_NODE" "$TRAIN_SCRIPT" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUT_ROOT/out_arc_easy_yuan_lora" \
  --stage "$STAGE" \
  --dataset arc-e --eval_dataset arc-e \
  --train_split train --eval_split validation \
  --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs "$EPOCHS" \
  --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --train_router_qkv "$TRAIN_ROUTER_QKV" --fp16 "$FP16" --bf16 "$BF16" \
  --num_proc "$NUM_PROC" --eval_max_samples "$EVAL_MAX_SAMPLES" \
  --stage1_epochs "$STAGE1_EPOCHS" --stage1_lr "$STAGE1_LR" \
  --stage1_grad_accum "$STAGE1_GRAD_ACCUM" --stage1_data_ratio "$STAGE1_DATA_RATIO"

run_one "arc_challenge_yuan" torchrun --nproc_per_node "$NPROC_PER_NODE" "$TRAIN_SCRIPT" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUT_ROOT/out_arc_challenge_yuan_lora" \
  --stage "$STAGE" \
  --dataset arc-c --eval_dataset arc-c \
  --train_split train --eval_split validation \
  --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs "$EPOCHS" \
  --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --train_router_qkv "$TRAIN_ROUTER_QKV" --fp16 "$FP16" --bf16 "$BF16" \
  --num_proc "$NUM_PROC" --eval_max_samples "$EVAL_MAX_SAMPLES" \
  --stage1_epochs "$STAGE1_EPOCHS" --stage1_lr "$STAGE1_LR" \
  --stage1_grad_accum "$STAGE1_GRAD_ACCUM" --stage1_data_ratio "$STAGE1_DATA_RATIO"

run_one "siqa_yuan" torchrun --nproc_per_node "$NPROC_PER_NODE" "$TRAIN_SCRIPT" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUT_ROOT/out_siqa_yuan_lora" \
  --stage "$STAGE" \
  --dataset siqa --eval_dataset siqa \
  --train_split train --eval_split validation \
  --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs "$EPOCHS" \
  --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --train_router_qkv "$TRAIN_ROUTER_QKV" --fp16 "$FP16" --bf16 "$BF16" \
  --num_proc "$NUM_PROC" --eval_max_samples "$EVAL_MAX_SAMPLES" \
  --stage1_epochs "$STAGE1_EPOCHS" --stage1_lr "$STAGE1_LR" \
  --stage1_grad_accum "$STAGE1_GRAD_ACCUM" --stage1_data_ratio "$STAGE1_DATA_RATIO"

run_one "oqa_yuan" torchrun --nproc_per_node "$NPROC_PER_NODE" "$TRAIN_SCRIPT" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUT_ROOT/out_oqa_yuan_lora" \
  --stage "$STAGE" \
  --dataset openbookqa --eval_dataset openbookqa \
  --train_split train --eval_split validation \
  --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs "$EPOCHS" \
  --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --train_router_qkv "$TRAIN_ROUTER_QKV" --fp16 "$FP16" --bf16 "$BF16" \
  --num_proc "$NUM_PROC" --eval_max_samples "$EVAL_MAX_SAMPLES" \
  --stage1_epochs "$STAGE1_EPOCHS" --stage1_lr "$STAGE1_LR" \
  --stage1_grad_accum "$STAGE1_GRAD_ACCUM" --stage1_data_ratio "$STAGE1_DATA_RATIO"


# run_one "hellaswag_yuan" torchrun --nproc_per_node "$NPROC_PER_NODE" "$TRAIN_SCRIPT" \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_hellaswag_yuan_lora" \
#   --dataset hellaswag --eval_dataset hellaswag \
#   --train_split train --eval_split validation \
#   --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs "$EPOCHS" \
#   --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
#   --num_proc "$NUM_PROC" --train_max_samples 15000 --eval_max_samples "$EVAL_MAX_SAMPLES"

# run_one "siqa_yuan" torchrun --nproc_per_node "$NPROC_PER_NODE" "$TRAIN_SCRIPT" \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_siqa_yuan_lora" \
#   --dataset siqa --eval_dataset siqa \
#   --train_split train --eval_split validation \
#   --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs 2 \
#   --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
#   --num_proc "$NUM_PROC" --train_max_samples 10000 --eval_max_samples "$EVAL_MAX_SAMPLES"

# run_one "csqa_yuan" torchrun --nproc_per_node "$NPROC_PER_NODE" "$TRAIN_SCRIPT" \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_csqa_yuan_lora" \
#   --dataset csqa --eval_dataset csqa \
#   --train_split train --eval_split validation \
#   --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs "$EPOCHS" \
#   --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
#   --num_proc "$NUM_PROC" --eval_max_samples "$EVAL_MAX_SAMPLES"

# run_one "mmlu_yuan" torchrun --nproc_per_node "$NPROC_PER_NODE" "$TRAIN_SCRIPT" \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_mmlu_yuan_lora" \
#   --dataset mmlu --eval_dataset mmlu \
#   --mmlu_subjects all --mmlu_answer_mode text \
#   --train_split dev --eval_split validation \
#   --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs "$EPOCHS" \
#   --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
#   --num_proc "$NUM_PROC" --eval_max_samples "$EVAL_MAX_SAMPLES"

# run_one "winogrande_yuan" torchrun --nproc_per_node "$NPROC_PER_NODE" "$TRAIN_SCRIPT" \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_winogrande_yuan_lora" \
#   --dataset winogrande --eval_dataset winogrande --winogrande_config winogrande_xl \
#   --train_split train --eval_split validation \
#   --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs 1 \
#   --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
#   --num_proc "$NUM_PROC" --eval_max_samples "$EVAL_MAX_SAMPLES"

# run_one "bbh_boolean_expressions_yuan" torchrun --nproc_per_node "$NPROC_PER_NODE" "$TRAIN_SCRIPT" \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_bbh_boolean_expr_yuan_lora" \
#   --dataset bbh --eval_dataset bbh \
#   --bbh_task boolean_expressions \
#   --train_split test --eval_split test \
#   --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs "$EPOCHS" \
#   --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
#   --num_proc "$NUM_PROC" --eval_max_samples "$EVAL_MAX_SAMPLES"

# run_one "openbookqa_yuan" torchrun --nproc_per_node "$NPROC_PER_NODE" "$TRAIN_SCRIPT" \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_openbookqa_yuan_lora" \
#   --dataset openbookqa --eval_dataset openbookqa \
#   --train_split train --eval_split validation \
#   --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs "$EPOCHS" \
#   --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
#   --num_proc "$NUM_PROC" --eval_max_samples "$EVAL_MAX_SAMPLES"

echo "ALL DONE $(date)"
