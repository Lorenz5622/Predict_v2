#!/usr/bin/env bash
set -Eeuo pipefail

# Unified defaults. Override per dataset below when needed.
MODEL_PATH="/path/to/qwen_moe_qw_ori"
OUT_ROOT="/path/to/outputs"
LOG_ROOT="$(pwd)/logs"
SCRIPT="finetune_qwen_moe_qw_ori_8bit.py"
NPROC_PER_NODE=2

COMMON_BLOCK_SIZE=192
COMMON_BATCH_SIZE=4
COMMON_GRAD_ACCUM=4
COMMON_EPOCHS=3
COMMON_LR=2e-4
COMMON_LORA_R=8
COMMON_LORA_ALPHA=16
COMMON_LORA_DROPOUT=0.05
COMMON_NUM_PROC=16
COMMON_EVAL_MAX_SAMPLES=64

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_one() {
  local name="$1"; shift
  local log="$LOG_ROOT/${name}_$(date +%Y%m%d_%H%M%S).log"

  echo "==== [$name] START $(date) ====" | tee -a "$log"
  echo "CMD: $*" | tee -a "$log"
  /usr/bin/time -v "$@" 2>&1 | tee -a "$log"
  echo "==== [$name] DONE $(date) ====" | tee -a "$log"
  echo
}

# Uncomment the datasets you want to run. Each block can override hyperparameters.

# run_one "piqa_qw_ori_8bit" torchrun --nproc_per_node "$NPROC_PER_NODE" "$SCRIPT" \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_piqa_qw_ori_8bit" \
#   --dataset piqa --eval_dataset piqa \
#   --train_split train --eval_split validation \
#   --block_size "$COMMON_BLOCK_SIZE" --batch_size "$COMMON_BATCH_SIZE" --grad_accum "$COMMON_GRAD_ACCUM" \
#   --epochs 3 --lr "$COMMON_LR" \
#   --lora_r "$COMMON_LORA_R" --lora_alpha "$COMMON_LORA_ALPHA" --lora_dropout "$COMMON_LORA_DROPOUT" \
#   --num_proc "$COMMON_NUM_PROC" --eval_max_samples "$COMMON_EVAL_MAX_SAMPLES" \
#   --gradient_checkpointing 1 --use_bnb_8bit 1 --router_topk 2

# run_one "siqa_qw_ori_8bit" torchrun --nproc_per_node "$NPROC_PER_NODE" "$SCRIPT" \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_siqa_qw_ori_8bit" \
#   --dataset siqa --eval_dataset siqa \
#   --train_split train --eval_split validation \
#   --block_size "$COMMON_BLOCK_SIZE" --batch_size "$COMMON_BATCH_SIZE" --grad_accum "$COMMON_GRAD_ACCUM" \
#   --epochs 2 --lr "$COMMON_LR" \
#   --lora_r "$COMMON_LORA_R" --lora_alpha "$COMMON_LORA_ALPHA" --lora_dropout "$COMMON_LORA_DROPOUT" \
#   --num_proc "$COMMON_NUM_PROC" --eval_max_samples "$COMMON_EVAL_MAX_SAMPLES" \
#   --gradient_checkpointing 1 --use_bnb_8bit 1 --router_topk 2

# run_one "hellaswag_qw_ori_8bit" torchrun --nproc_per_node "$NPROC_PER_NODE" "$SCRIPT" \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_hellaswag_qw_ori_8bit" \
#   --dataset hellaswag --eval_dataset hellaswag \
#   --train_split train --eval_split validation \
#   --block_size "$COMMON_BLOCK_SIZE" --batch_size "$COMMON_BATCH_SIZE" --grad_accum "$COMMON_GRAD_ACCUM" \
#   --epochs "$COMMON_EPOCHS" --lr "$COMMON_LR" \
#   --lora_r "$COMMON_LORA_R" --lora_alpha "$COMMON_LORA_ALPHA" --lora_dropout "$COMMON_LORA_DROPOUT" \
#   --num_proc "$COMMON_NUM_PROC" --train_max_samples 15000 --eval_max_samples "$COMMON_EVAL_MAX_SAMPLES" \
#   --gradient_checkpointing 1 --use_bnb_8bit 1 --router_topk 2

# run_one "arc_easy_qw_ori_8bit" torchrun --nproc_per_node "$NPROC_PER_NODE" "$SCRIPT" \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_arc_easy_qw_ori_8bit" \
#   --dataset arc-e --eval_dataset arc-e \
#   --train_split train --eval_split validation \
#   --block_size "$COMMON_BLOCK_SIZE" --batch_size "$COMMON_BATCH_SIZE" --grad_accum "$COMMON_GRAD_ACCUM" \
#   --epochs "$COMMON_EPOCHS" --lr "$COMMON_LR" \
#   --lora_r "$COMMON_LORA_R" --lora_alpha "$COMMON_LORA_ALPHA" --lora_dropout "$COMMON_LORA_DROPOUT" \
#   --num_proc "$COMMON_NUM_PROC" --eval_max_samples "$COMMON_EVAL_MAX_SAMPLES" \
#   --gradient_checkpointing 1 --use_bnb_8bit 1 --router_topk 2

# run_one "arc_challenge_qw_ori_8bit" torchrun --nproc_per_node "$NPROC_PER_NODE" "$SCRIPT" \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_arc_challenge_qw_ori_8bit" \
#   --dataset arc-c --eval_dataset arc-c \
#   --train_split train --eval_split validation \
#   --block_size "$COMMON_BLOCK_SIZE" --batch_size "$COMMON_BATCH_SIZE" --grad_accum "$COMMON_GRAD_ACCUM" \
#   --epochs "$COMMON_EPOCHS" --lr "$COMMON_LR" \
#   --lora_r "$COMMON_LORA_R" --lora_alpha "$COMMON_LORA_ALPHA" --lora_dropout "$COMMON_LORA_DROPOUT" \
#   --num_proc "$COMMON_NUM_PROC" --eval_max_samples "$COMMON_EVAL_MAX_SAMPLES" \
#   --gradient_checkpointing 1 --use_bnb_8bit 1 --router_topk 2

# run_one "csqa_qw_ori_8bit" torchrun --nproc_per_node "$NPROC_PER_NODE" "$SCRIPT" \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_csqa_qw_ori_8bit" \
#   --dataset csqa --eval_dataset csqa \
#   --train_split train --eval_split validation \
#   --block_size "$COMMON_BLOCK_SIZE" --batch_size "$COMMON_BATCH_SIZE" --grad_accum "$COMMON_GRAD_ACCUM" \
#   --epochs "$COMMON_EPOCHS" --lr "$COMMON_LR" \
#   --lora_r "$COMMON_LORA_R" --lora_alpha "$COMMON_LORA_ALPHA" --lora_dropout "$COMMON_LORA_DROPOUT" \
#   --num_proc "$COMMON_NUM_PROC" --eval_max_samples "$COMMON_EVAL_MAX_SAMPLES" \
#   --gradient_checkpointing 1 --use_bnb_8bit 1 --router_topk 2

# run_one "winogrande_qw_ori_8bit" torchrun --nproc_per_node "$NPROC_PER_NODE" "$SCRIPT" \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_winogrande_qw_ori_8bit" \
#   --dataset winogrande --eval_dataset winogrande --winogrande_config winogrande_xl \
#   --train_split train --eval_split validation \
#   --block_size "$COMMON_BLOCK_SIZE" --batch_size "$COMMON_BATCH_SIZE" --grad_accum "$COMMON_GRAD_ACCUM" \
#   --epochs 1 --lr "$COMMON_LR" \
#   --lora_r "$COMMON_LORA_R" --lora_alpha "$COMMON_LORA_ALPHA" --lora_dropout "$COMMON_LORA_DROPOUT" \
#   --num_proc "$COMMON_NUM_PROC" --eval_max_samples "$COMMON_EVAL_MAX_SAMPLES" \
#   --gradient_checkpointing 1 --use_bnb_8bit 1 --router_topk 2

# run_one "mmlu_qw_ori_8bit" torchrun --nproc_per_node "$NPROC_PER_NODE" "$SCRIPT" \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_mmlu_qw_ori_8bit" \
#   --dataset mmlu --eval_dataset mmlu \
#   --mmlu_subjects all --mmlu_answer_mode text \
#   --train_split dev --eval_split validation \
#   --block_size "$COMMON_BLOCK_SIZE" --batch_size "$COMMON_BATCH_SIZE" --grad_accum "$COMMON_GRAD_ACCUM" \
#   --epochs "$COMMON_EPOCHS" --lr "$COMMON_LR" \
#   --lora_r "$COMMON_LORA_R" --lora_alpha "$COMMON_LORA_ALPHA" --lora_dropout "$COMMON_LORA_DROPOUT" \
#   --num_proc "$COMMON_NUM_PROC" --eval_max_samples "$COMMON_EVAL_MAX_SAMPLES" \
#   --gradient_checkpointing 1 --use_bnb_8bit 1 --router_topk 2

# run_one "bbh_boolean_expressions_qw_ori_8bit" torchrun --nproc_per_node "$NPROC_PER_NODE" "$SCRIPT" \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_bbh_boolean_expr_qw_ori_8bit" \
#   --dataset bbh --eval_dataset bbh --bbh_task boolean_expressions \
#   --train_split test --eval_split test \
#   --block_size "$COMMON_BLOCK_SIZE" --batch_size "$COMMON_BATCH_SIZE" --grad_accum "$COMMON_GRAD_ACCUM" \
#   --epochs "$COMMON_EPOCHS" --lr "$COMMON_LR" \
#   --lora_r "$COMMON_LORA_R" --lora_alpha "$COMMON_LORA_ALPHA" --lora_dropout "$COMMON_LORA_DROPOUT" \
#   --num_proc "$COMMON_NUM_PROC" --eval_max_samples "$COMMON_EVAL_MAX_SAMPLES" \
#   --gradient_checkpointing 1 --use_bnb_8bit 1 --router_topk 2

# run_one "openbookqa_qw_ori_8bit" torchrun --nproc_per_node "$NPROC_PER_NODE" "$SCRIPT" \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_openbookqa_qw_ori_8bit" \
#   --dataset openbookqa --eval_dataset openbookqa \
#   --train_split train --eval_split validation \
#   --block_size "$COMMON_BLOCK_SIZE" --batch_size "$COMMON_BATCH_SIZE" --grad_accum "$COMMON_GRAD_ACCUM" \
#   --epochs "$COMMON_EPOCHS" --lr "$COMMON_LR" \
#   --lora_r "$COMMON_LORA_R" --lora_alpha "$COMMON_LORA_ALPHA" --lora_dropout "$COMMON_LORA_DROPOUT" \
#   --num_proc "$COMMON_NUM_PROC" --eval_max_samples "$COMMON_EVAL_MAX_SAMPLES" \
#   --gradient_checkpointing 1 --use_bnb_8bit 1 --router_topk 2

echo "No dataset block is enabled. Uncomment one or more run_one commands above."
