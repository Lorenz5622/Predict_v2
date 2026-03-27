#!/usr/bin/env bash
set -Eeuo pipefail

# ====== 你可以改的统一配置 ======
MODEL_PATH="/data/cyx/models/Dynamic_MoE"
OUT_ROOT="/data/cyx/models"
LOG_PATH="/home/cyx/qwen_moe"
BLOCK_SIZE=192
BATCH_SIZE=6
GRAD_ACCUM=3
EPOCHS=2
LR=1.5e-4
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.05
NUM_PROC=8

# 模型结构相关参数统一从 $MODEL_PATH 下的 configuration_moe_dm 加载；这里只保留训练参数。
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

# run_one "piqa" torchrun --nproc_per_node 2 finetune_dynamic_moe.py \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_piqa_entmax_15" \
#   --dataset piqa --eval_dataset piqa \
#   --train_split train --eval_split validation \
#   --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs 3 \
#   --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
#   --num_proc "$NUM_PROC" --load_in_8bit 0 --load_in_fp16 1 --bf16 0 --train_extra_params_in_fp32 1 \
#   --router_use_entmax 1 --router_entmax_alpha 1.5

# run_one "piqa" torchrun --nproc_per_node 2 finetune_dynamic_moe.py \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_piqa_entmax_17" \
#   --dataset piqa --eval_dataset piqa \
#   --train_split train --eval_split validation \
#   --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs 3 \
#   --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
#   --num_proc "$NUM_PROC" --load_in_8bit 0 --load_in_fp16 1 --bf16 0 --train_extra_params_in_fp32 1 \
#   --router_use_entmax 1 --router_entmax_alpha 1.7

# run_one "piqa" torchrun --nproc_per_node 2 finetune_dynamic_moe.py \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_piqa_entmax_15" \
#   --dataset piqa --eval_dataset piqa \
#   --train_split train --eval_split validation \
#   --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs 3 \
#   --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
#   --num_proc "$NUM_PROC" --load_in_8bit 0 --load_in_fp16 1 --bf16 0 --train_extra_params_in_fp32 1 \
#   --router_use_entmax 1 --router_entmax_alpha 1.5

# run_one "piqa" torchrun --nproc_per_node 2 finetune_dynamic_moe.py \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_piqa_entmax_17" \
#   --dataset piqa --eval_dataset piqa \
#   --train_split train --eval_split validation \
#   --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs 3 \
#   --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
#   --num_proc "$NUM_PROC" --load_in_8bit 0 --load_in_fp16 1 --bf16 0 --train_extra_params_in_fp32 1 \
#   --router_use_entmax 1 --router_entmax_alpha 1.7

# run_one "piqa" torchrun --nproc_per_node 2 finetune_dynamic_moe.py \
#   --model_path "$MODEL_PATH" \
#   --output_dir "$OUT_ROOT/out_piqa_entmax_19" \
#   --dataset piqa --eval_dataset piqa \
#   --train_split train --eval_split validation \
#   --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs 3 \
#   --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
#   --num_proc "$NUM_PROC" --load_in_8bit 0 --load_in_fp16 1 --bf16 0 --train_extra_params_in_fp32 1 \
#   --router_use_entmax 1 --router_entmax_alpha 1.9

run_one "piqa" torchrun --nproc_per_node 2 finetune_dynamic_moe.py \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUT_ROOT/out_piqa_each_embedding" \
  --dataset piqa --eval_dataset piqa \
  --train_split train --eval_split validation \
  --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs 3 \
  --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --num_proc "$NUM_PROC" --load_in_8bit 0 --load_in_fp16 1 --bf16 0 --train_extra_params_in_fp32 1 \
  --router_use_entmax 0 --share_router_expert_embedding  0 --router_entmax_alpha 1.7

run_one "piqa" torchrun --nproc_per_node 2 finetune_dynamic_moe.py \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUT_ROOT/out_piqa_each_embedding_aux_003" \
  --dataset piqa --eval_dataset piqa \
  --train_split train --eval_split validation \
  --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs 3 \
  --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --num_proc "$NUM_PROC" --load_in_8bit 0 --load_in_fp16 1 --bf16 0 --train_extra_params_in_fp32 1 \
  --router_use_entmax 0 --share_router_expert_embedding  0 --router_entmax_alpha 1.7 --router_aux_loss_coef 0.03

run_one "piqa" torchrun --nproc_per_node 2 finetune_dynamic_moe.py \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUT_ROOT/out_piqa_entmax_17_each_embedding" \
  --dataset piqa --eval_dataset piqa \
  --train_split train --eval_split validation \
  --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs 3 \
  --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --num_proc "$NUM_PROC" --load_in_8bit 0 --load_in_fp16 1 --bf16 0 --train_extra_params_in_fp32 1 \
  --router_use_entmax 1 --share_router_expert_embedding  0 --router_entmax_alpha 1.7

run_one "piqa" torchrun --nproc_per_node 2 finetune_dynamic_moe.py \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUT_ROOT/out_piqa_entmax_17_each_embedding_aux_003" \
  --dataset piqa --eval_dataset piqa \
  --train_split train --eval_split validation \
  --block_size "$BLOCK_SIZE" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" --epochs 3 \
  --lr "$LR" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --num_proc "$NUM_PROC" --load_in_8bit 0 --load_in_fp16 1 --bf16 0 --train_extra_params_in_fp32 1 \
  --router_use_entmax 1 --share_router_expert_embedding  0 --router_entmax_alpha 1.7 --router_aux_loss_coef 0.03

echo "ALL DONE ✅  $(date)"