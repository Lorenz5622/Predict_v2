#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPROC=2
LAUNCHER="$SCRIPT_DIR/launch_two_stage_torchrun.py"

CONFIGS=(
  # "configs/finetune_dm.json"
  # "configs/finetune_dm_hard_lr1_dylr_exp_lrmult10_reg_4e.json"
  # "configs/finetune_dm_hard_lr1_dylr_exp_lrmult125_minlr03_4e.json"
  # "configs/finetune_dm_hard_lr1_dylr_exp_lrmult10_minlr03_4e.json"
  # "configs/finetune_dm_hard_lr1_dylr_CB_load8.json"
  "configs/finetune_dm_hard_lr1_dylr_CB_v1.json"
  # "configs/finetune_dm_hard.json"
  # "configs/finetune_dm_hard_lr1.json"
  # "configs/finetune_dm_hard_lr1_dl.json"
  # "configs/finetune_dm_lr1_dl.json"
)

LOGDIR="$SCRIPT_DIR/logs/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

for cfg in "${CONFIGS[@]}"; do
  cfg_path="$SCRIPT_DIR/$cfg"
  name="$(basename "$cfg" .json)"
  echo "==== Running $name ($cfg_path) ===="

  python "$LAUNCHER" \
    --nproc_per_node "$NPROC" \
    -- \
    --config "$cfg_path" \
    1> >(tee "$LOGDIR/${name}.out.log") \
    2> >(tee "$LOGDIR/${name}.err.log" >&2)

  echo "==== Finished $name ===="
done
