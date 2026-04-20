#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPROC=2
ENTRYPOINT="$SCRIPT_DIR/finetune_qwen_dynamic_moe.py"

CONFIGS=(
  "configs/qw_cb_lowlr_norestrict_abl_newlr01_ctx03.json"
)

LOGDIR="$SCRIPT_DIR/logs/qwen_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

for cfg in "${CONFIGS[@]}"; do
  cfg_path="$SCRIPT_DIR/$cfg"
  name="$(basename "$cfg" .json)"
  echo "==== Running $name ($cfg_path) ===="

  torchrun \
    --nproc_per_node "$NPROC" \
    "$ENTRYPOINT" \
    --config "$cfg_path" \
    1> >(tee "$LOGDIR/${name}.out.log") \
    2> >(tee "$LOGDIR/${name}.err.log" >&2)

  echo "==== Finished $name ===="
done
