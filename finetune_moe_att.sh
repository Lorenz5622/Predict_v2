#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPROC=2
LAUNCHER="$SCRIPT_DIR/launch_two_stage_torchrun_moe_att.py"
EXTRA_ARGS=("$@")

CONFIGS=(
  "configs/moe_att_example.json"
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
    "${EXTRA_ARGS[@]}" \
    1> >(tee "$LOGDIR/${name}.out.log") \
    2> >(tee "$LOGDIR/${name}.err.log" >&2)

  echo "==== Finished $name ===="
done
