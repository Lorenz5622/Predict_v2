#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$SCRIPT_DIR/launch_two_stage_torchrun_qwen.py"

CONFIGS=(
  "configs/qw_cb_lowlr_norestrict_abl_newlr01_ctx03.json"
)

LOGDIR="$SCRIPT_DIR/logs/qwen_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

for cfg in "${CONFIGS[@]}"; do
  cfg_path="$SCRIPT_DIR/$cfg"
  name="$(basename "$cfg" .json)"
  nproc="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], "r", encoding="utf-8")).get("launcher", {}).get("nproc_per_node", 1))' "$cfg_path")"
  echo "==== Running $name ($cfg_path) ===="

  python "$LAUNCHER" \
    --nproc_per_node "$nproc" \
    -- \
    --config "$cfg_path" \
    1> >(tee "$LOGDIR/${name}.out.log") \
    2> >(tee "$LOGDIR/${name}.err.log" >&2)

  echo "==== Finished $name ===="
done
