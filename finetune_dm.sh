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
  # "configs/finetune_dm_hard_lr1_dylr_CB_withV_pull0.json"
  # "configs/finetune_dm_hard_lr1_dylr_CB_withV_stage1strong.json"
  # "configs/finetune_dm_hard_lr1_dylr_CB_withV_stage1strong_pull0.json" # 69.91
  # "configs/finetune_dm_hard_lr1_dylr_CB_withV_stage1strong_pull0_lrmult10_temp15_kl15_logit05_eval200.json" # 68.50
  # "configs/finetune_dm_hard_lr1_dylr_CB_withV_stage1strong_pull0_temp20_kl20_logit025_eval200.json" 68.39
  # "configs/finetune_dm_hard_lr1_dylr_CB_withV_stage1strong_pull0_temp15_kl15_logit05_eval200.json" 68.39
  # "configs/finetune_dm_hard_lr1_dylr_CB_withV_stage1strong_pull0_lrmult10_eval200.json" # 68.93
  # "configs/finetune_dm_hard_lr1_dylr_CB_withV_stage1strong_pull0_lrmult075_eval200.json" 
  # "configs/finetune_dm_hard.json"
  # "configs/finetune_dm_hard_lr1.json"
  # "configs/finetune_dm_hard_lr1_dl.json"
  # "configs/finetune_dm_lr1_dl.json"
  # "configs/dm_cb_r150.json" 68.28
  # "configs/dm_cb_r125.json" 68.88
  # "configs/dm_cb_r175.json" 69.21
  # "configs/dm_cb_t20_z02.json" 69.26
  # "configs/dm_cb_t20_za02.json" 69.80
  # "configs/dm_cb_t20_z05.json" 68.66
  # "configs/dm_cb_e15.json" 67.46
  # "configs/dm_cb_za015.json" 69.75
  # "configs/dm_cb_za025.json" 68.88
  # "configs/dm_cb_za02_r175.json" 70.02
  # "configs/dm_cb_za02_r225.json" 68.61
  # "configs/dm_cb_za02_m04.json" 69.42
  # "configs/dm_cb_za02_r175_tp028_b22.json" 69.48
  # "configs/dm_cb_za02_r175_tp032_b22_late.json" # 保留value，69.70；不保留value, 69.31
  # "configs/dm_cb_za02_r175_tp030_b22_later.json" 69.53
  # "configs/dm_cb_za02_r175_tp040_b22_highstart.json" 68.50
  # "configs/dm_cb_za02_r175_tp032_b22.json" 68.88
  # "configs/dm_cb_za02_r175_tp031_b22_late_s80.json" # 68.28 69.86
  # "configs/dm_cb_za02_r175_tp030_b22_late_s85.json" # 68.23 69.75
  # "configs/dm_cb_za02_r175_tp032_b22_late_t09.json" # 67.63 69.42
  # "configs/dm_cb_za02_r175_tp031_b22_late_t085.json" # 68.93 68.17
  # "configs/dm_cb_za02_r175_tp031_b22_late_s80_abl_aux0.json" 68.39
  # "configs/dm_cb_za02_r175_tp031_b22_late_s80_abl_z0.json" 69.21
  # "configs/dm_cb_za02_r175_tp031_b22_late_s80_s1t12.json" 68.88
  # "configs/dm_cb_za02_r175_tp031_b22_late_s80_s2temp085.json" 68.72
  # "configs/dm_cb_za02_r175_tp031_b22_late_s80_pull_sched.json" 68.50
  # "configs/dm_cb_za02_r175_tp032_b22_late_exp1_acc_recovery.json"
  # "configs/dm_cb_za02_r175_tp032_b22_late_exp2_slightly_stronger.json"
  # "configs/dm_cb_za02_r175_tp032_b22_late_exp3_low_router_lr.json"
  # "configs/dm_cb_za02_r175_tp032_b22_late_exp4_tiny_late_pull.json"
  # "configs/dm_cb_za02_r175_tp032_b22_late_a2_tpf0295_t205_c0012_lrm115.json" 68.50
  # "configs/dm_cb_za02_r175_tp032_b22_late_a2_tpf0290_t200_c0012_lrm115.json" # 69.70
  # "configs/dm_cb_za02_r175_tp032_b22_late_a2_tpf0290_t200_c0014_lrm115.json" 68.55
  # "configs/dm_cb_za02_r175_tp032_b22_late_a2_tpf0285_t195_c0014_lrm110.json" 68.28
#   "configs/dm_cb_za02_r175_tp032_b22_late_a2_withV20.json" 71.49
#   "configs/dm_cb_za02_r175_tp032_b22_late_a2_tpf0292_t202_c0011_lrm115.json" 68.55
#   "configs/dm_cb_za02_r175_tp032_b22_late_a2_tpf0290_t200_c0012_lrm112.json" 68.01
#   "configs/dm_cb_za02_r175_tp032_b22_late_a2_tpf0288_t198_c0012_lrm115.json" 68.17
#   "configs/dm_cb_za02_r175_tp032_b22_late_a2_tpf0288_t198_c0012_lrm112.json" 67.57
  # "configs/dm_cb_za02_r175_tp031_b22_late_s80_e4.json"
  # "configs/dm_cb_za02_r175_tp032_b22_late_a2_tpf0290_t200_c0012_lrm115.json"
  # "configs/dm_cb_za02_r175_tp032_b22_late_a2_tpf0291_t201_c0012_lrm115.json"
  # "configs/dm_cb_za02_r175_tp032_b22_late_a2_tpf0289_t199_c0012_lrm115.json"
  # "configs/CB.json" 69.21
  # "configs/CB_llr.json" 69.15
  #"configs/CB_norestrict.json" 69.86
  # "configs/CB_lowlr_norestrict.json" # 70.40
  # 新版本代码
  # "configs/CB_lowlr_norestrict_abl_newlr01.json" # 69.26
  # "configs/CB_lowlr_norestrict_abl_newlr01_ctx03.json" # 69.64
  "configs/CB_lowlr_norestrict.json" # 68.39
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
