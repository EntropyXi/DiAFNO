#!/usr/bin/env bash
# 用途：A5-centered 主实验收尾流水线：选模 → 汇总 → 统一 test-200 → 图 → 报告。
#
# Runs after all six val-200 candidates are finished.  Every step is
# fail-closed (set -e) and non-destructive: the runner refuses a
# non-empty test200 directory that already holds artifacts, the val
# sweep reuses finished validation.json files instead of resampling,
# and no existing result file is overwritten.
#
# Usage (on the server, from the repo root):
#   bash scripts/finalize_a5_centered_main.sh [GPU_INDEX]
set -euo pipefail

GPU_INDEX="${1:-1}"
ROOT="/data2/user/zzx/exam_preprocessed/DiAFNO_spatiotemporal_ablation"
PY="/data2/user/zzx/ENTER/envs/DiAFNO/bin/python"
EXP="experiments/a5_centered_diafno_v1_20260909"
VAL_SWEEP="$EXP/validation"
TEST_DIR="$EXP/test200"
H5="/data2/user/zzx/exam_preprocessed_data/ocean_temperature_data_patched.h5"
OLD_ROOT="/data2/user/zzx/exam_preprocessed/DiAFNO/experiments"

cd "$ROOT"
export PYTHONPATH=.

COMMON="--h5-path $H5 \
 --data-manifest artifacts/ostia_data_manifest_real.json \
 --sample-manifest experiments/a5_longtrain_v1_20260908/protocol/val200_manifest.json \
 --train-dir $EXP/train \
 --validation-dir $VAL_SWEEP \
 --num-epochs 30 --every 5"

echo "=== [1/5] frozen selection ($(TZ=Asia/Shanghai date '+%F %T')) ==="
# No GPU work: every candidate validation.json already exists, so this
# pass only re-reads them and writes the two frozen best artifacts.
CUDA_VISIBLE_DEVICES="" $PY -u scripts/validate_a5_centered_epochs.py \
  $COMMON --device cpu

echo "=== [2/5] val sweep summary ==="
$PY scripts/summarize_centered_val_sweep.py \
  --validation-dir "$VAL_SWEEP" \
  --output "$VAL_SWEEP/VAL_SWEEP.md" \
  --baseline-json "$VAL_SWEEP/baseline_a5_frozen.json"

echo "=== [3/5] CRPS-best row decision ==="
CRPS_EXTRA="$($PY - "$VAL_SWEEP" <<'PY'
import json, os, sys
directory = sys.argv[1]
rmse = json.load(open(os.path.join(directory, "best_val_mean_rmse.json")))
crps = json.load(open(os.path.join(directory, "best_val_crps.json")))
if rmse["source"] == crps["source"]:
    print("")
else:
    print("--centered-crps-checkpoint "
          + os.path.join(directory, "best_val_crps.pth"))
PY
)"
if [ -n "$CRPS_EXTRA" ]; then
  echo "CRPS-best differs -> adding the pre-declared secondary row"
else
  echo "CRPS-best equals RMSE-best -> single centered row"
fi

echo "=== [4/5] unified test-200 (GPU $GPU_INDEX) ==="
mkdir -p "$TEST_DIR"
CUDA_VISIBLE_DEVICES="$GPU_INDEX" $PY -u scripts/compare_ostia_protocol.py \
  --a5-checkpoint experiments/a5_longtrain_v1_20260908/validation/best_val_rmse.pth \
  --old-iafno-checkpoint "$OLD_ROOT/det_lead_standardized/epoch_015.pth" \
  --old-diafno-checkpoint "$OLD_ROOT/ostia_7day_to15day_residual_scratch/best_val_mean_rmse.pth" \
  --centered-checkpoint "$VAL_SWEEP/best_val_mean_rmse.pth" \
  $CRPS_EXTRA \
  --h5-path "$H5" \
  --data-manifest artifacts/ostia_data_manifest_real.json \
  --sample-manifest experiments/a5_longtrain_v1_20260908/protocol/test200_manifest.json \
  --output-dir "$TEST_DIR" \
  --device cuda:0 --ensemble-members 16 --sampling-steps 16 --s-churn 0.0 \
  --seed 123 --block-days 22 --bootstrap-replicates 2000 --figure-samples 4 \
  > "$TEST_DIR/run.log" 2>&1

echo "=== [5/5] figures + report ==="
$PY scripts/plot_ostia_protocol_figures.py \
  --fields "$TEST_DIR/figure_fields.npz" \
  --output-dir "$TEST_DIR/figures" > "$TEST_DIR/figures.log" 2>&1
$PY scripts/report_ostia_protocol.py \
  --metrics "$TEST_DIR/metrics.json" \
  --bootstrap "$TEST_DIR/bootstrap.json" \
  --figures "$TEST_DIR/figures/figures.json" \
  --output "$TEST_DIR/REPORT.md"

echo "=== finalize done ($(TZ=Asia/Shanghai date '+%F %T')) ==="
