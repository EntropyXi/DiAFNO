#!/usr/bin/env bash
# 用途：在独立输出目录执行 centered diffusion 的 GPU 结构烟测。
# ---------------------------------------------------------------------------
# smoke_ostia_centered.sh
#
# GPU structural smoke for the centered diffusion training.  Uses the
# independent output directory experiments/ostia_centered_smoke_scratch
# and NEVER writes to the canonical output dir or the archive.
#
# Phase 1 (default): 20 optimizer steps (structure check).
# Phase 2 (--phase 2): 1000 optimizer steps with the same target, stats,
#   sigma_data, LR and effective global batch (32) as the main config.
#
# The smoke config is derived from configs/ostia_centered_diffusion_main.json
# by overriding ONLY the output dir / step budget / batch fields.
# ---------------------------------------------------------------------------
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PHASE=1
GPUS=""
MEAN_CHECKPOINT=""
CENTERED_STATS=""
H5_PATH=""
RUN_ID="$(date +%Y%m%dT%H%M%S)"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN_JSON="$REPO_ROOT/configs/ostia_centered_diffusion_main.json"
SMOKE_BASE="$REPO_ROOT/experiments/ostia_centered_smoke_scratch"

usage() {
    cat <<'EOF'
Usage: smoke_ostia_centered.sh --gpus <id[,id...]> --mean-checkpoint <pth>
       --centered-stats <json> --h5-path <h5> [--phase 1|2]
       [--run-id <safe-name>]

phase 1: 20 optimizer steps, resume for another 20, then finite sample
phase 2: 1000 optimizer steps (formal smoke, effective batch 32)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --mean-checkpoint)
            MEAN_CHECKPOINT="$2"
            shift 2
            ;;
        --centered-stats)
            CENTERED_STATS="$2"
            shift 2
            ;;
        --h5-path)
            H5_PATH="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

[[ -n "$GPUS" ]] || { echo "ERROR: --gpus is required" >&2; exit 2; }
[[ -n "$MEAN_CHECKPOINT" ]] || { echo "ERROR: --mean-checkpoint is required" >&2; exit 2; }
[[ -n "$CENTERED_STATS" ]] || { echo "ERROR: --centered-stats is required" >&2; exit 2; }
[[ -n "$H5_PATH" ]] || { echo "ERROR: --h5-path is required" >&2; exit 2; }
[[ "$PHASE" == "1" || "$PHASE" == "2" ]] || { echo "ERROR: --phase must be 1 or 2" >&2; exit 2; }
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: unsafe --run-id" >&2; exit 2; }

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
NPROC="${#GPU_IDS[@]}"
for gpu_id in "${GPU_IDS[@]}"; do
    [[ "$gpu_id" =~ ^[0-9]+$ ]] || { echo "ERROR: bad gpu id: $gpu_id" >&2; exit 2; }
done
[[ "$NPROC" -ge 1 && "$NPROC" -le 2 ]] || { echo "ERROR: smoke supports 1 or 2 GPUs" >&2; exit 2; }
LOCKED_MEAN_SHA="cb09b15ce97e11800b83fcf7c8ef9df09aa47f8831a0a36fffa987e413fc53e6"
SMOKE_DIR="$SMOKE_BASE/phase${PHASE}_${RUN_ID}"
[[ ! -e "$SMOKE_DIR" ]] || { echo "ERROR: smoke run dir already exists: $SMOKE_DIR" >&2; exit 1; }

echo "=== centered smoke preflight (phase $PHASE, nproc=$NPROC) ==="
echo "commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"

MEAN_ABS="$(realpath "$MEAN_CHECKPOINT")"
STATS_ABS="$(realpath "$CENTERED_STATS")"
[[ -f "$MEAN_ABS" ]] || { echo "ERROR: frozen mean missing: $MEAN_ABS" >&2; exit 1; }
[[ -f "$STATS_ABS" ]] || { echo "ERROR: centered stats missing: $STATS_ABS" >&2; exit 1; }
[[ -f "$H5_PATH" ]] || { echo "ERROR: H5 missing: $H5_PATH" >&2; exit 1; }
MEAN_SHA="$(sha256sum "$MEAN_ABS" | awk '{print $1}')"
[[ "$MEAN_SHA" == "$LOCKED_MEAN_SHA" ]] \
    || { echo "ERROR: frozen mean SHA-256 mismatch" >&2; exit 1; }
echo "frozen mean SHA-256: $MEAN_SHA (locked OK)"
echo "centered stats SHA-256: $(sha256sum "$STATS_ABS" | awk '{print $1}')"

# Derive the smoke config from the authoritative JSON, overriding only
# the budget/batch/output fields.
SMOKE_JSON="$SMOKE_DIR/config/smoke_config.json"
mkdir -p "$SMOKE_DIR/config" "$SMOKE_DIR/logs"
if [[ "$PHASE" -eq 1 ]]; then
    # 20 optimizer steps: 20 x 4 x 1 x nproc samples per epoch.
    STEPS=20
    BATCH_PER_GPU=4
    GRAD_ACCUM=1
else
    # 1000 optimizer steps at effective batch 32 (plan 6.3 phase 2).
    STEPS=1000
    BATCH_PER_GPU=8
    if [[ "$NPROC" -eq 2 ]]; then
        GRAD_ACCUM=2
    else
        GRAD_ACCUM=4
    fi
fi
SAMPLES_PER_EPOCH=$(( STEPS * BATCH_PER_GPU * GRAD_ACCUM * NPROC ))

"$PYTHON_BIN" - "$MAIN_JSON" "$SMOKE_JSON" \
        "$SMOKE_DIR" "$SAMPLES_PER_EPOCH" "$BATCH_PER_GPU" "$GRAD_ACCUM" \
        "$MEAN_ABS" "$STATS_ABS" "$H5_PATH" <<'PYEOF'
import json, sys
(
    main_json, smoke_json, smoke_dir, samples, batch, grad_accum,
    mean_path, stats_path, h5_path,
) = sys.argv[1:]
with open(main_json, encoding="utf-8") as f:
    c = json.load(f)
c["output_dir"] = smoke_dir
c["mean_checkpoint_path"] = mean_path
c["centered_stats_path"] = stats_path
c["train_h5_path"] = h5_path
c["num_epochs"] = 1
c["samples_per_epoch"] = int(samples)
c["batch_per_gpu"] = int(batch)
c["gradient_accumulation"] = int(grad_accum)
c["num_workers"] = 2
c["prefetch_factor"] = 1
c["checkpoint_interval"] = 1
with open(smoke_json, "w", encoding="utf-8") as f:
    json.dump(c, f, indent=2, ensure_ascii=False)
    f.write("\n")
print("smoke config written:", smoke_json)
PYEOF

echo "smoke config SHA-256: $(sha256sum "$SMOKE_JSON" | awk '{print $1}')"
echo "optimizer steps: $STEPS (samples_per_epoch=$SAMPLES_PER_EPOCH, batch=$BATCH_PER_GPU, accum=$GRAD_ACCUM)"
echo "output dir: $SMOKE_DIR (independent; canonical/archive untouched)"

echo
echo "=== launching smoke trainer ==="
cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPUS" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON_BIN" -u -m torch.distributed.run \
    --standalone --nproc_per_node="$NPROC" \
    trainer_ostia.py \
    --config "$SMOKE_JSON" \
    2>&1 | tee "$SMOKE_DIR/logs/smoke_phase${PHASE}.log"

echo
echo "=== smoke verification ==="
SMOKE_LOG="$SMOKE_DIR/logs/smoke_phase${PHASE}.log"
if grep -q "AMP overflow skip" "$SMOKE_LOG"; then
    echo "ERROR: AMP overflow skips observed; smoke must have skipped_optimizer_steps==0" >&2
    exit 1
fi
if ! grep -q "epoch=1 .*skipped_optimizer_steps=" "$SMOKE_LOG"; then
    echo "ERROR: no epoch summary found in the smoke log" >&2
    exit 1
fi
grep "epoch=1 .*skipped_optimizer_steps=" "$SMOKE_LOG" | tail -1
echo "SMOKE PASS: finite training, no optimizer skips"

if [[ "$PHASE" -eq 1 ]]; then
    FIRST_STEP="$($PYTHON_BIN - "$SMOKE_DIR/latest.pth" <<'PYEOF'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(c["global_step"]))
PYEOF
)"
    [[ "$FIRST_STEP" -eq 20 ]] || { echo "ERROR: expected first checkpoint at step 20, got $FIRST_STEP" >&2; exit 1; }

    echo
    echo "=== resume continuity gate (one additional 20-step epoch) ==="
    CUDA_VISIBLE_DEVICES="$GPUS" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON_BIN" -u -m torch.distributed.run \
        --standalone --nproc_per_node="$NPROC" \
        trainer_ostia.py \
        --config "$SMOKE_JSON" --resume latest --num-epochs 2 \
        --allow-resume-override \
        2>&1 | tee "$SMOKE_DIR/logs/smoke_phase1_resume.log"
    grep -q "Starting epoch: 2 global step: 20" "$SMOKE_DIR/logs/smoke_phase1_resume.log" \
        || { echo "ERROR: resume did not restore epoch/global_step" >&2; exit 1; }
    SECOND_STEP="$($PYTHON_BIN - "$SMOKE_DIR/latest.pth" <<'PYEOF'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(c["global_step"]))
PYEOF
)"
    [[ "$SECOND_STEP" -eq 40 ]] || { echo "ERROR: expected resumed checkpoint at step 40, got $SECOND_STEP" >&2; exit 1; }

    echo
    echo "=== finite sampler/evaluator gate (two val samples, four steps) ==="
    CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" \
    "$PYTHON_BIN" -u validate_ostia.py \
        --checkpoint "$SMOKE_DIR/latest.pth" \
        --h5-path "$H5_PATH" --split val --max-samples 2 \
        --sampling-steps 4 --s-churn 0 --ensemble-members 1 \
        --batch-size 1 --num-workers 0 --device cuda:0 \
        --output-path "$SMOKE_DIR/logs/finite_sample.json"
    "$PYTHON_BIN" - "$SMOKE_DIR/logs/finite_sample.json" <<'PYEOF'
import json, math, sys
with open(sys.argv[1], encoding="utf-8") as f:
    result = json.load(f)
rmse = result.get("overall", {}).get("rmse")
if rmse is None or not math.isfinite(float(rmse)):
    raise SystemExit(f"missing/non-finite overall RMSE: {rmse}")
for key, value in result["overall"].items():
    if isinstance(value, (int, float)) and not math.isfinite(float(value)):
        raise SystemExit(f"non-finite evaluator metric {key}={value}")
print("finite sampler/evaluator gate PASS")
PYEOF
fi

echo "SMOKE ARTIFACTS RETAINED: $SMOKE_DIR"
