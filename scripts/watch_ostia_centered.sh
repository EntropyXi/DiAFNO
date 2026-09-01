#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# watch_ostia_centered.sh
#
# Read-only watcher for the centered main training.  It NEVER touches the
# test split: every validation runs on the frozen val-200 protocol
# (seed 123, s_churn=0, sampling_steps=16, ensemble_members=1 until the
# inference-side ablation is locked at epoch 5).
#
# Protocol state lives in <output_dir>/watcher_config.json:
#   {"locked": false, "sampling_steps": 16, "ensemble_members": 1,
#    "s_churn": 0.0, "ablation_epoch": null, "last_processed_epoch": 0}
#
# When the first epoch >= 5 checkpoint appears, the watcher runs the
# cheap inference ablations (steps 16 vs 32, members 1 vs 4) into
# watcher_ablation/, prints the table and stops, asking Codex to review
# and set "locked": true with the chosen profile.  All subsequent
# cross-epoch comparisons use only the locked profile.
# ---------------------------------------------------------------------------
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR=""
H5_PATH=""
POLL_SECONDS=60
ONCE=0
ABLATION_EPOCH=5
GPU_ID="0"
BATCH_SIZE=8

usage() {
    cat <<'EOF'
Usage: watch_ostia_centered.sh --output-dir <dir> --h5-path <h5>
                              [--gpu-id N] [--batch-size 8]
                              [--poll-seconds 60] [--once]

Fixed val-200 protocol only; the test split is never read.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --h5-path)
            H5_PATH="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --poll-seconds)
            POLL_SECONDS="$2"
            shift 2
            ;;
        --once)
            ONCE=1
            shift
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

[[ -n "$OUTPUT_DIR" ]] || { echo "ERROR: --output-dir is required" >&2; exit 2; }
[[ -n "$H5_PATH" ]] || { echo "ERROR: --h5-path is required" >&2; exit 2; }
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "ERROR: --gpu-id must be numeric" >&2; exit 2; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --batch-size must be positive" >&2; exit 2; }

OUT_ABS="$REPO_ROOT/$OUTPUT_DIR"
WATCHER_CONFIG="$OUT_ABS/watcher_config.json"
RESULTS_DIR="$OUT_ABS/watcher_results"
ABLATION_DIR="$OUT_ABS/watcher_ablation"
mkdir -p "$RESULTS_DIR"

if [[ ! -f "$WATCHER_CONFIG" ]]; then
    cat > "$WATCHER_CONFIG" <<'EOF'
{"locked": false, "sampling_steps": 16, "ensemble_members": 1,
 "s_churn": 0.0, "ablation_epoch": null, "last_processed_epoch": 0}
EOF
    echo "initialized watcher config: $WATCHER_CONFIG"
fi

read_watcher_field() {
    "$PYTHON_BIN" - "$WATCHER_CONFIG" "$1" <<'PYEOF'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    c = json.load(f)
print(json.dumps(c.get(sys.argv[2], None)))
PYEOF
}

run_validation() {
    local checkpoint="$1"
    local steps="$2"
    local members="$3"
    local output_path="$4"
    cd "$REPO_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    "$PYTHON_BIN" -u validate_ostia.py \
        --checkpoint "$checkpoint" \
        --h5-path "$H5_PATH" \
        --split val \
        --max-samples 200 \
        --seed 123 \
        --sampling-steps "$steps" \
        --s-churn 0 \
        --ensemble-members "$members" \
        --batch-size "$BATCH_SIZE" \
        --device cuda:0 \
        --output-path "$output_path"
}

run_ablation() {
    local checkpoint="$1"
    local epoch="$2"
    mkdir -p "$ABLATION_DIR"
    echo "=== epoch $epoch inference ablation (steps/members) ==="
    for steps in 16 32; do
        for members in 1 4; do
            local tag="epoch${epoch}_s${steps}_e${members}.json"
            run_validation "$checkpoint" "$steps" "$members" \
                "$ABLATION_DIR/$tag"
        done
    done
    "$PYTHON_BIN" - "$ABLATION_DIR" <<'PYEOF'
import glob, json, os, sys
rows = []
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    rmse = result["overall"]["rmse"]
    rows.append((os.path.basename(path), rmse))
print("ablation summary (val-200 RMSE, K):")
for name, rmse in rows:
    print(f"  {name}: {rmse:.6f}")
print("REVIEW REQUIRED: choose the profile, then set")
print("  watcher_config.json locked=true with sampling_steps/ensemble_members")
PYEOF
}

while true; do
    LATEST_EPOCH=0
    LATEST_CKPT=""
    for ckpt in "$OUT_ABS"/epoch_*.pth; do
        [[ -e "$ckpt" ]] || continue
        epoch="$(basename "$ckpt" .pth)"
        epoch="${epoch#epoch_}"
        epoch_num=$(( 10#$epoch ))
        if [[ "$epoch_num" -gt "$LATEST_EPOCH" ]]; then
            LATEST_EPOCH="$epoch_num"
            LATEST_CKPT="$ckpt"
        fi
    done

    LOCKED="$(read_watcher_field locked)"
    ABLATION_EPOCH_DONE="$(read_watcher_field ablation_epoch)"
    LAST_PROCESSED="$(read_watcher_field last_processed_epoch)"

    if [[ "$LOCKED" == "false" && "$ABLATION_EPOCH_DONE" != "null" ]]; then
        echo "Watcher awaiting profile lock from epoch $ABLATION_EPOCH_DONE; no ablation rerun."
        exit 0
    fi

    if [[ "$LATEST_EPOCH" -gt 0 && "$LATEST_EPOCH" -gt "$LAST_PROCESSED" ]]; then
        if [[ "$LOCKED" == "false" && "$LATEST_EPOCH" -ge "$ABLATION_EPOCH" ]]; then
            run_ablation "$LATEST_CKPT" "$LATEST_EPOCH"
            "$PYTHON_BIN" - "$WATCHER_CONFIG" "$LATEST_EPOCH" <<'PYEOF'
import json, sys
path, epoch = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as f:
    c = json.load(f)
c["ablation_epoch"] = epoch
c["last_processed_epoch"] = epoch
with open(path, "w", encoding="utf-8") as f:
    json.dump(c, f, indent=2)
print("watcher paused for ablation review at epoch", epoch)
PYEOF
            echo "Watcher stopped: review the ablation and lock the profile."
            exit 0
        fi
        STEPS="$(read_watcher_field sampling_steps)"
        MEMBERS="$(read_watcher_field ensemble_members)"
        OUTPUT_PATH="$RESULTS_DIR/val_epoch$(printf '%03d' "$LATEST_EPOCH").json"
        echo "validating epoch $LATEST_EPOCH (steps=$STEPS, members=$MEMBERS)"
        run_validation "$LATEST_CKPT" "$STEPS" "$MEMBERS" "$OUTPUT_PATH"
        "$PYTHON_BIN" - "$WATCHER_CONFIG" "$LATEST_EPOCH" <<'PYEOF'
import json, sys
path, epoch = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as f:
    c = json.load(f)
c["last_processed_epoch"] = epoch
with open(path, "w", encoding="utf-8") as f:
    json.dump(c, f, indent=2)
PYEOF
    fi

    [[ "$ONCE" -eq 1 ]] && break
    sleep "$POLL_SECONDS"
done
