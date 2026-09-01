#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_ostia_centered_main.sh
#
# Launch the Phase 2 centered diffusion main training from the single
# authoritative config JSON.  The script never invents config values:
# it reads configs/ostia_centered_diffusion_main.json, preflights the
# frozen mean / centered stats identity, prints every hash, then starts
# torchrun with --config.
#
#   run_ostia_centered_main.sh --gpus 0,1
#   run_ostia_centered_main.sh --gpus 3          # single-GPU fallback
#
# Effective global batch is always 32:
#   2 GPUs: batch_per_gpu=8  x grad_accum=2 x 2 = 32   (config JSON)
#   1 GPU : batch_per_gpu=8  x grad_accum=4 x 1 = 32   (CLI override)
# ---------------------------------------------------------------------------
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS=""
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_JSON="$REPO_ROOT/configs/ostia_centered_diffusion_main.json"
LOCKED_MEAN_SHA="cb09b15ce97e11800b83fcf7c8ef9df09aa47f8831a0a36fffa987e413fc53e6"

usage() {
    cat <<'EOF'
Usage: run_ostia_centered_main.sh --gpus <id[,id...]>

Reads configs/ostia_centered_diffusion_main.json (the only authoritative
config source), preflights the frozen-mean / centered-stats identity,
prints all hashes and launches the training.  Writes a launch manifest
into the output dir under logs/.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPUS="$2"
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

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
NPROC="${#GPU_IDS[@]}"
for gpu_id in "${GPU_IDS[@]}"; do
    [[ "$gpu_id" =~ ^[0-9]+$ ]] || { echo "ERROR: bad gpu id: $gpu_id" >&2; exit 2; }
done
[[ "$NPROC" -ge 1 && "$NPROC" -le 2 ]] || { echo "ERROR: 1 or 2 GPUs only (plan 7.1/7.2)" >&2; exit 2; }

echo "=== centered main launch preflight ==="
echo "repo:     $REPO_ROOT"
echo "commit:   $(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "gpus:     $GPUS (nproc=$NPROC)"
echo

[[ -f "$CONFIG_JSON" ]] || { echo "ERROR: config JSON missing: $CONFIG_JSON" >&2; exit 1; }
CONFIG_SHA="$(sha256sum "$CONFIG_JSON" | awk '{print $1}')"
echo "config JSON SHA-256: $CONFIG_SHA"

# JSON-derived values (no jq dependency: python reads the same file
# the trainer will read).
read -r BATCH_PER_GPU GRAD_ACCUM MEAN_PATH STATS_PATH OUT_DIR < <(
    "$PYTHON_BIN" - "$CONFIG_JSON" <<'PYEOF'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    c = json.load(f)
print(c["batch_per_gpu"], c["gradient_accumulation"],
      c["mean_checkpoint_path"], c["centered_stats_path"],
      c["output_dir"])
PYEOF
)

# Centered runs require sigma_data=1.0: verify the JSON itself.
"$PYTHON_BIN" - "$CONFIG_JSON" <<'PYEOF' \
    || { echo "ERROR: config JSON sigma_data != 1.0" >&2; exit 1; }
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    c = json.load(f)
assert c["model_type"] == "centered_diffusion", c.get("model_type")
assert c["sigma_data"] == 1.0, c.get("sigma_data")
PYEOF
echo "config JSON sigma_data=1.0 OK (model_type=centered_diffusion)"

MEAN_ABS="$REPO_ROOT/$MEAN_PATH"
STATS_ABS="$REPO_ROOT/$STATS_PATH"
[[ -f "$MEAN_ABS" ]] || { echo "ERROR: frozen mean missing: $MEAN_ABS" >&2; exit 1; }
[[ -f "$STATS_ABS" ]] || { echo "ERROR: centered stats missing: $STATS_ABS" >&2; exit 1; }
MEAN_SHA="$(sha256sum "$MEAN_ABS" | awk '{print $1}')"
STATS_SHA="$(sha256sum "$STATS_ABS" | awk '{print $1}')"
echo "frozen mean SHA-256:  $MEAN_SHA"
echo "centered stats SHA-256: $STATS_SHA"

if [[ "$MEAN_SHA" != "$LOCKED_MEAN_SHA" ]]; then
    echo "ERROR: frozen mean SHA-256 does not match the locked identity $LOCKED_MEAN_SHA" >&2
    exit 1
fi
echo "frozen mean identity LOCKED OK"

# Structural validation of the stats JSON (same validator the trainer runs).
PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" - "$STATS_ABS" <<'PYEOF' \
    || { echo "ERROR: centered stats validation failed" >&2; exit 1; }
import json, sys
from deterministic_iafno.centered_stats import validate_centered_stats_payload
with open(sys.argv[1], encoding="utf-8") as f:
    stats = json.load(f)
validate_centered_stats_payload(stats, 15, 7, 15)
print("centered stats provenance OK (split=train, 15 leads, positive stds)")
PYEOF

# Effective global batch.
EXTRA_ARGS=()
if [[ "$NPROC" -eq 1 ]]; then
    GRAD_ACCUM=4   # plan 7.2: single-GPU fallback keeps batch 32
    EXTRA_ARGS+=(--gradient-accumulation "$GRAD_ACCUM")
fi
EFFECTIVE_BATCH=$(( BATCH_PER_GPU * GRAD_ACCUM * NPROC ))
echo "effective global batch: $BATCH_PER_GPU x $GRAD_ACCUM x $NPROC = $EFFECTIVE_BATCH"
[[ "$EFFECTIVE_BATCH" -eq 32 ]] || { echo "ERROR: effective global batch must be 32" >&2; exit 1; }

# Launch manifest.
OUT_ABS="$REPO_ROOT/$OUT_DIR"
mkdir -p "$OUT_ABS/logs"
MANIFEST="$OUT_ABS/logs/launch_manifest.json"
cat > "$MANIFEST" <<EOF
{
  "git_commit": "$(git -C "$REPO_ROOT" rev-parse HEAD)",
  "config_json_sha256": "$CONFIG_SHA",
  "frozen_mean_sha256": "$MEAN_SHA",
  "centered_stats_sha256": "$STATS_SHA",
  "gpus": "$GPUS",
  "nproc": $NPROC,
  "batch_per_gpu": $BATCH_PER_GPU,
  "gradient_accumulation": $GRAD_ACCUM,
  "effective_global_batch": $EFFECTIVE_BATCH,
  "launched_at": "$(date -Is)"
}
EOF
echo "launch manifest: $MANIFEST"

echo
echo "=== launching trainer ==="
cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPUS" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON_BIN" -u -m torch.distributed.run \
    --standalone --nproc_per_node="$NPROC" \
    trainer_ostia.py \
    --config "$CONFIG_JSON" \
    "${EXTRA_ARGS[@]}"
