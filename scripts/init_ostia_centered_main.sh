#!/usr/bin/env bash
# Initialize the now-empty canonical main directory after verified archive.
# Default is dry-run. Never overwrites an existing config artifact.
set -euo pipefail

EXECUTE=0
EXPECTED_COMMIT=""
MEAN_CHECKPOINT=""
CENTERED_STATS=""
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CANONICAL="$REPO_ROOT/experiments/ostia_7day_to15day_residual_scratch"
CONFIG_SOURCE="$REPO_ROOT/configs/ostia_centered_diffusion_main.json"
LOCKED_MEAN_SHA="cb09b15ce97e11800b83fcf7c8ef9df09aa47f8831a0a36fffa987e413fc53e6"

usage() {
    cat <<'EOF'
Usage: init_ostia_centered_main.sh --mean-checkpoint <pth>
       --centered-stats <json> --expected-commit <sha> [--execute]

Default is dry-run. Run only after archive_legacy_ostia_main.sh succeeds.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --execute) EXECUTE=1; shift ;;
        --expected-commit) EXPECTED_COMMIT="$2"; shift 2 ;;
        --mean-checkpoint) MEAN_CHECKPOINT="$2"; shift 2 ;;
        --centered-stats) CENTERED_STATS="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

[[ -n "$EXPECTED_COMMIT" ]] || { echo "ERROR: --expected-commit is required" >&2; exit 2; }
[[ -f "$MEAN_CHECKPOINT" ]] || { echo "ERROR: frozen mean missing: $MEAN_CHECKPOINT" >&2; exit 1; }
[[ -f "$CENTERED_STATS" ]] || { echo "ERROR: centered stats missing: $CENTERED_STATS" >&2; exit 1; }
[[ -f "$CONFIG_SOURCE" ]] || { echo "ERROR: main config missing: $CONFIG_SOURCE" >&2; exit 1; }
HEAD="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$HEAD" == "$EXPECTED_COMMIT" ]] || { echo "ERROR: HEAD $HEAD != $EXPECTED_COMMIT" >&2; exit 1; }
MEAN_SHA="$(sha256sum "$MEAN_CHECKPOINT" | awk '{print $1}')"
[[ "$MEAN_SHA" == "$LOCKED_MEAN_SHA" ]] || { echo "ERROR: frozen mean SHA mismatch" >&2; exit 1; }
MEAN_SIDECAR="${MEAN_CHECKPOINT}.semantics.json"
[[ -f "$MEAN_SIDECAR" ]] || { echo "ERROR: frozen mean sidecar missing: $MEAN_SIDECAR" >&2; exit 1; }

[[ -d "$CANONICAL" ]] || { echo "ERROR: canonical directory absent; archive step did not initialize it" >&2; exit 1; }
if find "$CANONICAL" -mindepth 1 -maxdepth 1 ! -name logs -print -quit | grep -q .; then
    echo "ERROR: canonical directory contains non-log artifacts; refusing" >&2
    exit 1
fi
if [[ -d "$CANONICAL/logs" ]] && find "$CANONICAL/logs" -mindepth 1 -print -quit | grep -q .; then
    echo "ERROR: canonical logs directory is not empty; refusing" >&2
    exit 1
fi

echo "MODE: $([[ "$EXECUTE" -eq 1 ]] && echo EXECUTE || echo DRY-RUN)"
echo "commit: $HEAD"
echo "frozen mean: $MEAN_CHECKPOINT ($MEAN_SHA)"
echo "centered stats: $CENTERED_STATS ($(sha256sum "$CENTERED_STATS" | awk '{print $1}'))"
echo "target: $CANONICAL/config"
[[ "$EXECUTE" -eq 1 ]] || { echo "DRY-RUN COMPLETE"; exit 0; }

STAGING="$CANONICAL/.config.staging.$$"
[[ ! -e "$STAGING" ]] || { echo "ERROR: staging path already exists: $STAGING" >&2; exit 1; }
mkdir "$STAGING"
cleanup_staging() {
    if [[ -d "$STAGING" ]]; then
        [[ "$STAGING" == "$CANONICAL/.config.staging."* ]] \
            || { echo "ERROR: unsafe staging cleanup target: $STAGING" >&2; return 1; }
        rm -rf -- "$STAGING"
    fi
}
trap cleanup_staging EXIT

cp --no-clobber "$CONFIG_SOURCE" "$STAGING/ostia_centered_diffusion_main.json"
cp --no-clobber "$MEAN_CHECKPOINT" "$STAGING/frozen_mean.pth"
cp --no-clobber "$CENTERED_STATS" "$STAGING/centered_stats_train.json"
cp --no-clobber "$MEAN_SIDECAR" "$STAGING/frozen_mean.pth.semantics.json"
printf '%s\n' "$HEAD" > "$STAGING/source_git_commit.txt"
printf '%s\n' 'scripts/run_ostia_centered_main.sh --gpus <a,b>' > "$STAGING/launch_command.sh"
(
    cd "$STAGING"
    sha256sum \
        ostia_centered_diffusion_main.json frozen_mean.pth \
        frozen_mean.pth.semantics.json centered_stats_train.json \
        source_git_commit.txt launch_command.sh \
        > SHA256SUMS.txt
    sha256sum -c SHA256SUMS.txt
)
mv "$STAGING" "$CANONICAL/config"
trap - EXIT
echo "INITIALIZATION COMPLETE: $CANONICAL/config"
