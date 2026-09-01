#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# archive_legacy_ostia_main.sh
#
# Archive the legacy pre-centered main training directory so the new
# centered main training can reuse the canonical output path.
#
# DEFAULT: DRY-RUN.  The script only prints every check and the exact
# command it WOULD run.  Real archiving requires an explicit --execute
# and is performed by Codex on the training server, never by the local
# implementation agent.
#
# Fail-closed checks (PHASE2_MAIN_TRAINING_PLAN 5.2):
#   1. repo HEAD equals --expected-commit (when provided)
#   2. source directory exists and is not a symlink
#   3. archive target does not exist (no overwrite / no merge)
#   4. realpath of source and target both live under $REPO/experiments
#   5. no process or tmux pane is writing inside the source directory
#   6. filesystem has enough free space
#   7. legacy tmux pane tail is captured (last 500 lines)
#   8. new canonical directory is created ONLY after the moved tree
#      verifies (file count / total size / SHA-256)
#
# Rollback (plan 5.5): if the post-move verification fails, the tree is
# moved back atomically and SHA-256 re-verified.
# ---------------------------------------------------------------------------
set -euo pipefail

EXECUTE=0
EXPECTED_COMMIT=""
SOURCE_REL="experiments/ostia_7day_to15day_residual_scratch"
ARCHIVE_ROOT_REL="experiments/archive/pre_centered_20260901"
LEGACY_NAME="legacy_ostia_7day_to15day_residual_scratch"

usage() {
    cat <<'EOF'
Usage: archive_legacy_ostia_main.sh [--execute] [--expected-commit <sha>]

Default is DRY-RUN.  --execute performs the archive after all checks pass.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --execute)
            EXECUTE=1
            shift
            ;;
        --expected-commit)
            EXPECTED_COMMIT="$2"
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

REPO_ROOT="$(git rev-parse --show-toplevel)"
SOURCE="${REPO_ROOT}/${SOURCE_REL}"
ARCHIVE_ROOT="${REPO_ROOT}/${ARCHIVE_ROOT_REL}"
TARGET="${ARCHIVE_ROOT}/${LEGACY_NAME}"
CANONICAL="${REPO_ROOT}/experiments/ostia_7day_to15day_residual_scratch"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

pass() {
    echo "PASS: $*"
}

if [[ "$EXECUTE" -eq 1 && -z "$EXPECTED_COMMIT" ]]; then
    fail "--execute requires --expected-commit"
fi

reject_symlink_components() {
    local path="$1"
    local current="/"
    local relative="${path#/}"
    local component
    IFS='/' read -r -a components <<< "$relative"
    for component in "${components[@]}"; do
        [[ -n "$component" ]] || continue
        current="${current%/}/$component"
        [[ -L "$current" ]] && fail "symlink component in protected path: $current"
    done
    return 0
}

[[ "$EXECUTE" -eq 1 ]] \
    && echo "MODE: EXECUTE" \
    || echo "MODE: DRY-RUN (no filesystem changes)"

# ---- 1. HEAD check -------------------------------------------------------
HEAD="$(git rev-parse HEAD)"
if [[ -n "$EXPECTED_COMMIT" ]]; then
    if [[ "$HEAD" == "$EXPECTED_COMMIT" ]]; then
        pass "repo HEAD == expected commit $EXPECTED_COMMIT"
    else
        fail "repo HEAD=$HEAD does not match --expected-commit $EXPECTED_COMMIT"
    fi
else
    echo "NOTE: no --expected-commit given (permitted only in dry-run); HEAD=$HEAD"
fi

# ---- 2. source exists and is not a symlink ------------------------------
[[ -e "$SOURCE" ]] || fail "source $SOURCE does not exist"
[[ -d "$SOURCE" ]] || fail "source $SOURCE is not a directory"
[[ -L "$SOURCE" ]] && fail "source $SOURCE is a symlink; refusing"
pass "source exists: $SOURCE"

# ---- 3. archive target must not exist -----------------------------------
[[ -e "$TARGET" ]] && fail "archive target already exists: $TARGET (no overwrite/merge)"
[[ -e "$ARCHIVE_ROOT" ]] \
    && echo "NOTE: archive root exists (appending): $ARCHIVE_ROOT" \
    || echo "NOTE: archive root will be created: $ARCHIVE_ROOT"
pass "archive target does not exist: $TARGET"

# ---- 4. realpath containment --------------------------------------------
EXP_BASE="$(realpath "${REPO_ROOT}/experiments")"
SRC_REAL="$(realpath "$SOURCE")"
TGT_REAL="$(realpath -m "$TARGET")"
[[ "$SRC_REAL" == "${EXP_BASE}/"* ]] || fail "source realpath $SRC_REAL outside $EXP_BASE"
[[ "$TGT_REAL" == "${EXP_BASE}/"* ]] || fail "target realpath $TGT_REAL outside $EXP_BASE"
[[ "$SRC_REAL" == "$SOURCE" ]] || fail "source path contains symlink components"
reject_symlink_components "${REPO_ROOT}/experiments"
reject_symlink_components "$SOURCE"
reject_symlink_components "$ARCHIVE_ROOT"
reject_symlink_components "$(dirname "$TARGET")"
pass "realpath containment verified under $EXP_BASE"

# ---- 5. no active writers -----------------------------------------------
WRITERS=""
if command -v pgrep >/dev/null 2>&1; then
    WRITERS="$(pgrep -af "$SOURCE" 2>/dev/null || true)"
fi
if [[ -n "$WRITERS" ]]; then
    fail "processes reference the source directory; archive refused:
$WRITERS"
fi
pass "no process references the source directory"

TMUX_TAIL=""
if command -v tmux >/dev/null 2>&1; then
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        pane_path="${line##* }"
        [[ "$pane_path" == "${SOURCE}" || "$pane_path" == "${SOURCE}/"* ]] \
            && fail "tmux pane cwd is inside the source directory: $line"
    done < <(tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} #{pane_pid} #{pane_current_path}' 2>/dev/null || true)
    TMUX_TAIL="$(tmux capture-pane -p -S -500 -t "ostia_train" 2>/dev/null || true)"
fi
pass "no tmux pane is working inside the source directory"
if [[ -n "$TMUX_TAIL" ]]; then
    echo "NOTE: captured $(( $(printf '%s\n' "$TMUX_TAIL" | wc -l) )) lines of legacy ostia_train pane tail"
fi

# ---- 6. free space -------------------------------------------------------
SOURCE_BYTES="$(du -sb "$SOURCE" | cut -f1)"
FS_OF="$(df -Pk "$SOURCE" | awk 'NR==2 {print $4}')"
MARGIN_KB=$(( 10 * 1024 * 1024 ))  # 10 GiB margin for manifests + new run
NEEDED_KB=$(( SOURCE_BYTES / 1024 + MARGIN_KB ))
if [[ "$FS_OF" -lt "$NEEDED_KB" ]]; then
    fail "free space ${FS_OF} KiB < needed ~${NEEDED_KB} KiB"
fi
pass "free space ${FS_OF} KiB >= needed ~${NEEDED_KB} KiB (source ${SOURCE_BYTES} bytes)"

# ---- 7. new canonical dir handling --------------------------------------
if [[ -e "$CANONICAL" ]]; then
    if [[ "$EXECUTE" -eq 0 ]]; then
        echo "NOTE: canonical path currently exists (it IS the source); it will be moved away"
    fi
else
    fail "canonical path missing but source check passed; inconsistent state"
fi

# ---- plan the action -----------------------------------------------------
echo
echo "=== ARCHIVE PLAN (dry-run unless --execute) ==="
echo "  mkdir -p $ARCHIVE_ROOT"
echo "  manifest -> $ARCHIVE_ROOT/legacy_file_manifest.tsv"
echo "  sha256   -> $ARCHIVE_ROOT/legacy_sha256.txt"
echo "  commit   -> $ARCHIVE_ROOT/source_git_commit.txt"
echo "  tmux tail-> $ARCHIVE_ROOT/legacy_tmux_tail.txt"
echo "  mv $SOURCE -> $TARGET        (atomic, same filesystem)"
echo "  verify file count / total size / SHA-256"
echo "  mkdir -p $CANONICAL/logs"
echo

if [[ "$EXECUTE" -eq 0 ]]; then
    echo "DRY-RUN COMPLETE: no changes made. Re-run with --execute to archive."
    exit 0
fi

# ======================= EXECUTE MODE ====================================
mkdir -p "$ARCHIVE_ROOT"
git rev-parse HEAD > "$ARCHIVE_ROOT/source_git_commit.txt"
git status --porcelain >> "$ARCHIVE_ROOT/source_git_commit.txt" 2>/dev/null || true
[[ -n "$TMUX_TAIL" ]] && printf '%s\n' "$TMUX_TAIL" > "$ARCHIVE_ROOT/legacy_tmux_tail.txt"

( cd "$SOURCE" && find . -type f -printf '%P\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\n' | sort ) \
    > "$ARCHIVE_ROOT/legacy_file_manifest.tsv"

( cd "$SOURCE" && find . -type f \( \
        -name '*.json' -o -name '*.pth' -o -name '*.npz' \
        -o -name '*.png' -o -name '*.log' -o -name '*.txt' \
        -o -name '*.csv' -o -name '*.tsv' -o -name '*.md' \
    \) -print0 | sort -z | xargs -0 -r sha256sum ) \
    > "$ARCHIVE_ROOT/legacy_sha256.txt"

cat > "$ARCHIVE_ROOT/ARCHIVE_README.md" <<EOF
# Legacy pre-centered main training archive

- source: ${SOURCE_REL}
- archived: $(date -Is)
- archive commit: ${HEAD}
- target: ${ARCHIVE_ROOT_REL}/${LEGACY_NAME}
- checksums: legacy_sha256.txt
- manifest: legacy_file_manifest.tsv
- tmux tail: legacy_tmux_tail.txt

The new centered main training re-created the canonical path
${SOURCE_REL} only after this archive verified.
EOF

mv "$SOURCE" "$TARGET"
echo "moved: $SOURCE -> $TARGET"

# ---- post-move verification --------------------------------------------
COUNT_OK=0
SIZE_OK=0
SHA_OK=0
if [[ "$(cd "$TARGET" && find . -type f | wc -l)" == "$(wc -l < "$ARCHIVE_ROOT/legacy_file_manifest.tsv")" ]]; then
    COUNT_OK=1
else
    echo "VERIFY FAIL: file count changed after move"
fi
if [[ "$(du -sb "$TARGET" | cut -f1)" == "$SOURCE_BYTES" ]]; then
    SIZE_OK=1
else
    echo "VERIFY FAIL: total size changed after move"
fi
if ( cd "$TARGET" && find . -type f \( \
        -name '*.json' -o -name '*.pth' -o -name '*.npz' \
        -o -name '*.png' -o -name '*.log' -o -name '*.txt' \
        -o -name '*.csv' -o -name '*.tsv' -o -name '*.md' \
    \) -print0 | sort -z | xargs -0 -r sha256sum ) \
    | cmp -s - "$ARCHIVE_ROOT/legacy_sha256.txt"; then
    SHA_OK=1
else
    echo "VERIFY FAIL: SHA-256 changed after move"
fi

if [[ "$COUNT_OK" -eq 1 && "$SIZE_OK" -eq 1 && "$SHA_OK" -eq 1 ]]; then
    echo "VERIFY PASS: file count, total size and SHA-256 match"
    mkdir -p "$CANONICAL/logs"
    echo "created canonical directory: $CANONICAL/logs"
    echo "ARCHIVE COMPLETE."
else
    echo "ROLLBACK: moving the tree back to the canonical path"
    mv "$TARGET" "$SOURCE"
    ( cd "$SOURCE" && find . -type f \( \
        -name '*.json' -o -name '*.pth' -o -name '*.npz' \
        -o -name '*.png' -o -name '*.log' -o -name '*.txt' \
        -o -name '*.csv' -o -name '*.tsv' -o -name '*.md' \
    \) -print0 | sort -z | xargs -0 -r sha256sum ) \
        | cmp -s - "$ARCHIVE_ROOT/legacy_sha256.txt" \
        && echo "ROLLBACK VERIFIED: SHA-256 matches" \
        || echo "ROLLBACK FAILED: manual intervention required"
    exit 1
fi
