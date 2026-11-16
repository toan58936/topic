#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    echo "Usage: bash scripts/sync_results.sh SOURCE_RESULTS_DIR DEST_REPORTS_DIR [--split-stats PATH] [--include-text] [--overwrite] [--dry-run]"
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

[[ $# -ge 2 ]] || { usage >&2; exit 2; }

SOURCE="$1"
DEST="$2"
shift 2

SPLIT_STATS=""
INCLUDE_TEXT=0
OVERWRITE=0
DRY_RUN=0

while (($#)); do
    case "$1" in
        --split-stats) [[ $# -ge 2 ]] || fail "Missing split_stats path"; SPLIT_STATS="$2"; shift 2 ;;
        --include-text) INCLUDE_TEXT=1; shift ;;
        --overwrite) OVERWRITE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1" ;;
    esac
done

[[ -d "$SOURCE" ]] || fail "Source directory not found: $SOURCE"

SOURCE="$(cd "$SOURCE" && pwd -P)"

for file in test_predictions.csv model_summary.csv; do
    [[ -f "$SOURCE/$file" ]] || fail "Missing $file in source"
done

[[ -d "$SOURCE/preds" ]] || fail "Missing preds/ directory"

shopt -s nullglob
PRED_FILES=("$SOURCE"/preds/*_test.csv)
((${#PRED_FILES[@]} > 0)) || fail "No *_test.csv files found"

if [[ -z "$SPLIT_STATS" ]]; then
    if [[ -f "$SOURCE/../data/split_stats.json" ]]; then
        SPLIT_STATS="$SOURCE/../data/split_stats.json"
    elif [[ -f "$SOURCE/data/split_stats.json" ]]; then
        SPLIT_STATS="$SOURCE/data/split_stats.json"
    else
        fail "Cannot find split_stats.json; provide --split-stats PATH"
    fi
fi

[[ -f "$SPLIT_STATS" ]] || fail "split_stats.json not found: $SPLIT_STATS"

TARGETS=(
    "$DEST/test_predictions.csv"
    "$DEST/model_summary.csv"
    "$DEST/data/split_stats.json"
)

for file in "${PRED_FILES[@]}"; do
    TARGETS+=("$DEST/preds/$(basename "$file")")
done

for file in "$SOURCE"/preds/*_test_meta.json; do
    [[ -f "$file" ]] || continue
    TARGETS+=("$DEST/preds/$(basename "$file")")
done

if (( INCLUDE_TEXT )); then
    [[ -f "$SOURCE/test_predictions_with_text.csv" ]] || fail "with_text file not found"
    TARGETS+=("$DEST/test_predictions_with_text.csv")
fi

if (( ! OVERWRITE )); then
    for target in "${TARGETS[@]}"; do
        [[ ! -e "$target" ]] || fail "Target exists: $target. Use --overwrite to replace."
    done
fi

echo "Source: $SOURCE"
echo "Destination: $DEST"
echo "Individual prediction files: ${#PRED_FILES[@]}"

if (( DRY_RUN )); then
    echo "DRY RUN: no files copied."
    exit 0
fi

mkdir -p "$DEST/preds" "$DEST/data"

cp -p "$SOURCE/test_predictions.csv" "$DEST/test_predictions.csv"
cp -p "$SOURCE/model_summary.csv" "$DEST/model_summary.csv"
cp -p "$SPLIT_STATS" "$DEST/data/split_stats.json"
cp -p "${PRED_FILES[@]}" "$DEST/preds/"

if (( INCLUDE_TEXT )); then
    cp -p "$SOURCE/test_predictions_with_text.csv" "$DEST/"
    echo "WARNING: with_text contains review text. Do not commit it to Git."
fi

# Copy matching metadata files when present
for file in "$SOURCE"/preds/*_test_meta.json; do
    [[ -f "$file" ]] || continue
    target="$DEST/preds/$(basename "$file")"
    if [[ -e "$target" && $OVERWRITE -eq 0 ]]; then
        fail "Metadata target already exists: $target"
    fi
    cp -p "$file" "$target"
done

echo "SUCCESS: results synchronized. No destination files were deleted."