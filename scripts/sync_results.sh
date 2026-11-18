#!/usr/bin/env bash
# scripts/sync_results.sh
# Copy kết quả từ Google Drive về reports/ để commit lên GitHub
# Chạy SAU KHI Leader training xong và C validate PASS

set -euo pipefail

# Cấu hình đường dẫn (chỉnh theo môi trường thực tế)
DRIVE_RESULTS_DIR="${DRIVE_RESULTS_DIR:-/content/drive/MyDrive/yelp-topic7/results}"
REPORTS_DIR="${REPORTS_DIR:-$(git rev-parse --show-toplevel)/reports}"

echo "=== Sync Results from Drive to reports/ ==="
echo "Source: $DRIVE_RESULTS_DIR"
echo "Target: $REPORTS_DIR"
echo ""

# Kiểm tra thư mục nguồn
if [[ ! -d "$DRIVE_RESULTS_DIR" ]]; then
    echo "ERROR: Không tìm thấy thư mục Drive: $DRIVE_RESULTS_DIR"
    echo "Hãy set DRIVE_RESULTS_DIR hoặc mount Drive đúng chỗ."
    exit 1
fi

# Tạo reports/ nếu chưa có
mkdir -p "$REPORTS_DIR"

# Các file bắt buộc phải copy (theo Design §9.9)
REQUIRED_FILES=(
    "test_predictions.csv"
    "model_summary.csv"
)

# Các file tuỳ chọn (có thể thiếu nếu baseline/chưa chạy hết model)
OPTIONAL_FILES=(
    "test_predictions_with_text.csv"
    "preds/bert_test.csv"
    "preds/bert_test_meta.json"
    "preds/distilbert_test.csv"
    "preds/distilbert_test_meta.json"
    "preds/roberta_test.csv"
    "preds/roberta_test_meta.json"
    "preds/baseline_test.csv"
    "preds/baseline_test_meta.json"
)

echo "--- Copy required files ---"
for f in "${REQUIRED_FILES[@]}"; do
    src="$DRIVE_RESULTS_DIR/$f"
    dst="$REPORTS_DIR/$f"
    if [[ -f "$src" ]]; then
        cp -f "$src" "$dst"
        echo "  ✓ $f"
    else
        echo "  ✗ MISSING REQUIRED: $f"
        exit 1
    fi
done

echo ""
echo "--- Copy optional files ---"
for f in "${OPTIONAL_FILES[@]}"; do
    src="$DRIVE_RESULTS_DIR/$f"
    dst="$REPORTS_DIR/$f"
    if [[ -f "$src" ]]; then
        mkdir -p "$(dirname "$dst")"
        cp -f "$src" "$dst"
        echo "  ✓ $f"
    else
        echo "  ⚠ Missing optional: $f"
    fi
done

echo ""
echo "--- Verify copied files ---"
echo "test_predictions.csv columns:"
head -1 "$REPORTS_DIR/test_predictions.csv" | tr ',' '\n' | nl

echo ""
echo "model_summary.csv rows:"
wc -l "$REPORTS_DIR/model_summary.csv"

echo ""
echo "=== Sync completed ==="
echo "Kiểm tra: ls -la $REPORTS_DIR/"
echo "Sau đó chạy validation:"
echo "  python -m scripts.validate_outputs --results-dir $REPORTS_DIR"
echo "Nếu PASS: git add $REPORTS_DIR && git commit -m \"Week 2 results: test_predictions + model_summary\""