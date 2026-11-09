# Đặc tả Thiết kế Kiểm thử Tích hợp — `tests/test_integration.py`, `scripts/validate_outputs.py`

> **Tài liệu:** `docs/06_integration_test.md` (Define & Design cho end-to-end validation)
> **Module liên quan:** `tests/test_integration.py` (pytest), `scripts/validate_outputs.py` (CLI)
> **Kế hoạch kiểm thử:** `docs/06_integration_test_tests.md` (tách riêng — meta-test cho script validation)
> **Phụ thuộc:** Tất cả spec 01–05; `docs/design.md` §4, §9.9; `docs/01_data.md` (split_id), `docs/02_evaluation.md` (model_summary.csv), `docs/04_engine_and_utils.md` (predict/merge schemas)
> **Trạng thái:** Bản nháp trước khi hiện thực (Pre-implementation Design Specification)

---

## PHẦN I: DEFINE — Xác định Bài toán & Yêu cầu

### 1.1. Bối cảnh & Mục tiêu

Sau khi Tuần 2 kết thúc (chạy xong 3 backbone + baseline, xuất `test_predictions.csv` và `model_summary.csv` lên Drive), **thành viên F (Integration Tester)** chạy một quy trình xác thực tự động để đảm bảo:

1. **Tính nhất quán dữ liệu**: Các file dự đoán không bị trộn giữa các lần chia (`split_id`), không mất review, `review_id`/`true_stars` khớp nhau.
2. **Tính đúng đắn schema**: `test_predictions.csv`, `model_summary.csv`, `<model>_test.csv` đúng cột, kiểu, giá trị hợp lệ.
3. **Tính khớp số liệu**: Chỉ số trong `model_summary.csv` khớp với tính toán lại từ `test_predictions.csv` (dùng `src.metrics`).
4. **Sẵn sàng báo cáo**: Dashboard có thể chạy, không thiếu cột, không NaN lạ.

Hai công cụ:
- **`tests/test_integration.py`**: Chạy trong pytest (CI cục bộ/Colab), dùng fixture dữ liệu thật từ Drive hoặc artifact.
- **`scripts/validate_outputs.py`**: CLI độc lập, chạy sau khi pull kết quả từ Drive về `reports/`, exit code ≠ 0 nếu lỗi.

### 1.2. Đầu vào / Đầu ra

| | Mô tả |
| :--- | :--- |
| **Đầu vào** | Thư mục kết quả (Drive `results/` hoặc local `reports/`): `test_predictions.csv`, `test_predictions_with_text.csv` (tuỳ chọn), `model_summary.csv`, `preds/<model>_test.csv`, `preds/<model>_test_meta.json`, `split_stats.json` (từ `data/`) |
| **Đầu ra** | `stdout`: tóm tắt PASS/FAIL từng kiểm tra; `stderr`: chi tiết lỗi; exit code 0 nếu tất cả PASS, 1 nếu có FAIL. Không ghi file. |

### 1.3. Yêu cầu chức năng

| Mã | Yêu cầu |
| :--- | :--- |
| FR-1 | **Merge validation**: Đọc `test_predictions.csv` (và `test_predictions_with_text.csv` nếu có), kiểm tra: `split_id` khớp `split_stats.json`; tập `review_id` và `true_stars` khớp giữa các file; không trùng `review_id`; không NaN ở cột bắt buộc. |
| FR-2 | **Schema validation**: Mỗi file CSV/JSON đúng cột bắt buộc, kiểu dữ liệu, miền giá trị (theo `docs/design.md` §9.9). |
| FR-3 | **Metric reconciliation**: Đối chiếu `model_summary.csv` với tính toán lại từ `test_predictions.csv` dùng `src.metrics.compute_metrics`, `per_star_metrics`. Sai số tuyệt đối ≤ `1e-4` cho MAE/RMSE/R²; ≤ `1e-3` cho Pearson/Spearman (do làm tròn float). |
| FR-4 | **Per-model file validation**: Mỗi `<model>_test.csv` có cột `review_id`, `true_stars`, `pred_raw`, `pred`, `split_id`; `pred` đã clip 1.0-5.0; `pred_raw` không clip; `rmse_raw` trong `_test_meta.json` khớp tính toán từ `pred_raw`. |
| FR-5 | **Baseline validation**: Nếu có `pred_baseline`, kiểm tra nằm trong 1.0-5.0; `abs_err_baseline` = `|true_stars - pred_baseline|`. |
| FR-6 | **Dashboard readiness**: `model_summary.csv` đủ cột cho dashboard (§5 Design); `test_predictions.csv` có đủ cột `pred_*` và `abs_err_*` cho các mô hình có mặt. |
| FR-7 | **CLI `validate_outputs.py`**: `python -m scripts.validate_outputs --results-dir <path> [--split-stats <path>] [--tolerance <float>]`. Exit 0 = PASS, 1 = FAIL. In tóm tắt dạng bảng. |
| FR-8 | **Pytest `test_integration.py`**: Fixture `results_dir` trỏ tới thư mục kết quả (local `reports/` hoặc mount Drive). Các test function tương ứng FR-1..FR-6. |

### 1.4. Ràng buộc

* **Không import `torch`** trong `validate_outputs.py` (chạy trên máy không GPU). Chỉ dùng `pandas`, `numpy`, `scipy`, `src.metrics`.
* `tests/test_integration.py` có thể import `torch` nếu cần (ví dụ kiểm tra checkpoint), nhưng khuyến khích không.
* `split_stats.json` là nguồn thật cho `split_id` và phân phối sao.
* Dung lượng: `test_predictions.csv` ~1 MB (9k dòng), validation phải chạy < 30 giây.

### 1.5. Tiêu chí chấp nhận

1. Chạy `python -m scripts.validate_outputs --results-dir reports/` trên repo sau khi commit kết quả Tuần 2 → exit 0, in "ALL CHECKS PASSED".
2. Nếu cố ý sửa `model_summary.csv` sai một chỉ số → script báo FAIL ở metric reconciliation.
3. Nếu thiếu cột `split_id` trong `test_predictions.csv` → script báo FAIL ở merge validation.
4. Pytest `tests/test_integration.py -v` tất cả xanh khi dữ liệu đúng.
5. Script chạy xong < 30 giây trên 9k review.

---

## PHẦN II: DESIGN — Thiết kế Chi tiết

### 2.1. Kiến trúc validation

```text
scripts/validate_outputs.py
    │
    ├── parse_args() ──► results_dir, split_stats_path, tolerance
    │
    ├── load_all_data()
    │       ├── split_stats.json (từ data/ hoặc results/)
    │       ├── test_predictions.csv (và _with_text nếu có)
    │       ├── model_summary.csv
    │       ├── preds/<model>_test.csv + _test_meta.json (mỗi model)
    │       └── baseline.joblib (tuỳ chọn, không bắt buộc)
    │
    ├── check_merge_consistency()
    │       ├── split_id khớp split_stats.json
    │       ├── review_id unique, true_stars khớp giữa file
    │       ├── không NaN cột bắt buộc
    │       └── n_tokens > 0
    │
    ├── check_schemas()
    │       ├── test_predictions.csv: cột bắt buộc, pred_* trong 1.0-5.0, abs_err_* ≥ 0
    │       ├── model_summary.csv: cột §9.9.3 Design, params_total ≥ 0, train_seconds ≥ 0
    │       ├── <model>_test.csv: pred_raw không clip, pred đã clip, rmse_raw khớp meta
    │       └── _test_meta.json: device, inference_seconds, fp16, checkpoint_epoch
    │
    ├── reconcile_metrics()
    │       ├── Với mỗi model trong model_summary: đọc pred_<model> từ test_predictions.csv
    │       ├── Tính compute_metrics + per_star_metrics (src.metrics)
    │       ├── So sánh với dòng model_summary: |recalc - recorded| ≤ tolerance
    │       └── Báo cáo chi tiết: model, metric, recorded, recalc, diff
    │
    ├── check_dashboard_readiness()
    │       ├── model_summary.csv có cột: model, mae, rmse, r2, pearson, spearman, macro_mae,
    │       │   train_seconds, seconds_per_epoch, params_total, rmse_reduction_vs_baseline_pct,
    │       │   inference_seconds, split_id
    │       ├── test_predictions.csv có pred_* và abs_err_* cho mỗi model có mặt
    │       └── Không NaN toàn cột bắt buộc
    │
    └── print_summary_table() ──► exit(0 nếu all_pass else 1)
```

### 2.2. Schema chi tiết từng file (theo `docs/design.md` §9.9)

#### 2.2.1 `test_predictions.csv`
| Cột | Kiểu | Required | Validation |
| :--- | :--- | :--- | :--- |
| `review_id` | string | ✅ | Unique, không rỗng |
| `true_stars` | int | ✅ | ∈ {1,2,3,4,5} |
| `n_tokens` | int | ✅ | > 0, ≤ 512 (BERT tokenizer) |
| `split_id` | string | ✅ | Khớp `split_stats.json`, len=12 |
| `pred_baseline` | float | ⚠️ | ∈ [1.0, 5.0] nếu có |
| `pred_bert` | float | ⚠️ | ∈ [1.0, 5.0] nếu có |
| `pred_distilbert` | float | ⚠️ | ∈ [1.0, 5.0] nếu có |
| `pred_roberta` | float | ⚠️ | ∈ [1.0, 5.0] nếu có |
| `abs_err_*` | float | ⚠️ | = \|true_stars - pred_*\|, ≥ 0 |

#### 2.2.2 `model_summary.csv`
| Cột | Kiểu | Required | Validation |
| :--- | :--- | :--- | :--- |
| `model` | string | ✅ | ∈ {mean_predictor, baseline, bert, distilbert, roberta} |
| `mae`, `rmse`, `rmse_raw`, `r2`, `pearson`, `spearman`, `macro_mae` | float | ✅ (trừ model thiếu) | Hữu hạn; model thiếu → NaN/empty |
| `train_seconds` | float | ✅ | ≥ 0; mean_predictor → empty |
| `seconds_per_epoch` | float | ✅ | ≥ 0; mean_predictor, baseline → empty |
| `params_total` | int | ✅ | ≥ 0; baseline = Ridge coeffs |
| `rmse_reduction_vs_baseline_pct` | float | ✅ | mean_predictor, baseline → empty |
| `inference_seconds` | float | ✅ | ≥ 0; model thiếu → empty |
| `split_id` | string | ✅ | len=12, khớp `split_stats.json` |

#### 2.2.3 `<model>_test.csv`
| Cột | Kiểu | Required | Validation |
| :--- | :--- | :--- | :--- |
| `review_id` | string | ✅ | Unique, khớp `test_predictions.csv` |
| `true_stars` | int | ✅ | ∈ {1..5}, khớp `test_predictions.csv` |
| `pred_raw` | float | ✅ | Không clip (có thể <1 hoặc >5) |
| `pred` | float | ✅ | ∈ [1.0, 5.0] (đã clip) |
| `split_id` | string | ✅ | Khớp |

#### 2.2.4 `<model>_test_meta.json`
| Khoá | Kiểu | Validation |
| :--- | :--- | :--- |
| `model` | string | Khớp tên file |
| `run_name` | string | Tên lần chạy (configs) |
| `checkpoint_epoch` | int | ≥ 1 |
| `n` | int | = số dòng Test (9000) |
| `rmse_raw` | float | Khớp tính từ `pred_raw` trong CSV (±1e-4) |
| `inference_seconds` | float | > 0 |
| `fp16` | bool | True/False |
| `device` | string | "cpu" hoặc tên GPU (ví dụ "Tesla T4") |

### 2.3. Giao diện CLI — `scripts/validate_outputs.py`

```python
# scripts/validate_outputs.py

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate integration outputs")
    parser.add_argument("--results-dir", required=True, help="Thư mục chứa kết quả (reports/ hoặc results/)")
    parser.add_argument("--split-stats", help="Đường dẫn split_stats.json (mặc định: <results-dir>/../data/split_stats.json)")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Dung sai tuyệt đối cho metric reconciliation")
    parser.add_argument("--skip-dashboard", action="store_true", help="Bỏ qua check dashboard readiness")
    return parser.parse_args()

def load_all_data(results_dir: Path, split_stats_path: Path | None) -> dict:
    """Đọc tất cả file, trả về dict DataFrames."""
    ...

def check_merge_consistency(data: dict, split_stats: dict) -> list[str]:
    """Trả về list lỗi (rỗng = PASS)."""
    ...

def check_schemas(data: dict) -> list[str]:
    ...

def reconcile_metrics(data: dict, tolerance: float) -> list[str]:
    """Dùng src.metrics.compute_metrics, per_star_metrics."""
    ...

def check_dashboard_readiness(data: dict) -> list[str]:
    ...

def print_summary(checks: dict[str, list[str]]) -> bool:
    """In bảng PASS/FAIL, trả về True nếu all pass."""
    ...

def main() -> int:
    args = parse_args()
    data = load_all_data(Path(args.results_dir), Path(args.split_stats) if args.split_stats else None)
    split_stats = json.load(open(data["split_stats_path"])) if "split_stats_path" in data else {}

    all_errors = {}
    all_errors["merge"] = check_merge_consistency(data, split_stats)
    all_errors["schemas"] = check_schemas(data)
    all_errors["metrics"] = reconcile_metrics(data, args.tolerance)
    if not args.skip_dashboard:
        all_errors["dashboard"] = check_dashboard_readiness(data)

    ok = print_summary(all_errors)
    return 0 if ok else 1
```

### 2.4. Pytest — `tests/test_integration.py`

```python
# tests/test_integration.py

import pytest
from scripts.validate_outputs import (
    load_all_data, check_merge_consistency, check_schemas,
    reconcile_metrics, check_dashboard_readiness
)

@pytest.fixture(scope="session")
def results_dir():
    # Trỏ tới reports/ local hoặc mount Drive
    return Path("reports")  # hoặc /content/drive/MyDrive/project/results

@pytest.fixture(scope="session")
def split_stats_path(results_dir):
    return results_dir.parent / "data" / "split_stats.json"

@pytest.fixture(scope="session")
def validation_data(results_dir, split_stats_path):
    return load_all_data(results_dir, split_stats_path)

def test_merge_consistency(validation_data, split_stats_path):
    import json
    split_stats = json.load(open(split_stats_path))
    errors = check_merge_consistency(validation_data, split_stats)
    assert not errors, f"Merge consistency failed:\n" + "\n".join(errors)

def test_schemas(validation_data):
    errors = check_schemas(validation_data)
    assert not errors, f"Schema validation failed:\n" + "\n".join(errors)

def test_metric_reconciliation(validation_data):
    errors = reconcile_metrics(validation_data, tolerance=1e-4)
    assert not errors, f"Metric reconciliation failed:\n" + "\n".join(errors)

def test_dashboard_readiness(validation_data):
    errors = check_dashboard_readiness(validation_data)
    assert not errors, f"Dashboard readiness failed:\n" + "\n".join(errors)
```

### 2.5. Logic kiểm tra quan trọng

#### 2.5.1 Merge consistency
```python
def check_merge_consistency(data, split_stats):
    errors = []
    preds = data["test_predictions"]
    split_id = preds["split_id"].iloc[0]
    
    # 1. split_id khớp
    if split_id != split_stats.get("split_id"):
        errors.append(f"split_id mismatch: predictions={split_id}, split_stats={split_stats.get('split_id')}")
    
    # 2. review_id unique
    if preds["review_id"].duplicated().any():
        errors.append("Duplicate review_id in test_predictions.csv")
    
    # 3. true_stars hợp lệ
    invalid_stars = ~preds["true_stars"].isin([1,2,3,4,5])
    if invalid_stars.any():
        errors.append(f"Invalid true_stars values: {preds.loc[invalid_stars, 'true_stars'].unique()}")
    
    # 4. Nếu có with_text, review_id và true_stars phải khớp
    if "test_predictions_with_text" in data:
        with_text = data["test_predictions_with_text"]
        merged = preds[["review_id", "true_stars"]].merge(
            with_text[["review_id", "true_stars"]], on="review_id", suffixes=("", "_wt")
        )
        if not (merged["true_stars"] == merged["true_stars_wt"]).all():
            errors.append("true_stars mismatch between test_predictions.csv and _with_text version")
    
    return errors
```

#### 2.5.2 Metric reconciliation
```python
def reconcile_metrics(data, tolerance):
    from src.metrics import compute_metrics, per_star_metrics
    errors = []
    
    summary = data["model_summary"]
    preds = data["test_predictions"]
    
    for _, row in summary.iterrows():
        model = row["model"]
        if model == "mean_predictor":
            continue  # không có pred column
        
        pred_col = f"pred_{model}"
        if pred_col not in preds.columns:
            errors.append(f"Missing pred column {pred_col} for model {model}")
            continue
        
        y_true = preds["true_stars"].values
        y_pred = preds[pred_col].values
        
        # Global metrics
        recalc = compute_metrics(y_true, y_pred, clip=True)
        for metric in ["mae", "rmse", "r2", "pearson", "spearman"]:
            recorded = row[metric]
            if pd.isna(recorded):
                continue
            diff = abs(recalc[metric] - recorded)
            if diff > tolerance:
                errors.append(f"{model}.{metric}: recorded={recorded:.6f}, recalc={recalc[metric]:.6f}, diff={diff:.6f} > {tolerance}")
        
        # Per-star + macro_mae
        per_star = per_star_metrics(y_true, y_pred, clip=True)
        if pd.notna(row["macro_mae"]):
            diff = abs(per_star["macro_mae"] - row["macro_mae"])
            if diff > tolerance:
                errors.append(f"{model}.macro_mae: recorded={row['macro_mae']:.6f}, recalc={per_star['macro_mae']:.6f}, diff={diff:.6f}")
        
        # rmse_raw (từ pred_raw trong <model>_test.csv)
        if model in data["individual_preds"]:
            raw_preds = data["individual_preds"][model]["pred_raw"].values
            raw_rmse = np.sqrt(np.mean((raw_preds - y_true) ** 2))
            if pd.notna(row["rmse_raw"]):
                diff = abs(raw_rmse - row["rmse_raw"])
                if diff > tolerance:
                    errors.append(f"{model}.rmse_raw: recorded={row['rmse_raw']:.6f}, recalc={raw_rmse:.6f}, diff={diff:.6f}")
    
    return errors
```

### 2.6. Tích hợp với quy trình nhóm

| Bước | Ai | Công cụ |
| :--- | :--- | :--- |
| Sau Tuần 2, pull kết quả từ Drive về `reports/` | Leader / F | `git pull` + copy từ Drive |
| Chạy validation cục bộ | F | `python -m scripts.validate_outputs --results-dir reports/` |
| Commit `reports/` nếu PASS | Leader | `git add reports/ && git commit -m "Week 2 results"` |
| CI/CD (tuỳ chọn) | GitHub Actions | `pytest tests/test_integration.py` trên artifact |

### 2.7. Quyết định thiết kế và điểm mở

| Hạng mục | Quyết định |
| :--- | :--- |
| Hai công cụ: pytest + CLI độc lập | Đã chốt (Design §6, Table 6: thành viên F) |
| `validate_outputs.py` không import `torch` | Đã chốt (chạy máy không GPU) |
| Tolerance mặc định `1e-4` cho MAE/RMSE/R², `1e-3` cho Pearson/Spearman | Đề xuất; điều chỉnh ở lần chạy đầu |
| Kiểm tra `split_id` khớp `split_stats.json` | Bắt buộc (tránh trộn kết quả giữa các lần chia) |
| `rmse_raw` reconcile từ `<model>_test.csv` (`pred_raw`) | Đã chốt (Design §4: điểm thô không lưu trong `test_predictions.csv`) |
| Baseline validation tuỳ chọn (có thể thiếu nếu Tuần 1 chưa xong) | Đề xuất |
| Meta-test cho `validate_outputs.py` ở `06_integration_test_tests.md` | Sẽ viết (test script validation với dữ liệu故意 sai) |