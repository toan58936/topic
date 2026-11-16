# Đặc tả Thiết kế Kiểm thử Tích hợp — `tests/test_integration.py`, `scripts/validate_outputs.py`

> **Tài liệu:** `docs/define&designt/06_integration_test.md` (Define & Design cho end-to-end validation)
> **Module liên quan:** `tests/test_integration.py` (pytest), `scripts/validate_outputs.py` (CLI)
> **Kế hoạch kiểm thử:** `docs/tests/06_integration_test_tests.md` (tách riêng — meta-test cho script validation)
> **Phụ thuộc:** Tất cả spec 01–05; `docs/design.md` §4, §9.9; `docs/01_data.md` (split_id), `docs/02_evaluation.md` (model_summary.csv), `docs/04_engine_and_utils.md` (predict/merge schemas)
> **Trạng thái:** Đã triển khai một phần; chờ dữ liệu thật cho FR-8 và Leader review

---

## PHẦN I: DEFINE — Xác định Bài toán & Yêu cầu

### 1.1. Bối cảnh & Mục tiêu

Sau khi Tuần 2 kết thúc (chạy xong 3 backbone + baseline, xuất `test_predictions.csv` và `model_summary.csv` lên Drive), **thành viên C (Model + Integration Test)** chạy một quy trình xác thực tự động để đảm bảo:

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
| FR-3 | **Metric reconciliation**: Tính lại MAE, RMSE, R², Pearson, Spearman bằng `src.metrics.compute_metrics`; tính `macro_mae` bằng `per_star_metrics`. Bỏ qua model `mean_predictor`. Dung sai mặc định `1e-4`; Pearson/Spearman sử dụng `max(tolerance, 1e-3)`. |
| FR-4 | **Per-model file validation**: Kiểm tra các cột `review_id`, `true_stars`, `pred_raw`, `pred`, `split_id` trong từng `<model>_test.csv`; kiểm tra giá trị hợp lệ và tính nhất quán với `test_predictions.csv`. Kiểm tra metadata JSON, trong đó `rmse_raw` được tính lại với dung sai cố định `1e-4`. |
| FR-5 | **Prediction/Baseline validation**: Với mỗi cột `pred_*` hiện diện, kiểm tra giá trị hợp lệ trong khoảng [1,5]. Nếu có cột `abs_err_*` tương ứng thì đối chiếu với `|true_stars - pred_*|` bằng dung sai `1e-6`. |
| FR-6 | **Dashboard readiness**: Kiểm tra các cột cần thiết trong `model_summary.csv` và các cột dự đoán/sai số phục vụ dashboard trong `test_predictions.csv`. |
| FR-7 | **CLI `validate_outputs.py`**: Chạy `python -m scripts.validate_outputs --results-dir <path> [--split-stats <path>] [--tolerance <float>]`. Exit code 0 = PASS, 1 = FAIL. Có tùy chọn `--skip-dashboard`. |
| FR-8 | **Pytest `test_integration.py`**: Kiểm thử tích hợp bằng dữ liệu kết quả huấn luyện thật. Hiện chưa triển khai do chưa có đủ dữ liệu đầu ra thực tế từ Phase 5. |

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
    ├── parse_args() ──► results_dir, split_stats_path, tolerance, skip_dashboard
    │
    ├── load_all_data()
    │       ├── split_stats.json (từ data/ hoặc results/)
    │       ├── test_predictions.csv (và _with_text nếu có)
    │       ├── model_summary.csv
    │       └── preds/<model>_test.csv + _test_meta.json (mỗi model)
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
    └── main()
            ├── Thực hiện các bước validation
            ├── Bỏ qua check_dashboard_readiness() nếu --skip-dashboard
            ├── Gọi print_summary() để in PASS/FAIL
            └── Exit code 0 nếu tất cả PASS; ngược lại 1
```


**Ghi chú triển khai:**
- `load_all_data()` sử dụng đường dẫn `--split-stats` nếu được cung cấp.
- Nếu không chỉ định, chương trình tìm `split_stats.json` tại `results_dir.parent/data/`; nếu không có thì thử `results_dir/data/`.
- `load_all_data()` lưu nội dung đã đọc vào `data["split_stats"]` để `main()` sử dụng.
- `reconcile_metrics()` dùng tolerance mặc định `1e-4`; Pearson và Spearman dùng `max(tolerance, 1e-3)`.
- Kiểm tra `rmse_raw` trong metadata JSON sử dụng dung sai cố định `1e-4`.
- Tùy chọn `--skip-dashboard` cho phép bỏ qua bước dashboard readiness.

### 2.2. Schema chi tiết từng file (theo `docs/design.md` §9.9)

#### 2.2.1 `test_predictions.csv`

| Cột | Kiểu | Required | Validation thực tế |
| :--- | :--- | :--- | :--- |
| `review_id` | string | Có | Không rỗng, không trùng lặp |
| `true_stars` | int | Có | Thuộc {1,2,3,4,5} |
| `n_tokens` | numeric | Có | Lớn hơn 0 và không vượt quá 512 |
| `split_id` | string | Có | Có độ dài 12; tất cả dòng phải khớp `split_stats.json` |
| `pred_*` | float | Khi có | Giá trị nằm trong [1.0, 5.0], không chứa giá trị thiếu hoặc không hợp lệ |
| `abs_err_*` | float | Khi có cột tương ứng | Đối chiếu với `abs(true_stars - pred_*)`, dung sai tuyệt đối `1e-6` |

Ghi chú: `check_schemas()` kiểm tra các cột dự đoán đang có trong file. Cột `abs_err_*` chỉ được đối chiếu khi cột tương ứng tồn tại.

#### 2.2.2 `model_summary.csv`

| Cột | Kiểu | Required | Validation thực tế |
| :--- | :--- | :--- | :--- |
| `model` | string | Có | Thuộc {mean_predictor, baseline, bert, distilbert, roberta}; không trùng tên |
| `mae`, `rmse`, `rmse_raw`, `r2`, `pearson`, `spearman`, `macro_mae` | numeric | Có | Nếu có giá trị thì phải hữu hạn; có thể để trống khi model không có kết quả |
| `train_seconds`, `seconds_per_epoch`, `inference_seconds` | numeric | Có | Nếu có giá trị thì phải hữu hạn và không âm |
| `params_total` | numeric | Có | Nếu có giá trị thì phải hữu hạn và không âm |
| `rmse_reduction_vs_baseline_pct` | numeric | Có | Cột bắt buộc phải tồn tại; code hiện tại chưa kiểm tra miền giá trị riêng cho cột này |
| `split_id` | string | Có | Nếu có giá trị thì phải có độ dài 12; code hiện tại không đối chiếu trực tiếp cột này với `split_stats.json` |

Ghi chú: Các cột trong bảng là cột bắt buộc theo schema. Việc một ô có thể để trống phụ thuộc kết quả của model; code kiểm tra tính hữu hạn và miền giá trị đối với các trường được quy định trong `check_schemas()`.

#### 2.2.3 `<model>_test.csv`

| Cột | Kiểu | Required | Validation thực tế |
| :--- | :--- | :--- | :--- |
| `review_id` | string | Có | Không thiếu, không trùng; tập ID phải khớp `test_predictions.csv` |
| `true_stars` | int | Có | Thuộc {1,2,3,4,5}; phải khớp file tổng hợp |
| `pred_raw` | float | Có | Phải là giá trị số hữu hạn; không áp dụng giới hạn [1,5] |
| `pred` | float | Có | Phải là giá trị số hữu hạn trong [1.0,5.0] |
| `split_id` | string | Có | Không thiếu và phải khớp file tổng hợp |

Ngoài ra, các cột bắt buộc không được chứa giá trị thiếu. `check_schemas()` đối chiếu `review_id`, `true_stars` và `split_id` giữa file riêng của từng model với `test_predictions.csv`.

#### 2.2.4 `<model>_test_meta.json`

| Khóa | Kiểu | Required | Validation thực tế |
| :--- | :--- | :--- | :--- |
| `model` | string | Có | Phải khớp tên model |
| `run_name` | string | Có | Chuỗi không rỗng |
| `checkpoint_epoch` | int | Có | Số nguyên lớn hơn hoặc bằng 1 |
| `n` | int | Có | Bằng số dòng trong file `<model>_test.csv` |
| `rmse_raw` | float | Có | Hữu hạn và khớp RMSE tính lại từ `pred_raw`, dung sai cố định `1e-4` |
| `inference_seconds` | float | Có | Giá trị hữu hạn và lớn hơn 0 |
| `fp16` | bool | Có | Phải là boolean |
| `device` | string | Có | Chuỗi không rỗng |

Các khóa bắt buộc phải tồn tại. Nếu thiếu khóa hoặc metadata không khớp dữ liệu file tương ứng, validation sẽ báo lỗi.

### 2.3. Giao diện CLI — `scripts/validate_outputs.py`
**Ghi chú đối chiếu triển khai:**
- `load_all_data()` nạp split statistics, CSV tổng hợp, CSV dự đoán từng model và metadata JSON.
- `check_merge_consistency()` kiểm tra dữ liệu bắt buộc, split ID, review ID, điểm sao, token và file with_text nếu có.
- `check_schemas()` kiểm tra schema, miền giá trị, quan hệ giữa file tổng hợp và file từng model, cùng metadata.
- `reconcile_metrics()` tính lại metric từ `src.metrics`, bỏ qua `mean_predictor` và đối chiếu thêm `rmse_raw`.
- `check_dashboard_readiness()` kiểm tra các cột phục vụ dashboard; `main()` điều phối và `print_summary()` in kết quả.


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
**Trạng thái triển khai FR-8:**
- `tests/test_integration.py` hiện chưa tồn tại.
- Phần code mẫu trong mục này là thiết kế dự kiến, chưa phải implementation đã chạy.
- Việc triển khai FR-8 đang chờ dữ liệu kết quả huấn luyện thật từ Phase 5.
- Fixture dữ liệu giả của meta-test không thay thế kiểm thử tích hợp bằng kết quả huấn luyện thật.


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
**Đối chiếu với code hiện tại:**
- Kiểm tra các cột bắt buộc `review_id`, `true_stars`, `split_id`, `n_tokens`.
- Kiểm tra NaN, review ID trùng, điểm sao ngoài 1–5 và n_tokens không hợp lệ.
- Kiểm tra split ID của tất cả dòng với `split_stats.json`.
- Nếu có `test_predictions_with_text.csv`, dùng outer join để đối chiếu tập review ID và true stars.

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
**Đối chiếu với code hiện tại:**
- Bỏ qua model `mean_predictor` trong metric reconciliation.
- Tính lại MAE, RMSE, R², Pearson và Spearman bằng `compute_metrics(..., clip=True)`.
- Tính lại `macro_mae` bằng `per_star_metrics(..., clip=True)`.
- Tính RMSE từ `pred_raw` trong file `<model>_test.csv` và đối chiếu `rmse_raw` trong `model_summary.csv`.
- Dung sai mặc định là `1e-4`; Pearson và Spearman dùng `max(tolerance, 1e-3)`.
- `rmse_raw` trong metadata JSON được kiểm tra riêng với dung sai cố định `1e-4`.

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
**Cập nhật trạng thái kiểm thử:**
- Meta-test đã triển khai tại `tests/test_integration_validation.py`; lần chạy gần nhất có kết quả 16 passed.
- Gồm 1 test dữ liệu hợp lệ, 11 kịch bản lỗi C1–C11 và 4 unit test bổ sung.
- Kết quả này xác nhận validation script phát hiện lỗi trong fixture; không chứng minh FR-8 đã kiểm thử với dữ liệu huấn luyện thật.
- Spec 06 vẫn chờ dữ liệu thật cho FR-8 và Leader review.


| Hạng mục | Quyết định |
| :--- | :--- |
| Hai công cụ: pytest + CLI độc lập | Đã chốt (Design §6, Table 6: thành viên C) |
| `validate_outputs.py` không import `torch` | Đã chốt (chạy máy không GPU) |
| Tolerance mặc định `1e-4` cho MAE/RMSE/R², `1e-3` cho Pearson/Spearman | Đề xuất; điều chỉnh ở lần chạy đầu |
| Kiểm tra `split_id` khớp `split_stats.json` | Bắt buộc (tránh trộn kết quả giữa các lần chia) |
| `rmse_raw` reconcile từ `<model>_test.csv` (`pred_raw`) | Đã chốt (Design §4: điểm thô không lưu trong `test_predictions.csv`) |
| Baseline validation tuỳ chọn (có thể thiếu nếu Tuần 1 chưa xong) | Đề xuất |
| Meta-test cho `validate_outputs.py` ở `06_integration_test_tests.md` | Đã triển khai; 16 test PASS, gồm 1 test dữ liệu hợp lệ, 11 kịch bản lỗi C1–C11 và 4 unit test bổ sung. |
