# Kế hoạch kiểm thử Meta-test cho Validation — `tests/test_integration_validation.py`

> **Tài liệu:** `docs/06_integration_test_tests.md`
> **Đặc tả được kiểm thử:** `docs/06_integration_test.md` (script `scripts/validate_outputs.py` và pytest `tests/test_integration.py`)
> **Trạng thái:** Bản nháp trước khi hiện thực

---

## 1. Mục tiêu

Kiểm chứng **script validation tự bắt được lỗi** khi dữ liệu đầu vào sai. Đây là "test của test" — dùng dữ liệu giả cố ý hỏng để xác minh `validate_outputs.py` exit code = 1 và in ra lỗi đúng loại.

---

## 2. Nguyên tắc

1. **Tạo dữ liệu giả trong fixture**: `tmp_path` → ghi CSV/JSON hỏng theo các kịch bản.
2. **Gọi `main()` của script** (hoặc các hàm con) và kiểm tra `exit code` / danh sách lỗi.
3. **Mỗi test một loại lỗi**; không phụ thuộc kết quả training thật.
4. **Không cần `torch`**; chỉ dùng `pandas`, `numpy`, `src.metrics`.

---

## 3. Fixtures (`tests/conftest.py` bổ sung)

| Fixture | Nhiệm vụ |
| :--- | :--- |
| `valid_results_dir(tmp_path)` | Tạo bộ dữ liệu **đúng** đầy đủ: `test_predictions.csv`, `model_summary.csv`, `preds/<model>_test.csv`, `_test_meta.json`, `split_stats.json`. Trả về `Path`. |
| `corrupted_results_dir(tmp_path, valid_results_dir)` | Copy từ `valid_results_dir` rồi áp dụng một loại hỏng (parametrize). Trả về `(Path, corruption_type)`. |

---

## 4. Kịch bản hỏng (Corruption Scenarios)

| Mã | Loại hỏng | Cách làm hỏng | Kỳ vọng lỗi |
| :--- | :--- | :--- | :--- |
| C1 | `split_id_mismatch` | Đổi `split_id` trong `test_predictions.csv` khác `split_stats.json` | `check_merge_consistency`: "split_id mismatch" |
| C2 | `duplicate_review_id` | Lặp một dòng `review_id` trong `test_predictions.csv` | `check_merge_consistency`: "Duplicate review_id" |
| C3 | `invalid_true_stars` | Đặt `true_stars = 0` hoặc `6` | `check_merge_consistency`: "Invalid true_stars values" |
| C4 | `pred_out_of_range` | `pred_bert = 6.0` (không clip) | `check_schemas`: "pred_bert out of range [1,5]" |
| C5 | `abs_err_mismatch` | `abs_err_bert != |true_stars - pred_bert|` | `check_schemas`: "abs_err_bert mismatch" |
| C6 | `model_summary_metric_wrong` | Đổi `mae` của BERT +0.1 so với tính toán lại | `reconcile_metrics`: "bert.mae diff > tolerance" |
| C7 | `missing_pred_column` | Xóa cột `pred_distilbert` khỏi `test_predictions.csv` | `reconcile_metrics`: "Missing pred column pred_distilbert" |
| C8 | `rmse_raw_mismatch` | Đổi `rmse_raw` trong `model_summary.csv` khác tính từ `pred_raw` | `reconcile_metrics`: "bert.rmse_raw diff > tolerance" |
| C9 | `missing_split_id_column` | Xóa cột `split_id` khỏi `test_predictions.csv` | `check_schemas` hoặc `merge`: missing column |
| C10 | `individual_pred_raw_not_match` | `pred_raw` trong `bert_test.csv` khác `pred_raw` dùng tính `rmse_raw` | `reconcile_metrics`: `rmse_raw` mismatch |
| C11 | `dashboard_missing_column` | Xóa cột `seconds_per_epoch` khỏi `model_summary.csv` | `check_dashboard_readiness`: "Missing column seconds_per_epoch" |

---

## 5. Mô tả chi tiết test

### 5.1. Test fixture valid data PASS

```python
def test_valid_data_passes(valid_results_dir):
    from scripts.validate_outputs import main, parse_args
    import sys
    from unittest.mock import patch

    with patch.object(sys, 'argv', ['validate_outputs', '--results-dir', str(valid_results_dir)]):
        exit_code = main()
    assert exit_code == 0
```

### 5.2. Parametrize corruption tests

```python
@pytest.mark.parametrize("corruption_type,expected_error_substring", [
    ("split_id_mismatch", "split_id mismatch"),
    ("duplicate_review_id", "Duplicate review_id"),
    ("invalid_true_stars", "Invalid true_stars values"),
    ("pred_out_of_range", "out of range"),
    ("abs_err_mismatch", "abs_err.*mismatch"),
    ("model_summary_metric_wrong", "bert.mae.*diff"),
    ("missing_pred_column", "Missing pred column"),
    ("rmse_raw_mismatch", "rmse_raw.*diff"),
    ("missing_split_id_column", "split_id"),
    ("individual_pred_raw_not_match", "rmse_raw.*diff"),
    ("dashboard_missing_column", "seconds_per_epoch"),
])
def test_corruption_detected(corrupted_results_dir, corruption_type, expected_error_substring):
    from scripts.validate_outputs import main
    import sys
    from unittest.mock import patch

    results_dir, actual_corruption = corrupted_results_dir
    assert actual_corruption == corruption_type

    with patch.object(sys, 'argv', ['validate_outputs', '--results-dir', str(results_dir)]):
        exit_code = main()
    
    assert exit_code == 1, f"Corruption {corruption_type} should fail but exited 0"
    
    # Capture stderr/stdout để kiểm tra message lỗi (tuỳ chọn, dùng capsys fixture)
```

### 5.3. Test unit từng hàm kiểm tra (không qua CLI)

```python
def test_check_merge_consistency_split_id_mismatch(valid_results_dir):
    from scripts.validate_outputs import load_all_data, check_merge_consistency
    import json
    
    data = load_all_data(valid_results_dir, valid_results_dir.parent / "data" / "split_stats.json")
    # Hỏng split_id
    data["test_predictions"]["split_id"] = "different_split_id"
    split_stats = json.load(open(valid_results_dir.parent / "data" / "split_stats.json"))
    
    errors = check_merge_consistency(data, split_stats)
    assert any("split_id mismatch" in e for e in errors)

def test_reconcile_metrics_detects_wrong_mae(valid_results_dir):
    from scripts.validate_outputs import load_all_data, reconcile_metrics
    
    data = load_all_data(valid_results_dir, valid_results_dir.parent / "data" / "split_stats.json")
    # Hỏng MAE của BERT trong model_summary
    idx = data["model_summary"][data["model_summary"]["model"] == "bert"].index[0]
    data["model_summary"].loc[idx, "mae"] += 0.1
    
    errors = reconcile_metrics(data, tolerance=1e-4)
    assert any("bert.mae" in e and "diff" in e for e in errors)
```

### 5.4. Test schema validation từng file

```python
def test_check_schemas_pred_out_of_range(valid_results_dir):
    from scripts.validate_outputs import load_all_data, check_schemas
    
    data = load_all_data(valid_results_dir, valid_results_dir.parent / "data" / "split_stats.json")
    data["test_predictions"]["pred_bert"] = 6.0  # vượt 5.0
    
    errors = check_schemas(data)
    assert any("pred_bert" in e and "out of range" in e for e in errors)

def test_check_schemas_abs_err_mismatch(valid_results_dir):
    from scripts.validate_outputs import load_all_data, check_schemas
    
    data = load_all_data(valid_results_dir, valid_results_dir.parent / "data" / "split_stats.json")
    # abs_err_bert sai: true=4, pred=3.2 → abs_err nên là 0.8, set 9.9
    data["test_predictions"]["abs_err_bert"] = 9.9
    
    errors = check_schemas(data)
    assert any("abs_err_bert" in e and "mismatch" in e for e in errors)
```

---

## 6. Cách chạy

```bash
# Chạy meta-test validation
pytest tests/test_integration_validation.py -v

# Chạy kèm integration test thật (cần dữ liệu thật)
pytest tests/test_integration.py -v

# Chạy script validation trực tiếp trên reports/
python -m scripts.validate_outputs --results-dir reports/
```

---

## 7. Ánh xạ với FR và tiêu chí chấp nhận `06_integration_test.md`

| FR / Tiêu chí | Meta-test chứng minh |
| :--- | :--- |
| FR-1 Merge validation | C1, C2, C3, C9 |
| FR-2 Schema validation | C4, C5, C9 |
| FR-3 Metric reconciliation | C6, C7, C8, C10 |
| FR-6 Dashboard readiness | C11 |
| Tiêu chí 1: Valid data → exit 0 | `test_valid_data_passes` |
| Tiêu chí 2: Sai metric → FAIL | C6 |
| Tiêu chí 3: Thiếu split_id → FAIL | C9 |
| Tiêu chí 4: Pytest integration xanh | `test_integration.py` (yêu cầu dữ liệu thật) |
| Tiêu chí 5: Chạy < 30s | Không test thời gian ở meta-test; đo khi chạy thật |

---

## 8. Ghi chú triển khai fixture `corrupted_results_dir`

```python
# tests/conftest.py (bổ sung)

@pytest.fixture(params=[
    "split_id_mismatch",
    "duplicate_review_id", 
    "invalid_true_stars",
    "pred_out_of_range",
    "abs_err_mismatch",
    "model_summary_metric_wrong",
    "missing_pred_column",
    "rmse_raw_mismatch",
    "missing_split_id_column",
    "individual_pred_raw_not_match",
    "dashboard_missing_column",
])
def corrupted_results_dir(tmp_path, valid_results_dir, request):
    import shutil
    import pandas as pd
    
    corruption_type = request.param
    test_dir = tmp_path / f"corrupted_{corruption_type}"
    shutil.copytree(valid_results_dir, test_dir)
    
    # Áp dụng hỏng theo loại
    if corruption_type == "split_id_mismatch":
        df = pd.read_csv(test_dir / "test_predictions.csv")
        df["split_id"] = "wrong_split_id"
        df.to_csv(test_dir / "test_predictions.csv", index=False)
    elif corruption_type == "duplicate_review_id":
        df = pd.read_csv(test_dir / "test_predictions.csv")
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        df.to_csv(test_dir / "test_predictions.csv", index=False)
    elif corruption_type == "invalid_true_stars":
        df = pd.read_csv(test_dir / "test_predictions.csv")
        df.loc[0, "true_stars"] = 0
        df.to_csv(test_dir / "test_predictions.csv", index=False)
    elif corruption_type == "pred_out_of_range":
        df = pd.read_csv(test_dir / "test_predictions.csv")
        df.loc[0, "pred_bert"] = 6.0
        df.to_csv(test_dir / "test_predictions.csv", index=False)
    elif corruption_type == "abs_err_mismatch":
        df = pd.read_csv(test_dir / "test_predictions.csv")
        df.loc[0, "abs_err_bert"] = 9.9
        df.to_csv(test_dir / "test_predictions.csv", index=False)
    elif corruption_type == "model_summary_metric_wrong":
        df = pd.read_csv(test_dir / "model_summary.csv")
        idx = df[df["model"] == "bert"].index[0]
        df.loc[idx, "mae"] += 0.1
        df.to_csv(test_dir / "model_summary.csv", index=False)
    elif corruption_type == "missing_pred_column":
        df = pd.read_csv(test_dir / "test_predictions.csv")
        df = df.drop(columns=["pred_distilbert"])
        df.to_csv(test_dir / "test_predictions.csv", index=False)
    elif corruption_type == "rmse_raw_mismatch":
        df = pd.read_csv(test_dir / "model_summary.csv")
        idx = df[df["model"] == "bert"].index[0]
        df.loc[idx, "rmse_raw"] += 0.1
        df.to_csv(test_dir / "model_summary.csv", index=False)
    elif corruption_type == "missing_split_id_column":
        df = pd.read_csv(test_dir / "test_predictions.csv")
        df = df.drop(columns=["split_id"])
        df.to_csv(test_dir / "test_predictions.csv", index=False)
    elif corruption_type == "individual_pred_raw_not_match":
        df = pd.read_csv(test_dir / "preds" / "bert_test.csv")
        df.loc[0, "pred_raw"] += 10.0  # làm lệch rmse_raw
        df.to_csv(test_dir / "preds" / "bert_test.csv", index=False)
    elif corruption_type == "dashboard_missing_column":
        df = pd.read_csv(test_dir / "model_summary.csv")
        df = df.drop(columns=["seconds_per_epoch"])
        df.to_csv(test_dir / "model_summary.csv", index=False)
    
    yield test_dir, corruption_type
```

---

## 9. Giới hạn

* Meta-test chỉ kiểm tra script validation **phát hiện lỗi**; không kiểm tra logic sửa lỗi (do script chỉ validate, không sửa).
* Cần `valid_results_dir` fixture tạo bộ dữ liệu đúng đầy đủ — phần này có thể dùng lại từ `tests/fixtures/integration/` nếu có sẵn.
* Test `individual_pred_raw_not_match` giả định `preds/bert_test.csv` tồn tại; nếu Tuần 2 chỉ chạy một phần model thì skip tương ứng.