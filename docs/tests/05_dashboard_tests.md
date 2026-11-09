# Kế hoạch kiểm thử Dashboard — `tests/test_dashboard.py`

> **Tài liệu:** `docs/05_dashboard_tests.md`
> **Đặc tả được kiểm thử:** `docs/05_dashboard.md` (mục 1.5 Tiêu chí chấp nhận, mục 2.3 đến 2.6)
> **Trạng thái:** Bản nháp trước khi hiện thực

---

## 1. Mục tiêu và nguyên tắc

Bộ test kiểm chứng dashboard **không cần Streamlit server chạy thật**; test các hàm logic cốt lõi (`load_data`, `filter_predictions`, `get_model_columns`, `render_*` helpers) bằng cách gọi trực tiếp và kiểm tra giá trị trả về / side effect (ví dụ `st.dataframe` được gọi với DataFrame đúng).

1. **Không khởi động Streamlit**: dùng `streamlit.testing.v1.AppTest` (Streamlit ≥ 1.22) hoặc mock `streamlit` module.
2. **Dữ liệu giả**: CSV nhỏ (10-20 dòng) tạo trong `tests/fixtures/` hoặc fixture pytest.
3. **Test tích hợp nhẹ**: `AppTest.from_file("app/streamlit_dashboard.py").run()` kiểm tra không crash.
4. **Không test UI thẩm mỹ**; chỉ test dữ liệu hiển thị đúng cột, đúng lọc, đúng sắp xếp.

---

## 2. Fixtures (`tests/conftest.py` bổ sung)

| Fixture | Nhiệm vụ |
| :--- | :--- |
| `sample_summary_csv(tmp_path)` | Tạo `model_summary.csv` giả 5 dòng (mean_predictor, baseline, bert, distilbert, roberta) có đủ cột theo schema §9.9.3 Design. Trả về `Path`. |
| `sample_preds_csv(tmp_path)` | Tạo `test_predictions.csv` giả 20 dòng, 4 mô hình, có `abs_err_*`, `n_tokens`, `split_id`. Trả về `Path`. |
| `sample_preds_with_text_csv(tmp_path)` | Giống `sample_preds_csv` + cột `text`. Trả về `Path`. |
| `mock_reports_dir(tmp_path, sample_summary_csv, sample_preds_csv)` | Tạo thư mục `reports/` với hai file, thêm vào `sys.path` để import `src.metrics`. Trả về đường dẫn thư mục. |

---

## 3. Năm nhóm khía cạnh kiểm thử

| Nhóm | Khía cạnh | Test |
| :--- | :--- | :--- |
| A | Đọc dữ liệu & tự dò file | D1, D2 |
| B | Lọc & xác định mô hình | F1, F2, F3 |
| C | Tính per-star tại runtime | P1, P2 |
| D | Top-k worst cases & expander | W1, W2 |
| E | Smoke test Streamlit AppTest | S1 |

---

## 4. Mô tả chi tiết từng test

### Nhóm A: Đọc dữ liệu & tự dò file

| Mã | Hàm test | Cách kiểm thử | Kỳ vọng |
| :--- | :--- | :--- | :--- |
| D1 | `test_load_data_prefers_with_text` | Tạo cả `test_predictions.csv` và `test_predictions_with_text.csv` trong `reports/`. Gọi `load_data("reports")`. | Trả về `preds_df` có cột `text`; `has_text = True`. |
| D2 | `test_load_data_fallback_no_text` | Chỉ tạo `test_predictions.csv`. Gọi `load_data("reports")`. | Trả về `preds_df` **không** có cột `text`; `has_text = False`. |

### Nhóm B: Lọc & xác định mô hình

| Mã | Hàm test | Cách kiểm thử | Kỳ vọng |
| :--- | :--- | :--- | :--- |
| F1 | `test_get_model_columns` | `preds_df` có cột `pred_baseline`, `pred_bert`, `pred_distilbert`, `pred_roberta`. | `get_model_columns` trả về `["baseline", "bert", "distilbert", "roberta"]` (không có `mean_predictor`). |
| F2 | `test_filter_predictions_by_stars` | `preds_df` 20 dòng, `true_stars` phân bố đều. Lọc `stars=[1, 5]`. | Kết quả chỉ chứa dòng `true_stars` ∈ {1, 5}; shape đúng. |
| F3 | `test_filter_predictions_invalid_model` | Gọi `filter_predictions(df, "nonexistent", [1,2,3])`. | Ném `ValueError` rõ ràng. |

### Nhóm C: Tính per-star tại runtime

| Mã | Hàm test | Cách kiểm thử | Kỳ vọng |
| :--- | :--- | :--- | :--- |
| P1 | `test_render_per_star_metrics_matches_src_metrics` | Tạo `preds_df` 50 dòng, giá trị biết trước. Gọi `per_star_metrics` từ `src.metrics` và logic trong `render_per_star_metrics` (extract bảng). | Hai kết quả `per_star` và `macro_mae` khớp tuyệt đối (`atol=1e-6`). |
| P2 | `test_per_star_missing_star_handled` | `preds_df` thiếu sao 1. Gọi logic per-star. | Kết quả chỉ có sao 2-5; `macro_mae` là trung bình 4 mức sao có mặt. |

### Nhóm D: Top-k worst cases & expander

| Mã | Hàm test | Cách kiểm thử | Kỳ vọng |
| :--- | :--- | :--- | :--- |
| W1 | `test_render_worst_cases_top_k` | `preds_df` có `abs_err_bert` biết trước. Gọi logic `render_worst_cases(df, "bert", top_k=5)`. | Trả về DataFrame 5 dòng, sắp xếp `abs_err_bert` giảm dần, đúng `review_id` lớn nhất. |
| W2 | `test_worst_cases_includes_text_when_available` | `preds_df` có cột `text`. Gọi logic. | Kết quả có cột `text`; nội dung khớp `review_id`. |

### Nhóm E: Smoke test Streamlit AppTest

| Mã | Hàm test | Cách kiểm thử | Kỳ vọng |
| :--- | :--- | :--- | :--- |
| S1 | `test_app_runs_without_crash` | `AppTest.from_file("app/streamlit_dashboard.py").run(timeout=30)`. | Không exception; `app.exception` là `None`; các widget chính (radio model, multiselect stars) xuất hiện. |

---

## 5. Ví dụ đọc test chi tiết

### D1: Tự dò file ưu tiên `_with_text`

```python
def test_load_data_prefers_with_text(mock_reports_dir, sample_preds_with_text_csv, sample_summary_csv):
    from app.streamlit_dashboard import load_data

    summary_df, preds_df = load_data(mock_reports_dir)

    assert "text" in preds_df.columns
    assert len(preds_df) == len(pd.read_csv(sample_preds_with_text_csv))
```

### P1: Per-star khớp `src.metrics`

```python
def test_render_per_star_metrics_matches_src_metrics(mock_reports_dir, sample_preds_csv):
    from app.streamlit_dashboard import filter_predictions, get_model_columns
    from src.metrics import per_star_metrics
    import pandas as pd

    summary_df, preds_df = load_data(mock_reports_dir)
    model = get_model_columns(preds_df)[0]  # baseline
    filtered = filter_predictions(preds_df, model, [1, 2, 3, 4, 5])

    # Logic trong render_per_star_metrics
    y_true = filtered["true_stars"].values
    y_pred = filtered[f"pred_{model}"].values
    dash_result = per_star_metrics(y_true, y_pred, clip=True)

    # Gọi trực tiếp src.metrics
    direct_result = per_star_metrics(y_true, y_pred, clip=True)

    assert dash_result["macro_mae"] == pytest.approx(direct_result["macro_mae"])
    for star in dash_result["per_star"]:
        assert dash_result["per_star"][star]["mae"] == pytest.approx(direct_result["per_star"][star]["mae"])
        assert dash_result["per_star"][star]["rmse"] == pytest.approx(direct_result["per_star"][star]["rmse"])
```

### S1: AppTest smoke test

```python
def test_app_runs_without_crash(mock_reports_dir):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app/streamlit_dashboard.py").run(timeout=30)

    assert at.exception is None
    # Kiểm tra widget chính xuất hiện
    radio_models = [w for w in at.get("radio") if "model" in str(w.label).lower()]
    assert len(radio_models) >= 1
    multiselect_stars = at.get("multiselect")
    assert len(multiselect_stars) >= 1
```

---

## 6. Cách chạy

```bash
# Chạy tất cả test dashboard
pytest tests/test_dashboard.py -v

# Chạy kèm test khác (01-04)
pytest tests/ -v -m "not slow"

# Chạy smoke test Streamlit (cần streamlit ≥ 1.22)
pytest tests/test_dashboard.py::test_app_runs_without_crash -v
```

---

## 7. Ánh xạ tiêu chí chấp nhận (mục 1.5 `05_dashboard.md`)

| Tiêu chí | Test chứng minh |
| :--- | :--- |
| 1. Khởi động không lỗi trên máy không có PyTorch | S1 (AppTest), D1/D2 (import `src.metrics` không kéo `torch`) |
| 2. Bảng so sánh đúng số dòng/cột từ `model_summary.csv` | F1 (model columns), logic `render_comparison_table` |
| 3. Per-star MAE/RMSE/Macro-MAE khớp `src.metrics` | P1, P2 |
| 4. Bộ lọc sao và sắp xếp theo sai số | F2, W1 (sắp xếp `abs_err` giảm dần) |
| 5. Top-10 ca đoán sai đúng | W1 |
| 6. Có `with_text` thì hiển thị text, không thì chỉ `review_id` | D1, W2 |

---

## 8. Giới hạn

* Không test render chart (Altair/Streamlit native) chi tiết pixel; chỉ test dữ liệu truyền vào chart đúng.
* AppTest cần Streamlit ≥ 1.22; phiên bản cũ bỏ qua S1.
* Test không kiểm tra CSS/layout responsive.
* Fixture `mock_reports_dir` phải đảm bảo `sys.path` chứa repo root để `import src.metrics` hoạt động.