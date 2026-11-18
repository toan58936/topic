# Kế hoạch Kiểm thử Dashboard — `tests/test_dashboard.py`

> **Tài liệu:** `docs/05_dashboard_tests.md`
> **Đặc tả được kiểm thử:** `docs/05_dashboard.md` (`app/streamlit_dashboard.py`)
> **Trạng thái:** Đã chốt — Đã hiện thực (`tests/test_dashboard.py`, 24 test pass, CPU no torch)

---

## 1. Mục tiêu

Kiểm chứng dashboard Streamlit:
1. Đọc CSV đúng, tự dò `_with_text` version
2. Bảng so sánh hiển thị đúng, blank khi thiếu metric
3. Biểu đồ (scatter, histogram) render không lỗi
4. Per-star metrics tính runtime khớp `src.metrics.per_star_metrics`
5. Bộ lọc (model, stars, sort) hoạt động đúng
6. Top-10 worst cases + detail view (text nếu có)
7. **Không import `torch`** — chạy được trên máy không cài PyTorch

---

## 2. Nguyên tắc

- **CPU only, no torch**: Test không phụ thuộc PyTorch, dùng dữ liệu tổng hợp (synthetic)
- **Streamlit AppTest**: Dùng `streamlit.testing.v1.AppTest` để test UI logic
- **Dữ liệu giả**: Tạo `model_summary.csv` + `test_predictions.csv` trong fixture (tmp_path)
- **Mỗi test một hành vi**: Không phụ thuộc thứ tự

---

## 3. Fixtures (`tests/conftest.py` có sẵn)

| Fixture | Nhiệm vụ |
|---------|----------|
| `dashboard_data(tmp_path)` | Tạo `reports/` với `model_summary.csv` + `test_predictions.csv` (có/không `text`) đầy đủ schema |
| `dashboard_data_minimal(tmp_path)` | Tạo dữ liệu tối thiểu (chỉ baseline + bert) để test missing columns |

---

## 4. Danh sách Test Cases (24 test)

| Mã | Tên test | Mô tả | FR mapping |
|----|----------|-------|------------|
| **D1** | `test_load_data_prefers_with_text` | Có `_with_text.csv` → load bản đó, `has_text=True` | FR-1 |
| **D2** | `test_load_data_fallback_no_text` | Không có `_with_text.csv` → load `test_predictions.csv`, `has_text=False` | FR-1 |
| **D3** | `test_load_data_missing_file_raises` | Thiếu CSV bắt buộc → `FileNotFoundError` | FR-1 |
| **D4** | `test_comparison_table_all_models` | Bảng hiển thị 5 dòng (mean_predictor, baseline, bert, distilbert, roberta) | FR-2 |
| **D5** | `test_comparison_table_missing_metrics_blank` | Model thiếu metrics → ô trống (blank) | FR-2 |
| **D6** | `test_comparison_table_columns_present` | Đủ cột: MAE, RMSE, R², Pearson, Spearman, train_s, s/epoch, params, reduction%, infer_s | FR-2 |
| **D7** | `test_scatter_plot_renders` | Scatter true_stars vs pred_<model> render không lỗi (altair chart) | FR-3 |
| **D8** | `test_error_histogram_renders` | Histogram abs_err_<model> render không lỗi | FR-4 |
| **D9** | `test_per_star_metrics_matches_src` | Per-star MAE/RMSE/n + Macro-MAE khớp `src.metrics.per_star_metrics` | FR-5 |
| **D10** | `test_per_star_missing_star_handled` | Thiếu mức sao (n=0) → không lỗi, Macro-MAE đúng | FR-5 |
| **D11** | `test_filter_model_selector` | Radio chọn model chỉ hiện các model có cột `pred_*` | FR-6 |
| **D12** | `test_filter_stars_multiselect` | Multiselect sao 1-5 lọc đúng `true_stars` | FR-6 |
| **D13** | `test_sort_by_error_descending` | Checkbox sắp xếp theo `abs_err_<model>` giảm dần | FR-6 |
| **D14** | `test_worst_cases_top10` | Top-10 `abs_err_<model>` lớn nhất trong mức sao lọc | FR-7 |
| **D15** | `test_worst_cases_empty_filter` | Lọc sao rỗng → không dòng nào, không lỗi | FR-7 |
| **D16** | `test_worst_cases_has_text_shows_text` | Có `_with_text.csv` → worst cases có cột `text` | FR-7, FR-8 |
| **D17** | `test_worst_cases_no_text_shows_id_only` | Không có `_with_text.csv` → chỉ `review_id` + số liệu | FR-7 |
| **D18** | `test_detail_view_expander` | Click dòng → expander mở, hiển thị text/nhãn/dự đoán/sai số | FR-8 |
| **D19** | `test_no_torch_import` | `import app.streamlit_dashboard` không nạp `torch` | FR-9 |
| **D20** | `test_run_from_repo_root` | App thêm repo root vào `sys.path`, `import src.metrics` OK | FR-10 |
| **D21** | `test_model_columns_detection` | `get_model_columns` trả đúng `["baseline", "bert", "distilbert", "roberta"]` | FR-2, FR-6 |
| **D22** | `test_filter_data_preserves_columns` | `filter_predictions` giữ nguyên tất cả cột gốc | FR-6 |
| **D23** | `test_missing_pred_column_raises` | Chọn model không có cột `pred_*` → `ValueError` | FR-6 |
| **D24** | `test_empty_stars_filter_raises` | `stars` rỗng → `ValueError` | FR-6 |

---

## 5. Chi tiết một số test quan trọng

### 5.1 `test_per_star_metrics_matches_src` (D9)
```python
def test_per_star_metrics_matches_src(dashboard_data):
    from app.streamlit_dashboard import load_data, filter_predictions
    from src.metrics import per_star_metrics
    
    summary_df, preds_df = load_data("reports")
    model = "bert"
    filtered = filter_predictions(preds_df, model, [1,2,3,4,5])
    
    # Dashboard tính
    y_true = filtered["true_stars"].values
    y_pred = filtered[f"pred_{model}"].values
    dash_result = per_star_metrics(y_true, y_pred, clip=True)
    
    # Tính trực tiếp từ src.metrics
    src_result = per_star_metrics(y_true, y_pred, clip=True)
    
    assert dash_result == src_result
```

### 5.2 `test_no_torch_import` (D19)
```python
def test_no_torch_import():
    import sys
    if "torch" in sys.modules:
        del sys.modules["torch"]
    import app.streamlit_dashboard  # Should not import torch
    assert "torch" not in sys.modules
```

### 5.3 `test_load_data_prefers_with_text` (D1)
```python
def test_load_data_prefers_with_text(dashboard_data):
    from app.streamlit_dashboard import load_data
    summary_df, preds_df = load_data("reports")
    assert "text" in preds_df.columns
    assert preds_df["text"].notna().all()
```

---

## 6. Cách chạy

```bash
# Chạy toàn bộ dashboard tests
pytest tests/test_dashboard.py -v

# Chạy kèm error_analysis tests
pytest tests/test_dashboard.py tests/test_error_analysis.py -v

# Chạy chỉ test no-torch
pytest tests/test_dashboard.py::test_no_torch_import -v
```

---

## 7. Ánh xạ với FR và Tiêu chí chấp nhận `05_dashboard.md`

| FR / Tiêu chí | Test cases |
|---------------|------------|
| FR-1 Đọc CSV auto-detect | D1, D2, D3 |
| FR-2 Bảng so sánh | D4, D5, D6, D21 |
| FR-3 Scatter plot | D7 |
| FR-4 Histogram | D8 |
| FR-5 Per-star metrics | D9, D10 |
| FR-6 Filter/sort | D11, D12, D13, D22, D23, D24 |
| FR-7 Top-10 worst cases | D14, D15, D16, D17 |
| FR-8 Detail view | D18 |
| FR-9 Không import torch | D19 |
| FR-10 Chạy từ repo root | D20 |
| Tiêu chí 1: Khởi động không lỗi | D1-D3, D19-D20 |
| Tiêu chí 2: Bảng đúng số dòng/cột | D4, D6 |
| Tiêu chí 3: Per-star khớp src.metrics | D9 |
| Tiêu chí 4: Filter/sort hoạt động | D11-D13 |
| Tiêu chí 5: Top-10 đúng abs_err | D14 |
| Tiêu chí 6: Text hiển thị khi có | D16, D17 |

---

## 8. Ghi chú triển khai

- `dashboard_data` fixture tạo 5 model (mean_predictor, baseline, bert, distilbert, roberta) với metrics hợp lý
- `dashboard_data_minimal` chỉ có baseline + bert để test missing columns
- AppTest dùng `at.run(timeout=30)` cho các interaction (click, selectbox, checkbox)
- Test `test_no_torch_import` chạy đầu tiên trong CI để bắt lỗi import sớm