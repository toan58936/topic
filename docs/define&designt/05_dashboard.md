# Đặc tả Thiết kế Dashboard Kết quả — `app/streamlit_dashboard.py`

> **Tài liệu:** `docs/05_dashboard.md` (Define & Design cho dashboard Streamlit)
> **Module liên quan:** `app/streamlit_dashboard.py`
> **Kế hoạch kiểm thử:** `docs/05_dashboard_tests.md` (tách riêng)
> **Phụ thuộc:** `docs/define.md`, `docs/design.md`; `docs/02_evaluation.md` (`metrics.py` để tính lại chỉ số theo bộ lọc); `docs/04_engine_and_utils.md` (schema `test_predictions.csv` và `model_summary.csv`)
> **Trạng thái:** Bản nháp trước khi hiện thực (Pre-implementation Design Specification)

---

## PHẦN I: DEFINE — Xác định Bài toán & Yêu cầu

### 1.1. Bối cảnh & Mục tiêu

Dashboard là giao diện trình bày kết quả cuối cùng cho buổi báo cáo. **Yêu cầu cốt lõi: không cần nạp mô hình, không cần GPU, chỉ đọc hai file CSV nhẹ đã commit vào `reports/`**. Chạy trên máy cá nhân của bất kỳ thành viên nào bằng `streamlit run app/streamlit_dashboard.py`.

Mục tiêu:
- So sánh mốc trung bình, baseline và ba mô hình Transformer trên các chỉ số chính
- Trực quan hóa dự đoán so với nhãn thật, phân phối sai số
- Phân tích theo từng mức sao (MAE/RMSE per-star + Macro-MAE)
- Xem chi tiết ca đoán sai điển hình (Top-10 sai số lớn nhất)

### 1.2. Đầu vào / Đầu ra

| | Mô tả |
| :--- | :--- |
| **Đầu vào bắt buộc** | `reports/model_summary.csv`, `reports/test_predictions.csv` (không có cột `text`) |
| **Đầu vào tuỳ chọn** | `reports/test_predictions_with_text.csv` (có cột `text`, từ Drive, bị `.gitignore`) |
| **Đầu ra** | Giao diện web Streamlit; không ghi file |

### 1.3. Yêu cầu chức năng

| Mã | Yêu cầu |
| :--- | :--- |
| FR-1 | Đọc `model_summary.csv` và `test_predictions.csv` từ `reports/`; tự dò: nếu có `test_predictions_with_text.csv` thì dùng bản đầy đủ, không thì dùng bản không text. |
| FR-2 | Bảng so sánh: hiển thị tất cả mô hình có mặt trong `model_summary.csv` (cột `model`) cùng các chỉ số: MAE, RMSE, R², Pearson, Spearman, `train_seconds`, `seconds_per_epoch`, `params_total`, `rmse_reduction_vs_baseline_pct`. Mô hình thiếu chỉ số để trống ô. |
| FR-3 | Biểu đồ 1: Scatter điểm dự đoán vs điểm thật (trục X: `true_stars`, trục Y: `pred_<model>`), một đồ thị cho mỗi mô hình được chọn. |
| FR-4 | Biểu đồ 2: Histogram/KDE phân phối sai số tuyệt đối (`abs_err_<model>`) cho mô hình đang chọn. |
| FR-5 | Bảng per-star: MAE, RMSE, số mẫu `n` cho từng mức sao 1-5 (tính từ `test_predictions.csv` dùng `src.metrics.per_star_metrics`), kèm Macro-MAE. |
| FR-6 | Bộ lọc: chọn mô hình (radio), lọc theo mức sao (multiselect 1-5), sắp xếp theo sai số của mô hình đang chọn (giảm dần). |
| FR-7 | Bảng "Ca đoán sai điển hình": Top-10 `review_id` có `abs_err_<model>` lớn nhất trong mức sao đang lọc (hoặc toàn bộ nếu không lọc). Hiển thị: `review_id`, `true_stars`, `pred_<model>`, `abs_err_<model>`, và `text` nếu có bản `with_text`. |
| FR-8 | Xem chi tiết một review: click vào dòng trong bảng ca đoán sai → mở expander hiển thị toàn bộ `text` (nếu có), nhãn thật, dự đoán, sai số. |
| FR-9 | Dashboard **không import `torch`**; chỉ dùng `streamlit`, `pandas`, `numpy`, `scipy`, `src.metrics`. `app/requirements.txt` không chứa `torch`. |
| FR-10 | Chạy được từ thư mục gốc repo: `streamlit run app/streamlit_dashboard.py` (app tự thêm thư mục gốc vào `sys.path` để `import src.metrics`). |

### 1.4. Ràng buộc

* `src/__init__.py` để trống (theo `docs/02_evaluation.md`), nên `import src.metrics` hoạt động khi không có `torch`.
* Dashboard **không** cho phép nhập review mới dự đoán (cần nạp trọng số, ngoài phạm vi).
* `test_predictions.csv` không có cột `text` (đã commit GitHub); bản có `text` chỉ nằm local/Drive.
* Mọi chỉ số per-star và Macro-MAE tính **tại thời điểm chạy dashboard** từ `test_predictions.csv` bằng `src.metrics.per_star_metrics` để đảm bảo khớp với `model_summary.csv`.

### 1.5. Tiêu chí chấp nhận

1. `streamlit run app/streamlit_dashboard.py` khởi động không lỗi trên máy không cài PyTorch.
2. Bảng so sánh hiển thị đúng số dòng và cột từ `model_summary.csv`.
3. Per-star MAE/RMSE/Macro-MAE khớp với `src.metrics.per_star_metrics` trên cùng dữ liệu.
4. Bộ lọc sao và sắp xếp theo sai số hoạt động đúng.
5. Top-10 ca đoán sai đúng theo `abs_err_<model>` trong mức sao đang lọc.
6. Nếu có `test_predictions_with_text.csv`, nội dung review hiển thị; nếu không, chỉ hiển thị `review_id` + số liệu.

---

## PHẦN II: DESIGN — Thiết kế Chi tiết

### 2.1. Kiến trúc

```text
app/streamlit_dashboard.py
    │
    ├── load_data() ──► reports/model_summary.csv, reports/test_predictions.csv (hoặc _with_text)
    │
    ├── sidebar_controls()
    │       ├── select_model: radio ["baseline", "bert", "distilbert", "roberta"] (chỉ các mô hình có cột pred_*)
    │       ├── filter_stars: multiselect [1,2,3,4,5] (mặc định chọn hết)
    │       └── sort_by_error: checkbox "Sắp xếp theo sai số giảm dần"
    │
    ├── filter_data(df, model, stars) ──► DataFrame đã lọc
    │
    ├── compute_per_star(filtered_df, model) ──► src.metrics.per_star_metrics
    │
    ├── render_comparison_table(model_summary_df)
    │
    ├── render_scatter(filtered_df, model)
    │
    ├── render_error_distribution(filtered_df, model)
    │
    ├── render_per_star_table(per_star_dict)
    │
    └── render_worst_cases(filtered_df, model) ──► Top-10 abs_err, expander chi tiết
```

### 2.2. Schema dữ liệu đầu vào

**`reports/model_summary.csv`** (theo `docs/design.md` §9.9.3):
| Cột | Kiểu | Ghi chú |
| :--- | :--- | :--- |
| `model` | string | `mean_predictor`, `baseline`, `bert`, `distilbert`, `roberta` |
| `mae`, `rmse`, `rmse_raw`, `r2`, `pearson`, `spearman`, `macro_mae` | float | Trống nếu model thiếu |
| `train_seconds`, `seconds_per_epoch` | float | Trống cho mean_predictor, baseline |
| `params_total` | int | Baseline = số hệ số Ridge |
| `rmse_reduction_vs_baseline_pct` | float | Trống cho mean_predictor, baseline |
| `inference_seconds` | float | Trống nếu model thiếu |
| `split_id` | string | 12 chars |

**`reports/test_predictions.csv`** (theo `docs/design.md` §4):
| Cột | Kiểu | Ghi chú |
| :--- | :--- | :--- |
| `review_id` | string | Primary key |
| `true_stars` | int | 1-5 |
| `n_tokens` | int | Tokenizer BERT, không cắt |
| `split_id` | string | Validation merge |
| `pred_baseline`, `pred_bert`, `pred_distilbert`, `pred_roberta` | float | Chỉ các mô hình có kết quả; đã clip 1.0-5.0 |
| `abs_err_baseline`, `abs_err_bert`, `abs_err_distilbert`, `abs_err_roberta` | float | Tương ứng |

**`reports/test_predictions_with_text.csv`**: Các cột trên + `text` (sau `review_id`).

### 2.3. Giao diện hàm chính

```python
# app/streamlit_dashboard.py

def load_data(reports_dir: str = "reports") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Đọc model_summary.csv và test_predictions.csv (ưu tiên _with_text nếu có).
    Trả về (summary_df, preds_df).
    """

def filter_predictions(preds_df: pd.DataFrame, model: str, stars: list[int]) -> pd.DataFrame:
    """
    Lọc preds_df theo true_stars trong stars; trả về bản sao.
    """

def get_model_columns(preds_df: pd.DataFrame) -> list[str]:
    """
    Trả về danh sách các mô hình có cột pred_* trong preds_df (bỏ qua mean_predictor).
    """

def render_comparison_table(summary_df: pd.DataFrame) -> None:
    """
    Hiển thị bảng so sánh Streamlit (st.dataframe).
    Chỉ hiện các cột có ít nhất một giá trị không null.
    """

def render_scatter_plot(preds_df: pd.DataFrame, model: str) -> None:
    """
    Scatter true_stars vs pred_<model> (st.altair_chart hoặc st.scatter_chart).
    """

def render_error_histogram(preds_df: pd.DataFrame, model: str) -> None:
    """
    Histogram abs_err_<model> (st.bar_chart hoặc altair).
    """

def render_per_star_metrics(preds_df: pd.DataFrame, model: str) -> None:
    """
    Gọi src.metrics.per_star_metrics(y_true, y_pred) và hiển thị bảng.
    """

def render_worst_cases(preds_df: pd.DataFrame, model: str, top_k: int = 10) -> None:
    """
    Top-k abs_err_<model> lớn nhất; expander cho mỗi dòng hiển thị text (nếu có).
    """

def main() -> None:
    """
    Entry point: st.set_page_config, load_data, sidebar, render all sections.
    """
```

### 2.4. Logic xử lý chi tiết

**Tự dò file input:**
```python
preds_path = Path("reports/test_predictions.csv")
preds_with_text_path = Path("reports/test_predictions_with_text.csv")
if preds_with_text_path.exists():
    preds_df = pd.read_csv(preds_with_text_path)
    has_text = True
else:
    preds_df = pd.read_csv(preds_path)
    has_text = False
```

**Xác định mô hình có mặt:**
```python
model_cols = [c for c in preds_df.columns if c.startswith("pred_") and c != "pred_mean_predictor"]
models = [c.replace("pred_", "") for c in model_cols]  # ["baseline", "bert", ...]
```

**Per-star metrics (dùng src.metrics):**
```python
from src.metrics import per_star_metrics
y_true = filtered_df["true_stars"].values
y_pred = filtered_df[f"pred_{model}"].values
result = per_star_metrics(y_true, y_pred, clip=True)  # {"per_star": {...}, "macro_mae": float}
```

**Top-k worst cases:**
```python
err_col = f"abs_err_{model}"
worst = filtered_df.nlargest(top_k, err_col)[["review_id", "true_stars", f"pred_{model}", err_col]]
if has_text:
    worst = worst.merge(preds_df[["review_id", "text"]], on="review_id")
```

### 2.5. Ngoại lệ

| Hàm | Ngoại lệ |
| :--- | :--- |
| `load_data` | `FileNotFoundError` nếu thiếu `model_summary.csv` hoặc `test_predictions.csv` |
| `filter_predictions` | `ValueError` nếu `model` không có cột `pred_<model>` hoặc `stars` rỗng |
| `render_*` | `KeyError` nếu cột kỳ vọng không có trong DataFrame (bug dữ liệu) |

### 2.6. Tích hợp với các module khác

| Module | Cách phối hợp |
| :--- | :--- |
| `metrics.py` (`02`) | Gọi `per_star_metrics` để tính lại MAE/RMSE per-star + Macro-MAE tại runtime (đảm bảo khớp `model_summary.csv`). Không gọi `compute_metrics` hay `build_model_summary`. |
| `data_setup.py` (`01`) | Không import trực tiếp; `text` trong `with_text` version lấy từ `test.parquet` qua `merge_test_predictions` của `04`. |
| `engine.py`, `model_builder.py` | Không import (dashboard không cài `torch`). |

### 2.7. Quyết định thiết kế và điểm mở

| Hạng mục | Quyết định |
| :--- | :--- |
| Công cụ: Streamlit | Đã chốt (Design §9.1) |
| Đọc CSV từ `reports/` (GitHub) + tự dò `_with_text` (Drive/local) | Đã chốt (Design §9.2, §5) |
| Không nhập review mới dự đoán | Ngoài phạm vi (Design §5) |
| Tính per-star tại runtime bằng `src.metrics` | Đã chốt (đảm bảo số khớp báo cáo) |
| `app/requirements.txt`: streamlit, pandas, numpy, scipy; **không torch** | Đã chốt (Design §5, §8) |
| Chạy từ repo root: `streamlit run app/streamlit_dashboard.py` | Đã chốt (Design §5) |
| Biểu đồ: Altair hoặc Streamlit native | Đề xuất; Altair cho tương tác tốt hơn |
| Caching: `@st.cache_data` cho `load_data` | Đề xuất, tránh đọc lại CSV khi tương tác |
| Layout: `st.tabs` hoặc `st.expander` cho các section | Đề xuất |