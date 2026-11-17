# Đặc tả Dashboard kết quả — `app/streamlit_dashboard.py`

> **Tài liệu:** `docs/define&designt/05_dashboard.md`
> **Kế hoạch kiểm thử:** [05_dashboard_tests.md](../tests/05_dashboard_tests.md)
> **Phạm vi E:** Dashboard và Error Analysis theo `plans/assignments.md` mục 7.
> **Tham chiếu:** `docs/define&designt/Tài liệu Define.md`, `docs/define&designt/Tài liệu Design.md`, `docs/define&designt/02_evaluation.md`, `docs/define&designt/04_engine_and_utils.md`.
> **Trạng thái:** Đã hiện thực; đồng bộ với code ngày 2026-11-17. Phân tích trên dữ liệu thật vẫn chờ các phụ thuộc ở mục 2.8.

## PHẦN I: DEFINE — Yêu cầu đã hiện thực

### 1.1. Bối cảnh và mục tiêu

Dashboard đọc CSV, không nạp mô hình hay trọng số, không cần GPU và không import `torch`. Chạy từ thư mục gốc repo:

```powershell
streamlit run app/streamlit_dashboard.py
```

Mục tiêu là so sánh kết quả Test, xem sai số theo mức sao và đọc các ca sai lớn nhất. Dashboard không nhận review mới để suy luận. Kiểm thử giao diện dùng dữ liệu tổng hợp; chưa có kết quả training thật trong `reports/` tại lần rà soát này.

### 1.2. Đầu vào và đầu ra

| Thành phần | Hành vi thực tế |
| :--- | :--- |
| Thư mục CSV | Mặc định `ROOT/reports`; đặt `YELP_REPORTS_DIR` hoặc nhập tại sidebar để thay đổi. |
| Bảng tổng hợp | Bắt buộc có `model_summary.csv`. |
| Dự đoán | Ưu tiên `test_predictions_with_text.csv` nếu tồn tại; nếu không, đọc `test_predictions.csv`. Không cần đồng thời cả hai bản. |
| Bản có text bị lỗi | Báo lỗi; không âm thầm chuyển sang bản không text. |
| Đầu ra Dashboard | Giao diện Streamlit; không ghi file. |
| Dữ liệu demo | Có `DEMO_ONLY.txt` thì hiện cảnh báo dữ liệu giả lập, không dùng làm kết quả báo cáo. |

### 1.3. Yêu cầu chức năng

| Mã | Hành vi đã hiện thực |
| :--- | :--- |
| FR-1 | Đọc CSV và tự dò bản `_with_text` theo mục 1.2. |
| FR-2 | Bảng so sánh giữ mọi dòng model trong summary. Hiện các cột thuộc `COMPARISON_COLUMNS` có ít nhất một giá trị; ô thiếu để trống, cột thiếu hoặc toàn null bị bỏ. Bảng luôn là kết quả toàn Test, không tính lại theo bộ lọc. |
| FR-3 | Scatter cho model đang chọn: X = `true_stars`, Y = `pred_<model>`, miền [1, 5], tooltip ID/nhãn/dự đoán, đường y=x bằng Vega-Lite. |
| FR-4 | Histogram sai số tuyệt đối bằng `numpy.histogram`: 20 bins trong [0, 4], hiển thị bằng `st.bar_chart`. Không có KDE. |
| FR-5 | Bảng và biểu đồ per-star MAE/RMSE/n, kèm Macro-MAE, tính trên dữ liệu đang lọc bằng `src.metrics.per_star_metrics`. |
| FR-6 | Radio chọn model có ít nhất một dự đoán; multiselect sao 1–5 mặc định chọn hết; checkbox sort sai số giảm dần mặc định bật. Review thiếu dự đoán của model đang chọn được bỏ qua kèm cảnh báo số lượng. |
| FR-7 | Top-10 theo `abs_err_<model>` trong dữ liệu đang lọc, luôn giảm dần dù checkbox sort toàn bộ bảng đã tắt; có text nếu đầu vào có. |
| FR-8 | Chọn một dòng Top-10 để mở expander chi tiết; khi chưa chọn dòng, dùng selectbox ID. Chi tiết có nhãn thật, dự đoán, sai số và toàn bộ text; thiếu text thì hiện hướng dẫn. |
| FR-9 | Dashboard dùng `streamlit`, `pandas`, `numpy`, `scipy` qua `src.metrics`; `app/requirements.txt` không có `torch`. |
| FR-10 | App tự thêm repo root vào `sys.path`, hỗ trợ lệnh chạy từ repo root. |

### 1.4. Ràng buộc và xử lý thiếu dữ liệu

- Thiếu `n_tokens` hoặc `split_id`: Dashboard vẫn hiển thị và cảnh báo, không tự tạo metadata. Error Analysis bắt buộc có cả hai.
- Có metadata thì kiểm tra giá trị; predictions phải thuộc một split, và các split không null trong summary phải khớp nếu cả hai file có cột này.
- Model để vẽ lấy từ `pred_*`, loại `pred_mean_predictor`; bảng summary vẫn giữ dòng `mean_predictor` nếu có.
- Chưa có model với dự đoán, không chọn sao hoặc không còn review sau lọc: hiện thông báo rồi dừng phần biểu đồ.
- Text thật chỉ dùng local/Drive; bản `_with_text` bị gitignore.
- Per-star/Macro-MAE tính lại theo bộ lọc; chỉ đối chiếu summary khi dùng cùng tập review và model.

### 1.5. Tiêu chí chấp nhận

1. AppTest chạy với/không text; subprocess import Dashboard không nạp `torch`.
2. Bảng so sánh giữ mọi model và xử lý đúng metrics thiếu.
3. Per-star MAE/RMSE/n và Macro-MAE khớp `src.metrics.per_star_metrics`; bỏ qua sao không có mẫu.
4. Lọc sao, đổi model, bật/tắt sort và lọc rỗng hoạt động đúng.
5. Top-k đúng ID/thứ tự/text; chọn dòng hoặc selectbox mở đúng chi tiết.
6. Thiếu metadata vẫn xem Dashboard kèm cảnh báo; Error Analysis từ chối chuẩn bị mẫu.

Tên test và giới hạn kiểm chứng nằm trong [kế hoạch kiểm thử](../tests/05_dashboard_tests.md). Test không chứng minh training, chất lượng mô hình thật hoặc hoàn tất gán nhãn người.

## PHẦN II: DESIGN — Theo code hiện tại

### 2.1. Luồng thực thi

`main()` cấu hình trang/sidebar thư mục → `load_data()` → cảnh báo metadata → bảng so sánh → radio model/multiselect sao/checkbox sort → `filter_predictions()` → bỏ dự đoán thiếu, sort nếu bật → scatter/histogram trong hai cột → per-star → Top-10/chi tiết → expander tất cả dự đoán đang lọc.

Không có hàm `sidebar_controls()` riêng, `st.tabs` hay `@st.cache_data`; CSV được đọc lại khi app chạy lại.

### 2.2. Schema đầu vào và validation

**`model_summary.csv`:**

| Cột | Cách sử dụng |
| :--- | :--- |
| `model` | Bắt buộc, không null, không trùng tên. Không ép đủ 5 dòng. |
| `mae`, `rmse`, `rmse_raw`, `r2`, `pearson`, `spearman`, `macro_mae` | Hiển thị khi có giá trị. Error Analysis cần `rmse` Test để chọn model hợp lệ tốt nhất. |
| `train_seconds`, `seconds_per_epoch`, `params_total`, `rmse_reduction_vs_baseline_pct`, `inference_seconds` | Hiển thị khi có giá trị; không tự ước lượng chi phí. |
| `split_id` | Đọc dạng chuỗi; đối chiếu predictions nếu cả hai file có cột này, không đưa vào bảng so sánh. |

**Hai bản predictions:**

| Cột | Kiểm tra của Dashboard |
| :--- | :--- |
| `review_id` | Bắt buộc, duy nhất, không null; đọc dạng chuỗi để giữ số 0 đầu ID. Bảng không được rỗng. |
| `true_stars` | Bắt buộc, giá trị nguyên 1–5. |
| `pred_<model>` | Giá trị không null phải trong [1, 5]; cho phép thiếu dự đoán một số dòng. |
| `abs_err_<model>` | Bắt buộc cho mỗi `pred_*`, kể cả `pred_mean_predictor` nếu có; cùng vị trí null và khớp sai số tuyệt đối (`atol=2e-6`, `rtol=0`). |
| `n_tokens` | Có thể thiếu ở Dashboard; nếu có phải hữu hạn, nguyên không âm. Hợp đồng dữ liệu: số token BERT trước cắt, ngưỡng >254 cho cờ cắt. |
| `split_id` | Có thể thiếu ở Dashboard; nếu có phải không null và duy nhất. Hợp đồng quy định 12 ký tự; Dashboard không kiểm tra độ dài. |
| `text` | Bắt buộc có cột khi đọc `_with_text`; Dashboard cho phép giá trị thiếu. Error Analysis cần nội dung không rỗng ở mọi review. |

Schema production vẫn cần `n_tokens` và `split_id` theo tài liệu Design mục 4/9.9.1; việc Dashboard cho phép thiếu chỉ hỗ trợ xem output chưa đầy đủ.

### 2.3. Giao diện hàm hiện có

| Hàm | Kết quả/hành vi |
| :--- | :--- |
| `get_model_columns(preds_df)` | Hậu tố model của `pred_*` theo thứ tự cột, loại `mean_predictor`. |
| `validate_predictions(df)` | Kiểm tra mục 2.2, không sửa DataFrame. |
| `load_data(reports_dir="reports")` | Trả `(summary_df, preds_df)`; không trả `has_text`. |
| `filter_predictions(preds_df, model, stars)` | Bản sao lọc sao, không tự sort/bỏ dự đoán null. |
| `comparison_table(summary_df)` | Bản sao các cột được hiển thị. |
| `compute_per_star(preds_df, model)` | `(table, macro_mae)`, bảng gồm `stars`, `mae`, `rmse`, `n`. |
| `get_worst_cases(preds_df, model, top_k=10)` | DataFrame Top-k, giữ text trực tiếp từ cùng dòng nếu có. |
| `render_comparison_table`, `render_scatter_plot`, `render_error_histogram`, `render_per_star_metrics`, `render_worst_cases` | Hiển thị bằng Streamlit, không trả DataFrame. |
| `main()` | Điều phối toàn bộ trang. |

### 2.4. Metrics và Top-k

`compute_per_star()` gọi `per_star_metrics(y_true, y_pred)` với clipping mặc định, không truyền keyword `clip`. Chỉ sao có mẫu xuất hiện trong bảng; Macro-MAE là trung bình MAE các sao đó.

`get_worst_cases()` dùng `nlargest(top_k, abs_err_<model>)`; kiểm tra `top_k >= 1`, có thể trả ít hơn k nếu không đủ mẫu. `render_worst_cases()` reset index trước khi dùng vị trí dòng từ `event.selection.rows`, còn selectbox là lựa chọn khi chưa chọn dòng.

### 2.5. Ngoại lệ

| Nơi xử lý | Hành vi |
| :--- | :--- |
| `load_data()` | Truyền lỗi filesystem/CSV ra ngoài; `ValueError` khi schema, giá trị, split hoặc cột text không hợp lệ. |
| `filter_predictions()` | `ValueError` khi không có cột model, sao rỗng hoặc ngoài 1–5. |
| `get_worst_cases()` | `ValueError` nếu `top_k < 1`. |
| `main()` | Bắt `OSError`, `ValueError`, `TypeError`, `ParserError` trong lúc load; hiển thị lỗi/hướng dẫn rồi trả về. |

### 2.6. Tích hợp

- B — `src.metrics`: chỉ import `per_star_metrics`; không sửa module B.
- F — `src.predict`: xuất hai bản predictions; E chỉ đọc CSV, không import module suy luận. Blocker ở mục 2.8.
- A — Data: không import trực tiếp; text gắn theo `review_id` của Test.
- Leader/C: Leader training/xuất kết quả, C validate output trước E hoàn tất phân tích thật.

### 2.7. Môi trường

`app/requirements.txt`: `streamlit>=1.50,<2`, `pandas>=2,<3`, `numpy>=1.24,<3`, `scipy>=1.10,<2`. Scatter dùng Vega-Lite qua Streamlit, không thêm dependency chart riêng.

Dashboard không import `torch`, nhưng `tests/conftest.py` hiện import `torch` ở cấp module nên pytest của repo cần dependency này khi collection. E không sửa fixture chung trong đợt này. Test import dùng subprocess riêng để kiểm tra Dashboard không nạp `torch`.

### 2.8. Error Analysis và blocker

`scripts/error_analysis.py` giữ hai lệnh hiện có:

```powershell
python -m scripts.error_analysis prepare --reports-dir reports
python -m scripts.error_analysis summarize --reports-dir reports
```

Đây là quy trình sau khi có dữ liệu thật đã validate; đợt này chỉ chạy workflow bằng fixture pytest trong thư mục tạm. `--output-dir` cho phép tách worksheet khỏi nguồn CSV.

1. `prepare` chọn model có RMSE Test hữu hạn, không âm thấp nhất; hòa điểm thì theo tên model. Model tốt nhất thiếu dự đoán thì dừng, không thay model khác. Lấy 20 worst và 20 random (4/sao) ngoài top-20, sort ID trước sampling, seed=42; thiếu quota thì dừng.
2. Xuất `error_sample.csv`, `error_sample_with_text.csv`, `error_analysis.csv`, `leader_crosscheck_with_text.csv`, `error_analysis_manifest.json`. Nhãn/notes để trống; worksheet Leader có 4 review cố định và không chứa nhãn E. Không ghi đè file chuẩn bị đã tồn tại.
3. E đọc text, gán một trong `Nhãn nhiễu`, `Cảm xúc lẫn lộn`, `Bị cắt 256 token`, `Mỉa mai`, `Khác`, kèm notes có bằng chứng. Không sửa dữ liệu gốc; nhóm cắt chỉ hợp lệ khi `n_tokens > 254`. Cờ token không tự chứng minh cắt gây lỗi.
4. Leader gán độc lập đúng 4 review; `summarize` kiểm tra manifest, đủ 40 nhãn/notes và agreement ít nhất 3/4 (75%) trước khi ghi tổng hợp.
5. Script hiện xuất `error_group_summary.csv`, `error_analysis_summary.md` trong output-dir. Chưa tạo báo cáo thật trong repo ở đợt này; không thêm tài liệu Markdown mới hay khôi phục guide đã bị main xóa.

**Blocker F:** `src/predict.py::merge_test_predictions()` hiện khởi tạo bảng bằng `review_id`/`true_stars`, thêm `pred_*`/`abs_err_*` rồi text; không tính/đưa `n_tokens` và `split_id` vào hai CSV cuối. `export_test_predictions()` cũng chưa đưa hai cột này vào CSV từng model. Error Analysis sẽ dừng bằng `ValueError` khi dùng output này. E không điền metadata giả, bỏ kiểm tra hoặc sửa code F.

**Phối hợp:** F bổ sung số token BERT trước cắt theo quy ước ngưỡng 254 (không tính hai token đặc biệt), truyền/kiểm tra `split_id` đúng lần chia và xuất đủ metadata ở cả hai bản CSV. Leader cung cấp kết quả training thật; C xác nhận schema/ID/nhãn/split. Sau đó E mới `prepare`, gán nhãn người, nhận Leader cross-check và `summarize`. Theo dõi tại [status_thanh_vien_e.md](../analysts/status_thanh_vien_e.md).
