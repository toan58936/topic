# Kế hoạch kiểm thử Dashboard và Error Analysis

> **Tài liệu:** `docs/tests/05_dashboard_tests.md` — file đã tồn tại, cập nhật tại chỗ.
> **Đặc tả:** [05_dashboard.md](../define&designt/05_dashboard.md)
> **Code:** `app/streamlit_dashboard.py`, `scripts/error_analysis.py`
> **Test:** `tests/test_dashboard.py`, `tests/test_error_analysis.py`
> **Trạng thái:** Đã hiện thực; đồng bộ tên test, fixture và hành vi ngày 2026-11-17. Kết quả mới nhất ghi trong [trạng thái E](../analysts/status_thanh_vien_e.md) và `docs/ai_log.md`.

## 1. Nguyên tắc và môi trường

- Test logic gọi trực tiếp helper; render dùng `MagicMock` để kiểm tra DataFrame/chart spec/chi tiết truyền cho Streamlit.
- `streamlit.testing.v1.AppTest` chạy app và tương tác widget, không mở server/trình duyệt. Kiểm tra không lỗi bằng `not at.exception`, không so sánh `at.exception is None`.
- Dùng Streamlit theo `app/requirements.txt` (`>=1.50,<2`), không tự skip test trên phiên bản cũ.
- `tests/conftest.py` hiện import `torch` khi collection; pytest cần torch, dù Dashboard không import thư viện này. Test subprocess kiểm tra riêng import Dashboard không nạp torch; không khẳng định cả suite chạy trong môi trường không cài torch.
- Chỉ dùng dữ liệu tổng hợp trong `tmp_path`. Không training/inference thật, không tạo kết quả nghiên cứu, nhãn E hay cross-check Leader thật.
- Summary tiếng Việt được ghi/đọc rõ `encoding='utf-8'`, không phụ thuộc encoding mặc định của Windows.
- Không kiểm tra pixel/CSS/responsive, GPU, chất lượng mô hình hoặc output production của F.

## 2. Fixture thực tế

Fixture nằm ngay trong từng file test; không thêm fixture vào `tests/conftest.py`.

| Fixture/helper | Vị trí và dữ liệu |
| :--- | :--- |
| `reports(tmp_path)` | Cả hai file gọi `scripts.create_dashboard_demo.create_demo(tmp_path / 'demo')`: 5 dòng summary, 200 review, 4 model dự đoán và sai số, `n_tokens`, `split_id`, hai bản CSV có/không text, marker `DEMO_ONLY.txt`. |
| `preds(reports)` | `tests/test_dashboard.py`: predictions từ `load_data(reports)`. |
| `sample(reports)` | `tests/test_error_analysis.py`: `sample_reviews(*load_data(reports))`, 40 review chưa gán nhãn. |
| `labeled(sample)` | Helper test Error Analysis, tạo nhãn `Khác` và notes ghi rõ synthetic để kiểm tra validation/tổng hợp; không phải gán nhãn người. |

Không có các fixture `sample_summary_csv`, `sample_preds_csv`, `sample_preds_with_text_csv`, `mock_reports_dir` của bản nháp cũ. AppTest được trỏ tới fixture bằng `YELP_REPORTS_DIR` qua `monkeypatch`, không đọc `reports/` thật.

## 3. Coverage Dashboard — `tests/test_dashboard.py`

| Test thực tế | Nội dung và kỳ vọng |
| :--- | :--- |
| `test_prefers_text_and_fallback` | `load_data` trả `(summary, preds)`, ưu tiên bản có text; xóa bản này thì đọc CSV không text. Không trả `has_text`. |
| `test_missing_files` | Thiếu file bắt buộc → `FileNotFoundError`. |
| `test_model_columns` | Tìm model theo `pred_*`, loại `mean_predictor`. |
| `test_filter_copy` | Lọc sao 1/5 đúng số lượng, sửa kết quả không đổi đầu vào. |
| `test_invalid_filter` (3 case) | Model không tồn tại, sao rỗng, sao ngoài 1–5 → `ValueError`. |
| `test_render_per_star_matches_source` | Bảng truyền cho `st.dataframe` khớp MAE/RMSE/n từ `src.metrics.per_star_metrics` (`abs=1e-6`); Macro-MAE hiển thị 4 chữ số. |
| `test_missing_star` | Thiếu sao 1 thì bảng có sao 2–5; Macro-MAE là trung bình MAE các sao hiện có. |
| `test_top_k_text_alignment` | `get_worst_cases` trên sao 1/5: Top-5 giảm dần, đúng ID/text; ít hơn k thì trả số dòng hiện có. `render_worst_cases` không trả DataFrame. |
| `test_comparison_missing_metrics` | Giữ 5 dòng; ô thiếu là null, giữ cột có ít nhất một giá trị và bỏ cột toàn null. |
| `test_chart_data` | Tổng số review histogram khớp đầu vào; scatter có đường y=x từ (1,1) đến (5,5). |
| `test_bad_values` (4 case) | Sao ngoài miền, dự đoán ngoài [1,5], sai số không khớp, token âm → `ValueError`. |
| `test_duplicate_and_split_mismatch` | Chặn ID trùng và split summary khác predictions. |
| `test_numeric_id_preserved` | ID `00001` giữ số 0 đầu sau khi đọc CSV. |
| `test_app_runs_without_crash` (2 case) | AppTest có/không text: widget chính, đổi model, lọc sao, Top-10 giảm dần, selectbox mở đúng chi tiết, tắt sort, chọn sao rỗng. |
| `test_missing_input_ui` | Thiếu file hiện lỗi/hướng dẫn, không exception. |
| `test_no_torch_import` | Subprocess import Dashboard; `torch` không có trong `sys.modules`. |
| `test_legacy_producer_metadata_is_optional` (3 case) | Thiếu `n_tokens`, `split_id`, cả hai: loader vẫn đọc đủ; AppTest cảnh báo nêu các cột thiếu và yêu cầu Error Analysis, vẫn có radio, bảng 5 model và Top-10. |
| `test_selected_table_row_opens_detail` | Mock selection dòng thứ ba, mở đúng text/ID, không gọi selectbox. AppTest ở trên kiểm tra selectbox; click dòng được kiểm tra riêng ở đây. |

## 4. Coverage Error Analysis — `tests/test_error_analysis.py`

| Test thực tế | Nội dung và kỳ vọng |
| :--- | :--- |
| `test_counts_strata_and_worst` | Đúng 40 ID duy nhất, model RMSE Test tốt nhất trong fixture, 20 worst + 4 random/sao, nhãn rỗng. |
| `test_reproducible_input_order` | Shuffle summary/predictions không đổi mẫu hoặc 4 ID Leader, seed=42. |
| `test_need_text` | Thiếu text → lỗi yêu cầu nội dung để gán nhãn. |
| `test_prepare_requires_producer_metadata` (3 case) | Thiếu riêng `n_tokens`, riêng `split_id`, cả hai: `prepare` báo lỗi; không tạo output-dir/worksheet. Bảo vệ hợp đồng metadata, không sửa producer của F. |
| `test_insufficient_strata` | Thiếu quota một sao ngoài top-20 → lỗi, không lấy mẫu thay thế. |
| `test_missing_best_not_replaced` | Model RMSE tốt nhất thiếu dự đoán → lỗi, không tự đổi model. |
| `test_prepare_no_overwrite_blind_leader` | Đúng schema annotation, 40 nhãn trống; worksheet Leader đúng 4 ID, không chứa nhãn E; prepare lần hai không ghi đè. |
| `test_bad_annotations` (7 case) | Chặn nhãn trống/sai, ID trùng, sửa dự đoán/model, notes trống, gán nhóm cắt khi token không vượt 254. |
| `test_crosscheck_threshold` | 3/4 cho agreement 75%; ID Leader sai mẫu → lỗi. |
| `test_complete_workflow_requires_agreement` | Chưa đủ nhãn → lỗi; nhãn synthetic agreement 2/4 → FAIL, không tạo Markdown tổng hợp; 3/4 → summary trong thư mục tạm, 5 nhóm tổng 40 và worst/random 20/20. |

## 5. Ánh xạ tiêu chí chấp nhận

| Tiêu chí ở đặc tả §1.5 | Test chứng minh |
| :--- | :--- |
| 1. App chạy, import không nạp torch | `test_app_runs_without_crash`, `test_no_torch_import`. |
| 2. Bảng đủ dòng, xử lý metrics thiếu | `test_comparison_missing_metrics`. |
| 3. Per-star/Macro-MAE khớp module B | `test_render_per_star_matches_source`, `test_missing_star`. |
| 4. Model, bộ lọc, sort | `test_filter_copy`, `test_invalid_filter`, `test_app_runs_without_crash`. |
| 5. Top-k, text, chi tiết | `test_top_k_text_alignment`, `test_app_runs_without_crash`, `test_selected_table_row_opens_detail`. |
| 6. Thiếu metadata: Dashboard cảnh báo, Error Analysis dừng | `test_legacy_producer_metadata_is_optional`, `test_prepare_requires_producer_metadata`. |

## 6. Lệnh kiểm chứng và giới hạn kết quả

Chạy từ repo root, với interpreter có đầy đủ dependency:

```powershell
python -m pytest tests/test_dashboard.py tests/test_error_analysis.py -q
```

Có thể chạy riêng Dashboard/AppTest:

```powershell
python -m pytest tests/test_dashboard.py -q
python -m pytest tests/test_dashboard.py::test_app_runs_without_crash -q
```

PASS của suite này chỉ chứng minh hành vi trên fixture tổng hợp. Chưa có dữ liệu training thật trong `reports/` ở lần rà soát 2026-11-17. `src/predict.py` hiện chưa xuất `n_tokens`/`split_id`; F cần bổ sung và C validate output trước E chạy `prepare` với dữ liệu thật. Sau đó vẫn cần E gán nhãn 40 review và Leader cross-check độc lập 4 review ≥75% trước `summarize`.
