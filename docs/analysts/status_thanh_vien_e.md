# Báo cáo trạng thái — Thành viên E (Dashboard / Error Analysis)

**Cập nhật:** 2026-11-17

**Nhánh:** `feature-cuong`

**Phạm vi:** `plans/assignments.md` mục 7; chỉ sửa file E và tài liệu liên quan trực tiếp.

## 1. Trạng thái sau đối chiếu feedback

| Hạng mục | Kết luận có bằng chứng |
| :--- | :--- |
| Dashboard | Đã hiện thực FR-1–FR-10; kiểm chứng bằng CSV tổng hợp/AppTest. Chưa demo với kết quả training thật. |
| Error Analysis | Có code sampling, validation, cross-check và tổng hợp; workflow được test bằng fixture. Chưa thực hiện phân tích/gán nhãn người trên dữ liệu thật. |
| Tài liệu | Spec chuyển sang Đã hiện thực, đồng bộ code. Test plan đã tồn tại tại `docs/tests/05_dashboard_tests.md`, được sửa tại chỗ. |
| Kiểm thử | **44 passed in 17.50s**: 26 case Dashboard + 18 case Error Analysis. |
| Dữ liệu hiện có | `reports/` chỉ có `.gitkeep` tại lần rà soát; không có CSV production để xác nhận kết quả mô hình. |

## 2. Feedback đã xử lý

| Feedback/vấn đề | Đối chiếu và thay đổi |
| :--- | :--- |
| Spec vẫn Bản nháp | Sửa `docs/define&designt/05_dashboard.md`: trạng thái, đường dẫn, hàm thực có, CSV validation, widget/detail view, histogram/Vega-Lite, per-star theo bộ lọc, metadata và phụ thuộc. Bỏ mô tả caching/layout chỉ mới đề xuất. |
| Báo cáo nói không có test plan | Nhận định cũ không còn đúng. `docs/tests/05_dashboard_tests.md` đã có; sửa đường dẫn, fixture, tên test, API trả về, AppTest/version, ma trận coverage và lệnh chạy, không tạo test plan mới. |
| Dashboard khi thiếu metadata | Code đã cho phép thiếu và cảnh báo. Mở rộng test cho thiếu riêng `n_tokens`, riêng `split_id`, cả hai: loader/UI vẫn có bảng/model/Top-10 và cảnh báo đúng. |
| Error Analysis cần metadata | Code đã dừng khi thiếu một trong hai cột. Thêm 3 case `prepare` từ chối input thiếu metadata và không tạo thư mục worksheet; không nới điều kiện hoặc sửa producer F. |
| Lỗi kiểm thử trên Windows | Lần chạy phát hiện `UnicodeDecodeError`: test đọc summary UTF-8 bằng encoding mặc định cp1252. Sửa `read_text(encoding='utf-8')`; script đã ghi UTF-8 đúng nên không cần đổi. |
| Chú thích dependency metrics lỗi thời | Sửa comment tại `compute_per_star`: sử dụng clipping mặc định; `src.metrics` hiện có tham số `clip`, không còn đúng khi nói chỉ nhận hai tham số. Không đổi lời gọi/API. |
| Báo cáo/AI log | Cập nhật theo thay đổi và lệnh test thực chạy; không suy diễn trạng thái training, nhãn người hoặc lịch sử PR/merge. |

## 3. Blocker `n_tokens` / `split_id` — thuộc F

Đã đọc `src/predict.py` để đối chiếu, **không sửa file này**:

- `export_test_predictions()` tạo CSV từng model gồm `review_id`, `true_stars`, `pred_raw`, `pred`; chưa xuất hai cột metadata.
- `merge_test_predictions()` khởi tạo bảng bằng `review_id`/`true_stars`, thêm dự đoán/sai số và text; không tính/truyền `n_tokens` hay `split_id` vào hai bản CSV cuối. Việc có split trong nguồn metadata của summary không bổ sung cột cho predictions.
- `scripts/error_analysis.py::sample_reviews()` bắt buộc có cả hai. `n_tokens` dùng xác định cờ >254, `split_id` dùng manifest và đối chiếu phiên dữ liệu. Dashboard có thể cảnh báo rồi chạy; Error Analysis không thể thay metadata bằng giá trị giả.

**Cách phối hợp để gỡ chặn:**

1. **F:** thống nhất schema với Leader theo tài liệu Design mục 4/9.9.1; xuất `n_tokens` từ tokenizer BERT trước cắt, theo quy ước không tính hai token đặc biệt cho ngưỡng 254; truyền/kiểm tra `split_id` đúng lần chia và giữ metadata ở cả hai CSV cuối.
2. **Leader:** chạy training và cung cấp `model_summary.csv`, `test_predictions.csv`, bản `_with_text` thật.
3. **C:** validate schema, ID/nhãn/split và kết quả xuất; thông báo output hợp lệ trước E phân tích.
4. **E:** `prepare` → đọc/gán nhãn đủ 40 review kèm bằng chứng → **Leader** gán độc lập 4 review → `summarize` khi agreement ≥75%. Lệnh và file output được mô tả trong spec hiện có, mục 2.8.

Chưa tạo `reports/error_analysis.csv` hoặc summary thật; chưa có nhãn E hay cross-check Leader thật. Không nhận PASS của workflow synthetic làm bằng chứng hoàn tất deliverable sau training.

## 4. Kiểm chứng thực hiện

Chạy từ repo root, đưa `.venv/Scripts` lên đầu PATH để `python` dùng môi trường riêng:

```powershell
python -m pytest tests/test_dashboard.py tests/test_error_analysis.py -q
```

**Kết quả cuối:** `44 passed in 17.50s` (exit code 0), Python 3.12.10, pytest 9.2.0, Streamlit 1.67.0, torch CPU 2.15.0+cpu.

- Python mặc định ban đầu thiếu `pytest` và dependency; đã tạo `.venv` (gitignore) và cài dependency Dashboard/pytest, torch CPU phục vụ fixture chung. Một lần collection khi torch chưa cài xong lỗi thiếu `torch.utils`; chạy lại sau khi cài đủ.
- Lần suite trước sửa encoding: `1 failed, 43 passed in 49.88s`; sau khi chỉ định UTF-8, chạy lại nguyên lệnh và đạt kết quả cuối ở trên.
- `tests/conftest.py` hiện import torch khi collection; không sửa fixture chung. `test_no_torch_import` chạy subprocess riêng chứng minh import Dashboard không nạp torch, không tuyên bố cả môi trường không cài torch.
- Đối chiếu tên test trong test plan bằng AST và kiểm tra các link tương đối trong hai tài liệu: hợp lệ.
- Không chạy suite module khác, production training/inference, gán nhãn thật hoặc cross-check thật.

## 5. Giới hạn thay đổi

Chỉ sửa 7 file hiện có: `app/streamlit_dashboard.py` (comment), `tests/test_dashboard.py`, `tests/test_error_analysis.py`, `docs/define&designt/05_dashboard.md`, `docs/tests/05_dashboard_tests.md`, báo cáo này và `docs/ai_log.md`.

Không đổi API, không sửa module thành viên khác (bao gồm `src/predict.py`)
