# AI Log — Đề tài 7 Text Regression (Yelp)

> Running log theo `plans/workflow.md` mục 6. Mỗi entry: Phase - ngày - owner, Prompt, Smoke Test, Run Model, Check Description.

---

## Phase 1 - 2026-11-09 - A (Data)

### Prompt
> Viết `src/data_setup.py` (`prepare_splits`, `create_dataloaders`, `ReviewDataset`, `_allocate_counts`) theo spec `docs/define&designt/01_data.md` và test plan `docs/tests/01_data_tests.md`. Kèm `tests/conftest.py` + `tests/test_data_setup.py` (D1–D15, S1–S2 slow).

### Smoke Test
- `pytest tests/test_data_setup.py -m "not slow"`: **PASS 15/15** (0.70s, CPU synthetic)
- S1/S2 (`slow`): skip — chưa có `review.json` thật trên máy local.

### Run Model
- Chưa chạy `prepare_splits()` trên `review.json` thật (chờ Leader/Colab, Phase 1.3).

### Check Description
- [x] Đọc file theo khối, không nạp RAM (2 lượt đọc + hash sha256 stream)
- [x] Lấy mẫu phân tầng giữ phân phối tự nhiên (`_allocate_counts` + rng riêng `default_rng(seed)`)
- [x] Chia 70/15/15 phân tầng theo sao, tái lập theo seed (D4)
- [x] Làm sạch NFC/strip/lower/collapse whitespace; loại text rỗng (D6)
- [x] Ghi `train/val/test.parquet` + `split_stats.json` đủ schema (+ khoá `quota` phục vụ test D2)
- [x] Không ghi đè trừ khi `overwrite=True` (D8)
- [x] `ReviewDataset`/`create_dataloaders`: shape `[B,256]`, labels `float32`, review_id list[str], shuffle chỉ Train (D10–D13)
- [x] Val/Test không trùng text với Train sau chuẩn hoá (D15, `duplicate_text_removed`)

### Vấn đề môi trường (local Windows)
- **Smart App Control = On** chặn DLL của pandas/pyarrow/torch bản mới trong `site-packages` (`WinError 4551 / DLL load failed`).
- Workaround đã dùng để chạy test: cài bản cũ hơn vào `%TEMP%` và set `PYTHONPATH`:
  ```powershell
  pip install "pandas==2.2.3" --target $env:TEMP\pdtest
  pip install "pyarrow==18.1.0" --target $env:TEMP\lab2deps
  pip install "torch==2.8.0+cpu" --index-url https://download.pytorch.org/whl/cpu --target $env:TEMP\lab2deps
  $env:PYTHONPATH = "$env:TEMP\pdtest;$env:TEMP\lab2deps;."
  pytest tests/test_data_setup.py -m "not slow"
  ```
- Trên Colab/Linux không gặp vấn đề này. Cân nhắc tắt Smart App Control hoặc chuẩn hoá mọi người dev trên Colab.

### Ghi chú quyết định
- `split_stats.json` có thêm khoá `quota` (`{split: {star: count}}`) ngoài schema gốc — cần cho test D2 đối chiếu hạn ngạch.
- `data_version` = `yelp_YYYY_MM_<sha8>` với YYYY_MM theo thời điểm chạy (UTC), sha8 = 8 ký tự đầu sha256 của `review.json`.
- Phát hiện thêm: tên file `docs/define&designt/Tài liệu Define.md` / `Tài liệu Design.md` bị lỗi encoding trên Windows ("T�i li?u") — cần rename.
\# AI Log



\## Phase 2 – Model Building (Member C)



\*\*Thành viên:\*\* C – Model  

\*\*Module:\*\* `src/model\_builder.py`  

\*\*Ngày thực hiện:\*\* 14/11/2026  

\*\*Git branch:\*\* `feature/model-builder`  

\*\*Pull Request:\*\* \[PR #7](https://github.com/toan58936/topic/pull/7)



\### 1. Mục tiêu



Sử dụng AI hỗ trợ xây dựng module Transformer Model Builder phục vụ bài toán dự đoán điểm đánh giá (Text Regression).



Các yêu cầu chính:

\- Hỗ trợ ba backbone: BERT, RoBERTa và DistilBERT.

\- Xây dựng `TextRegressor` với Mean Pooling hoặc CLS Pooling.

\- Sử dụng Dropout và Linear Regression Head.

\- Xây dựng các hàm `build\_model()`, `build\_tokenizer()` và `count\_parameters()`.

\- Cho phép khởi tạo model bằng Tiny Random Config để kiểm thử trên CPU.

\- Không sử dụng `token\_type\_ids` trong forward.



\### 2. AI hỗ trợ thực hiện



\*\*Prompt summary (tóm tắt yêu cầu đã trao đổi với AI):\*\*



Yêu cầu AI hướng dẫn xây dựng `src/model\_builder.py` theo tài liệu thiết kế của dự án, hỗ trợ BERT, RoBERTa và DistilBERT; triển khai pooling, regression head, tokenizer, đếm tham số và giải thích từng phần code.



\*\*Các nội dung AI hỗ trợ:\*\*

\- Đề xuất cấu trúc module và cách khởi tạo ba loại Transformer backbone.

\- Hướng dẫn triển khai Mean Pooling có sử dụng `attention\_mask` để bỏ qua padding.

\- Hướng dẫn triển khai CLS Pooling bằng hidden state tại vị trí đầu tiên.

\- Xây dựng Regression Head gồm Dropout và Linear.

\- Hướng dẫn khởi tạo model bằng Tiny Config để kiểm thử nhanh trên CPU.

\- Hướng dẫn viết test cho output shape, pooling, parameter count, tokenizer và các trường hợp kiểm tra khác.



\### 3. Kiểm thử và xác minh kết quả



\*\*Tiny Config Test:\*\*



Sử dụng cấu hình Transformer nhỏ với:

\- `hidden\_size = 32`

\- `num\_hidden\_layers = 2`

\- `num\_attention\_heads = 2`

\- `intermediate\_size = 64` (BERT/RoBERTa)

\- Cấu hình tương ứng của DistilBERT.



Kiểm tra cả ba backbone có thể khởi tạo và thực hiện forward trên dữ liệu giả lập.



\*\*Các nội dung đã kiểm tra:\*\*

\- Input batch có kích thước `\[2, 16]` cho output `\[2]`.

\- Mean Pooling không bị ảnh hưởng bởi padding.

\- CLS Pooling lấy hidden state tại vị trí 0.

\- Regression Head hỗ trợ khởi tạo bias theo giá trị trung bình nhãn train khi được cung cấp.

\- Parameter count bao gồm backbone và head, không tính pooler đã loại bỏ.

\- Forward không sử dụng `token\_type\_ids`.



\*\*Lệnh kiểm thử:\*\*



```powershell

pytest -v

```



\*\*Kết quả thực tế:\*\*



`49 passed in 25.35s`



Tất cả 49 bài kiểm thử hiện có trong bộ test Model Builder đều thành công trên môi trường CPU.



\### 4. Đánh giá và kiểm tra của thành viên



Sau khi nhận code hỗ trợ từ AI, thành viên đã:

\- Đọc và kiểm tra lại nội dung `src/model\_builder.py`.

\- Đối chiếu implementation với tài liệu thiết kế và tài liệu kiểm thử của Model Builder.

\- Chạy bộ test bằng pytest và kiểm tra kết quả thực tế.

\- Kiểm tra các file cần commit trước khi đưa lên GitHub.

\- Thực hiện commit và push code lên nhánh riêng.



AI được sử dụng như công cụ hỗ trợ viết code, giải thích và hướng dẫn kiểm thử. Kết quả cuối cùng được xác nhận bằng việc chạy test thực tế và kiểm tra Pull Request trên GitHub.



\### 5. GitHub và kết quả bàn giao



\*\*Commit:\*\*

\- Commit ID: `9959764`

\- Message: `feat: add transformer model builder`



\*\*Các file đã commit:\*\*

\- `src/model\_builder.py`

\- `tests/conftest.py`

\- `tests/test\_model\_builder.py`



\*\*Pull Request:\*\*

\- PR #7

\- Branch: `feature/model-builder`

\- Target branch: `develop`

\- Trạng thái: Merged



\### 6. Kết luận



Thành viên C đã hoàn thành phần xây dựng Model Builder, kiểm thử CPU với 49 bài test thành công và hợp nhất Pull Request vào nhánh `develop`.



Phần cần tiếp tục phối hợp là xác nhận giao diện đếm tham số với module `metrics.py` của thành viên B và kiểm tra tích hợp trong toàn bộ pipeline của dự án.

