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

---

## Phase 2 - 2026-11-11 - C (Model) + D (Training Infra)

### Prompt
> Viết `src/model_builder.py`, `src/utils.py`, `src/engine.py`, `src/train.py`, `src/predict.py` theo spec `docs/03_model.md`, `docs/04_engine_and_utils.md`; kèm `tests/test_model_builder.py`, `test_utils.py`, `test_engine.py`, `test_train.py`, `test_predict.py`. `src/metrics.py` viết phần lõi (clip_scores, rmse, mae, compute_metrics, per_star_metrics) làm dependency — phần bootstrap/summary/baseline của member B chưa làm.

### Smoke Test
- `pytest tests/ -m "not slow"`: **PASS 60/60** (10.2s, CPU, mô hình tiny random config)
- Gồm: 15 test data_setup + 14 model_builder + 4 utils + 13 engine + 5 train + 9 predict
- S1/S2 (`slow`/`gpu`): skip — chờ Colab/GPU thật.

### Run Model
- Chưa chạy trên `review.json`/checkpoint thật (Phase 3 trở đi, Colab).

### Check Description
- [x] `fit` chạy đúng epochs, history.csv đủ cột, best_model.pt = epoch RMSE Val thấp nhất (F1)
- [x] Resume sau `stop_after_epoch` cho kết quả y hệt chạy liền trên CPU/dropout=0 (F2)
- [x] Guard resume/overwrite: config khác -> ValueError nêu khoá; completed -> FileExistsError; overwrite thắng resume (F3)
- [x] `fit` không nhận test_loader; `run_training` với ExplodingLoader hoàn tất (F5)
- [x] fp16 trên CPU -> UserWarning + tự tắt; loss không hữu hạn -> FloatingPointError (L5, L7)
- [x] `predict` clip [1,5], không phụ thuộc batch_size, khôi phục chế độ train/eval (P1-P4)
- [x] `load_trained_model` đọc backbone_config từ checkpoint, không cần mạng (P5)
- [x] `export/merge_test_predictions` đúng cột, chặn ghi đè, kiểm tra khớp review_id/true_stars (P6, P7)
- [x] `set_seed` gọi lại ngay trước `build_model` (T4)

### Ghi chú quyết định / điểm lệch spec nhỏ
- `fit` nhận `epoch` làm keyword nội bộ của `train_one_epoch` (mặc định 0) để thông báo số epoch trong FloatingPointError.
- `history.csv`/`run_meta.json`/checkpoint ghi **mỗi epoch** bằng `write_atomic` (os.replace).
- `best_model.pt` lưu `model_state_dict` đã `.cpu()`; `last.pt` xoá khi hoàn tất trừ khi `keep_last=True`.
- `predict.py` `AutoConfig.for_model` cần bỏ khoá `model_type` khỏi config dict (transformers 5.x báo trùng tham số).
- `metrics.py` hiện chỉ có phần lõi; `bootstrap_ci`, `paired_bootstrap_diff`, `mean_predictor`, `build_model_summary`, `write_model_summary` và toàn bộ `baseline.py` còn lại của member B.
- `test_model_builder.py` mới viết theo checklist phase2 (spec `03_model_tests.md` có thêm nhóm slow/gpu chưa cover).

---

## Phase 6 - 2026-11-13 - E (Dashboard/Error Analysis)

### Smoke Test
- `python -m pytest tests/test_dashboard.py tests/test_error_analysis.py -q`: **39 passed**, 8.53s, Python 3.12, Streamlit 1.67.0; môi trường **không cài torch**.
- AppTest kiểm tra bản có/không text, đổi model, lọc sao 1/5 và rỗng, sắp xếp, dropdown xem chi tiết; test render riêng kiểm tra click chọn dòng.
- Test chuẩn bị/tổng hợp Error Analysis bằng dữ liệu giả: 40 ID duy nhất, 4 random/sao, seed cố định, worksheet Leader không lộ nhãn E, chặn nhãn thiếu/đổi dữ liệu gốc, chặn agreement 2/4, chấp nhận 3/4.
- Một warning từ plugin anyio của môi trường về API pytest bị deprecated, không phải lỗi ứng dụng.
- Không chạy lại suite training A–D (môi trường không cài torch/transformers).

### Run Model
- Không chạy production training hoặc inference; dashboard chỉ đọc CSV.
- Chưa có kết quả training thật trong ZIP. Chưa gán nhãn 40 review thật, chưa Leader cross-check, chưa PR/merge.

### Check Description
- [x] Đọc CSV ưu tiên with_text, bảng summary giữ mọi model có mặt, blank khi thiếu metrics.
- [x] Scatter + identity line, histogram, per-star MAE/RMSE/n + Macro-MAE qua src.metrics.
- [x] Filter/sort, Top-10, detail text; không import torch.
- [x] Hàm/công cụ sampling + annotation validation + summary, tài liệu hướng dẫn Windows.
- [x] Demo riêng có banner SYNTHETIC, không tạo báo cáo thật giả mạo.
- [ ] Nhận CSV thật đầy đủ sau Leader/F validation.
- [ ] E gán nhãn 40 review, Leader kiểm tra 4 review ≥75%, tạo báo cáo cuối.
- [ ] Review/merge PR, ảnh demo với dữ liệu thật.

### Quyết định và điểm tích hợp
- Gọi `per_star_metrics(y_true, y_pred)` theo code hiện có (tự clip), không truyền keyword `clip` mà spec minh họa.
- Chuyển torch import trong tests/conftest.py sang lazy import ở helper để test E chạy độc lập; Dataset giả dùng __len__/__getitem__ protocol.
- Bổ sung scripts/error_analysis.py thay vì sửa module metrics của B. Các CLI `src.metrics --sample-errors` trong design chưa được hiện thực; guide đưa lệnh thực chạy được.
- Loại top-20 khỏi random pool để có đúng 40 review duy nhất; sort ID trước random, seed=42, thiếu 4/sao thì báo lỗi.
- merge_test_predictions hiện thiếu n_tokens/split_id: dashboard chấp nhận kèm cảnh báo; Error Analysis cần đầy đủ, chờ D/F cung cấp đúng schema.
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

