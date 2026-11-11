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
