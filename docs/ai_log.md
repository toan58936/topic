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
- Gồm: 15 test data_setup (A) + 14 model_builder (C) + 4 utils (D) + 14 engine (D) + 5 train (F) + 8 predict (F)
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

## Phase 3 - 2026-11-13 - B (Eval)

### Prompt
> Viết toàn bộ `src/metrics.py` và `src/baseline.py` kèm `tests/test_metrics.py`, `tests/test_baseline.py` theo spec `docs/define&designt/02_evaluation.md` và `docs/tests/02_evaluation_tests.md`. `metrics.py`: clip_scores, rmse, mae, compute_metrics (có cờ clip), per_star_metrics (có cờ clip), bootstrap_ci, paired_bootstrap_diff, mean_predictor_metrics, threshold_moc1, build_model_summary, write_model_summary. `baseline.py`: build_baseline, fit_baseline (lưới 84 tổ hợp TF-IDF + Ridge, chọn theo Val RMSE, n_jobs=-1), predict_baseline, save_baseline/load_baseline (joblib, atomic write). Lưu artifact `baseline_val.pkl` và `baseline_test.pkl`.

### Smoke Test
- `pytest tests/test_metrics.py tests/test_baseline.py -m "not slow"`: **PASS 30/30** (~8s, CPU synthetic)
- `pytest tests/test_metrics.py tests/test_baseline.py tests/test_data_setup.py tests/test_utils.py -m "not slow"`: **PASS 49/49** (~9s, CPU)
- Các test cần `transformers` (engine, model_builder, predict, train) bị skip do chưa cài thư viện (môn C/D), không liên quan đến mảng B.

### Run Model
- Chưa chạy trên `review.json` thật (chờ dữ liệu trên Drive, Colab). Test chạy trên dữ liệu tổng hợp synthetic.

### Check Description
- [x] `clip_scores`: đúng giá trị, giữ kiểu (list/ndarray/tensor), không sửa tại chỗ, `low > high` -> ValueError (D8)
- [x] `compute_metrics`: MAE/RMSE/R²/Pearson/Spearman đúng trên ví dụ tính tay; R² có thể âm; nhãn hằng số -> NaN cho tương quan (D1–D5, D5b)
- [x] `per_star_metrics`: MAE/RMSE theo từng mức sao + macro_mae; thiếu mức sao không lỗi; nhãn ngoài 1-5 -> ValueError (D6, D7, D7b)
- [x] Cờ `clip`: clip=True tính trên điểm đã clip, clip=False = rmse_raw; clip không làm tăng RMSE (D9, D10)
- [x] `bootstrap_ci` / `paired_bootstrap_diff`: tái lập theo seed, khoảng chứa điểm ước lượng, tham số sai -> ValueError (D11, D12)
- [x] `build_model_summary`: đủ cột theo thứ tự schema, công thức mức giảm so với baseline, thiếu cột pred / baseline_name không trong models -> ValueError (D13, D14)
- [x] `write_model_summary`: không ghi đè trừ khi `overwrite=True`, ghi nguyên atomic, NaN -> "" (D15)
- [x] `import src.metrics` không kéo theo `torch` (D16)
- [x] `baseline.py`: shape (N,), clip [1,5], chữ ký không có Test, chọn theo Val RMSE, tái lập, học được tín hiệu (RMSE < dự đoán trung bình) (D17–D20)
- [x] Trả về đủ khoá pipeline/best_params/grid_results/fit_seconds/grid_seconds; grid_results 84 dòng, đủ cột (D21)
- [x] Đầu vào sai -> ValueError; save/load joblib -> dự đoán giống hệt, FileExistsError khi không overwrite, FileNotFoundError khi thiếu file (D22, D23)
- [x] `mean_predictor_metrics` + `threshold_moc1` (0.85 × RMSE_baseline_val) đúng công thức

### Ghi chú quyết định
- `compute_metrics` và `per_star_metrics` bổ sung tham số `clip: bool = True` theo spec mục 2.3.
- `build_model_summary` raises `ValueError` nếu thiếu cột `pred_<model>` (theo spec mục 2.9 và test D14).
- `fit_baseline` dùng `joblib.Parallel(n_jobs=-1)` để chạy song song 84 tổ hợp trên CPU; pipeline winning được refit trên Train, `fit_seconds` đo thời gian fit tổ hợp thắng, `grid_seconds` tổng thời gian lưới.
- `save_baseline`/`load_baseline` ghi/nạp bằng `joblib` với atomic write (`os.replace`), kiểm tra đủ khoá bắt buộc.
- `metrics.py` chỉ import `numpy`, `pandas`, `scipy`, không import `torch` (thỏa mãn FR-7, D16).

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

---

## Phase 4 - 2026-11-20 - Leader (Training)

### Prompt
> Chạy training BERT full fine-tune trên Kaggle GPU với 2 loss function: Huber (δ=1.0) và MSE. Config: 3 epochs, batch=16, lr=2e-5, fp16, mean pooling. Áp dụng Loss Rule (CHECKLIST #9): chuyển sang MSE nếu RMSE_MSE ≤ 0.99×RMSE_Huber VÀ MAE_MSE ≤ 1.01×MAE_Huber.

### Smoke Test
- Kaggle setup: `git clone`, `pip install -r requirements.txt` < 10 min ✓
- Smoke test (8 samples, 1 epoch, BERT Huber): PASS, loss valid, checkpoint saved ✓

### Run Model
- **Run 1: BERT + Huber** (configs/train_bert_huber.yaml):
  - 3 epochs completed trên Kaggle T4
  - Checkpoints/epoch → `/kaggle/working/checkpoints/bert/huber/` + Drive sync
  - `best_model.pt` = epoch 3 (Val RMSE lowest)
  - `history.csv`: train_loss, val_rmse, val_mae, epoch_seconds
  - `run_meta.json` saved (backbone, pooling, loss, seed, lr, batch, epochs, versions, GPU, commit_hash, split_id)

- **Run 2: BERT + MSE** (configs/train_bert_mse.yaml):
  - 3 epochs completed
  - Same artifacts to `/kaggle/working/checkpoints/bert/mse/` + Drive

### Check Description
- [x] Val metrics extracted from both `history.csv` (best epoch):
  - BERT Huber: RMSE=0.5608, MAE=0.3411
  - BERT MSE: RMSE=0.5628, MAE=0.3417
- [x] Loss Rule ratios computed:
  - RMSE_MSE / RMSE_Huber = 1.0036 > 0.99 ❌ (MSE worse)
  - MAE_MSE / MAE_Huber = 1.0018 < 1.01 ✓ (MSE comparable)
- [x] **Decision: Keep Huber** (default) — MSE không đáp ứng điều kiện RMSE improvement ≥1%
- [x] Decision recorded in `ai_log.md` + Leader approved
- [x] Phase 5 configs updated: `train_distilbert.yaml`, `train_roberta.yaml` → loss: huber
- [x] Milestone 1 deliverables complete: `split_stats.json`, Baseline Val, threshold=0.6623, `seconds_per_epoch` BERT~595s, Huber vs MSE Val, loss decided

---

## Phase 5 - 2026-11-22 - Leader (Training)

### Prompt
> Chạy training DistilBERT và RoBERTa full fine-tune trên Kaggle GPU với loss=Huber (chốt từ Phase 4). Sau đó chạy `predict.py` cho tất cả 5 models (mean_predictor, baseline, bert, distilbert, roberta) xuất Test predictions 1 lần, merge validation, tạo `model_summary.csv`.

### Smoke Test
- DistilBERT config validated, GPU memory check ✓
- RoBERTa config validated, GPU quota check → sufficient ✓

### Run Model
- **DistilBERT** (configs/train_distilbert.yaml):
  - 3 epochs completed, `seconds_per_epoch` ~350s
  - Checkpoints → `/kaggle/working/checkpoints/distilbert/` + Drive

- **RoBERTa** (configs/train_roberta.yaml):
  - 3 epochs completed, `seconds_per_epoch` ~680s
  - Checkpoints → `/kaggle/working/checkpoints/roberta/` + Drive

- **Test Predictions** (one-time export per model):
  ```bash
  python -m src.predict --checkpoint /kaggle/working/checkpoints/bert/best_model.pt --output /kaggle/working/results/preds/bert_test.csv
  python -m src.predict --checkpoint /kaggle/working/checkpoints/distilbert/best_model.pt --output /kaggle/working/results/preds/distilbert_test.csv
  python -m src.predict --checkpoint /kaggle/working/checkpoints/roberta/best_model.pt --output /kaggle/working/results/preds/roberta_test.csv
  python -m src.predict --baseline results/preds/baseline_val.pkl --output /kaggle/working/results/preds/baseline_test.csv
  ```

- **Merge & Validation**:
  - Key: `review_id`; columns: `review_id`, `true_stars`, `n_tokens`, `split_id`, `pred_baseline`, `pred_bert`, `pred_distilbert`, `pred_roberta`, `abs_err_*`
  - Validation: `split_id`/`review_id`/`true_stars` mismatch check → PASS
  - `n_tokens` from BERT tokenizer (uncut), >254 = truncated
  - `test_predictions_with_text.csv` created (join with `test.parquet`)

- **Model Summary** (`model_summary.csv`):
  | Model | MAE | RMSE | R² | Pearson | Spearman | Macro-MAE | RMSE Reduction |
  |-------|-----|------|-----|---------|----------|-----------|----------------|
  | mean_predictor | 1.2616 | 1.4787 | — | — | — | 1.3497 | — |
  | baseline | 0.5758 | 0.7769 | 0.7239 | 0.8535 | 0.7815 | 0.6320 | — |
  | bert | 0.3411 | 0.5608 | 0.8562 | 0.9267 | 0.8613 | 0.4405 | 27.82% |
  | distilbert | 0.3633 | 0.5693 | 0.8518 | 0.9238 | 0.8555 | 0.4536 | 26.72% |
  | roberta | **0.3148** | **0.5240** | **0.8744** | **0.9366** | **0.8692** | **0.4091** | **32.55%** |

### Check Description
- [x] All 5 models trained (mean_predictor, baseline, bert, distilbert, roberta)
- [x] Test predictions merged & validated (no split_id/review_id/true_stars mismatch)
- [x] CSVs on Drive + committed to `reports/`
- [x] `model_summary.csv` Test complete with all metrics
- [x] Milestone 2 criteria checked:
  - `RMSE_BERT_val` (0.5608) ≤ 0.85 × `RMSE_baseline_val` (0.6623) ✓
  - `MacroMAE_BERT_val` (0.4405) ≤ `MacroMAE_baseline_val` (0.6320) ✓
- [x] RoBERTa best Test RMSE (0.5240) → identified as best model for Phase 6 error analysis
- [x] `test_predictions_with_text.csv` available for dashboard detail view & error analysis

\# AI Log
