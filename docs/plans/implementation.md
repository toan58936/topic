# Implementation Plan — Đề tài 7 Text Regression (Yelp)

> Thứ tự triển khai chi tiết, mapping từ spec → code → test → run. File này được tham chiếu từ `docs/define&designt/Tài liệu Define.md` Mục 13.

---

## 1. Thứ tự viết Spec → Code → Test (6 nhóm module)

| Thứ tự | Spec (docs/) | Module | Test (tests/) | Điều kiện bắt đầu |
|--------|--------------|--------|---------------|-------------------|
| **01** | `01_data.md` | `data_setup.py` | `test_data_setup.py` | — |
| **02** | `02_evaluation.md` | `metrics.py`, `baseline.py` | `test_metrics.py`, `test_baseline.py` | 01 spec xong (cần `create_dataloaders` signature) |
| **03** | `03_model.md` | `model_builder.py` | `test_model_builder.py` | 01, 02 spec xong (cần tokenizer, param count) |
| **04** | `04_engine_and_utils.md` | `utils.py`, `engine.py`, `train.py`, `predict.py` | `test_utils.py`, `test_engine.py`, `test_train.py`, `test_predict.py` | 03 spec xong (cần `build_model`, `count_parameters`) |
| **05** | `05_dashboard.md` | `app/streamlit_dashboard.py` | `test_dashboard.py` | 02, 04 spec xong (cần CSV schema) — **có thể làm song song từ đầu với CSV dummy** |
| **06** | `06_integration_test.md` | `tests/test_integration.py`, `scripts/validate_outputs.py` | `test_integration.py` | 04 spec xong (cần train/predict CLI) — **có thể làm song song Phase 0-1** |

**Quy tắc**: Spec phải chốt trước khi code. Code phải pass smoke test + pytest CPU trước khi chạy GPU/Kaggle.

---

## 2. Chi tiết từng giai đoạn (Execution Phases)

### Phase 0: Khởi tạo repo & môi trường — **COMPLETED**

| Task | Output |
|------|--------|
| Tạo GitHub private repo, cấu trúc thư mục | Repo với `src/`, `docs/`, `tests/`, `configs/`, `notebooks/`, `scripts/`, `app/`, `reports/`, `requirements.txt`, `README.md`, `.gitignore` |
| Cài đặt local dev env (venv) | `pip install -r requirements.txt` |
| Survey Kaggle GPU access | Danh sách fallback runners |
| Script sync Kaggle → Drive | `scripts/sync_results.sh` (copy `/kaggle/working/` → `drive/project/`, commit) |
| Setup Google Drive folder structure | `drive/project/{data,checkpoints,results}` shared |

---

### Phase 1: Data & Baseline — **COMPLETED**

#### 1.1 Data Setup

| Task | Verify |
|------|--------|
| Viết spec `docs/01_data.md` | Spec review bởi Leader ✓ |
| Code `src/data_setup.py` (`prepare_splits`, `create_dataloaders`) | Pass `test_data_setup.py` (CPU, synthetic data) ✓ |
| Chạy `prepare_splits()` trên Kaggle CPU: tải `review.json`, stratified sample 60k, split 70/15/15, lưu parquet + `split_stats.json` lên Drive | `split_stats.json` hợp lệ schema; Val/Test không trùng text với Train ✓ |
| Commit `data_setup.py`, tests; push | PR merged ✓ |

#### 1.2 Metrics & Baseline

| Task | Verify |
|------|--------|
| Viết spec `docs/02_evaluation.md` | Spec review bởi Leader ✓ |
| Code `src/metrics.py` (clip_scores, MAE/RMSE/R²/Pearson/Spearman, per-star, macro_mae, mean_predictor, model_summary) | Pass `test_metrics.py` ✓ |
| Code `src/baseline.py` (TF-IDF + Ridge, grid 84 combos, Val cố định, joblib save/load) | Pass `test_baseline.py` ✓ |
| Chạy Baseline trên Val (Kaggle CPU), tính ngưỡng mục tiêu: `threshold = 0.85 * RMSE_baseline_val` | Ngưỡng ghi vào Mốc 1: **0.6623** ✓ |
| Commit `metrics.py`, `baseline.py`, tests; push | PR merged ✓ |

**Deliverable Phase 1**: `split_stats.json` trên Drive, Baseline Val metrics (RMSE=0.7769), ngưỡng mục tiêu 0.6623, code 01-02 pass tests.

---

### Phase 2: Model & Training Infra — **COMPLETED**

#### 2.1 Model Builder

| Task | Verify |
|------|--------|
| Viết spec `docs/03_model.md` | Spec review bởi Leader ✓ |
| Code `src/model_builder.py` (BACKBONES, build_model, mean/cls pooling, build_tokenizer, count_parameters) | Pass `test_model_builder.py` (tiny random config cho 3 backbone) ✓ |
| Commit; push | PR merged ✓ |

#### 2.2 Engine, Utils, Train, Predict

| Task | Verify |
|------|--------|
| Viết spec `docs/04_engine_and_utils.md` | Spec review bởi Leader ✓ |
| Code `src/utils.py` (set_seed, atomic checkpoint save/load, weights_only=True) | Pass `test_utils.py` ✓ |
| Code `src/engine.py` (train_epoch, eval_epoch, Huber/MSE, fp16, checkpoint/epoch, history.csv) | Pass `test_engine.py` ✓ |
| Code `src/train.py` (CLI config, set_seed trước build_model, log run_meta) | Pass `test_train.py` ✓ |
| Code `src/predict.py` (load weights, predict, clip, export test_predictions.csv, merge validation) | Pass `test_predict.py` ✓ |
| Commit; push | PR merged ✓ |

**Deliverable Phase 2**: Toàn bộ training pipeline code pass CPU tests (60/60).

---

### Phase 3: Smoke Test & Short GPU Run (Leader chạy trên Kaggle) — **COMPLETED**

| Task | Verify |
|------|--------|
| Kaggle: `git clone`, `pip install -r requirements.txt` | Setup < 10 phút ✓ |
| Smoke test: 8 samples, 1 epoch, batch=16, fp16, BERT Huber | Không lỗi; loss giảm; checkpoint ghi `/kaggle/working/checkpoints/` ✓ |
| Short GPU run: 100 steps (hoặc 1 epoch), đo `seconds_per_epoch`, VRAM usage | `seconds_per_epoch` ghi Mốc 1: BERT~595s, DistilBERT~350s, RoBERTa~680s; VRAM < 14GB ✓ |
| Fix bug nếu có (OOM → batch=8 + gradient_checkpointing; import error → patch) | Chạy ổn định ✓ |

**Deliverable Phase 3**: `seconds_per_epoch` BERT/DistilBERT/RoBERTa, xác nhận pipeline GPU chạy được.

---

### Phase 4: Full BERT Training (Huber + MSE) — Loss Rule — **COMPLETED**

| Task | Verify |
|------|--------|
| **Run 1**: BERT full fine-tune, Huber (δ=1.0), 3 epochs, batch=16, fp16 | Checkpoint/epoch, history.csv, best_model.pt ✓ |
| **Run 2**: BERT full fine-tune, MSE, 3 epochs, same setup | Checkpoint/epoch, history.csv, best_model.pt ✓ |
| Đọc Val RMSE/MAE từ 2 history.csv; áp dụng **Loss Rule** (CHECKLIST #9):<br>• Chuyển sang MSE nếu `RMSE_MSE <= 0.99 * RMSE_Huber` **và** `MAE_MSE <= 1.01 * MAE_Huber`<br>• Ngược lại giữ Huber | **Decision: Keep Huber** (RMSE_Huber=0.5608, RMSE_MSE=0.5628, ratio=1.0036 > 0.99). Ghi `ai_log.md` ✓ |
| Cập nhật config default loss cho Phase 5 | `configs/train_*.yaml` dùng loss=Huber ✓ |

**Deliverable Phase 4**: Loss đã chốt (Huber), BERT Huber/MSE checkpoints trên Drive, Mốc 1 docs hoàn tất.

---

### Phase 5: DistilBERT → RoBERTa (Leader chạy trên Kaggle) — **COMPLETED**

| Task | Verify |
|------|--------|
| DistilBERT: full fine-tune 3 epochs, loss=Huber, same setup | Checkpoint, history, Val metrics ✓ |
| RoBERTa: full fine-tune 3 epochs, loss=Huber, same setup | Checkpoint, history, Val metrics ✓ (quota đủ) |
| `predict.py` dự đoán Test **một lần** cho mọi model + baseline | `test_predictions.csv`, `test_predictions_with_text.csv`, `model_summary.csv` trên Drive ✓ |
| Merge validation: kiểm tra `split_id`, `review_id`, `true_stars` khớp | Không lỗi merge ✓ |
| Copy CSV → `reports/`, commit, push | GitHub có kết quả Test ✓ |

**Test Results (model_summary.csv):**
| Model | MAE | RMSE | R² | Pearson | Spearman | Macro-MAE | RMSE Reduction vs Baseline |
|-------|-----|------|-----|---------|----------|-----------|---------------------------|
| mean_predictor | 1.2616 | 1.4787 | — | — | — | 1.3497 | — |
| baseline | 0.5758 | 0.7769 | 0.7239 | 0.8535 | 0.7815 | 0.6320 | — |
| bert | 0.3411 | 0.5608 | 0.8562 | 0.9267 | 0.8613 | 0.4405 | 27.82% |
| distilbert | 0.3633 | 0.5693 | 0.8518 | 0.9238 | 0.8555 | 0.4536 | 26.72% |
| roberta | **0.3148** | **0.5240** | **0.8744** | **0.9366** | **0.8692** | **0.4091** | **32.55%** |

**Deliverable Phase 5**: `model_summary.csv` Test, `test_predictions.csv` Test, tiêu chí đạt checked (RMSE_BERT=0.5608 ≤ 0.6623 ✓, MacroMAE_BERT=0.4405 ≤ 0.6320 ✓).

---

### Phase 6: Validate Outputs → Dashboard Finalize → Error Analysis → Report — **IN PROGRESS**

| Step | Owner | Task | Verify |
|------|-------|------|--------|
| 6.1 | C | Validate pipeline outputs sau Leader training: checkpoints, history.csv, run_meta.json, merge predictions, model_summary.csv, dashboard load | Dashboard reads CSVs ✓; features pending |
| 6.2 | E | Finalize dashboard: charts, filters, detail view với CSV thực | Code done, tests pass; charts/filters pending |
| 6.3 | E | **Error Analysis (automated via `scripts/error_analysis.py`):**<br>1. `prepare` → auto-sample 40 reviews (20 worst + 20 random stratified), create templates<br>2. E annotates `error_analysis.csv` (40 rows, 5 groups)<br>3. Leader cross-checks 4 reviews via `leader_crosscheck_with_text.csv`<br>4. `summarize` → validates, checks ≥75% agreement, outputs summary MD/CSV | Pending |
| 6.4 | Leader | Tổng hợp `ai_log.md`, viết báo cáo PDF, chuẩn bị demo | Pending |

---

## 3. Mapping Phase → Mốc (Milestones)

| Mốc | Thời hạn | Phase | Tiêu chí đạt |
|-----|----------|-------|--------------|
| **Mốc 1** | Hết Phase 4 | Phase 0–4 | `split_stats.json`, Baseline Val, ngưỡng 0.85×, `seconds_per_epoch` BERT/DistilBERT/RoBERTa, Huber vs MSE Val, loss đã chốt (Huber) ✓ |
| **Mốc 2** | Hết Phase 5 | Phase 5 | 3 models Test predictions, `model_summary.csv` Test, tiêu chí đạt checked ✓ |
| **Mốc 3** | Hết Phase 6 | Phase 6 | Dashboard, error analysis, báo cáo, ai_log |

---

## 4. Rủi ro & Contingency (Tóm tắt từ Design §7)

| Rủi ro | Trigger | Contingency |
|--------|---------|-------------|
| Kaggle ngắt/OOM | Phase 3, 4, 5 | Checkpoint resume; batch=8 + gradient_checkpointing; fallback freeze_backbone |
| Hết GPU quota | Phase 5 (RoBERTa) | Bỏ RoBERTa, báo cáo 2 models |
| Data download/split fail | Phase 1 | Sample 30k test trước; dùng IMDb fallback |
| Baseline grid search chậm | Phase 1 | `n_jobs=-1`, giảm grid nếu cần |
| Spec/code review chậm | Mọi phase | Leader review trong 4h; async approve |
| Dashboard bug | Phase 6 | CSV dummy sẵn từ đầu; làm độc lập |

---

## 5. Checklist thực thi (Definition of Done per Phase)

### Phase 0 Done khi:
- [x] GitHub private repo ready, structure đúng (local)
- [x] All members: local env verified, `pytest tests/` PASS (60/60 not slow)
- [x] `requirements.txt` pinned exact versions (torch 2.4.0+cu121, transformers 4.44.0, etc.)
- [x] Script sync Kaggle → Drive working
- [x] Google Drive folder structure created, shared
- [x] Kaggle GPU access confirmed
- [x] Docs synced, `ai_log.md` created

### Phase 1 Done khi:
- [x] `split_stats.json` trên Drive, schema đúng (split_id: f0e6776f8a1a)
- [x] Baseline Val MAE/RMSE/R² + mean predictor (MAE=0.5758, RMSE=0.7769, R²=0.7239)
- [x] Ngưỡng `0.85 * RMSE_baseline_val` = 0.6623 tính được
- [x] `data_setup.py`, `metrics.py`, `baseline.py` pass pytest CPU
- [x] PR merged vào main

### Phase 2 Done khi:
- [x] `model_builder.py`, `utils.py`, `engine.py`, `train.py`, `predict.py` pass pytest CPU
- [x] 3 backbone build được với tiny config
- [x] PR merged

### Phase 3 Done khi:
- [x] Kaggle setup < 10p, clone OK
- [x] Smoke test 8 samples pass
- [x] Short run 100 steps: `seconds_per_epoch` ghi nhận (BERT 595s, DistilBERT 350s, RoBERTa 680s), VRAM < 14GB

### Phase 4 Done khi:
- [x] BERT Huber 3 epochs xong, checkpoint `/kaggle/working/checkpoints/bert/` + Drive
- [x] BERT MSE 3 epochs xong, checkpoint `/kaggle/working/checkpoints/bert/` + Drive
- [x] Loss rule áp dụng, quyết định Keep Huber ghi `ai_log.md`
- [x] Mốc 1 docs hoàn tất

### Phase 5 Done khi:
- [x] DistilBERT 3 epochs xong (RoBERTa trained — quota cho phép)
- [x] Test predictions xuất 1 lần, merge validation pass
- [x] `model_summary.csv` Test có đủ chỉ số cho 5 models
- [x] Tiêu chí đạt (Define Mục 4) checked: RMSE_BERT ≤ 0.85× baseline Val (0.5608 ≤ 0.6623 ✓), MacroMAE_BERT ≤ baseline Val (0.4405 ≤ 0.6320 ✓)

### Phase 6 Done khi:
- [x] Dashboard code chạy local, đọc CSV đúng (code viết xong, test pass, CSV thật có sẵn)
- [ ] `error_analysis.csv` 40 review, 5 nhóm, Leader cross-check ≥75% (via `scripts/error_analysis.py` prepare → annotate → summarize)
- [ ] `error_group_summary.csv` + `error_analysis_summary.md` generated cho báo cáo
- [ ] Báo cáo PDF + `ai_log.md` hoàn chỉnh (ai_log có entries Phase 1,2,3; cần thêm 4,5,6)
- [ ] Demo sẵn sàng

---

## 6. Liên kết tài liệu

| File | Vai trò |
|------|---------|
| `docs/define&designt/Tài liệu Define.md` | Cái gì, dữ liệu, tiêu chí, phạm vi, sản phẩm |
| `docs/define&designt/Tài liệu Design.md` | Kiến trúc, module, luồng dữ liệu, config, schema, rủi ro |
| `docs/define&designt/CHECKLIST.md` | 10 mục track chi tiết (hyperparams, loss, baseline, data, checkpoint, output, error, versioning, loss rule, criteria) |
| `docs/01_data.md` – `06_integration_test.md` | Spec chi tiết từng module (chữ ký hàm, acceptance criteria) |
| `docs/*_tests.md` | Kế hoạch pytest cho từng module |
| `configs/train_*.yaml` | Cấu hình mỗi lần chạy (backbone, loss, hyperparams) |
| `plans/implementation.md` | **File này** — thứ tự triển khai chi tiết, phase gate, DoD |
| `plans/assignments.md` | Phân công 7 thành viên, trách nhiệm cá nhân |
| `plans/workflow.md` | Quy trình Git, AI collaboration, Phase gate rules |

---

> **Lưu ý**: Plan này thay thế lịch theo tuần ở Define Mục 13. Mốc 1/2/3 giữ nguyên định nghĩa, chỉ chi tiết hóa phase gate để dễ theo dõi và quản trị rủi ro. Training chạy trên **Kaggle GPU**, lưu data/checkpoints trên **Google Drive**, dev/test chạy **local**.