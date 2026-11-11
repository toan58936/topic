# Task Assignments — 7 Members (Leader + A–F Implementers)

> **6 Implementers (A–F)**: Spec + Code + Unit Test + PR cho 1 module (C gộp thêm Integration Test)
> **Leader**: Chạy training Colab, Review PR, Điều phối, Báo cáo, Demo

---

## Quy trình chung (All Members) — Xem `plans/workflow.md`

```
1. ĐỌC SPEC: docs/NN_*.md + docs/NN_*_tests.md
2. VIẾT CODE: src/<module>.py + tests/test_<module>.py
3. LOCAL TEST: pytest tests/test_<module>.py -v --tb=short  → PASS
4. GIT: feature branch → push → PR → develop
5. REVIEW: Leader + 1 người cross-module approve
6. MERGE: Squash merge, delete branch
7. AI LOG: Cập nhật docs/ai_log.md (prompt, smoke test, check description)
```

**Công cụ**: GitHub (PR, CI), Colab (training), Drive (data, checkpoints), Local (dev, dashboard).

---

## 1. Leader — Training Orchestration, PR Review, Report, Demo

| Hạng mục | Chi tiết |
|----------|----------|
| **Scope** | Chạy training Colab (Phase 3-5), Review 6 PR, Loss rule decision, Quota management, Tổng hợp báo cáo, Demo |
| **Specs tham chiếu** | `Define.md`, `Design.md`, `workflow.md`, `implementation.md`, `assignments.md` |
| **Deliverables** | - Training runs: BERT Huber/MSE, DistilBERT, RoBERTa (nếu quota)<br>- Test predictions merge & commit `reports/`<br>- Loss rule decision (Phase 4)<br>- RoBERTa skip decision (Phase 5)<br>- Báo cáo PDF hoàn chỉnh<br>- Demo script & rehearsal<br>- `docs/ai_log.md` entries: Phase 0, 3, 4, 5, 6 |
| **GitHub** | - Tạo repo, invite 6 thành viên<br>- Cấu hình branch protection, CI<br>- Review & merge 6 PR (A–F) trong 4h<br>- Commit `reports/test_predictions.csv`, `model_summary.csv` sau training |
| **Dependencies** | Cần A–F PR merged trước Phase 3; Cần C validate outputs trước Phase 6 |

---

## 2. Thành viên A (Data) — `data_setup.py`

| Hạng mục | Chi tiết |
|----------|----------|
| **Module** | `src/data_setup.py` |
| **Functions** | `prepare_splits()`, `create_dataloaders()` |
| **Specs** | `docs/01_data.md`, `docs/01_data_tests.md` |
| **Tests** | `tests/test_data_setup.py` (CPU, synthetic data) |
| **Deliverables** | - `split_stats.json` trên Drive (schema valid)<br>- `train.parquet`, `val.parquet`, `test.parquet` trên Drive<br>- Code pass pytest CPU<br>- PR merged → `develop` |
| **Verification** | - 60k stratified sample, 70/15/15 split<br>- Val/Test không trùng text với Train<br>- `split_stats.json` đủ schema (split_id, stars_per_split, overlap rates, duplicate counts) |
| **GitHub** | `feature/data-setup` → PR → `develop` |
| **AI Log** | Phase 0 (data download), Phase 1 (prepare_splits run) |
| **Dependencies** | Chạy đầu tiên (không phụ thuộc module khác) |

---

## 3. Thành viên B (Eval) — `metrics.py`, `baseline.py`

| Hạng mục | Chi tiết |
|----------|----------|
| **Modules** | `src/metrics.py`, `src/baseline.py` |
| **Specs** | `docs/02_evaluation.md`, `docs/02_evaluation_tests.md` |
| **Tests** | `tests/test_metrics.py`, `tests/test_baseline.py` |
| **Deliverables** | - Baseline Val metrics (MAE, RMSE, R², Pearson, Spearman, per-star, Macro-MAE)<br>- Mean predictor Val metrics<br>- Threshold: `0.85 * RMSE_baseline_val` (ghi Mốc 1)<br>- `baseline_val.pkl`, `baseline_test.pkl` joblib<br>- Code pass pytest CPU<br>- PR merged → `develop` |
| **Verification** | - Grid search 84 combos (TF-IDF + Ridge) trên Val cố định<br>- Best params by Val RMSE<br>- `n_jobs=-1` parallel CPU |
| **GitHub** | `feature/metrics-baseline` → PR → `develop` |
| **AI Log** | Phase 1 (baseline training, threshold calc) |
| **Dependencies** | Cần A xong `create_dataloaders` signature (để biết Val set format) |

---

## 4. Thành viên C (Model + Integration Test) — `model_builder.py` + Test tích hợp

| Hạng mục | Chi tiết |
|----------|----------|
| **Modules** | `src/model_builder.py` + `tests/test_integration.py`, `scripts/validate_outputs.py` |
| **Specs** | `docs/03_model.md`, `docs/03_model_tests.md` + `docs/06_integration_test.md`, `docs/06_integration_test_tests.md` |
| **Tests** | `tests/test_model_builder.py` + `tests/test_integration.py` |
| **Deliverables (Model)** | - `BACKBONES` dict: bert, roberta, distilbert (hf_name, model_type, drop_pooler)<br>- `build_model()`: TextRegressor với mean/cls pooling, dropout, Linear head<br>- `build_tokenizer()`: tokenizer đúng cặp backbone<br>- `count_parameters()`: total + trainable<br>- 3 backbone build được với tiny random config<br>- Code pass pytest CPU<br>- PR merged → `develop` |
| **Deliverables (Integration Test)** | - **Integration test suite**: Data → Model → Train → Predict → Dashboard<br>- **Validation scripts**: `scripts/validate_outputs.py` (check checkpoints, history, run_meta, predictions merge)<br>- **Pipeline validation report** sau mỗi lần Leader chạy training:<br>  &nbsp;&nbsp;✓ Checkpoints tồn tại, history.csv 3 epochs<br>  &nbsp;&nbsp;✓ run_meta.json đúng spec (backbone, loss, seed, split_id, versions)<br>  &nbsp;&nbsp;✓ Merge predictions: split_id/review_id/true_stars khớp<br>  &nbsp;&nbsp;✓ model_summary.csv đủ 5 dòng, metrics hợp lý<br>  &nbsp;&nbsp;✓ Dashboard load CSV không lỗi<br>- Báo cáo Leader: "Pipeline output VALID" hoặc "Issue: ..." |
| **Verification** | - Chạy pipeline full trên Colab staging (sample data) trước production<br>- Validate resume từ checkpoint<br>- Validate `sync_results.sh` copy Drive → reports/ → commit |
| **GitHub** | `feature/model-builder` → PR → `develop` (model); `feature/integration-test` → PR → `develop` (integration test, ưu tiên merge sớm) |
| **AI Log** | Phase 0 (integration test setup), Phase 2 (model building), Phase 3 (staging validation), Phase 5 (production validation) |
| **Dependencies** | Cần B xong `metrics.py` (param count interface); Cần F xong training CLI (`train.py`, `predict.py`); Cần Leader chạy training production |

---

## 5. Thành viên D (Train Infra Core) — `utils.py`, `engine.py`

| Hạng mục | Chi tiết |
|----------|----------|
| **Modules** | `src/utils.py`, `src/engine.py` |
| **Specs** | `docs/04_engine_and_utils.md`, `docs/04_engine_and_utils_tests.md` |
| **Tests** | `tests/test_utils.py`, `tests/test_engine.py` |
| **Deliverables (Code)** | - `set_seed()`, checkpoint save/load (weights_only=True)<br>- `train_epoch()`, `eval_epoch()`: Huber/MSE, fp16, grad clip<br>- Checkpoint/epoch: model, optimizer, scheduler, rng_state, history.csv<br>- Code pass pytest CPU<br>- PR merged → `develop` |
| **Verification** | - Resume từ checkpoint works<br>- fp16 autocast, output .float() trước loss |
| **GitHub** | `feature/training-infra-core` → PR → `develop` |
| **AI Log** | Phase 2 (infra code) |
| **Dependencies** | Cần C xong `model_builder.py` (build_model, count_parameters) |

---

## 6. Thành viên F (Train CLI) — `train.py`, `predict.py`

| Hạng mục | Chi tiết |
|----------|----------|
| **Modules** | `src/train.py`, `src/predict.py` |
| **Specs** | `docs/04_engine_and_utils.md`, `docs/04_engine_and_utils_tests.md` |
| **Tests** | `tests/test_train.py`, `tests/test_predict.py` |
| **Deliverables (Code)** | - `train.py`: CLI config, log run_meta<br>- `predict.py`: load weights, clip, export test_predictions.csv, merge validation<br>- Code pass pytest CPU<br>- PR merged → `develop` |
| **Verification** | - Merge validation: stop nếu split_id/review_id mismatch |
| **GitHub** | `feature/training-cli` → PR → `develop` |
| **AI Log** | Phase 2 (infra code), Phase 3 (smoke test support) |
| **Dependencies** | Cần D xong `engine.py` (train_epoch, eval_epoch); Cần C xong `model_builder.py` |

---

## 7. Thành viên E (Dashboard/Error Analysis) — `app/streamlit_dashboard.py` + Error Analysis

| Hạng mục | Chi tiết |
|----------|----------|
| **Module** | `app/streamlit_dashboard.py` |
| **Specs** | `docs/05_dashboard.md`, `docs/05_dashboard_tests.md` |
| **Tests** | `tests/test_dashboard.py` (CPU, no torch) |
| **Deliverables (Dashboard)** | - Đọc `reports/test_predictions.csv` + `model_summary.csv`<br>- Auto-detect `test_predictions_with_text.csv`<br>- Bảng so sánh 5 models (metrics + cost)<br>- Charts: Pred vs True, Error dist, Per-star MAE/RMSE<br>- Error cases table (top 10, filterable by star)<br>- Detail view (text nếu có with_text)<br>- Code pass pytest CPU<br>- PR merged → `develop` |
| **Deliverables (Error Analysis - SAU KHI LEADER TRAINING XONG)** | - Sample 40 reviews: 20 worst + 20 random stratified (4/star × 5)<br>- Annotate 5 groups: Nhãn nhiễu, Cảm xúc lẫn lộn, Bị cắt, Mỉa mai, Khác<br>- Leader cross-check 4 reviews (≥75% agreement)<br>- `reports/error_analysis.csv` + summary cho báo cáo |
| **GitHub** | `feature/dashboard` → PR → `develop` (có thể dev với CSV dummy từ đầu) |
| **AI Log** | Phase 6 (dashboard dev, error analysis) |
| **Dependencies** | Cần Leader xong training → C validate outputs → có `test_predictions.csv`, `model_summary.csv` thực |

---

## Ma trận phụ thuộc (Dependency Matrix)

| | A (Data) | B (Eval) | C (Model + Integration) | D (Train Infra Core) | F (Train CLI) | E (Dashboard) | Leader (Training) |
|---|----------|----------|-------------------------|-----------------|---------------|----------------|-------------------|
| **A (Data)** | — | Val set format | — | — | — | `prepare_splits()` |
| **B (Eval)** | Val set format | — | — | — | — | Threshold |
| **C (Model + Integration)** | — | Param count interface | — | `build_model`, `count_parameters` | — | Backbones, Validate outputs |
| **D (Train Infra Core)** | — | — | Model interface | — | — | `train_epoch`, `eval_epoch`, checkpoint |
| **F (Train CLI)** | — | — | Model interface | `train_epoch`, `eval_epoch` | — | `train.py`, `predict.py` CLI |
| **E (Dashboard)** | — | — | — | CSV schema | — | Real CSV |
| **Leader (Training)** | Data splits | Threshold | Model code, Validation | Core infra, CLI | — | **Chạy training** |

**Critical Path**: A → B → C → D → F → **Leader Training (Phase 3-5)** → C Validate → E Finalize → Leader Report

---

## RACI Summary

| Task | Leader | A | B | C | D | F | E |
|------|--------|---|---|---|---|---|---|---|
| Repo & CI setup | R/A | I | I | R | I | I | I |
| Data splits | A | R | I | I | I | I | I |
| Baseline & threshold | A | I | R | I | I | I | I |
| Model builder | A | I | I | R | I | I | I |
| Training infra | A | I | I | I | R | C | I |
| **Colab Training (Phase 3-5)** | **R** | I | I | I | C | C | I |
| Integration Test | A | I | I | **R** | C | C | I |
| Dashboard | A | I | I | I | I | I | R |
| Error Analysis | A | I | I | I | I | I | R |
| Validate Pipeline Outputs | A | I | I | **R** | I | C | I |
| Report & Demo | R/A | C | C | C | C | C | C |

**R** = Responsible, **A** = Accountable, **C** = Consulted, **I** = Informed

---

## Communication & Sync

| Ceremony | Tần suất | Tham gia | Mục đích |
|----------|----------|----------|----------|
| **Daily Standup** | Mỗi ngày 15p | All | Blocker, progress, next |
| **PR Review** | Async (4h SLA) | Leader + 1 | Code quality, spec compliance |
| **Phase Gate Review** | Mỗi phase end | Leader + Owner | DoD check, sign-off |
| **Post-Training Validation** | Sau mỗi training run | Leader + C | Validate outputs trước Phase tiếp theo |
| **Retro** | Cuối Phase 3, 6 | All | Process improvement |

---

> **Lưu ý**: Mọi deliverable phải có **PR merged** + **pytest PASS** + **AI log entry** mới coi là done. Leader không merge PR nếu thiếu 3 tiêu chí này.
> **Leader là người duy nhất chạy training production trên Colab**. C validate outputs, E finalize dashboard/error analysis sau khi validate PASS.