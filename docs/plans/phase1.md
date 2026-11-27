# Phase 1 Checklist: Data & Baseline

> Data setup, metrics, baseline. Owner: A (Data), B (Baseline), Leader (review).

---

## 1.1 Spec & Code

### Data Setup (A)
- [x] Spec `docs/01_data.md` written
- [x] Spec reviewed by Leader
- [x] Code `src/data_setup.py` written (`prepare_splits`, `create_dataloaders`)
- [x] Code `src/data_setup.py` follows spec signatures exactly

### Metrics (B)
- [x] Spec `docs/02_evaluation.md` written
- [x] Spec reviewed by Leader
- [x] Code `src/metrics.py` written (clip_scores, all metrics, mean_predictor, model_summary)

### Baseline (B)
- [x] Code `src/baseline.py` written (TF-IDF + Ridge, grid 84 combos, Val fixed, joblib)

---

## 1.2 Tests (CPU, Synthetic Data)

- [x] `tests/test_data_setup.py` written & PASS (15/15, `-m "not slow"`; S1/S2 chờ review.json thật)
  - [x] Dataset shape [B, 256] for input_ids, attention_mask (D10, D12)
  - [x] Labels float32 (D10, D12)
  - [x] Duplicate text removal works (D15 + `duplicate_text_removed` trong split_stats)
  - [x] split_id stable with same seed (D4, D9)

- [x] `tests/test_metrics.py` written & PASS (part of 30/30, `-m "not slow"`)
  - [x] Perfect prediction: MAE=0, RMSE=0, R²=1
  - [x] Per-star n counts correct
  - [x] Mean predictor works

- [x] `tests/test_baseline.py` written & PASS (part of 30/30, `-m "not slow"`)
  - [x] Output shape (N,)
  - [x] Predictions in [1.0, 5.0]
  - [x] Train fn doesn't accept Test set

---

## 1.3 Verification (Kaggle GPU) — **COMPLETED**

### Data Preparation (A + Leader)
- [x] `review.json` downloaded, version recorded in README
- [x] `prepare_splits()` runs on Kaggle CPU: 60k stratified, 70/15/15
- [x] `train.parquet`, `val.parquet`, `test.parquet` saved to Drive
- [x] `split_stats.json` saved to Drive, **schema valid**

### split_stats.json Schema Validation
- [x] `split_id` (12 chars): f0e6776f8a1a
- [x] `seed`: 42
- [x] `n_samples`: 60000
- [x] `train_size`: 42000, `val_size`: 9001, `test_size`: 8999
- [x] `stars_per_split` per split per star (1-5)
- [x] `stars_distribution_original` per star
- [x] `duplicate_text_removed`: val, test counts
- [x] `user_id_overlap_rate`: train_val, train_test
- [x] `business_id_overlap_rate`: train_val, train_test
- [x] `created_at`: ISO timestamp
- [x] Val/Test **no duplicate text** with Train (after normalization)

### Baseline Evaluation (B)
- [x] Baseline grid search runs on Val (84 combos, CPU, n_jobs=-1)
- [x] Best params selected by Val RMSE
- [x] Baseline Val metrics: MAE, RMSE, R², Pearson, Spearman, per-star MAE/RMSE (with n), Macro-MAE
- [x] Mean predictor Val metrics computed
- [x] **Threshold calculated**: `threshold = 0.85 * RMSE_baseline_val` = 0.6623
- [x] Results saved to `results/preds/baseline_val.pkl`, `baseline_test.csv`

---

## 1.4 Deliverables

- [x] `split_stats.json` on Drive (validated)
- [x] Baseline Val metrics recorded (RMSE=0.7769, MAE=0.5758, Macro-MAE=0.6320)
- [x] Threshold `0.85 × RMSE_baseline_val` = 0.6623 recorded for Milestone 1
- [x] `data_setup.py`, `metrics.py`, `baseline.py` + tests committed
- [x] PRs merged to `develop`

---

## 1.5 AI Log

- [x] `docs/ai_log.md` updated with Phase 1 entries (A, B)

---

## 1.6 Phase Gate: Phase 1 → Phase 2

**PASS Criteria** (Leader approve):
- [x] All tests PASS (pytest CPU)
- [x] `split_stats.json` schema valid, no data leakage (duplicate text removed)
- [x] Baseline Val metrics + threshold computed
- [x] Code merged to `develop`

**Sign-off**:
- Owner A: _______________ / Date: _______________
- Owner B: _______________ / Date: _______________
- Leader: _______________ / Date: _______________