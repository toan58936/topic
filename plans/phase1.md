                    # Phase 1 Checklist: Data & Baseline

> Data setup, metrics, baseline. Owner: A (Data), B (Baseline), Leader (review).

---

## 1.1 Spec & Code

### Data Setup (A)
- [x] Spec `docs/01_data.md` written
- [ ] Spec reviewed by Leader
- [x] Code `src/data_setup.py` written (`prepare_splits`, `create_dataloaders`)
- [x] Code `src/data_setup.py` follows spec signatures exactly

### Metrics (B)
- [ ] Spec `docs/02_evaluation.md` written
- [ ] Spec reviewed by Leader
- [ ] Code `src/metrics.py` written (clip_scores, all metrics, mean_predictor, model_summary)

### Baseline (B)
- [ ] Code `src/baseline.py` written (TF-IDF + Ridge, grid 84 combos, Val fixed, joblib)

---

## 1.2 Tests (CPU, Synthetic Data)

- [x] `tests/test_data_setup.py` written & PASS (15/15, `-m "not slow"`; S1/S2 chờ review.json thật)
  - [x] Dataset shape [B, 256] for input_ids, attention_mask (D10, D12)
  - [x] Labels float32 (D10, D12)
  - [x] Duplicate text removal works (D15 + `duplicate_text_removed` trong split_stats)
  - [x] split_id stable with same seed (D4, D9)

- [ ] `tests/test_metrics.py` written & PASS
  - [ ] Perfect prediction: MAE=0, RMSE=0, R²=1
  - [ ] Per-star n counts correct
  - [ ] Mean predictor works

- [ ] `tests/test_baseline.py` written & PASS
  - [ ] Output shape (N,)
  - [ ] Predictions in [1.0, 5.0]
  - [ ] Train fn doesn't accept Test set

---

## 1.3 Verification (GPU/Colab)

### Data Preparation (A + Leader)
- [ ] `review.json` downloaded, version recorded in README
- [ ] `prepare_splits()` runs on Colab: 60k stratified, 70/15/15
- [ ] `train.parquet`, `val.parquet`, `test.parquet` saved to Drive
- [ ] `split_stats.json` saved to Drive, **schema valid** (see below)

### split_stats.json Schema Validation
- [ ] `split_id` (12 chars)
- [ ] `seed`: 42
- [ ] `n_samples`: 60000
- [ ] `train_size`: 42000, `val_size`: 9000, `test_size`: 9000
- [ ] `stars_per_split` per split per star (1-5)
- [ ] `stars_distribution_original` per star
- [ ] `duplicate_text_removed`: val, test counts
- [ ] `user_id_overlap_rate`: train_val, train_test
- [ ] `business_id_overlap_rate`: train_val, train_test
- [ ] `created_at`: ISO timestamp
- [ ] Val/Test **no duplicate text** with Train (after normalization)

### Baseline Evaluation (B)
- [ ] Baseline grid search runs on Val (84 combos, CPU, n_jobs=-1)
- [ ] Best params selected by Val RMSE
- [ ] Baseline Val metrics: MAE, RMSE, R², Pearson, Spearman, per-star MAE/RMSE (with n), Macro-MAE
- [ ] Mean predictor Val metrics computed
- [ ] **Threshold calculated**: `threshold = 0.85 * RMSE_baseline_val`
- [ ] Results saved to `results/preds/baseline_val.csv`, `baseline_test.csv`

---

## 1.4 Deliverables

- [ ] `split_stats.json` on Drive (validated)
- [ ] Baseline Val metrics recorded
- [ ] Threshold `0.85 × RMSE_baseline_val` recorded for Milestone 1
- [ ] `data_setup.py`, `metrics.py`, `baseline.py` + tests committed
- [ ] PRs merged to `develop`

---

## 1.5 AI Log

- [x] `docs/ai_log.md` updated with Phase 1 entries (A; B pending)

---

## 1.6 Phase Gate: Phase 1 → Phase 2

**PASS Criteria** (Leader approve):
- [ ] All tests PASS (pytest CPU)
- [ ] `split_stats.json` schema valid, no data leakage (duplicate text removed)
- [ ] Baseline Val metrics + threshold computed
- [ ] Code merged to `develop`

**Sign-off**:
- Owner A: _______________ / Date: _______________
- Owner B: _______________ / Date: _______________
- Leader: _______________ / Date: _______________