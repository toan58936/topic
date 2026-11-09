# Phase 5 Checklist: DistilBERT → RoBERTa Training

> Train remaining backbones with chosen loss, produce Test predictions. Owner: D (Training), Leader (quota decision).

---

## 5.1 DistilBERT Training

- [ ] Config: `configs/train_distilbert.yaml`
  - backbone: distilbert, pooling: mean, loss: [chosen from Phase 4], epochs: 3, batch: 16, lr: 2e-5, fp16: true
- [ ] Run on Colab
- [ ] Verify:
  - [ ] 3 epochs completed
  - [ ] Checkpoints + best_model.pt + history.csv + run_meta.json on Drive (`drive/project/checkpoints/distilbert/`)

---

## 5.2 RoBERTa Training (Conditional)

- [ ] **Check GPU quota remaining** before starting
- [ ] If quota sufficient:
  - [ ] Config: `configs/train_roberta.yaml` (same setup, backbone: roberta)
  - [ ] Run on Colab
  - [ ] Verify artifacts on Drive (`drive/project/checkpoints/roberta/`)
- [ ] If quota **insufficient**:
  - [ ] **Skip RoBERTa entirely** (do not reduce epochs/batch for any model)
  - [ ] Document in `ai_log.md`: "RoBERTa skipped due to GPU quota limit"
  - [ ] Leader approves skip decision
  - [ ] Report will compare 2 models (BERT, DistilBERT) — still valid per Define §5

---

## 5.3 Test Predictions (One-Time Export)

- [ ] Run `predict.py` for **each completed model + baseline**:
  ```bash
  python -m src.predict --checkpoint drive/project/checkpoints/bert/best_model.pt --output results/preds/bert_test.csv
  python -m src.predict --checkpoint drive/project/checkpoints/distilbert/best_model.pt --output results/preds/distilbert_test.csv
  # python -m src.predict --checkpoint drive/project/checkpoints/roberta/best_model.pt --output results/preds/roberta_test.csv  # if run
  python -m src.predict --baseline results/preds/baseline_val.pkl --output results/preds/baseline_test.csv
  ```
- [ ] Each output: `review_id`, `true_stars`, `pred`, `split_id`

---

## 5.4 Merge & Validation

- [ ] Merge all predictions into `test_predictions.csv`:
  - [ ] Key: `review_id`
  - [ ] Columns: `review_id`, `true_stars`, `n_tokens`, `split_id`, `pred_baseline`, `pred_bert`, `pred_distilbert`, `pred_roberta` (if exists), `abs_err_*`
  - [ ] **Validation**: Stop if any `split_id` mismatch OR `review_id` set mismatch OR `true_stars` mismatch
  - [ ] `n_tokens`: from BERT tokenizer (uncut), `> 254` = truncated
- [ ] Create `test_predictions_with_text.csv`: join with `test.parquet` for `text` column
- [ ] Generate `model_summary.csv`:
  - [ ] Rows: mean_predictor, baseline, bert, distilbert, roberta (if exists) — **fixed order**
  - [ ] Columns: all per Define §12 + Design §9.9 (mae, rmse, rmse_raw, r2, pearson, spearman, macro_mae, train_seconds, seconds_per_epoch, params_total, rmse_reduction_vs_baseline_pct, inference_seconds, split_id)
  - [ ] Missing model → row exists, metric columns empty string `""`

---

## 5.5 Copy to Repo & Commit

- [ ] Copy `test_predictions.csv`, `model_summary.csv` from Drive → `reports/`
- [ ] `git add reports/` → commit → push
- [ ] `test_predictions_with_text.csv` → **NOT committed** (`.gitignore`)

---

## 5.6 Milestone 2 Check (Define Mục 4)

- [ ] Compute Test metrics for all models
- [ ] Check criteria:
  - [ ] `RMSE_BERT_val <= 0.85 * RMSE_baseline_val` (Val-based, already checked Milestone 1)
  - [ ] `MacroMAE_BERT_val <= MacroMAE_baseline_val` (Val-based)
  - [ ] Report Test metrics for reference (not for pass/fail)
- [ ] Record in `ai_log.md`

---

## 5.7 Deliverables

- [ ] All model checkpoints on Drive
- [ ] `test_predictions.csv`, `test_predictions_with_text.csv`, `model_summary.csv` on Drive
- [ ] `test_predictions.csv`, `model_summary.csv` committed to `reports/`
- [ ] Milestone 2 criteria checked & recorded

---

## 5.8 AI Log

- [ ] `docs/ai_log.md` updated with Phase 5 entries (D, Leader)

---

## 5.9 Phase Gate: Phase 5 → Phase 6

**PASS Criteria** (Leader approve):
- [ ] DistilBERT trained (RoBERTa if quota allowed)
- [ ] Test predictions merged & validated
- [ ] CSVs on Drive + committed to repo
- [ ] `model_summary.csv` Test complete
- [ ] Milestone 2 criteria recorded

**Sign-off**:
- Owner D: _______________ / Date: _______________
- Leader: _______________ / Date: _______________