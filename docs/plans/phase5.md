# Phase 5 Checklist: DistilBERT → RoBERTa Training

> Train remaining backbones with chosen loss, produce Test predictions. Owner: D (Training), Leader (quota decision).

---

## 5.1 DistilBERT Training — **COMPLETED**

- [x] Config: `configs/train_distilbert.yaml`
  - backbone: distilbert, pooling: mean, loss: huber, epochs: 3, batch: 16, lr: 2e-5, fp16: true
- [x] Run on Kaggle
- [x] Verify:
  - [x] 3 epochs completed
  - [x] Checkpoints + best_model.pt + history.csv + run_meta.json on `/kaggle/working/checkpoints/distilbert/` + Drive

---

## 5.2 RoBERTa Training (Conditional) — **COMPLETED**

- [x] **Check GPU quota remaining** before starting — sufficient quota
- [x] Config: `configs/train_roberta.yaml` (same setup, backbone: roberta)
- [x] Run on Kaggle
- [x] Verify artifacts on `/kaggle/working/checkpoints/roberta/` + Drive

---

## 5.3 Test Predictions (One-Time Export) — **COMPLETED**

- [x] Run `predict.py` for **each completed model + baseline**:
  ```bash
  python -m src.predict --checkpoint /kaggle/working/checkpoints/bert/best_model.pt --output /kaggle/working/results/preds/bert_test.csv
  python -m src.predict --checkpoint /kaggle/working/checkpoints/distilbert/best_model.pt --output /kaggle/working/results/preds/distilbert_test.csv
  python -m src.predict --checkpoint /kaggle/working/checkpoints/roberta/best_model.pt --output /kaggle/working/results/preds/roberta_test.csv
  python -m src.predict --baseline results/preds/baseline_val.pkl --output /kaggle/working/results/preds/baseline_test.csv
  ```
- [x] Each output: `review_id`, `true_stars`, `pred`, `split_id`
- [x] Sync predictions về Drive (`drive/project/results/preds/`)

---

## 5.4 Merge & Validation — **COMPLETED**

- [x] Merge all predictions into `test_predictions.csv`:
  - [x] Key: `review_id`
  - [x] Columns: `review_id`, `true_stars`, `n_tokens`, `split_id`, `pred_baseline`, `pred_bert`, `pred_distilbert`, `pred_roberta`, `abs_err_*`
  - [x] **Validation**: Stop if any `split_id` mismatch OR `review_id` set mismatch OR `true_stars` mismatch
  - [x] `n_tokens`: from BERT tokenizer (uncut), `> 254` = truncated
- [x] Create `test_predictions_with_text.csv`: join with `test.parquet` for `text` column
- [x] Generate `model_summary.csv`:
  - [x] Rows: mean_predictor, baseline, bert, distilbert, roberta — **fixed order**
  - [x] Columns: all per Define §12 + Design §9.9 (mae, rmse, rmse_raw, r2, pearson, spearman, macro_mae, train_seconds, seconds_per_epoch, params_total, rmse_reduction_vs_baseline_pct, inference_seconds, split_id)
  - [x] All 5 models present with complete metrics

---

## 5.5 Copy to Repo & Commit — **COMPLETED**

- [x] Copy `test_predictions.csv`, `model_summary.csv` from Drive → `reports/`
- [x] `git add reports/` → commit → push
- [x] `test_predictions_with_text.csv` → **NOT committed** (`.gitignore`)

---

## 5.6 Milestone 2 Check (Define Mục 4) — **COMPLETED**

- [x] Compute Test metrics for all models
- [x] Check criteria:
  - [x] `RMSE_BERT_val <= 0.85 * RMSE_baseline_val` (Val-based): 0.5608 <= 0.6623 ✓
  - [x] `MacroMAE_BERT_val <= MacroMAE_baseline_val` (Val-based): 0.4405 <= 0.6320 ✓
  - [x] Report Test metrics for reference (not for pass/fail)
- [x] Record in `ai_log.md`

---

## 5.7 Deliverables — **ALL COMPLETED**

- [x] All model checkpoints on `/kaggle/working/checkpoints/` + Drive
- [x] `test_predictions.csv`, `test_predictions_with_text.csv`, `model_summary.csv` on Drive
- [x] `test_predictions.csv`, `model_summary.csv` committed to `reports/`
- [x] Milestone 2 criteria checked & recorded

---

## 5.8 AI Log

- [x] `docs/ai_log.md` updated with Phase 5 entries (D, Leader)

---

## 5.9 Phase Gate: Phase 5 → Phase 6

**PASS Criteria** (Leader approve):
- [x] DistilBERT trained (RoBERTa trained — quota allowed)
- [x] Test predictions merged & validated
- [x] CSVs on Drive + committed to repo
- [x] `model_summary.csv` Test complete
- [x] Milestone 2 criteria recorded

**Sign-off**:
- Owner D: _______________ / Date: _______________
- Leader: _______________ / Date: _______________