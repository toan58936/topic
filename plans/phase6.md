# Phase 6 Checklist: Dashboard, Error Analysis, Report

> Final deliverables: Dashboard, Error Analysis, Report, AI Log. Owner: E (Dashboard/Error), Leader (Report), All (AI Log).

---

## 6.1 Dashboard (E)

### Development
- [ ] Spec `docs/05_dashboard.md` written & reviewed
- [ ] Code `app/streamlit_dashboard.py` written
- [ ] `tests/test_dashboard.py` written & PASS (CPU, no torch needed)

### Features (per Design §5)
- [ ] Reads `reports/test_predictions.csv` + `reports/model_summary.csv`
- [ ] Auto-detects `test_predictions_with_text.csv` if present
- [ ] **Comparison Table**: mean_predictor, baseline, bert, distilbert, roberta (if exists)
  - Columns: MAE, RMSE, R², Pearson, Spearman, train_seconds, seconds_per_epoch, params_total, rmse_reduction_vs_baseline_pct
- [ ] **Charts**:
  - [ ] Predicted vs True (scatter, identity line)
  - [ ] Error distribution (histogram)
  - [ ] Per-star MAE/RMSE (bar chart with n)
- [ ] **Error Cases Table**: Top 10 worst `abs_err` for selected model, filterable by star rating
- [ ] **Detail View**: Click review → show text (if with_text), true/pred/error
- [ ] **Interactions**: Model select, star filter, sort by error

### Verification
- [ ] Run local: `cd app && streamlit run streamlit_dashboard.py`
- [ ] All charts render, no errors
- [ ] Works with `test_predictions.csv` (no text) AND `test_predictions_with_text.csv`
- [ ] Screenshots saved for report backup

---

## 6.2 Error Analysis (E + Leader Cross-check)

### Preparation
- [ ] Ensure `test_predictions_with_text.csv` exists on Drive/local
- [ ] Identify **best model** = lowest RMSE on Test (from `model_summary.csv`)

### Sampling (40 reviews total)
- [ ] **20 Worst-case**: Top 20 `abs_err` of best model on Test
- [ ] **20 Random stratified**: 4 reviews per star (1-5) × 5 stars = 20, seed=42
- [ ] Export `error_sample.csv`: `review_id`, `model`, `abs_error`, `true_stars`, `pred`

### Annotation (E reads 40 reviews, assigns 1 of 5 groups)
| Group | Definition |
|-------|------------|
| **Nhãn nhiễu** | Nội dung tích cực (good, great, love...) nhưng sao ≤2; hoặc tiêu cực (bad, terrible...) nhưng sao ≥4 |
| **Cảm xúc lẫn lộn** | Có cả khen và chê, từ khóa chuyển tiếp: but, however, although, though |
| **Bị cắt 256 token** | `n_tokens > 254` (đã trừ [CLS] [SEP]) |
| **Mỉa mai** | Ngữ nghĩa đảo ngược so với từ khóa, cần đọc hiểu context |
| **Khác** | Không rơi 4 nhóm trên, ghi note cụ thể |

- [ ] E fills `error_group` + `notes` in spreadsheet
- [ ] Export `reports/error_analysis.csv`: `review_id,model,abs_error,true_stars,pred,error_group,notes`

### Leader Cross-check (10% = 4 reviews)
- [ ] Leader independently annotates 4 random reviews from the 40
- [ ] Agreement ≥ 75% (3/4) → PASS
- [ ] If < 75%: discuss, re-annotate disputed, repeat

### Summary for Report
- [ ] Compute % per error group
- [ ] Write error analysis section in report

---

## 6.3 Report (Leader + All)

### Structure (per Define Mục 13, Yêu cầu gốc)
- [ ] **Vấn đề & Phạm vi**: Business problem, scope, constraints
- [ ] **Dữ liệu**: Source, 60k stratified, split 70/15/15, `split_stats.json` summary, limitations (label noise, user/business overlap, no language filter)
- [ ] **Phương pháp**: Baseline TF-IDF+Ridge, BERT/DistilBERT/RoBERTa + Linear head, Mean pooling, Huber/MSE, training config (3 epochs, batch 16, lr 2e-5, fp16)
- [ ] **Kết quả**: 
  - Val metrics (Huber vs MSE, loss rule)
  - Test metrics table (MAE, RMSE, R², Pearson, Spearman, per-star, Macro-MAE)
  - Cost: train_seconds, seconds_per_epoch, params_total
  - rmse_reduction_vs_baseline_pct
- [ ] **Phân tích lỗi**: 40 reviews, 5 groups, percentages, examples
- [ ] **Giới hạn** (bắt buộc): Integer labels, single seed, 3 checkpoints not pure architectures, skewed distribution, label noise, user/business overlap, non-English reviews, Mean pooling theoretical not empirical, near-reproducibility
- [ ] **Nhật ký AI**: Appendix hoặc link `docs/ai_log.md`

### Writing
- [ ] Leader coordinates, assigns sections
- [ ] All members contribute their module sections
- [ ] Leader compiles, edits, formats
- [ ] PDF export

---

## 6.4 AI Log Finalization

- [ ] `docs/ai_log.md` complete for all 6 phases
- [ ] Format: Phase, Date, Owner, Prompt, Smoke Test, Run Model, Check Description (per workflow.md §6)
- [ ] Leader verifies completeness

---

## 6.5 Demo Preparation

- [ ] Dashboard running on presenter machine
- [ ] `test_predictions_with_text.csv` available locally (for detail view)
- [ ] Screenshots backup (in case network issues)
- [ ] 5-min demo script: Problem → Data → Models → Results → Error Analysis → Limitations

---

## 6.6 Deliverables

- [ ] `app/streamlit_dashboard.py` + tests committed
- [ ] Dashboard runs locally, reads `reports/` CSVs
- [ ] `reports/error_analysis.csv` (40 reviews, 5 groups)
- [ ] Report PDF
- [ ] `docs/ai_log.md` complete
- [ ] Demo ready

---

## 6.7 AI Log

- [ ] `docs/ai_log.md` updated with Phase 6 entries (E, Leader, All)

---

## 6.8 Phase Gate: Phase 6 → DONE

**PASS Criteria** (Leader approve):
- [ ] Dashboard functional, all features work
- [ ] Error analysis: 40 reviews annotated, Leader cross-check ≥75% agreement
- [ ] Report PDF complete with all required sections
- [ ] AI log complete for all phases
- [ ] Demo rehearsed

**Sign-off**:
- Owner E: _______________ / Date: _______________
- Leader: _______________ / Date: _______________
- All members: _______________ / Date: _______________

---

> **Project Complete** 🎉