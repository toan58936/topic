# Phase 6 Checklist: Dashboard, Error Analysis, Report

> Final deliverables: Dashboard, Error Analysis, Report, AI Log. Owner: E (Dashboard/Error), Leader (Report), All (AI Log).

---

## 6.1 Dashboard (E) — **CODE COMPLETE, FEATURES PENDING**

### Development

- [ ] Spec `docs/05_dashboard.md` written & reviewed
- [x] Code `app/streamlit_dashboard.py` written
- [x] `tests/test_dashboard.py` written & PASS (CPU, no torch needed) — 39 passed

### Features (per Design §5)

- [x] Reads `reports/test_predictions.csv` + `reports/model_summary.csv`
- [x] Auto-detects `test_predictions_with_text.csv` if present (on Drive/local)
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

## 6.2 Error Analysis (E + Leader Cross-check) — **AUTOMATED via `scripts/error_analysis.py`**

### Workflow (2 commands)

**Step 1: Prepare sample (E runs once)**
```bash
python -m scripts.error_analysis prepare --reports-dir reports
```
- Auto-selects best model (lowest RMSE Test from `model_summary.csv`) → **RoBERTa**
- Samples 40 reviews: 20 worst `abs_err` + 20 random stratified (4/star × 5, seed=42)
- Creates 5 files in `reports/`:
  - `error_sample.csv` — template for E (5 core cols + `error_group`, `notes`)
  - `error_sample_with_text.csv` — full with `text`, `n_tokens`, `truncation_flag` for reading
  - `error_analysis.csv` — E edits this (fills `error_group`, `notes` for 40 rows)
  - `leader_crosscheck_with_text.csv` — 4 reviews for Leader (seed=42), with `leader_error_group`, `leader_notes`
  - `error_analysis_manifest.json` — metadata (model, split_id, review_ids, leader_ids, seed)

**Step 2: Annotation (E + Leader)**
- E reads `error_sample_with_text.csv`, fills `error_group` + `notes` in `error_analysis.csv` (40 rows)
- Leader reads `leader_crosscheck_with_text.csv`, fills `leader_error_group` + `leader_notes` (4 rows)

**Step 3: Validate & Summarize (E runs)**
```bash
python -m scripts.error_analysis summarize --reports-dir reports
```
- Validates: 40 unique reviews, no data tampering, `error_group` ∈ 5 groups, `notes` non-empty, truncation flag consistency
- Cross-check: Leader labels vs E labels on 4 reviews → agreement ≥ 75% (3/4) required
- Outputs:
  - `error_group_summary.csv` — table: group, n, pct, worst_n, random_n
  - `error_analysis_summary.md` — markdown ready for report (table + examples + disclaimer)

### 5 Error Groups (hardcoded in script)
| Group | Definition |
|-------|------------|
| **Nhãn nhiễu** | Positive words (good, great, love...) but stars ≤2; negative words (bad, terrible...) but stars ≥4 |
| **Cảm xúc lẫn lộn** | Mixed sentiment with transition words: but, however, although, though |
| **Bị cắt 256 token** | `n_tokens > 254` (from predictions column) |
| **Mỉa mai** | Semantic reversal, needs context understanding |
| **Khác** | None of above, write specific note |

### Guardrails
- `prepare` fails if output files exist (no overwrite)
- `summarize` fails if data changed since `prepare` (manifest mismatch)
- `summarize` fails if cross-check agreement < 75% (must discuss + re-annotate)
- All validation raises clear `ValueError` with actionable message

---

## 6.3 Report (Leader + All) — **PENDING**

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

## 6.4 AI Log Finalization — **PARTIAL**

- [x] `docs/ai_log.md` complete for Phase 1, 2, 3 (entries exist)
- [ ] `docs/ai_log.md` entries for Phase 4, 5, 6 to be added
- [ ] Format: Phase, Date, Owner, Prompt, Smoke Test, Run Model, Check Description (per workflow.md §6)
- [ ] Leader verifies completeness

---

## 6.5 Demo Preparation — **PENDING**

- [ ] Dashboard running on presenter machine
- [ ] `test_predictions_with_text.csv` available locally (for detail view)
- [ ] Screenshots backup (in case network issues)
- [ ] 5-min demo script: Problem → Data → Models → Results → Error Analysis → Limitations

---

## 6.6 Deliverables — **PENDING**

- [x] `app/streamlit_dashboard.py` + tests committed
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