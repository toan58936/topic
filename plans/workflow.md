# Workflow & AI Collaboration Guide — Đề tài 7 Text Regression (Yelp)

> Quy trình làm việc chuẩn cho toàn nhóm: Git/GitHub, AI collaboration, Phase gates, Definition of Done.

---

## 1. Git & GitHub Workflow

### 1.1 Branching Model
```
main (protected)
  │
  ├── develop (integration branch, PR target)
  │     │
  │     ├── feature/data-setup (A)
  │     ├── feature/metrics-baseline (B)
  │     ├── feature/model-builder (C)
  │     ├── feature/training-infra (D)
  │     └── feature/dashboard (E)
  │
  └── release/v1.0 (tag khi Mốc 3 done)
```

### 1.2 Branch Naming
| Loại | Format | Ví dụ |
|------|--------|-------|
| Feature | `feature/<module>-<short-desc>` | `feature/data-setup-prepare-splits` |
| Bugfix | `fix/<module>-<issue>` | `fix/engine-fp16-nan` |
| Docs | `docs/<scope>` | `docs/update-define-scope` |
| Config | `config/<purpose>` | `config/train-bert-huber` |

### 1.3 Commit Convention (Conventional Commits)
```
<type>(<scope>): <subject>

<body> (optional)

<footer> (optional)
```

| Type | Khi nào dùng |
|------|--------------|
| `feat` | Tính năng mới (module, function) |
| `fix` | Sửa bug |
| `refactor` | Refactor code không đổi behavior |
| `docs` | Chỉ sửa docs/spec |
| `test` | Thêm/sửa test |
| `chore` | Config, deps, CI, repo setup |
| `perf` | Cải thiện tốc độ/bo nhớ |

**Ví dụ**:
```
feat(data): add prepare_splits with stratified sampling

- Read review.json in chunks
- Stratified sample 60k by stars
- Split 70/15/15, remove duplicate text val/test vs train
- Save parquet + split_stats.json to Drive

Closes #1
```

### 1.4 Pull Request Process
1. **Tạo PR** từ feature branch → `develop`
2. **PR Template** (`.github/pull_request_template.md`):
   - Mô tả thay đổi
   - Link spec (docs/NN_*.md)
   - Test results (pytest output, smoke test log)
   - Checklist DoD (xem Phase checklist)
3. **Review**: Leader + 1 thành viên khác (cross-module)
4. **Require**: CI pass (pytest CPU), no merge conflicts
5. **Squash merge** vào `develop`
6. **Delete branch** sau merge

### 1.5 Protected Branch Rules (`main`, `develop`)
- Require PR review (1 approval)
- Require status checks (pytest CPU)
- Require linear history (squash merge)
- No force push

---

## 2. AI Collaboration Workflow (4 Bước)

> Theo quy trình giảng viên: **AI gen code → Smoke test → Run model → Check description**

### 2.1 Bước 1: AI Gen Code
**Input cho AI**:
- Spec file: `docs/NN_*.md` (chữ ký hàm, acceptance criteria)
- Define + Design (context)
- Related module specs (dependencies)
- Current codebase structure (`src/`, `tests/`)

**Prompt template**:
```
Bạn là senior ML engineer. Viết module `<module_name>` theo spec dưới đây.

SPEC: <đọc docs/NN_*.md>
CONTEXT: Define.md Mục X, Design.md Mục Y
DEPENDENCIES: <list imports từ modules đã xong>
REQUIREMENTS:
- Chỉ viết code trong src/<module>.py
- Tuân thủ chữ ký hàm, type hints, docstring
- Không import module chưa có trong spec
- Pass pytest CPU với synthetic data
- Không hardcode path, dùng config/argument

Hãy xuất code hoàn chỉnh.
```

**Output**: Code file + test file (nếu spec yêu cầu)

### 2.2 Bước 2: Smoke Test (CPU + Tiny GPU)
```bash
# Local CPU
pytest tests/test_<module>.py -v --tb=short

# Colab GPU (8 samples, 1 epoch)
python -m src.train --config configs/smoke_test.yaml --epochs 1 --max-steps 8
```

**Pass criteria**:
- ✅ Không lỗi import/runtime
- ✅ Mỗi review ra 1 số thực
- ✅ Loss hợp lệ (không NaN/Inf)
- ✅ Checkpoint ghi được (Drive/Colab local)

### 2.3 Bước 3: Run Model (Full GPU)
```bash
# Colab
python -m src.train --config configs/train_bert_huber.yaml
```

**Pass criteria**:
- ✅ Chạy hoàn tất 3 epochs
- ✅ Checkpoint mỗi epoch lên Drive
- ✅ `history.csv` có train_loss, val_rmse, val_mae, epoch_seconds
- ✅ `best_model.pt` là epoch có Val RMSE thấp nhất

### 2.4 Bước 4: Check Description (Đối chiếu Define)
**Verify checklist**:
- [ ] Điểm dự đoán ∈ [1.0, 5.0] (clip áp dụng)
- [ ] `seed = 42` cố định
- [ ] `max_length = 256` token
- [ ] Checkpoint mỗi epoch
- [ ] `split_id` khớp giữa config & predictions
- [ ] Đủ chỉ số: MAE, RMSE, R², Pearson, Spearman, per-star, Macro-MAE
- [ ] Có mean predictor & baseline
- [ ] Std dự đoán Val ≥ 0.5 × Std nhãn Val
- [ ] Loss Val giảm sau epoch 1
- [ ] RMSE Val tốt hơn mean predictor & baseline

**Nếu FAIL**: Quay lại Bước 1 (AI fix) → lặp lại 2-3-4

---

## 3. Phase Gate & Definition of Done

Mỗi Phase có file checklist riêng: `plans/phase1.md`, `phase2.md`... 

### 3.1 Phase Gate Rules
| Gate | Điều kiện PASS | Ai approve |
|------|----------------|------------|
| **Phase 0 → 1** | Repo tạo xong, structure đúng, pre-commit hook chạy | Leader |
| **Phase 1 → 2** | `split_stats.json` valid, Baseline Val metrics có, ngưỡng tính được, 01-02 pass pytest | Leader |
| **Phase 2 → 3** | 03-04 pass pytest CPU, 3 backbone build được tiny config | Leader |
| **Phase 3 → 4** | Smoke test pass, short GPU run có `seconds_per_epoch`, VRAM < 14GB | Leader, D |
| **Phase 4 → 5** | BERT Huber/MSE xong, loss rule áp dụng, quyết định ghi `ai_log.md` | Leader |
| **Phase 5 → 6** | DistilBERT xong (RoBERTa nếu quota), Test predictions merge valid, `model_summary.csv` Test | Leader |
| **Phase 6 → Done** | Dashboard chạy, error analysis 40 review, báo cáo PDF, `ai_log.md` | Leader |

### 3.2 Checklist Format (mỗi phase file)
```markdown
## Phase X: <Tên>

### Spec & Code
- [ ] Spec `docs/NN_X.md` written & reviewed
- [ ] Code `src/<module>.py` written
- [ ] Test `tests/test_<module>.py` written

### Verification
- [ ] `pytest tests/test_<module>.py` PASS (CPU)
- [ ] Smoke test PASS (Colab GPU, 8 samples)
- [ ] Short GPU run PASS (100 steps, metrics logged)

### Deliverables
- [ ] <Artifact 1> (ví dụ: `split_stats.json` trên Drive)
- [ ] <Artifact 2>
- [ ] PR merged to `develop`

### AI Log
- [ ] `docs/ai_log.md` updated cho phase này

### Sign-off
- [ ] Owner: <name> / <date>
- [ ] Leader: <name> / <date>
```

---

## 4. File Structure Reference

```
project/
├── .github/
│   ├── pull_request_template.md
│   └── workflows/ci.yml          # pytest CPU on PR
├── docs/
│   ├── define&designt/
│   │   ├── Tài liệu Define.md
│   │   ├── Tài liệu Design.md
│   │   ├── CHECKLIST.md
│   │   ├── 01_data.md ... 05_dashboard.md
│   │   └── *_tests.md
│   └── ai_log.md                 # Running log: phase, prompt, smoke, run, check
├── src/
│   ├── data_setup.py
│   ├── metrics.py
│   ├── baseline.py
│   ├── model_builder.py
│   ├── utils.py
│   ├── engine.py
│   ├── train.py
│   └── predict.py
├── tests/
│   ├── test_data_setup.py
│   ├── test_metrics.py
│   ├── test_baseline.py
│   ├── test_model_builder.py
│   ├── test_utils.py
│   ├── test_engine.py
│   ├── test_train.py
│   └── test_predict.py
├── configs/
│   ├── train_bert_huber.yaml
│   ├── train_bert_mse.yaml
│   ├── train_distilbert.yaml
│   ├── train_roberta.yaml
│   └── smoke_test.yaml
├── notebooks/
│   └── colab_runner.ipynb
├── app/
│   ├── streamlit_dashboard.py
│   └── requirements.txt
├── reports/                      # committed: test_predictions.csv, model_summary.csv
├── plans/
│   ├── implementation.md         # Master plan (Phase 0-6)
│   ├── workflow.md               # This file
│   ├── phase1.md                 # Phase 1 checklist
│   ├── phase2.md
│   ├── phase3.md
│   ├── phase4.md
│   ├── phase5.md
│   └── phase6.md
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 5. Daily Standup Template (Optional)

Mỗi ngày Leader cập nhật board:

| Phase | Owner | Status | Blocker | Next |
|-------|-------|--------|---------|------|
| Phase 1 | A | Spec done, coding data_setup | Data download chậm | Finish code, run prepare_splits |
| Phase 2 | C | Spec review pending | Waiting Leader | Leader review spec today |

---

## 6. AI Log Format (`docs/ai_log.md`)

Mỗi entry một dòng:

```
## Phase 1 - 2026-11-09 - A (Data)

### Prompt
> Viết data_setup.py với prepare_splits(), create_dataloaders()... [link spec]

### Smoke Test
- pytest: PASS (12/12)
- Colab 8 samples: PASS, loss=1.23, checkpoint saved

### Run Model
- prepare_splits(): 60k sampled, split 42k/9k/9k, split_stats.json uploaded
- Time: 4m 32s

### Check Description
- split_stats.json schema: VALID
- Duplicate text removed: val=12, test=8
- User overlap: train_val=0.03, train_test=0.02
- ✅ PASS Phase 1 gate
```

---

## 7. Quick Commands Reference

```bash
# Setup local
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install

# Run tests (CPU)
pytest tests/ -v --tb=short

# Smoke test (Colab)
python -m src.train --config configs/smoke_test.yaml

# Full train (Colab)
python -m src.train --config configs/train_bert_huber.yaml

# Predict Test
python -m src.predict --checkpoint drive/project/checkpoints/bert/best_model.pt --output results/preds/bert_test.csv

# Dashboard (local)
cd app && streamlit run streamlit_dashboard.py

# Git workflow
git checkout -b feature/data-setup-prepare-splits
git add . && git commit -m "feat(data): add prepare_splits"
git push origin feature/data-setup-prepare-splits
# -> tạo PR trên GitHub
```

---

> **Nguyên tắc cốt lõi**: Spec trước → Code sau → Test bắt buộc → AI log đầy đủ → Phase gate mới chuyển tiếp. Không bỏ qua gate.