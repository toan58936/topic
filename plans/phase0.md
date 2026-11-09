# Phase 0 Checklist: Khởi tạo Repo & Môi trường

> Repo setup, dev environment, local GPU survey. Owner: Leader (repo), All (local setup), Leader (survey).

---

## 0.1 GitHub Repository Setup (Leader)

- [ ] Tạo **private repository** trên GitHub (organization hoặc personal)
- [ ] Cấu trúc thư mục khởi tạo:
  ```
  project/
  ├── .github/
  │   ├── pull_request_template.md
  │   └── workflows/ci.yml
  ├── docs/
  │   ├── define&designt/
  │   │   ├── Tài liệu Define.md
  │   │   ├── Tài liệu Design.md
  │   │   ├── CHECKLIST.md
  │   │   ├── 01_data.md ... 05_dashboard.md
  │   │   └── *_tests.md
  │   └── ai_log.md
  ├── src/
  │   └── __init__.py (empty)
  ├── tests/
  │   └── fixtures/ (synthetic data for tests)
  ├── configs/
  ├── notebooks/
  ├── app/
  ├── reports/
  ├── plans/
  ├── requirements.txt
  ├── README.md
  └── .gitignore
  ```
- [ ] Cấu hình **Protected Branches**: `main`, `develop`
  - Require PR review (1 approval)
  - Require status checks (pytest CPU)
  - Require linear history (squash merge)
  - No force push
- [ ] Thêm `pull_request_template.md` (xem workflow.md)
- [ ] Thêm GitHub Actions CI: `pytest tests/ -v` trên PR
- [ ] Invite 5 thành viên vào repo với quyền Write

---

## 0.2 Local Development Environment (All Members)

- [ ] Clone repo: `git clone <repo-url>`
- [ ] Tạo virtual environment:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Windows: .venv\Scripts\activate
  ```
- [ ] Cài dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- [ ] Cài pre-commit hooks:
  ```bash
  pre-commit install
  ```
- [ ] Verify pytest chạy:
  ```bash
  pytest tests/ -v --tb=short
  ```
  (Ban đầu chỉ có test fixtures, sẽ thêm test theo phase)

---

## 0.3 Requirements.txt Pinned (Leader chuẩn bị)

- [ ] `requirements.txt` pin exact versions:
  ```
  torch==2.4.0+cu121
  transformers==4.44.0
  accelerate==0.33.0
  scikit-learn==1.5.1
  pandas==2.2.2
  numpy==1.26.4
  scipy==1.13.1
  joblib==1.4.2
  streamlit==1.37.0
  pytest==8.2.2
  pre-commit==3.7.1
  pyyaml==6.0.1
  polars==0.20.15  # for data processing
  ```
- [ ] `app/requirements.txt` (dashboard only, no torch):
  ```
  streamlit==1.37.0
  pandas==2.2.2
  numpy==1.26.4
  scipy==1.13.1
  plotly==5.22.0
  ```

---

## 0.4 Local GPU Survey (Leader)

- [ ] Gửi form cho 5 thành viên:
  - GPU model (RTX 3060 12GB / 3070 8GB / 3080 10GB / 3090 24GB / 4070 12GB / 4090 24GB / other)
  - VRAM
  - CUDA version installed (`nvidia-smi`)
  - PyTorch CUDA working? (`python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`)
  - Sẵn sàng chạy training local 30-60 phút?
- [ ] Tổng hợp kết quả → danh sách fallback runners
- [ ] Nếu ≥2 người có RTX 3070+ → Plan B: chia chạy song song trên local

---

## 0.5 Google Drive Setup (Leader)

- [ ] Tạo folder chung: `drive/project/`
  ```
  drive/project/
  ├── data/               # train.parquet, val.parquet, test.parquet, split_stats.json
  ├── checkpoints/        # bert/, distilbert/, roberta/ (best_model.pt, epoch_*.pt)
  └── results/            # preds/, test_predictions.csv, model_summary.csv, charts
  ```
- [ ] Chia sẻ folder với 6 thành viên (Editor access)
- [ ] Không đặt link công khai (restricted access)

---

## 0.6 Documentation Sync

- [ ] Copy `Tài liệu Define.md`, `Tài liệu Design.md`, `CHECKLIST.md` vào `docs/define&designt/`
- [ ] Copy `implementation.md`, `workflow.md` vào `plans/`
- [ ] Tạo `docs/ai_log.md` với header:
  ```markdown
  # AI Collaboration Log — Đề tài 7 Text Regression (Yelp)

  Format: Phase | Date | Owner | Prompt | Smoke Test | Run Model | Check Description
  ```

---

## 0.7 Deliverables

- [ ] GitHub private repo với structure đúng, protected branches, CI
- [ ] Tất cả 6 thành viên: local env chạy `pytest tests/` PASS
- [ ] `requirements.txt` pinned, `app/requirements.txt` nhẹ
- [ ] Local GPU survey kết quả ghi nhận
- [ ] Google Drive folder structure created, shared
- [ ] Docs synced, `ai_log.md` created

---

## 0.8 AI Log

- [ ] `docs/ai_log.md` updated với Phase 0 entries (Leader)

---

## 0.9 Phase Gate: Phase 0 → Phase 1

**PASS Criteria** (Leader approve):
- [ ] Repo ready, CI passing (pytest on empty tests)
- [ ] All members local env verified
- [ ] Drive folder ready
- [ ] GPU survey done

**Sign-off**:
- Leader: _______________ / Date: _______________