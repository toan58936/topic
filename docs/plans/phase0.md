# Phase 0 Checklist: Khởi tạo Repo & Môi trường

> Repo setup, dev environment, Kaggle GPU access survey. Owner: Leader (repo), All (local setup), Leader (survey).

---

## 0.1 GitHub Repository Setup (Leader)

- [x] Tạo **private repository** trên GitHub (organization hoặc personal)
- [x] Cấu trúc thư mục khởi tạo:
  ```
  project/
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
  ├── scripts/
  ├── requirements.txt
  ├── README.md
  └── .gitignore
  ```
- [ ] Invite 5 thành viên vào repo với quyền Write

---

## 0.2 Local Development Environment (All Members)

- [x] Clone repo: `git clone <repo-url>`
- [x] Tạo virtual environment:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Windows: .venv\Scripts\activate
  ```
- [x] Cài dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- [x] Verify pytest chạy:
  ```bash
  pytest tests/ -v --tb=short
  ```
  (Ban đầu chỉ có test fixtures, sẽ thêm test theo phase)

---

## 0.3 Requirements.txt Pinned (Leader chuẩn bị)

- [x] `requirements.txt` pin exact versions:
  ```
  torch==2.4.0+cu121
  transformers==4.44.0
  scikit-learn==1.5.1
  pandas==2.2.2
  numpy==1.26.4
  scipy==1.13.1
  joblib==1.4.2
  streamlit==1.37.0
  pytest==8.2.2
  pyyaml==6.0.1
  ```
- [x] `app/requirements.txt` (dashboard only, no torch):
  ```
  streamlit==1.37.0
  pandas==2.2.2
  numpy==1.26.4
  scipy==1.13.1
  plotly==5.22.0
  ```

---

## 0.4 Kaggle GPU Access Survey (Leader)

- [ ] Gửi form cho 5 thành viên:
  - Có tài khoản Kaggle? (có/không)
  - Đã thêm số điện thoại để bật GPU (T4/P100) chưa?
  - Thời gian sẵn sàng chạy training trên Kaggle (giờ/ngày)?
  - Có dùng Google Drive để lưu data/checkpoints không?
- [ ] Tổng hợp kết quả → danh sách fallback runners (Kaggle account)
- [ ] Xác nhận ≥2 thành viên có thể chạy Kaggle GPU để backup lẫn nhau

---

## 0.5 Google Drive Storage Setup (Leader)

- [ ] Tạo folder chung trên Drive: `data/`, `checkpoints/`, `results/`
  ```
  My Drive/YelpTopic/
  ├── data/               # train.parquet, val.parquet, test.parquet, split_stats.json
  ├── checkpoints/        # bert/, distilbert/, roberta/ (best_model.pt, epoch_*.pt)
  └── results/            # preds/, test_predictions.csv, model_summary.csv, charts
  ```
- [ ] `.gitignore` bỏ qua `data/`, `*.pt`, `__pycache__/`, `reports/*_with_text.csv`, `kaggle/`

---

## 0.6 Documentation Sync

- [x] Copy `Tài liệu Define.md`, `Tài liệu Design.md`, `CHECKLIST.md` vào `docs/define&designt/`
- [x] Copy `implementation.md`, `workflow.md` vào `plans/`
- [x] Tạo `docs/ai_log.md` với header:
  ```markdown
  # AI Collaboration Log — Đề tài 7 Text Regression (Yelp)

  Format: Phase | Date | Owner | Prompt | Smoke Test | Run Model | Check Description
  ```

---

## 0.7 Deliverables

- [ ] GitHub private repo với structure đúng
- [x] Tất cả 6 thành viên: local env chạy `pytest tests/` PASS (60/60 not slow)
- [x] `requirements.txt` pinned, `app/requirements.txt` nhẹ
- [ ] Kaggle GPU access kết quả ghi nhận (≥2 thành viên)
- [ ] Google Drive storage folder created (`data/`, `checkpoints/`, `results/`)
- [x] Docs synced, `ai_log.md` created

---

## 0.8 AI Log

- [x] `docs/ai_log.md` updated với Phase 0 entries (Leader)

---

## 0.9 Phase Gate: Phase 0 → Phase 1

**PASS Criteria** (Leader approve):
- [ ] Repo ready
- [x] All members local env verified
- [ ] Kaggle GPU access confirmed
- [ ] Drive storage ready

**Sign-off**:
- Leader: _______________ / Date: _______________