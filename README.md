# Đề tài 7: Text Regression (Yelp)

Dự đoán điểm cảm xúc liên tục (1.0-5.0) từ review bằng Transformer Encoder + Linear regression head.

**Hạ tầng:** Kaggle GPU (T4/P100) để training, Google Drive để lưu dữ liệu và checkpoint, local để dev & test.

Tài liệu: thư mục `docs/`.

---

## Project Structure

| Thư mục | Mô tả |
|---------|-------|
| `src/` | Source code chính: data processing, model building, training, evaluation, prediction |
| `app/` | Streamlit dashboard để trực quan hóa kết quả (chạy local, không cần torch) |
| `docs/` | Dokumentation: Define, Design, specs 01–06, AI log, plans (phase 0–6, implementation, workflow, assignments) |
| `configs/` | Cấu hình YAML cho từng lần chạy (bert, distilbert, roberta, smoke_test) |
| `notebooks/` | Jupyter notebooks cho thử nghiệm (Kaggle runner) |
| `scripts/` | Script validate outputs, sync kết quả Kaggle → Drive |
| `reports/` | Kết quả test (commit GitHub): `test_predictions.csv`, `model_summary.csv` |
| `tests/` | Test files (pytest chạy trên CPU) |
| `data/` | Dữ liệu đã chia (local, không commit): `train.parquet`, `val.parquet`, `test.parquet`, `split_stats.json` |
| `checkpoints/` | Trọng số mô hình (local, không commit): `best_model.pt`, `last.pt` |

---

## Cài đặt môi trường local

```bash
# Tạo venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Cài dependencies
pip install -r requirements.txt

# Chạy test (CPU)
pytest tests/ -v --tb=short
```

---

## Chạy training trên Kaggle

1. Mở Kaggle Notebook (GPU T4/P100), gắn Yelp Dataset làm Input.
2. Clone repo:
   ```bash
   git clone <repo-url> repo
   cd repo
   pip install -q -r requirements.txt
   ```
3. Copy dữ liệu đã chia từ Input Dataset về working:
   ```bash
   cp /kaggle/input/yelp-topic7-splits/*.parquet /kaggle/working/data/
   cp /kaggle/input/yelp-topic7-splits/split_stats.json /kaggle/working/data/
   ```
4. Smoke test:
   ```bash
   python -m src.train --config configs/smoke_test.yaml
   ```
5. Huấn luyện đầy đủ:
   ```bash
   python -m src.train --config configs/train_bert_huber.yaml
   python -m src.train --config configs/train_bert_mse.yaml
   python -m src.train --config configs/train_distilbert.yaml
   python -m src.train --config configs/train_roberta.yaml  # nếu quota còn
   ```
6. Dự đoán Test:
   ```bash
   python -m src.predict --checkpoint /kaggle/working/checkpoints/bert/best_model.pt --output /kaggle/working/results/preds/bert_test.csv
   ```

Kết quả checkpoint và preds lưu tại `/kaggle/working/`, sau đó sync về Google Drive (`drive/project/{checkpoints,results}`).

---

## Chạy dashboard local

```bash
# Cài dependencies dashboard (không cần torch)
python -m pip install -r app/requirements.txt

# Chạy dashboard
python -m streamlit run app/streamlit_dashboard.py
```

Dashboard đọc `reports/test_predictions.csv` và `reports/model_summary.csv` (đã commit GitHub), tùy chọn `test_predictions_with_text.csv` từ Drive/local.

---

## Quy trình làm việc

Xem chi tiết tại:
- `docs/plans/implementation.md` — kế hoạch triển khai chi tiết
- `docs/plans/workflow.md` — Git flow, AI collaboration, phase gates
- `docs/plans/assignments.md` — phân công 7 thành viên

Quy trình cốt lõi: **Spec → Code → Test (CPU) → Smoke test (Kaggle) → Run model (Kaggle) → Check description → Commit**.

---

## Thành viên E — Dashboard & Error Analysis

Hướng dẫn đầy đủ (cài đặt Windows, demo, CSV đầu vào, gán nhãn, cross-check và kiểm thử): [docs/member_e_guide.md](docs/member_e_guide.md).

Chưa có dữ liệu training? Chạy `python -m scripts.create_dashboard_demo`, rồi nhập `demo_reports` ở ô **Thư mục CSV**. Dữ liệu demo được ghi rõ là giả lập, không dùng cho báo cáo kết quả.