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

