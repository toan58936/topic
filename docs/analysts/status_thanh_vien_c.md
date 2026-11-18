# Báo Cáo Trạng Thái Công Việc — Thành Viên C (Model + Integration Test)
**Người thực hiện:** Thành viên C — Model + Integration Test
**Ngày đánh giá:** 2026-11-18 (cập nhật từ 2026-11-15)
**Nhánh:** `feature/model-builder` → `main` (PR #7); `feature/integration-test` → chưa tạo

---

## 1. Tổng quan trạng thái

| Phần | Module | Trạng thái | Mức hoàn thành |
| :--- | :--- | :---: | :---: |
| **Model** | `src/model_builder.py` + `tests/test_model_builder.py` | ✅ Hoàn thành (code + test) | ~85% |
| **Integration Test** | `tests/test_integration.py`, `scripts/validate_outputs.py`, `tests/test_integration_validation.py` | ✅ **Hoàn thành code + meta-test** | 100% code / 0% real data |

**Kết luận nhanh:** Phần Model đã xong và đúng spec; phần Integration Test **code + meta-test đã hoàn thành** (commit `47df8d4`). Lỗi tích hợp chéo với F (`predict.py` thiếu `n_tokens`/`split_id`) **đã được F sửa** (commit `93875da`). Chỉ còn chờ data thật để chạy integration test thực tế.

---

## 2. Phần Model — Chi tiết

### 2.1 Đã hoàn thành (tốt, đúng spec `docs/define&designt/03_model.md`)

| Hạng mục | Trạng thái | Bằng chứng |
| :--- | :---: | :--- |
| `BACKBONES` (3 backbone: bert/roberta/distilbert) | ✅ | `model_builder.py:10-14` |
| `build_model()` — mean/cls pooling, dropout, Linear head | ✅ | `model_builder.py:76-109` |
| `mean_pool()` — float32, nhân mask, `clamp(min=1.0)` → không NaN | ✅ | `model_builder.py:28-34` |
| `cls_pool()` — lấy vị trí 0 | ✅ | `model_builder.py:37-39` |
| `build_tokenizer()` | ✅ | `model_builder.py:112-115` |
| `count_parameters(trainable_only)` | ✅ | `model_builder.py:118-122` |
| `TextRegressor.forward` — check 2 chiều + cùng shape, `squeeze(-1)` | ✅ | `model_builder.py:58-73` |
| `output_bias_init` | ✅ | `model_builder.py:54-56` |
| Code pass pytest CPU | ✅ **14/14** | AI log Phase 2 |

### 2.2 ⚠️ Vấn đề: bộ test rút gọn so với kế hoạch `docs/tests/03_model_tests.md`

File `tests/test_model_builder.py` chỉ có **9 hàm / 14 case**, trong khi kế hoạch mô tả **M1–M15 + S1–S2 (17 test)**. Đã phủ: M1, M2, M3, M4, M5, M6 (một phần), M9 (một phần), M11, M14.

**Các test còn thiếu (quan trọng):**

| Test thiếu | Vì sao cần | Mức độ |
| :--- | :--- | :---: |
| **M12** — gradients flow | Xác minh model thực sự train được (không detach nhầm) | 🔴 Cao |
| **M15** — overfit batch nhỏ | Xác minh model học được (sanity check) | 🔴 Cao |
| **M13** — dropout train vs eval | Xác minh dropout đúng chế độ | 🟠 Trung bình |
| **M10** — `count_parameters(trainable_only=True)` | Hàm có tham số này nhưng chưa test | 🟠 Trung bình |
| **M7** — batch `[4,256]` từ dataloader | Định dạng đầu vào thực tế | 🟡 Thấp |
| **M8** — head structure (Linear in/out, hidden_size DistilBERT) | Kiểm tra `hidden_size` đọc được cho DistilBERT | 🟡 Thấp |
| **S1, S2** — tokenizer/model thật | Cần mạng (chấp nhận skip, đánh dấu `slow`) | 🟢 Tùy chọn |

---

## 3. Phần Integration Test — **ĐÃ HOÀN THÀNH CODE + META-TEST**

### 3.1 Code đã viết (100%)

| Cần có | Nguồn yêu cầu | Trạng thái | File |
| :--- | :--- | :---: | :--- |
| `scripts/validate_outputs.py` | `06_integration_test.md` §2.3 | ✅ **XONG** | `scripts/validate_outputs.py` |
| `tests/test_integration.py` | `06_integration_test.md` §2.4 | ✅ **XONG** | `tests/test_integration.py` |
| `tests/test_integration_validation.py` (meta-test) | `06_integration_test_tests.md` | ✅ **XONG** | `tests/test_integration_validation.py` |
| Fixtures `valid_results_dir`, `corrupted_results_dir` | `06_integration_test_tests.md` §8 | ✅ **XONG** | `tests/conftest.py` |
| `scripts/sync_results.sh` | Design §3, Phase 0 | ❌ Chưa có | — |

### 3.2 Spec còn "Bản nháp"
`06_integration_test.md` và `06_integration_test_tests.md` đều ghi trạng thái **"Bản nháp trước khi hiện thực"** → chưa được review/duyệt như spec 01–05. **Cần chốt spec sau khi review với Leader.**

### 3.3 ✅ Lỗi tích hợp chéo F ↔ C: **ĐÃ KHẮC PHỤC**

F đã sửa `predict.py` (commit `93875da`):
- `export_test_predictions()`: ghi `split_id` vào `<model>_test.csv` và meta JSON
- `merge_test_predictions()`: thêm cột `n_tokens` (tokenizer BERT không cắt) + `split_id` vào `test_predictions.csv` và `test_predictions_with_text.csv`
- `collect_summary_inputs()`: đọc được `split_id` từ `run_meta.json` và pred meta

### 3.4 Phụ thuộc dữ liệu (chặn phần "real data", KHÔNG chặn phần code)
- `test_predictions.csv`, `model_summary.csv`, `split_stats.json`, `preds/*.csv` — **chưa có** (chờ training Phase 5). → `test_integration.py` (real data) không chạy được.
- **Nhưng** `validate_outputs.py` + meta-test dùng **dữ liệu giả tổng hợp** → **đã viết + test được ngay**.

### 3.5 Quy trình
- Chưa có nhánh `feature/integration-test` (assignment ghi "ưu tiên merge sớm" — chưa thực hiện).
- PR #7 (`feature/model-builder`) merge vào **main** thay vì **develop** (giống B) — sai quy trình assignment.

---

## 4. Danh sách hành động cần cải thiện (theo thứ tự ưu tiên)

| # | Việc cần làm | Mức độ | Ghi chú |
| :-: | :--- | :---: | :--- |
| 1 | **Bổ sung test còn thiếu của Model**: M10, M12, M13, M15 (và M7, M8 nếu có thời gian) | 🟠 Cao | Nâng độ phủ test từ ~60% lên đủ |
| 2 | **Chạy integration test thật** khi có data thật (sau Phase 5 training) | 🟠 Cao | `python -m scripts.validate_outputs --results-dir reports/` + `pytest tests/test_integration.py` |
| 3 | **Chốt spec 06** (chuyển từ "Bản nháp" → "Đã chốt") sau khi review với Leader | 🟡 Nên làm | |
| 4 | **Tạo `scripts/sync_results.sh`** (copy Drive → reports/) | 🟢 Thấp | Thuộc Phase 0, có thể để Leader làm |
| 5 | **Tạo nhánh `feature/integration-test`** và merge vào đúng `develop` (theo quy trình) | 🟢 Thấp | |

---

## 5. Kết luận

- **Model:** Đạt yêu cầu (code đúng spec, 14/14 pass). Điểm cần cải thiện: bổ sung các test M10/M12/M13/M15 để bộ test đủ như kế hoạch.
- **Integration Test:** **Code + meta-test HOÀN THÀNH** (100%). Đã có `validate_outputs.py`, `test_integration.py`, `test_integration_validation.py` với 11 kịch bản corruption (C1–C11). Chỉ chờ data thật để chạy validation thực tế.
- **Lỗi tích hợp chéo F↔C:** **ĐÃ KHẮC PHỤC** — F đã sửa `predict.py` xuất đủ `n_tokens` + `split_id`.
- **Ưu tiên tiếp theo:** (1) Bổ sung test Model còn thiếu; (2) Chạy integration test thật khi có data; (3) Chốt spec 06.