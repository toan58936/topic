# Báo Cáo Trạng Thái Công Việc — Thành Viên F (Train CLI)
**Người thực hiện:** Thành viên F — Train CLI (`train.py`, `predict.py`)
**Ngày đánh giá:** 2026-11-15
**Nhánh:** Nằm trong PR #4 `feature/model-training-infra` → `main` (chưa có nhánh `feature/training-cli` riêng)

---

## 1. Tổng quan trạng thái

| Module | Trạng thái | Mức hoàn thành |
| :--- | :---: | :---: |
| `src/train.py` (CLI huấn luyện) | ✅ Hoàn thành | 100% |
| `src/predict.py` (dự đoán + xuất kết quả) | ⚠️ Hoàn thành theo spec 04, nhưng **thiếu `n_tokens`/`split_id`** | ~85% |
| Test (`test_train.py`, `test_predict.py`) | ✅ Hoàn thành | 100% |

**Kết luận nhanh:** Code + test hoàn thành đúng spec `04_engine_and_utils.md`. Tuy nhiên tồn tại **gap schema nghiêm trọng** (`n_tokens`, `split_id`) do **xung đột giữa spec 04 §2.7 và Design §9.9.1/06** — chặn Integration Test của C và làm `collect_summary_inputs` trả `split_id=None`.

---

## 2. `src/train.py` — Chi tiết (206 dòng)

| Hàm | Nhiệm vụ | Trạng thái | Dẫn chứng |
| :--- | :--- | :---: | :--- |
| `load_config` | Đọc YAML, defaults, overrides, ép kiểu số, validate | ✅ | `train.py:52-116` (FR-15) |
| `resolve_output_bias` | train_mean (chỉ đọc `train.parquet`), số, null | ✅ | `train.py:119-130` (FR-16) |
| `run_training` | set_seed ×2, build_tokenizer/dataloaders, build_model, config.yaml, fit | ✅ | `train.py:133-171` (FR-17) |
| `main` | CLI `--config/--set/--overwrite/--no-resume`, exit code | ✅ | `train.py:174-202` (FR-18) |

**Điểm chất lượng:**
- `load_config` ép `lr: 2e-5` (PyYAML đọc thành chuỗi) → float — `train.py:75-76`, chặn khoá lạ — `train.py:65-67`.
- `run_training` set_seed **2 lần** (trước `create_dataloaders` + **ngay trước** `build_model`) → head khởi tạo tái lập — `train.py:135, 150`.

---

## 3. `src/predict.py` — Chi tiết (312 dòng)

| Hàm | Nhiệm vụ | Trạng thái | Dẫn chứng |
| :--- | :--- | :---: | :--- |
| `predict` | 1 chuỗi→float, list→ndarray, clip 1–5, restore mode | ✅ | `predict.py:48-94` (FR-19) |
| `load_trained_model` | Nạp `best_model.pt`, backbone_config từ checkpoint | ✅ | `predict.py:97-141` (FR-20) |
| `export_test_predictions` | Ghi `<model>_test.csv` + meta, rmse_raw, no overwrite | ⚠️ Thiếu `split_id` | `predict.py:144-185` (FR-21) |
| `merge_test_predictions` | Ghép + baseline, check review_id/true_stars | ⚠️ Thiếu `n_tokens`/`split_id` | `predict.py:196-276` (FR-22) |
| `collect_summary_inputs` | Gom số liệu cho model_summary | ⚠️ `split_id=None` | `predict.py:279-312` (FR-23) |

---

## 4. 🔴 VẤN ĐỀ LỚN NHẤT: Schema gap `n_tokens` + `split_id` (xung đột spec)

Đây là **xung đột giữa các spec**, không phải lỗi code của F — nhưng F là nơi phải sửa.

| File | Spec 04 §2.7 (F tuân theo) | Design §4/§9.9.1 + 06 (C/dashboard cần) | Thực tế ghi |
| :--- | :--- | :--- | :--- |
| `test_predictions.csv` | `review_id, true_stars, pred_*, abs_err_*` | **+ `n_tokens`, `split_id`** | Thiếu cả 2 |
| `<model>_test.csv` | `review_id, true_stars, pred_raw, pred` | **+ `split_id`** | Thiếu `split_id` |

**Hệ quả cụ thể:**
1. **C (Integration Test) sẽ FAIL** trên data thật — `validate_outputs.py` cần `split_id`/`n_tokens` (`06_integration_test.md` §2.2).
2. **`collect_summary_inputs` trả `split_id=None`** — `run_meta.json` (engine.py) và pred meta json đều **không ghi** `split_id`; `predict.py:293-303` đọc nhưng không tìm thấy → `model_summary.csv` thiếu `split_id`.
3. **Design §4** yêu cầu "predict.py tính n_tokens bằng tokenizer BERT (không cắt)" — spec 04 §2.7 bỏ sót, code không làm.

→ **Lỗi tích hợp chéo F↔C↔D**, cần: (a) chốt spec 04 §2.7 vs Design §9.9.1; (b) F sửa `predict.py` (thêm `n_tokens` + `split_id`); (c) D sửa `engine.py` ghi `split_id` vào `run_meta.json`.

---

## 5. Test coverage — Đầy đủ

| File | Số test | Khớp kế hoạch `04_engine_and_utils_tests.md` |
| :--- | :---: | :--- |
| `test_train.py` | 5 (T1–T5) | ✅ Đủ nhóm D |
| `test_predict.py` | 8 (P1–P8) | ✅ Đủ nhóm E |

- **T1** load_config (ép kiểu `lr`, khoá lạ, 8 case lỗi), **T4** seed đặt trước build_model.
- **P1** clip, **P2** batch-size invariance, **P6** export (rmse_raw, no overwrite), **P7** merge (check review_id/true_stars/baseline).

---

## 6. Các điểm cần lưu ý khác

| # | Vấn đề | Chi tiết | Mức độ |
| :-: | :--- | :--- | :---: |
| 1 | **AI log Phase 2 stale** | Ghi "8/8 (train) + 12/12 (predict)" — thực tế 5 + 8 test. Thuộc phiên bản code cũ. | 🟡 Tài liệu |
| 2 | **Không có nhánh/PR riêng F** | train.py+predict.py nằm trong PR #4 (cấu trúc 5 thành viên cũ, D làm hết). D→F split chỉ là local. | 🟡 Quy trình |
| 3 | **`predict_cli.py` (72 dòng)** | Module CLI dự đoán **ngoài** assignment F — deliverable phát sinh (không có trong spec 04). | 🟢 Ghi chú |
| 4 | **Verification "split_id mismatch"** | Code check được `review_id` mismatch (`predict.py:216-219`), nhưng **không check được `split_id`** (không có cột) → chỉ đáp ứng một nửa. | 🟠 Trung bình |

---

## 7. Danh sách hành động cần cải thiện

| # | Việc cần làm | Mức độ | Ghi chú |
| :-: | :--- | :---: | :--- |
| 1 | **Chốt spec**: thống nhất 04 §2.7 ↔ Design §9.9.1 (quyết định có `n_tokens`/`split_id` hay không) | 🔴 Chặn | Ảnh hưởng C + E + D |
| 2 | **F sửa `predict.py`**: thêm `n_tokens` (tokenizer BERT, không cắt) vào `<model>_test.csv`/`test_predictions.csv`; thêm `split_id` | 🔴 Cao | Sau khi chốt spec |
| 3 | **D sửa `engine.py`**: ghi `split_id` vào `run_meta.json` | 🟠 Cao | Để `collect_summary_inputs` lấy được |
| 4 | **Cập nhật AI log Phase 2**: sửa số test thực tế (5 + 8), bỏ `output_embeds`/count cũ | 🟡 Nên làm | |
| 5 | **Tạo nhánh `feature/training-cli`** + PR → develop cho các thay đổi tới | 🟡 Nên làm | Theo quy trình |
| 6 | **Xác minh lại pytest** khi có torch (Colab/Linux) | 🟢 Khi có môi trường | |

---

## 8. Kết luận

- **Code + test: HOÀN THÀNH** (đúng spec 04 §2.6/§2.7, 13 test đủ plan). train.py chất lượng tốt (set_seed 2 lần, ép kiểu, validate đầy đủ).
- **Gap schema nghiêm trọng** (`n_tokens`/`split_id`) do **xung đột spec 04 §2.7 ↔ Design §9.9.1/06** — chặn Integration Test của C, làm `split_id=None` trong `collect_summary_inputs`, và dashboard thiếu `n_tokens` để phân tích.
- **Ưu tiên số 1:** Chốt spec → F sửa `predict.py` + D sửa `engine.py` → cập nhật AI log. Đây là mắt xích chặn C hoàn thành Integration Test.