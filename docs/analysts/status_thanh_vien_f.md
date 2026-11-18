# Báo Cáo Trạng Thái Công Việc — Thành Viên F (Train CLI)
**Người thực hiện:** Thành viên F — Train CLI (`train.py`, `predict.py`)
**Ngày đánh giá:** 2026-11-18 (cập nhật từ 2026-11-15)
**Nhánh:** Nằm trong PR #4 `feature/model-training-infra` → `main` (chưa có nhánh `feature/training-cli` riêng)

---

## 1. Tổng quan trạng thái

| Module | Trạng thái | Mức hoàn thành |
| :--- | :---: | :---: |
| `src/train.py` (CLI huấn luyện) | ✅ Hoàn thành | 100% |
| `src/predict.py` (dự đoán + xuất kết quả) | ✅ **Hoàn thành** (đã fix schema gap) | 100% |
| Test (`test_train.py`, `test_predict.py`) | ✅ Hoàn thành | 100% |

**Kết luận nhanh:** Code + test hoàn thành đúng spec `04_engine_and_utils.md`. **Gap schema nghiêm trọng (`n_tokens`, `split_id`) ĐÃ ĐƯỢC SỬA** (commit `93875da`). `predict.py` giờ xuất đủ cột theo Design §9.9.1 và 06_integration_test.md.

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

## 3. `src/predict.py` — Chi tiết (349 dòng, **đã fix**)

| Hàm | Nhiệm vụ | Trạng thái | Dẫn chứng |
| :--- | :--- | :---: | :--- |
| `predict` | 1 chuỗi→float, list→ndarray, clip 1–5, restore mode | ✅ | `predict.py:70-116` (FR-19) |
| `load_trained_model` | Nạp `best_model.pt`, backbone_config từ checkpoint | ✅ | `predict.py:119-163` (FR-20) |
| `export_test_predictions` | Ghi `<model>_test.csv` + meta, rmse_raw, **có split_id**, no overwrite | ✅ | `predict.py:166-211` (FR-21) |
| `merge_test_predictions` | Ghép + baseline, check review_id/true_stars/**split_id**, **thêm n_tokens + split_id** | ✅ | `predict.py:222-313` (FR-22) |
| `collect_summary_inputs` | Gom số liệu cho model_summary, **đọc được split_id** | ✅ | `predict.py:316-349` (FR-23) |

---

## 4. ✅ VẤN ĐỀ LỚN NHẤT: Schema gap `n_tokens` + `split_id` — **ĐÃ KHẮC PHỤC**

**Commit `93875da`** đã fix:
- Thêm `AutoTokenizer` import, helper `_count_tokens_bert()` (tokenizer BERT, không cắt), `_resolve_split_id()`
- `export_test_predictions()`: param `run_config`, `data_dir`; ghi `split_id` vào CSV và meta JSON
- `merge_test_predictions()`: validate `split_id` khớp giữa các file; insert `n_tokens` (index 2) + `split_id` (index 3) vào `test_predictions.csv` và `_with_text.csv`
- `collect_summary_inputs()`: đọc `split_id` từ `run_meta.json` và pred meta JSON

**Kết quả:** Giờ khớp Design §9.9.1 + 06_integration_test.md §2.2.1/2.2.3. Integration Test của C sẽ PASS trên data thật.

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

---

## 7. Danh sách hành động cần cải thiện

| # | Việc cần làm | Mức độ | Ghi chú |
| :-: | :--- | :---: | :--- |
| 1 | **Cập nhật AI log Phase 2**: sửa số test thực tế (5 + 8), bỏ `output_embeds`/count cũ | 🟡 Nên làm | |
| 2 | **Tạo nhánh `feature/training-cli`** + PR → develop cho các thay đổi tới | 🟡 Nên làm | Theo quy trình |
| 3 | **Xác minh lại pytest** khi có torch (Colab/Linux) | 🟢 Khi có môi trường | |

---

## 8. Kết luận

- **Code + test: HOÀN THÀNH 100%** (đúng spec 04 §2.6/§2.7, 13 test đủ plan). train.py chất lượng tốt (set_seed 2 lần, ép kiểu, validate đầy đủ).
- **Gap schema nghiêm trọng (`n_tokens`/`split_id`) ĐÃ ĐƯỢC SỬA** (commit `93875da`) — không còn chặn Integration Test của C, `collect_summary_inputs` giờ trả `split_id` đúng.
- **Ưu tiên tiếp theo:** Cập nhật AI log, tạo nhánh/PR riêng cho F, xác minh pytest khi có torch.