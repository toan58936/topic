# Báo Cáo Trạng Thái Công Việc — Thành Viên D (Train Infra Core)
**Người thực hiện:** Thành viên D — Train Infra Core (`utils.py`, `engine.py`)
**Ngày đánh giá:** 2026-11-15
**Nhánh:** `feature/model-training-infra` → `main` (PR #4)

---

## 1. Tổng quan trạng thái

| Module | Trạng thái | Mức hoàn thành |
| :--- | :---: | :---: |
| `src/utils.py` | ✅ Hoàn thành | 100% |
| `src/engine.py` | ✅ Hoàn thành | 100% |
| Test (`test_utils.py`, `test_engine.py`) | ✅ Hoàn thành | 100% |

**Kết luận nhanh:** Đây là module **hoàn chỉnh và chất lượng cao nhất** trong các thành viên đã đánh giá — code khớp spec đến từng FR, test phủ đủ kế hoạch, không có gap code/test. Chỉ còn vài điểm hành chính cần dọn (AI log stale, quy trình PR).

---

## 2. `src/utils.py` — Chi tiết (65 dòng)

| Hàm | Nhiệm vụ | Trạng thái | Dẫn chứng |
| :--- | :--- | :---: | :--- |
| `set_seed(seed, deterministic)` | Seed random/numpy/torch/cuda + cuDNN deterministic | ✅ | `utils.py:15-23` (FR-1) |
| `get_device(prefer)` | auto/cpu/cuda, báo lỗi khi thiếu GPU | ✅ | `utils.py:26-36` (FR-2) |
| `write_atomic(path, write_fn)` | Ghi temp + `os.replace` + cleanup khi lỗi | ✅ | `utils.py:39-50` (FR-3) |
| `save_checkpoint(path, state)` | `torch.save` qua write_atomic, check dict | ✅ | `utils.py:53-57` (FR-4) |
| `load_checkpoint(path, map_location)` | `torch.load(weights_only=True)`, FileNotFoundError | ✅ | `utils.py:60-65` (FR-4) |

---

## 3. `src/engine.py` — Chi tiết (499 dòng)

| Hàm | Nhiệm vụ | Trạng thái | Dẫn chứng |
| :--- | :--- | :---: | :--- |
| `build_loss(name, delta)` | Huber/MSE, validate | ✅ | `engine.py:36-44` (FR-5) |
| `build_optimizer_and_scheduler` | AdamW (no-decay cho ndim≤1) + warmup tuyến tính | ✅ | `engine.py:47-67` (FR-6) |
| `train_one_epoch` | fp16 autocast, GradScaler, grad clip, `out.float()` trước loss, `FloatingPointError` | ✅ | `engine.py:84-157` (FR-7) |
| `evaluate` | eval, no_grad, trả review_id/y_true/y_pred_raw/loss, khôi phục mode | ✅ | `engine.py:160-198` (FR-8) |
| `fit` | checkpoint/resume/history/run_meta/guards, `generator.manual_seed(seed+epoch)` | ✅ | `engine.py:244-470` (FR-9→FR-13) |
| `compare_runs` | Bảng so sánh + `rel_diff_pct_vs_first` | ✅ | `engine.py:473-499` (FR-14) |

**Điểm chất lượng nổi bật:**
- `train_one_epoch`: chỉ gọi `scheduler.step()` khi batch không bị GradScaler bỏ (tránh lệch lịch) — `engine.py:140-141`.
- `evaluate`: `fp16=None` → auto (chỉ CUDA, **không cảnh báo** trên CPU) — đúng spec §2.5.
- `fit`: `last.pt` xóa khi hoàn tất (trừ `keep_last`), `run_meta.status="completed"`, resume khớp config trừ `RESUME_IGNORE_KEYS` — `engine.py:25-28, 270-311, 454-457`.

---

## 4. Test coverage — Đầy đủ

| File | Số test | Khớp kế hoạch `04_engine_and_utils_tests.md` |
| :--- | :---: | :--- |
| `test_utils.py` | 4 (U1–U4) | ✅ Đủ nhóm A |
| `test_engine.py` | 14 (L1–L7, F1–F6 + F3b) | ✅ Đủ nhóm B, C (+F3b bonus) |

- **U1** seed tái lập, **U3** checkpoint roundtrip + `weights_only=True`, **U4** write_atomic fail an toàn.
- **F2** resume == chạy liền (`test_engine.py:203-257`), **F3** guards (resume/overwrite), **F5** Test không bị chạm (`ExplodingLoader`).

---

## 5. Các điểm cần lưu ý (không phải lỗi code)

| # | Vấn đề | Chi tiết | Mức độ |
| :-: | :--- | :--- | :---: |
| 1 | **`rng_state` trong assignment là ghi nhầm** | Assignment (dòng 90) ghi "checkpoint: ... rng_state", nhưng **spec §2.4 (dòng 228) KHÔNG yêu cầu** `rng_state`; §2.11 ghi rõ "trạng thái ngẫu nhiên dropout không lưu", tái lập bằng `generator.manual_seed(seed+epoch)`. Code **đúng spec** → cần sửa lại assignment cho khớp. | 🟡 Tài liệu |
| 2 | **AI log Phase 2 stale** | Ghi "11/11 pass (output_embeds=True)" — nhưng code hiện tại **không có `output_embeds`**, và test thực tế là 4 (utils) + 14 (engine), không phải 11/11. Entry này thuộc phiên bản code cũ. | 🟡 Tài liệu |
| 3 | **PR #4 → main (không develop)** | Branch `feature/model-training-infra` (khác `feature/training-infra-core` trong assignment); merge vào main thay vì develop. | 🟡 Quy trình |
| 4 | **Không chạy lại được pytest** | Môi trường thiếu torch → kết quả "pass" theo AI log, chưa xác minh độc lập. | 🟢 Chờ môi trường |

---

## 6. Danh sách hành động cần cải thiện

| # | Việc cần làm | Mức độ | Ghi chú |
| :-: | :--- | :---: | :--- |
| 1 | **Sửa `assignments.md`**: bỏ `rng_state` khỏi mô tả checkpoint (spec §2.4 không yêu cầu) | 🟡 Nên làm | Tránh hiểu nhầm khi review |
| 2 | **Cập nhật AI log Phase 2**: sửa số test thực tế (4 + 14) + bỏ `output_embeds` (không còn tồn tại) | 🟡 Nên làm | |
| 3 | **Xác minh lại pytest** khi có torch (Colab/Linux) để xác nhận 18/18 pass | 🟢 Khi có môi trường | |
| 4 | **Thống nhất tên nhánh/PR** với quy trình `feature/training-infra-core` → `develop` | 🟢 Thấp | Áp dụng cho các PR sau |

---

## 7. Kết luận

- **Code + test: XUẤT SẮC.** `utils.py` (5 hàm) và `engine.py` (7 hàm, 499 dòng) khớp spec §2.4/§2.5 đến từng FR; test phủ đủ U1–U4, L1–L7, F1–F6.
- **Không có gap code/test** như C (integration) hay B (test rút gọn). Đây là module tham chiếu chất lượng cho cả nhóm.
- **Việc còn lại chỉ là hành chính:** cập nhật AI log, sửa mô tả `rng_state` trong assignment, xác minh lại pytest khi có torch, và tuân thủ quy trình PR đúng (→ develop).