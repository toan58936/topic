# Báo Cáo Trạng Thái Công Việc — Thành Viên E (Dashboard / Error Analysis)
**Người thực hiện:** Thành viên E — Dashboard + Error Analysis
**Ngày đánh giá:** 2026-11-15
**Nhánh:** `feature-cuong` → PR #5 → `develop` (sau đó develop → main qua merge hợp nhất)

---

## 1. Tổng quan trạng thái

| Phần | Module | Trạng thái | Mức hoàn thành |
| :--- | :--- | :---: | :---: |
| **Dashboard** | `app/streamlit_dashboard.py` | ✅ Hoàn thành | 100% |
| **Error Analysis (code)** | `scripts/error_analysis.py` | ✅ Code xong, **chưa chạy** | 100% code / 0% execution |
| Test | `test_dashboard.py`, `test_error_analysis.py` | ✅ 39 pass | 100% |

**Kết luận nhanh:** Dashboard là deliverable **hoàn chỉnh xuất sắc** (đủ FR-1→FR-10, 24 test, không torch). Error Analysis code xong nhưng **execution 0%** — bị chặn bởi gap `n_tokens`/`split_id` của F + cần data thật + gán nhãn người. E là thành viên **duy nhất** làm đúng quy trình PR (→ develop) và có AI log **chính xác**.

---

## 2. Dashboard — `app/streamlit_dashboard.py` (220 dòng)

| FR | Yêu cầu | Trạng thái | Dẫn chứng |
| :--- | :--- | :---: | :--- |
| FR-1 | Đọc CSV, auto-detect `_with_text` | ✅ | `streamlit_dashboard.py:56-72` |
| FR-2 | Bảng so sánh, blank khi thiếu metric | ✅ | `:83-91` |
| FR-3 | Scatter Pred vs True + identity line | ✅ | `:94-109` |
| FR-4 | Histogram error | ✅ | `:112-116` |
| FR-5 | Per-star MAE/RMSE/n + Macro-MAE qua `src.metrics` | ✅ | `:119-133` |
| FR-6 | Filter model/stars/sort | ✅ | `:189-204` |
| FR-7 | Top-10 worst cases + filter | ✅ | `:136-164` |
| FR-8 | Detail view (click → text) | ✅ | `:158-164` |
| FR-9 | Không import torch | ✅ | `test_no_torch_import` |
| FR-10 | Chạy từ repo root | ✅ | `:6-8` |

**Điểm chất lượng:** vega-lite scatter có identity line; per-star tính runtime (khớp báo cáo); **robust khi thiếu `n_tokens`/`split_id`** — chỉ cảnh báo, vẫn chạy (`:180-183`).

---

## 3. Error Analysis — `scripts/error_analysis.py` (189 dòng)

| Thành phần | Trạng thái | Dẫn chứng |
| :--- | :---: | :--- |
| Sample 40 (20 worst + 20 random 4/sao) | ✅ | `:19-54` |
| 5 nhóm annotate + validate | ✅ | `:15, 94-113` |
| Leader cross-check 4 review ≥75% | ✅ | `:57-60, 116-125` |
| `error_analysis.csv` + summary | ✅ | `:67-91, 128-172` |
| **Chạy thực tế (gán nhãn 40 review + cross-check)** | ❌ 0% | Chờ data + annotate người |

**Điểm chất lượng:** seed=42 cố định; tách worst khỏi random pool (40 review duy nhất); **không lộ nhãn E cho Leader** (`:80-84`); chặn sửa dữ liệu gốc (`:103-105`); chặn "Bị cắt" khi `n_tokens≤254` (`:110-112`); agreement <75% → FAIL (`:141-142`).

---

## 4. Test coverage — Đầy đủ (39 test, khớp AI log)

| File | Số test | Nội dung |
| :--- | :---: | :--- |
| `test_dashboard.py` | 24 case | load_data, filter, per-star khớp source, worst-case, charts, validate, AppTest click/chi tiết, no-torch |
| `test_error_analysis.py` | 15 case | 40 unique/4-sao, seed tái lập, thiếu text/strata, no-overwrite, 7 mutation nhãn, crosscheck 75% |

→ AI log Phase 6 ghi **"39 passed"** — **chính xác** (khác D/F bị stale).

---

## 5. Các điểm cần lưu ý

| # | Vấn đề | Chi tiết | Mức độ |
| :-: | :--- | :--- | :---: |
| 1 | **Error Analysis bị chặn bởi F** | `error_analysis.py:20-21` **bắt buộc** `n_tokens` + `split_id` → thiếu sẽ `ValueError`. F chưa xuất 2 cột này → error analysis không chạy được. | 🔴 Chặn |
| 2 | **Error Analysis execution 0%** | Cần: F sửa schema → Leader training → C validate → E gán nhãn 40 review (người) → Leader cross-check. Đúng assignment ("SAU KHI LEADER TRAINING XONG"). | 🟡 Chờ (expected) |
| 3 | **Spec `05_dashboard.md` vẫn "Bản nháp"** | Code đã hiện thực nhưng spec vẫn ghi "Pre-implementation" → nên cập nhật "Đã hiện thực". | 🟡 Tài liệu |
| 4 | **Không có `05_dashboard_tests.md`** | Assignment tham chiếu file này nhưng không tồn tại — E viết test không có test plan chính thức. | 🟢 Ghi chú |
| 5 | **PR #5 → develop (ĐÚNG)** | `feature-cuong` → PR #5 → **develop** (rồi develop → main). E là **duy nhất** merge đúng develop. | ✅ Tốt |

---

## 6. Danh sách hành động cần cải thiện

| # | Việc cần làm | Mức độ | Ghi chú |
| :-: | :--- | :---: | :--- |
| 1 | **Chờ F sửa `predict.py`** (n_tokens + split_id) để Error Analysis chạy được | 🔴 Chặn | Phụ thuộc F |
| 2 | **Chạy Error Analysis sau khi có data thật**: `prepare` → gán nhãn 40 review → Leader cross-check → `summarize` | 🟠 Cao | Cần Leader training + C validate trước |
| 3 | **Cập nhật spec `05_dashboard.md`** sang "Đã hiện thực" + bổ sung `05_dashboard_tests.md` | 🟡 Nên làm | |
| 4 | **Cập nhật AI log** sau khi chạy error analysis thật (hiện Phase 6 đúng, chỉ thiếu bước execution) | 🟡 Nên làm | |
| 5 | **PR/merge dashboard** khi có CSV thật + ảnh demo | 🟢 Thấp | |

---

## 7. Kết luận

- **Dashboard: XUẤT SẮC** — hoàn thành đủ FR-1→FR-10, 24 test, không torch, robust, UI đầy đủ. Dùng được ngay với CSV thật sau khi Leader training.
- **Error Analysis code: HOÀN THÀNH** (15 test) — nhưng **execution 0%**: bị chặn bởi gap `n_tokens`/`split_id` của F + cần data thật + gán nhãn người + Leader cross-check.
- **Điểm sáng:** E là thành viên **duy nhất** đúng quy trình PR (→ develop) và có **AI log chính xác** (39 pass khớp thực tế).
- **Ưu tiên:** (1) F sửa `predict.py` → (2) E chạy error analysis sau khi có data + annotate → (3) cập nhật spec 05 + AI log.