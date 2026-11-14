# Phân Tích Đặc Tả Kỹ Thuật Module `src/data_setup.py` (Thành Viên A)
**Người thực hiện:** Võ Phạm Phương Anh  
**Nhánh phát triển:** `feature/data-setup` → `develop`

---

## 1. Phạm vi nhiệm vụ & Tài liệu tham chiếu
Theo phân công của nhóm (`assignments.md`), Thành viên A đảm nhận module đầu tiên trong chuỗi xử lý dữ liệu (`src/data_setup.py`), hoạt động độc lập và không phụ thuộc vào các module khác:
- **Tài liệu đặc tả kỹ thuật (Specs):** `docs/01_data.md` và `docs/01_data_tests.md`.
- **Tệp kiểm thử tự động (Unit Tests):** `tests/test_data_setup.py` (chạy kiểm thử trên CPU với dữ liệu giả lập synthetic data).
- **Lộ trình triển khai:** Phase 0 (tải dữ liệu gốc) và Phase 1 (chạy hàm chia tập dữ liệu `prepare_splits`).

---

## 2. Phân tích chức năng các hàm cốt lõi

| Tên hàm | Nhiệm vụ xử lý | Yêu cầu kỹ thuật & Đầu ra |
| :--- | :--- | :--- |
| **`prepare_splits()`** | Lấy mẫu phân tầng và phân chia tập dữ liệu | - Quy mô mẫu: **60.000 mẫu** lấy mẫu phân tầng (Stratified Sample) theo số sao (`stars`).<br>- Tỷ lệ chia chuẩn: **70% Train / 15% Validation / 15% Test**.<br>- Đảm bảo tính độc lập dữ liệu: văn bản (`text`) trong tập `Val` và `Test` không được trùng lặp với tập `Train`.<br>- Xuất ra 3 tệp dữ liệu chuẩn `.parquet` (`train.parquet`, `val.parquet`, `test.parquet`) và tệp thống kê `split_stats.json`. |
| **`create_dataloaders()`** | Đóng gói dữ liệu thành các lô (Batch) PyTorch | - Đọc dữ liệu từ các tệp `.parquet` đã phân chia.<br>- Khởi tạo các `DataLoader` cho Train, Validation và Test phục vụ quá trình huấn luyện và đánh giá mô hình. |

---

## 3. Tiêu chuẩn nghiệm thu (Deliverables & Verification)
1. **Tính hợp lệ của `split_stats.json`:** Bắt buộc chứa đầy đủ các trường thông tin kiểm chứng gồm `split_id`, phân bố nhãn sao trên từng tập (`stars_per_split`), tỷ lệ chồng chéo (`overlap rates`) và số lượng bản ghi trùng lặp (`duplicate counts`).
2. **Lưu trữ kết quả:** Đầy đủ `train.parquet`, `val.parquet`, `test.parquet` và `split_stats.json` hợp lệ trên Google Drive.
3. **Chất lượng mã nguồn:** Toàn bộ hàm trong `src/data_setup.py` vượt qua kiểm thử `pytest` trên CPU trước khi gộp vào nhánh `develop`.