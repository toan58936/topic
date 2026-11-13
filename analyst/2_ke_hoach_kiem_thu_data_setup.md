# Kế Hoạch Kiểm Thử & Quy Trình Bàn Giao Dữ Liệu (Thành Viên A)
**Người thực hiện:** Võ Phạm Phương Anh  
**Module:** `src/data_setup.py`

---

## 1. Danh mục kịch bản kiểm thử đơn vị (`tests/test_data_setup.py`)
Trước khi gửi yêu cầu gộp mã nguồn vào nhánh `develop`, module `data_setup.py` cần đảm bảo vượt qua các điều kiện kiểm thử trên CPU:

| Mã kiểm thử | Hạng mục kiểm tra | Tiêu chí đạt chuẩn (Expected Outcome) |
| :-: | :--- | :--- |
| **TC-01** | Tổng số lượng mẫu & Tỷ lệ chia | Tổng mẫu đạt 60.000 dòng; tỷ lệ chia chính xác 70% Train (42.000), 15% Val (9.000), 15% Test (9.000). |
| **TC-02** | Tính phân tầng (Stratification) | Tỷ lệ phân bố số sao (`stars`) trên cả 3 tập Train, Val, Test đồng nhất với tập gốc. |
| **TC-03** | Chống rò rỉ dữ liệu (Zero Overlap) | Giao của tập văn bản (`text`) giữa `Train` với `Val` và `Train` với `Test` bằng rỗng (`overlap rate == 0`). |
| **TC-04** | Định dạng tệp đầu ra | Xuất thành công `train.parquet`, `val.parquet`, `test.parquet` và `split_stats.json` đúng cấu trúc Schema. |
| **TC-05** | Kiểm thử `create_dataloaders()` | Trả về các đối tượng `DataLoader` hợp lệ, đúng kích thước batch và chạy ổn định trên CPU với synthetic data. |

---

## 2. Nhật ký thực hiện trên GitHub
- Khởi tạo nhánh tính năng `feature/data-setup` tách biệt với nhánh `develop` và `main`.
- Đọc hiểu toàn bộ đặc tả trong `docs/01_data.md`, `docs/01_data_tests.md` và `assignments.md`.
- Hoàn thiện bộ tài liệu phân tích và kế hoạch kiểm thử trong thư mục `analyst/`.