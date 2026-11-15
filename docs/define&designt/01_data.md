# Đặc tả Thiết kế Module Dữ liệu — `data_setup.py`

> **Tài liệu:** `docs/01_data.md` (Define & Design cho pipeline dữ liệu)
> **Module liên quan:** `src/data_setup.py`
> **Kế hoạch kiểm thử:** `docs/01_data_tests.md` (tách riêng)
> **Phụ thuộc:** `docs/define.md`, `docs/design.md`; `docs/02_model.md` (tokenizer đi kèm backbone), `docs/03_engine_and_utils.md` (huấn luyện, seed) — hai tài liệu này chưa viết
> **Trạng thái:** Đã hiện thực — `src/data_setup.py` + `tests/test_data_setup.py` (15/15 PASS, `-m "not slow"`, 2026-11-09). Chờ chạy `prepare_splits` trên `review.json` thật.

---

## PHẦN I: DEFINE — Xác định Bài toán & Yêu cầu

### 1.1. Bối cảnh & Mục tiêu

Module `data_setup.py` chịu trách nhiệm **biến file `review.json` của Yelp thành ba tập Train/Val/Test cố định, rồi đóng gói thành ba `DataLoader` sẵn sàng cho BERT, RoBERTa và DistilBERT**. Module không chứa mô hình và không chứa vòng lặp huấn luyện.

Mục tiêu:

* Đọc `review.json` theo khối, lấy mẫu `60,000` review phân tầng theo số sao và giữ phân phối sao tự nhiên.
* Chia mẫu thành Train/Val/Test theo `70/15/15`, phân tầng theo sao, tái lập được, ghi ra ba file `parquet` dùng chung cho cả nhóm.
* Tạo `DataLoader` cho từng backbone từ **cùng ba file đó**, chỉ khác tokenizer.

**Đặc điểm dữ liệu đầu vào (Yelp Dataset):**

| Thuộc tính | Giá trị |
| :--- | :--- |
| Nguồn | Yelp Dataset trên Kaggle, file `review.json` (mỗi dòng một đối tượng JSON) |
| Quy mô | Hàng triệu review, dung lượng vài GB (phụ thuộc phiên bản đã tải) |
| Trường dùng | `review_id`, `user_id`, `business_id`, `stars` (1-5), `text` |
| Phân phối nhãn | Lệch về sao cao (nhiều 4-5 sao, ít 2-3 sao) |
| Phân chia gốc | Không có; nhóm tự chia Train/Val/Test |
| Ngôn ngữ | Tiếng Anh |

**Bốn bài toán cốt lõi cần giải quyết:**

1. **File quá lớn để nạp vào RAM.** Colab có bộ nhớ hạn chế, nên phải đọc theo khối và chỉ giữ lại phần mẫu cần dùng.
2. **Lấy mẫu và chia phân tầng, giữ phân phối tự nhiên.** Nếu chia ngẫu nhiên thuần túy, tỉ lệ sao ở ba tập sẽ lệch nhau, nhất là các mức sao hiếm, làm số đo trên Val không dự báo được số đo trên Test. Mẫu cũng phải giữ đúng tỉ lệ sao của dữ liệu gốc để tập Test phản ánh phân phối thật.
3. **Cố định phép chia để so sánh công bằng.** Ba mô hình phải học và được đánh giá trên cùng một phép chia. Nếu mỗi phiên Colab chia lại, kết quả giữa các mô hình không so sánh được và dễ rò rỉ dữ liệu giữa Train và Test. Vì vậy phép chia chạy một lần và lưu thành file.
4. **Tương thích với nhiều backbone và nối được kết quả.** Mỗi backbone có tokenizer riêng; nhãn phải là số thực `float32` cho hàm loss hồi quy; mỗi mẫu phải mang `review_id` để sau này nối điểm dự đoán với nội dung review khi dựng dashboard.

### 1.2. Đầu vào / Đầu ra

| | Mô tả |
| :--- | :--- |
| **Đầu vào** | `prepare_splits`: đường dẫn `review.json`, thư mục ghi, `n_samples`, `ratios`, `seed`. `create_dataloaders`: thư mục chứa ba file `parquet`, `tokenizer`, `batch_size`, `max_length` |
| **Đầu ra** | `train_loader`, `val_loader`, `test_loader`; mỗi batch gồm `input_ids` và `attention_mask` shape `[B, 256]` kiểu `long`, `labels` shape `[B]` kiểu `float32`, `review_id` là danh sách chuỗi |
| **Đầu ra phụ** | Ba file `train.parquet`, `val.parquet`, `test.parquet` và `split_stats.json` |

### 1.3. Yêu cầu chức năng

| Mã | Yêu cầu |
| :--- | :--- |
| FR-1 | Đọc `review.json` theo khối, không nạp cả file vào bộ nhớ. |
| FR-2 | Lấy mẫu `n_samples` review sao cho tỉ lệ từng mức sao xấp xỉ tỉ lệ trong cả file. |
| FR-3 | Chia mẫu thành Train/Val/Test theo `ratios`, **phân tầng theo sao**, tái lập 100% theo `seed`. |
| FR-4 | Làm sạch review trước khi lấy mẫu: (1) bỏ `text` rỗng hoặc chỉ whitespace; (2) **chuẩn hoá văn bản**: `unicodedata.normalize('NFC', text.strip().lower())` rồi `collapse whitespace` (thay nhiều space/tab/newline liên tiếp bằng 1 space); văn bản đã chuẩn hoá dùng cho lấy mẫu, chia tập, **và so sánh trùng lặp** giữa Train/Val/Test. |
| FR-5 | Ghi ba file `parquet` có cùng cột; ghi `split_stats.json` gồm `split_id`, `seed`, `data_version`, số mẫu mỗi tập, số mẫu theo từng mức sao ở mỗi tập (`stars_per_split`), phân phối sao gốc (`stars_distribution_original`), số review trùng văn bản bị loại ở Val/Test (`duplicate_text_removed`), và tỉ lệ `user_id`, `business_id` của Test xuất hiện trong Train (`user_id_overlap_rate`, `business_id_overlap_rate`). |
| FR-6 | Không tự ghi đè ba file đã có, trừ khi người dùng yêu cầu `overwrite=True`. |
| FR-7 | `create_dataloaders` nhận tokenizer từ bên ngoài, trả về ba `DataLoader`; `shuffle=True` chỉ cho Train. |
| FR-8 | Mỗi mẫu mang `review_id`; nhãn là `float32` giữ nguyên thang 1.0-5.0, không chuẩn hoá. |
| FR-9 | Kiểm tra đầu vào hợp lệ và báo lỗi rõ ràng. |

### 1.4. Ràng buộc

* Dữ liệu gốc là `review.json` của Yelp; `n_samples = 60,000` (tối đa `100,000`), `ratios = (0.70, 0.15, 0.15)`, `seed = 42`, `max_length = 256` (theo `docs/define.md`).
* Phép lấy mẫu và chia dùng bộ sinh số ngẫu nhiên **riêng** (không phụ thuộc seed toàn cục do `utils.py` đặt).
* `prepare_splits` chỉ chạy một lần; các phiên huấn luyện chỉ đọc ba file đã chia.
* Ba file `parquet` nằm trên Drive (không đưa vào GitHub).
* `num_workers` mặc định `0` để chạy ổn định trên Google Colab.

### 1.5. Tiêu chí chấp nhận

1. Ba tập không có `review_id` chung; hợp của ba tập đúng bằng mẫu đã lấy.
2. Tỉ lệ từng mức sao ở mỗi tập xấp xỉ tỉ lệ trong cả mẫu (lệch tối đa vài mẫu do phần dư); tỉ lệ trong mẫu xấp xỉ tỉ lệ trong cả file.
3. Mỗi batch có đúng shape và kiểu dữ liệu ở mục 1.2.
4. Cùng `seed` cho cùng kết quả; `seed` khác cho kết quả khác.
5. Không có `text` rỗng trong ba tập; `train_loader` xáo trộn, `val_loader` và `test_loader` không xáo trộn.
6. Bộ test trong `docs/01_data_tests.md` đều đạt.

---

## PHẦN II: DESIGN — Thiết kế Chi tiết

### 2.1. Vị trí trong kiến trúc hệ thống

```text
notebooks/colab_runner.ipynb
        │  chạy prepare_splits(...) một lần
        ▼
src/data_setup.py ──► train.parquet, val.parquet, test.parquet, split_stats.json  (Drive)
        │
        │  mỗi phiên huấn luyện: create_dataloaders(data_dir, tokenizer, ...)
        ▼
(train_loader, val_loader, test_loader)
        ├──► engine.py              (huấn luyện, chọn best checkpoint bằng val_loader)
        └──► metrics.py / predict.py (suy luận trên test_loader, xuất test_predictions.csv)
```

### 2.2. Kiến trúc pipeline dữ liệu

```text
[1. review.json (hàng triệu dòng)]
                 │  đọc theo khối + làm sạch
                 ▼
[2. Lượt 1: đếm số review hợp lệ theo từng mức sao (N_1..N_5)]
                 ▼
[3. Hạn ngạch mẫu m_c cho từng mức sao, tổng = n_samples]
                 ▼
[4. Lượt 2: chọn ngẫu nhiên m_c review mỗi mức sao (rng riêng, seed=42)]
                 ▼
[5. Chia từng mức sao thành Train/Val/Test theo 70/15/15, gộp lại và xáo trộn]
                 ▼
[6. Ghi train.parquet, val.parquet, test.parquet, split_stats.json]
                 │
                 ▼  (các phiên sau, mỗi backbone một tokenizer)
[7. ReviewDataset + DataLoader]
       ├── train_loader (shuffle=True)
       ├── val_loader   (shuffle=False)
       └── test_loader  (shuffle=False)
```

### 2.3. Thiết kế tham số cấu hình

**`prepare_splits`:**

| Tham số | Kiểu | Mặc định | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| `review_path` | `str` | bắt buộc | Đường dẫn tới `review.json`. |
| `output_dir` | `str` | bắt buộc | Thư mục ghi ba file `parquet` và `split_stats.json`. |
| `n_samples` | `int` | `60000` | Số review trong mẫu. |
| `ratios` | `Tuple[float, float, float]` | `(0.70, 0.15, 0.15)` | Tỉ lệ Train, Val, Test; phải dương và cộng bằng 1. |
| `seed` | `int` | `42` | Hạt giống cho lấy mẫu và chia. |
| `chunk_size` | `int` | `100000` | Số dòng đọc mỗi khối. |
| `overwrite` | `bool` | `False` | Cho phép ghi đè file đã có. |

**`create_dataloaders`:**

| Tham số | Kiểu | Mặc định | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| `data_dir` | `str` | bắt buộc | Thư mục chứa ba file `parquet`. |
| `tokenizer` | tokenizer Hugging Face | bắt buộc | Tokenizer của backbone đang dùng. |
| `batch_size` | `int` | `16` | Kích thước lô (đề xuất). |
| `max_length` | `int` | `256` | Số token tối đa mỗi review. |
| `num_workers` | `int` | `0` | Số tiến trình nạp dữ liệu. |
| `seed` | `int` | `42` | Hạt giống cho việc xáo trộn của `train_loader`. |

### 2.4. Thuật toán lấy mẫu và chia phân tầng — `prepare_splits`

**Mục tiêu:** từ `review.json` tạo ba tập cố định, giữ phân phối sao tự nhiên, với số mẫu mỗi tập khớp `ratios`.

**Thuật toán:**

1. **Đếm (lượt 1):** đọc theo khối, làm sạch, đếm số review hợp lệ $N_c$ cho từng mức sao $c \in \{1,\dots,5\}$; $N = \sum N_c$.
2. **Hạn ngạch mẫu:** $q_c = \dfrac{n\_samples \times N_c}{N}$. Gán $m_c = \lfloor q_c \rfloor$. Nếu $\sum m_c < n\_samples$, cộng thêm $1$ cho các mức sao có phần lẻ lớn nhất cho đến khi tổng bằng đúng `n_samples`; khi phần lẻ bằng nhau, ưu tiên mức sao nhỏ hơn. Quy tắc này nằm trong hàm phụ `_allocate_counts(total, weights)` (tên dự kiến), dùng chung cho bước 2 và bước 4.
3. **Chọn mẫu (lượt 2):** với mỗi mức sao, chọn ngẫu nhiên $m_c$ vị trí trong $N_c$ review hợp lệ của mức sao đó bằng bộ sinh riêng `numpy.random.default_rng(seed)`, rồi đọc lại file để lấy đúng các dòng này.
4. **Chia:** với mỗi mức sao, xáo trộn $m_c$ review rồi chia thành Train/Val/Test theo `ratios` bằng cùng quy tắc phần dư của bước 2 (tổng ba phần bằng $m_c$).
5. **Ghi:** gộp theo từng tập, xáo trộn thứ tự dòng trong tập, ghi `parquet` và `split_stats.json`.

**Ví dụ phần dư.** Một mức sao có $m_c = 10$ review, `ratios = (0.7, 0.15, 0.15)`: hạn ngạch $(7.0, 1.5, 1.5)$, phần nguyên $(7, 1, 1)$, tổng $9 < 10$. Hai mức có phần lẻ $0.5$ bằng nhau, nên ưu tiên theo thứ tự Train, Val, Test: Val được cộng thêm $1$. Kết quả $(7, 2, 1)$.

**Với dữ liệu thật:** mỗi tập có xấp xỉ `42,000 / 9,000 / 9,000` review; tổng từng tập có thể lệch vài mẫu do phần dư ở các mức sao.

**Tính độc lập với seed toàn cục:** hàm dùng bộ sinh riêng nên kết quả chia **không phụ thuộc** trạng thái bộ sinh số ngẫu nhiên toàn cục.

### `split_stats.json` Schema

```json
{
  "split_id": "a1b2c3d4e5f6",
  "seed": 42,
  "data_version": "yelp_2024_01_a1b2c3d4",
  "n_samples": 60000,
  "train_size": 42000,
  "val_size": 9000,
  "test_size": 9000,
  "stars_per_split": {"train": {"1": ..., "2": ..., "3": ..., "4": ..., "5": ...}, "val": {...}, "test": {...}},
  "stars_distribution_original": {"1": ..., "2": ..., "3": ..., "4": ..., "5": ...},
  "duplicate_text_removed": {"val": N, "test": N},
  "user_id_overlap_rate": {"train_val": 0.xx, "train_test": 0.xx},
  "business_id_overlap_rate": {"train_val": 0.xx, "train_test": 0.xx},
  "created_at": "2026-11-09T03:15:00Z"
}
```

* `split_id`: 12 ký tự đầu sha256 trên danh sách `review_id` Test đã sắp xếp.
* `data_version`: format `yelp_YYYY_MM_<sha8>` (hash 8 ký tự đầu của file `review.json` gốc).

### Helper Function — `_allocate_counts`

**Chữ ký:**
```python
def _allocate_counts(total: int, weights: list[float]) -> list[int]:
    """
    Phân bổ `total` samples theo trọng số `weights` (tổng weights = 1.0).
    Trả về list[int] độ dài = len(weights), tổng = total.
    
    Quy tắc:
    1. q_i = total * weights[i]
    2. m_i = floor(q_i)
    3. remainder = total - sum(m_i)
    4. Sắp xếp index theo (q_i - m_i) giảm dần, tie-break: index nhỏ hơn ưu tiên
    5. Cộng 1 cho `remainder` index đầu tiên
    """
```

**Ví dụ:**
```python
# total=10, weights=[0.7, 0.15, 0.15]
# q = [7.0, 1.5, 1.5], m = [7, 1, 1], sum=9, remainder=1
# phần lẻ: [0.0, 0.5, 0.5] → index 1, 2 tie → ưu tiên index 1
# kết quả: [7, 2, 1]
```

**Dùng ở 2 chỗ:**
1. Bước 2: Phân bổ `n_samples` cho 5 mức sao (weights = tỷ lệ sao trong dữ liệu gốc)
2. Bước 4: Phân bổ `m_c` review của mỗi mức sao cho Train/Val/Test (weights = ratios)

**Định dạng ba file `parquet`:**

| Cột | Kiểu | Ý nghĩa |
| :--- | :--- | :--- |
| `review_id` | chuỗi | Mã review, dùng để nối kết quả dự đoán với nội dung |
| `user_id` | chuỗi | Giữ lại để đo tỉ lệ trùng người dùng giữa Train và Test |
| `business_id` | chuỗi | Giữ lại với mục đích như trên |
| `text` | chuỗi | Nội dung review đã làm sạch, không rỗng |
| `stars` | số nguyên 1-5 | Nhãn gốc |

### 2.5. Lớp `ReviewDataset`

**Mục tiêu:** lớp kế thừa `torch.utils.data.Dataset`, biến một dòng của file `parquet` thành đầu vào của mô hình.

| Thành phần | Mô tả |
| :--- | :--- |
| `__init__(df, tokenizer, max_length=256)` | Nhận DataFrame đã đọc từ `parquet`, tokenizer, độ dài tối đa |
| `__len__()` | Số dòng của DataFrame |
| `__getitem__(i)` | Token hoá `text` của dòng `i` ngay lúc truy xuất (cắt hoặc đệm tới đúng `max_length`), trả về từ điển |

Từ điển trả về:

| Khóa | Kiểu | Shape | Ghi chú |
| :--- | :--- | :--- | :--- |
| `input_ids` | `torch.long` | `(256,)` | |
| `attention_mask` | `torch.long` | `(256,)` | 1 cho token thật, 0 cho phần đệm |
| `labels` | `torch.float32` | `()` | Số sao 1.0-5.0, không chuẩn hoá |
| `review_id` | chuỗi | | Để nối kết quả dự đoán |

Người viết code tự quyết: token hoá ngay trong `__getitem__` hay token hoá trước cả tập (nhanh hơn khi huấn luyện, tốn bộ nhớ hơn).

### 2.6. Hàm factory khởi tạo DataLoader — `create_dataloaders`

**Chữ ký hàm:**

```python
create_dataloaders(
    data_dir: str,
    tokenizer,
    batch_size: int = 16,
    max_length: int = 256,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]
```

**Luồng tích hợp:**

```text
1. đọc train.parquet, val.parquet, test.parquet từ data_dir
2. tạo 3 ReviewDataset với cùng tokenizer và max_length
3. tạo 3 DataLoader
     train: shuffle=True, dùng torch.Generator().manual_seed(seed)
     val, test: shuffle=False (thứ tự ổn định khi xuất kết quả)
     pin_memory = torch.cuda.is_available()
4. return train_loader, val_loader, test_loader
```

### 2.7. Giao diện hàm và ngoại lệ

| Hàm | Đầu vào | Đầu ra | Ngoại lệ |
| :--- | :--- | :--- | :--- |
| `prepare_splits(...)` | xem mục 2.3 | `dict` (nội dung `split_stats.json`) | `FileNotFoundError`, `FileExistsError`, `ValueError` (xem bên dưới) |
| `ReviewDataset(df, tokenizer, max_length)` | `DataFrame`, tokenizer, `int` | `Dataset` | `ValueError` nếu thiếu cột |
| `create_dataloaders(...)` | xem mục 2.3 | 3 `DataLoader` | `FileNotFoundError` nếu thiếu file; `ValueError` nếu thiếu cột |

`prepare_splits` báo lỗi khi:

* `review_path` không tồn tại: `FileNotFoundError`.
* Ba file đầu ra đã có và `overwrite=False`: `FileExistsError`, kèm thông báo cách xử lý.
* `ratios` có giá trị không dương hoặc không cộng bằng 1; `n_samples <= 0`: `ValueError`.
* File có ít review hợp lệ hơn `n_samples`, hoặc một mức sao không đủ review để mỗi tập có ít nhất 1 mẫu: `ValueError`.
* Thiếu một trong các trường `review_id`, `user_id`, `business_id`, `stars`, `text`: `ValueError`.

### 2.8. Tích hợp với các module khác

| Module | Cách phối hợp |
| :--- | :--- |
| `model_builder.py` | Mỗi backbone có tokenizer riêng; gọi `create_dataloaders` một lần cho mỗi backbone với cùng `data_dir`, nên ba mô hình học và được đánh giá trên cùng phép chia. |
| `engine.py` | `train_loader` để huấn luyện; `val_loader` để chọn checkpoint tốt nhất mỗi epoch; `test_loader` **không** được dùng để chọn mô hình. |
| `metrics.py`, `predict.py` | Dùng `test_loader` (có `review_id`, thứ tự cố định) để suy luận và xuất `test_predictions.csv`. |
| `utils.py` | `set_seed(42)` cố định bộ sinh ngẫu nhiên toàn cục (ảnh hưởng quá trình huấn luyện); `prepare_splits` đã độc lập nhờ bộ sinh riêng. |
| Dashboard | Nối `review_id` trong `test_predictions.csv` với `test.parquet` để lấy nội dung review khi cần hiển thị các ca đoán sai. |

### 2.9. Quyết định thiết kế và điểm mở

| Hạng mục | Quyết định |
| :--- | :--- |
| Chia Train/Val/Test phân tầng theo sao, giữ phân phối tự nhiên | Đã chốt (theo `docs/define.md`) |
| Phép chia chạy một lần, lưu thành file `parquet` | Đề xuất của nhóm |
| Lấy mẫu hai lượt (đếm rồi chọn) | Đề xuất; người viết code có thể chọn cách khác nếu đúng các hành vi ở mục 1.5 |
| Giữ cột `user_id`, `business_id` và ghi `split_stats.json` | Đề xuất, để đo tỉ lệ trùng ở Tuần 1 |
| Giữ cột `text` trong `test.parquet` | Đề xuất: có, vì dashboard cần xem nội dung ca đoán sai |
| Chia theo nhóm người dùng hoặc doanh nghiệp | Chưa làm; chỉ làm nếu `split_stats.json` cho thấy tỉ lệ trùng cao |
