# Kế hoạch kiểm thử Module Dữ liệu — `tests/test_data_setup.py`

> **Tài liệu:** `docs/01_data_tests.md`
> **Đặc tả được kiểm thử:** `docs/01_data.md` (mục 1.5 Tiêu chí chấp nhận, mục 2.4 đến 2.7)
> **Trạng thái:** Bản nháp trước khi hiện thực

---

## 1. Mục tiêu và nguyên tắc

Bộ test kiểm chứng `data_setup.py` **độc lập** trước khi tích hợp. Test không đánh giá độ chính xác mô hình; test kiểm tra **tính đúng đắn của phép lấy mẫu, chia, làm sạch và đóng gói dữ liệu**.

1. **Nhanh, không cần mạng và không cần file thật:** dùng file `review.json` giả nhỏ (khoảng 300 dòng, phân phối sao lệch về 4-5 sao, có vài dòng `text` rỗng) và tokenizer giả (stub) trả về đúng shape.
2. **Kiểm tra hành vi quan sát được:** so sánh tập `review_id`, shape, kiểu dữ liệu, số đếm; không kiểm tra cách cài đặt bên trong.
3. **Mỗi test một ý chính:** khi test thất bại, tên test cho biết ngay lỗi nằm ở đâu.
4. **Test lỗi dùng `pytest.raises`:** đầu vào sai phải báo đúng loại lỗi.
5. **Test cần file thật đánh dấu `slow`:** tự bỏ qua nếu không có `review.json`.

Dữ liệu giả và tokenizer giả đặt trong `tests/conftest.py`.

## 2. Năm nhóm khía cạnh kiểm thử

| Nhóm | Khía cạnh | Câu hỏi cần trả lời | Test |
| :--- | :--- | :--- | :--- |
| A | Chia phân tầng đúng | Ba tập có tách biệt, đủ số lượng, đúng tỉ lệ sao không? | D1, D2, D3 |
| B | Tái lập và phần dư | Cùng seed có cùng kết quả? Phần dư được phân bổ đúng? | D4, D5 |
| C | Làm sạch và kiểm tra đầu vào | Dữ liệu bẩn được loại? Đầu vào sai bị chặn bằng lỗi rõ ràng? | D6, D7, D8, D9 |
| D | Dataset và DataLoader | Đúng shape, kiểu, cờ xáo trộn, thứ tự ổn định? | D10, D11, D12, D13, D14 |
| S | Dữ liệu thật (chậm) | Trên `review.json` thật, kích thước và tỉ lệ có đúng không? | S1, S2 |

## 3. Mô tả chi tiết từng test

**Nhóm A: Chia phân tầng đúng**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| D1 | `test_d1_disjoint_and_complete` | Chạy `prepare_splits` trên file giả với `n_samples=100`; đọc ba file, chuyển `review_id` thành `set`. | Ba tập đôi một không giao nhau; hợp có đúng `100` phần tử | Một review nằm ở cả hai tập (rò rỉ); mất review. |
| D2 | `test_d2_stratified_per_split` | Đếm số mẫu theo `stars` ở mỗi tập; so với hạn ngạch kỳ vọng từ `ratios`. | Mỗi mức sao ở mỗi tập lệch không quá $1$ mẫu so với kỳ vọng | Chia ngẫu nhiên thuần túy làm lệch tỉ lệ sao giữa các tập. |
| D3 | `test_d3_sample_proportional` | So tỉ lệ sao trong mẫu với tỉ lệ sao trong file giả (đếm các dòng hợp lệ). | Số mẫu mỗi mức sao lệch không quá $1$ so với $n \times N_c / N$ | Lấy mẫu cân bằng theo sao thay vì giữ phân phối tự nhiên. |

**Nhóm B: Tái lập và phần dư**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| D4 | `test_d4_reproducible_by_seed` | Chạy hai lần với `seed=42` (đặt seed toàn cục khác nhau giữa hai lần), thêm một lần `seed=7`. | Hai lần `seed=42` cho DataFrame giống hệt; `seed=7` cho kết quả khác | Dùng bộ sinh toàn cục thay vì bộ sinh riêng. |
| D5 | `test_d5_remainder_distribution` | Gọi hàm phụ `_allocate_counts(10, (0.7, 0.15, 0.15))` (tên dự kiến). | Kết quả `(7, 2, 1)`, tổng đúng $10$ | Phân bổ phần dư sai (tổng lệch, sót phần). |

**Nhóm C: Làm sạch và kiểm tra đầu vào**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| D6 | `test_d6_cleaning` | File giả có dòng `text` rỗng, dòng chỉ gồm khoảng trắng, và dòng có nhiều khoảng trắng liền nhau. | Không còn `text` rỗng trong ba tập; khoảng trắng thừa được chuẩn hoá | Review rỗng lọt vào tập huấn luyện. |
| D7 | `test_d7_invalid_inputs` | Đường dẫn không tồn tại; `ratios=(0.5, 0.3, 0.3)`; `ratios=(0.7, 0.3, 0.0)`; `n_samples=0`; `n_samples` lớn hơn số review hợp lệ. | Lần lượt `FileNotFoundError`, rồi `ValueError` cho bốn trường hợp còn lại | Đầu vào sai lọt qua và tạo tập rỗng hoặc lỗi khó đoán. |
| D8 | `test_d8_no_silent_overwrite` | Chạy hai lần liên tiếp không `overwrite`; chạy thêm một lần với `overwrite=True`. | Lần hai `FileExistsError`; lần ba chạy được | Âm thầm ghi đè phép chia đang dùng. |
| D9 | `test_d9_stats_file` | Đọc `split_stats.json` sau khi chạy. | Có đủ khoá: số mẫu mỗi tập, số mẫu theo mức sao mỗi tập, tỉ lệ trùng `user_id` và `business_id`; tổng khớp với ba file | Thiếu số liệu cần cho việc xác nhận ở Tuần 1. |

**Nhóm D: Dataset và DataLoader**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| D10 | `test_d10_item_shapes_dtypes` | `ReviewDataset` với $5$ dòng giả và tokenizer giả; lấy một phần tử. | `input_ids`, `attention_mask` shape `(256,)` kiểu `long`; `labels` kiểu `float32` bằng số sao; `review_id` là chuỗi | Nhãn là `int` làm hàm loss hồi quy báo lỗi; sai shape. |
| D11 | `test_d11_truncate_and_pad` | Một review dài hơn $256$ token và một review ngắn. | Cả hai ra đúng $256$; review dài có `attention_mask` toàn $1$; review ngắn có phần đệm bằng $0$ | Quên cắt hoặc đệm. |
| D12 | `test_d12_loader_flags_and_batch` | Tạo ba loader từ thư mục dữ liệu giả; kiểm tra `loader.sampler` và lấy một batch. | Train dùng `RandomSampler`; Val và Test dùng `SequentialSampler`; `labels` shape `(B,)` kiểu `float32`; `review_id` là danh sách chuỗi | Val bị xáo trộn làm kết quả không so sánh được. |
| D13 | `test_d13_loader_order_stable` | Duyệt `val_loader` và `test_loader` hai lần liên tiếp trên cùng loader. Với Train: dựng **hai** bộ loader riêng từ cùng thư mục và cùng `seed`, duyệt `train_loader` của mỗi bộ một lần; thêm một bộ thứ ba với `seed` khác. (Không duyệt cùng một `train_loader` hai lần: generator tiến lên sau mỗi lượt nên lượt hai khác lượt một.) | Thứ tự `review_id` của Val và Test giống nhau giữa hai lần; hai bộ Train cùng `seed` cho cùng thứ tự; bộ khác `seed` cho thứ tự khác; hai lượt duyệt liên tiếp của cùng một `train_loader` cho thứ tự khác nhau (xác nhận việc xáo trộn mỗi epoch) | Không nối được dự đoán với `review_id` khi xuất `test_predictions.csv`. |
| D14 | `test_d14_missing_files_or_columns` | Xoá một file `parquet`; ghi một file thiếu cột `stars`. | `FileNotFoundError`; `ValueError` | Lỗi khó hiểu xuất hiện giữa lúc huấn luyện. |

**Nhóm S: Dữ liệu thật (đánh dấu `slow`)**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| S1 | `test_s1_real_file_sizes` | Chạy `prepare_splits` trên `review.json` thật với `n_samples=60000`. | Ba tập xấp xỉ `42,000 / 9,000 / 9,000` (lệch vài mẫu); không có `text` rỗng | Chia sai kích thước trên dữ liệu thật. |
| S2 | `test_s2_star_distribution_matches_file` | So tỉ lệ sao trong mẫu với số đếm từ lượt 1. | Mỗi mức sao lệch không quá $1$ mẫu so với hạn ngạch | Lấy mẫu làm lệch phân phối tự nhiên. |

Nếu không có `review.json` thật, các test nhóm S **bỏ qua** thay vì báo lỗi.

## 4. Hai ví dụ đọc test chi tiết

**Ví dụ 1: D2 (chia phân tầng theo sao)**

```python
def test_d2_stratified_per_split(fake_review_file, tmp_path):
    stats = prepare_splits(fake_review_file, tmp_path, n_samples=100, seed=42)

    for split in ("train", "val", "test"):
        counts = stats["stars_per_split"][split]          # {1: .., 2: .., ..., 5: ..}
        for star in range(1, 6):
            expected = stats["quota"][split][star]        # hạn ngạch tính từ ratios
            assert abs(counts[star] - expected) <= 1
```

Test chỉ dùng file giả nên chạy tức thì. Nếu thuật toán chia ngẫu nhiên thuần túy, số mẫu mỗi mức sao sẽ dao động và test thất bại ở dòng `assert`.

**Ví dụ 2: D10 (shape và kiểu của một mẫu)**

```python
def test_d10_item_shapes_dtypes(fake_df, fake_tokenizer):
    ds = ReviewDataset(fake_df.head(5), fake_tokenizer, max_length=256)
    item = ds[0]

    assert item["input_ids"].shape == (256,) and item["input_ids"].dtype == torch.long
    assert item["attention_mask"].shape == (256,)
    assert item["labels"].dtype == torch.float32
    assert isinstance(item["review_id"], str)
```

Cặp kiểm tra này bắt lỗi thường gặp ở code do AI viết: nhãn để kiểu `int`, hoặc quên đệm về đúng độ dài.

## 5. Cách chạy, ánh xạ tiêu chí chấp nhận và giới hạn

Lệnh chạy:

```bash
pytest tests/test_data_setup.py -v                 # tất cả
pytest tests/test_data_setup.py -v -m "not slow"   # bỏ qua test cần review.json thật
```

Marker `slow` được đăng ký trong `tests/conftest.py`.

Ánh xạ với tiêu chí chấp nhận ở mục 1.5 của `docs/01_data.md`:

| Tiêu chí | Test chứng minh |
| :--- | :--- |
| 1. Ba tập không giao nhau, hợp đúng bằng mẫu | D1 |
| 2. Tỉ lệ sao đúng ở mỗi tập và so với cả file | D2, D3, S2 |
| 3. Batch đúng shape và kiểu | D10, D12 |
| 4. Tái lập theo `seed` | D4 |
| 5. Không có `text` rỗng; cờ xáo trộn đúng | D6, D12 |
| 6. Toàn bộ test đạt | Mọi test ở trên |

Giới hạn của bộ test:

* Không kiểm tra tốc độ đọc file thật, hiệu năng đa tiến trình (`num_workers > 0`) hay `pin_memory` trên GPU.
* Dùng tokenizer giả nên không kiểm tra chất lượng token hoá của tokenizer thật.
* Nhóm S cần file `review.json` thật (vài GB) và thời gian chạy lâu.
