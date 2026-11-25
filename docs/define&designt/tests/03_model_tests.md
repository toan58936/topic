# Kế hoạch kiểm thử Module Mô hình — `tests/test_model_builder.py`

> **Tài liệu:** `docs/03_model_tests.md`
> **Đặc tả được kiểm thử:** `docs/03_model.md` (mục 1.5 Tiêu chí chấp nhận, mục 2.3 đến 2.11)
> **Trạng thái:** Bản nháp trước khi hiện thực

---

## 1. Mục tiêu và nguyên tắc

Bộ test kiểm chứng `model_builder.py` **độc lập** với huấn luyện và dữ liệu thật. Test không đánh giá chất lượng mô hình; test kiểm tra **hình dạng đầu ra, phép gộp vector, cấu trúc head, số tham số và tính tái lập**.

1. **Dùng mô hình nhỏ khởi tạo ngẫu nhiên** (`pretrained=False` cùng cấu hình nhỏ trong `tests/conftest.py`): chạy vài giây trên CPU, không cần mạng, không cần GPU.
2. **Ưu tiên ví dụ tính tay** cho phép gộp vector (hàm thuần, không cần backbone).
3. **Chạy cho cả ba backbone** bằng `pytest.mark.parametrize`, vì lỗi thường nằm ở sự khác biệt giữa các backbone.
4. **Kiểm tra hành vi quan sát được** (shape, giá trị, tên thuộc tính trong giao diện), không kiểm tra cách cài đặt bên trong.
5. **Mỗi test một ý chính; test lỗi dùng `pytest.raises`.**
6. **Cố định seed** (`torch.manual_seed`) trong mọi test có ngẫu nhiên.
7. **Test cần mạng hoặc trọng số thật đánh dấu `slow`:** tự bỏ qua khi không có mạng hoặc khi `HF_HUB_OFFLINE=1`.

**Hàm hỗ trợ trong `tests/conftest.py`** (dùng chung với các file test khác, cần thống nhất với `docs/01_data_tests.md` và `docs/02_evaluation_tests.md`):

| Hàm hoặc fixture | Nhiệm vụ |
| :--- | :--- |
| `no_dropout(config)` | Đặt mọi dropout của backbone về 0 và trả về `config`: `hidden_dropout_prob`, `attention_probs_dropout_prob` cho BERT/RoBERTa; `dropout`, `attention_dropout` (và `seq_classif_dropout`) cho DistilBERT. Dùng chung với test 04 (cùng định nghĩa, đặt trong `conftest.py`) |
| `tiny_config(backbone)` | Trả về cấu hình nhỏ: `BertConfig` và `RobertaConfig` với `hidden_size=32`, 2 lớp, 2 đầu chú ý, `intermediate_size=64`, `vocab_size=1000`; `DistilBertConfig` với `dim=32`, `n_layers=2`, `n_heads=2`, `hidden_dim=64`, `vocab_size=1000` |
| `make_batch(backbone, batch_size, seq_len)` | `input_ids` ngẫu nhiên trong `[5, 1000)` và `attention_mask` toàn 1, kiểu `long` |
| `pad_batch(backbone, ids, mask, to_len)` | Thêm token đệm đúng id của backbone (RoBERTa là 1, hai backbone còn lại là 0) vào cuối, `attention_mask` bằng 0 ở phần thêm |

## 2. Năm nhóm khía cạnh kiểm thử

| Nhóm | Khía cạnh | Câu hỏi cần trả lời | Test |
| :--- | :--- | :--- | :--- |
| A | Phép gộp vector | Đúng giá trị tính tay? Phần đệm bị bỏ? Hàng toàn đệm có `NaN` không? | M1, M2, M3 |
| B | Hình dạng và giao diện | Cả ba backbone và hai kiểu gộp cho shape `[B]`? Phần đệm không đổi điểm? Tham số sai báo lỗi? | M4, M5, M6, M7 |
| C | Head và tham số | Head đúng cấu trúc? Pooler đã bỏ? Đếm tham số đúng? Khởi tạo bias đúng? | M8, M9, M10, M11 |
| D | Huấn luyện và tái lập | Gradient chảy? Dropout đúng chế độ? Cùng seed cho cùng khởi tạo? Học được batch nhỏ? | M12, M13, M14, M15 |
| E | Mô hình và tokenizer thật (chậm) | Tokenizer đúng cặp, token đầu đúng? Số tham số và `hidden_size` thật đúng? | S1, S2 |

## 3. Mô tả chi tiết từng test

**Nhóm A: Phép gộp vector**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| M1 | `test_m1_mean_pool_known_values` | `hidden = [[[1,2],[3,4],[100,100]], [[5,6],[9,9],[9,9]]]`, `mask = [[1,1,0],[1,0,0]]`. | `[[2, 3], [5, 6]]` (phần đệm giá trị 100 và 9 bị bỏ) | Quên nhân `attention_mask`; chia cho độ dài chuỗi thay vì số token thật. |
| M2 | `test_m2_mean_pool_zero_mask` | Một hàng có `mask` toàn 0 trong batch hai mẫu. | Hàng đó là vector 0, mọi giá trị hữu hạn, hàng còn lại đúng | Chia cho 0 cho ra `NaN`. |
| M3 | `test_m3_cls_pool_first_token` | `hidden` ngẫu nhiên `[3, 5, 4]`. | `cls_pool(hidden)` bằng `hidden[:, 0]` đúng từng phần tử | Lấy nhầm vị trí hoặc nhầm trục. |

**Nhóm B: Hình dạng và giao diện**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| M4 | `test_m4_forward_shape` | Tham số hoá 3 backbone × 2 kiểu gộp. Batch `batch_size=2`, `seq_len=16`; thêm batch một mẫu. | Shape `(2,)` và `(1,)`; kiểu `float32`; mọi giá trị hữu hạn | Dùng `squeeze()` làm batch một mẫu ra tensor 0 chiều; một backbone không chạy được. |
| M5 | `test_m5_padding_invariance` | Chế độ `eval`. Cùng một batch ở độ dài 8, rồi đệm thêm tới độ dài 16 (`pad_batch`). Tham số hoá 3 backbone × 2 kiểu gộp. | Hai kết quả giống nhau với `atol=1e-5` | Mean pooling không che phần đệm; điểm phụ thuộc vào batch. |
| M6 | `test_m6_registry_and_invalid_args` | Đọc `BACKBONES`; gọi `build_model` với backbone lạ, `pooling="max"`, `dropout=1.0`, `backbone_config` của BERT truyền cho `backbone="roberta"`; gọi `forward` với `input_ids` một chiều và với hai tensor khác shape. | `BACKBONES` có đúng ba khoá, `hf_name` của `bert` là `bert-base-uncased`; mỗi lời gọi sai cho `ValueError` | Tham số sai lọt qua và lỗi muộn khó hiểu; cấu hình lệch backbone. |
| M7 | `test_m7_batch_format_from_dataloader` | Batch giả `[4, 256]` kiểu `long` cho mỗi backbone nhỏ. | Chạy được, đầu ra shape `(4,)` | Độ dài 256 vượt giới hạn vị trí; sai kiểu `long`. |

**Nhóm C: Head và tham số**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| M8 | `test_m8_head_structure` | Dựng mô hình nhỏ cho mỗi backbone; đọc `model.head`, `model.backbone.config.hidden_size`. | `model.head` là `nn.Linear` với `in_features == hidden_size` và `out_features == 1`; `hidden_size` đọc được với cả DistilBERT (là 32 theo cấu hình nhỏ) | Head sai kích thước; `hidden_size` không đọc được ở DistilBERT; tên thuộc tính đổi làm `engine.py` hỏng. |
| M9 | `test_m9_pooler_dropped_and_param_count` | Với `bert` và `roberta` kiểm `model.backbone.pooler is None`; với `distilbert` kiểm không có thuộc tính `pooler`. Mọi backbone: `count_parameters(model)` so với tổng tham số của `backbone` cộng `hidden_size + 1`. | Hai giá trị bằng nhau | Còn pooler không dùng làm lệch số tham số; head có thêm lớp ngoài dự kiến. |
| M10 | `test_m10_count_parameters` | Đếm toàn bộ; sau đó `requires_grad = False` cho head rồi đếm `trainable_only=True`. | Toàn bộ bằng `sum(p.numel())`; số có thể huấn luyện nhỏ hơn đúng `hidden_size + 1` | Đếm sai khi đóng băng; bỏ sót tham số. |
| M11 | `test_m11_output_bias_init` | `build_model(..., output_bias_init=3.7)` và `output_bias_init=None`. | Bias bằng `3.7` ở trường hợp đầu; trường hợp sau bias không bị đặt về `3.7` | Tham số không có tác dụng. |

**Nhóm D: Huấn luyện và tái lập**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| M12 | `test_m12_gradients_flow` | Một lần `forward`, tính MSE với nhãn ngẫu nhiên, `backward`. | Mọi tham số có `requires_grad` có `grad` khác `None` và hữu hạn | Tham số bị cắt khỏi đồ thị tính toán (ví dụ `detach` nhầm); `NaN`. |
| M13 | `test_m13_dropout_train_vs_eval` | `dropout=0.5`. Hai lần `forward` trong `eval`; hai lần trong `train`. | Hai lần ở `eval` giống hệt nhau; hai lần ở `train` khác nhau | Dropout không được áp dụng, hoặc áp dụng cả khi `eval` làm dự đoán không ổn định. |
| M14 | `test_m14_seed_reproducible_init` | `torch.manual_seed(42)` rồi `build_model`, lặp lại; thêm một lần với seed `7`. | Hai `state_dict` seed 42 giống hệt nhau; seed 7 khác | Khởi tạo không tái lập theo seed. |
| M15 | `test_m15_overfit_small_batch` | 8 mẫu cố định, nhãn trong 1-5, `dropout=0` ở head **và** `backbone_config=no_dropout(tiny_config(backbone))` (nếu không, dropout 0.1 của backbone làm loss nhiễu), AdamW (`lr=1e-3`), 100 bước, loss Huber trên đầu ra thô. | Loss cuối nhỏ hơn 0.5 lần loss đầu (đề xuất; hiệu chỉnh ngưỡng ở lần chạy đầu) | Mô hình hoặc đồ thị tính toán không học được gì (ví dụ đầu ra không phụ thuộc đầu vào). |

**Nhóm E: Mô hình và tokenizer thật (đánh dấu `slow`)**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| S1 | `test_s1_tokenizer_special_tokens` | `build_tokenizer` cho từng backbone; mã hoá câu `"great food"`. | Phần tử đầu của `input_ids` bằng `cls_token_id` (là `<s>` với RoBERTa), phần tử cuối bằng `sep_token_id` | `cls` pooling lấy nhầm token vì tokenizer không thêm token đặc biệt. |
| S2 | `test_s2_pretrained_dims_and_param_counts` | `build_model(pretrained=True)` cho từng backbone; một batch hai review thật qua tokenizer. | `hidden_size = 768`; số tham số trong khoảng ±1% của ước tính (BERT 108,9 triệu, RoBERTa 124,1 triệu, DistilBERT 66,4 triệu); đầu ra shape `(2,)`, hữu hạn | Nạp nhầm checkpoint; pooler còn lại; lỗi chỉ lộ ra trên mô hình thật. |

Nếu không có mạng hoặc `HF_HUB_OFFLINE=1`, nhóm E **bỏ qua** thay vì báo lỗi.

## 4. Hai ví dụ đọc test chi tiết

**Ví dụ 1: M1 (giá trị tính tay)**

```python
def test_m1_mean_pool_known_values():
    hidden = torch.tensor([[[1., 2.], [3., 4.], [100., 100.]],
                           [[5., 6.], [9., 9.], [9., 9.]]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    out = mean_pool(hidden, mask)

    expected = torch.tensor([[2., 3.], [5., 6.]])
    assert torch.allclose(out, expected)
```

Mẫu 0 có hai token thật `[1, 2]` và `[3, 4]`, trung bình `[2, 3]`; mẫu 1 chỉ có một token thật `[5, 6]`. Nếu quên nhân mặt nạ, kết quả mẫu 0 thành khoảng `[34.67, 35.33]` và test thất bại ngay.

**Ví dụ 2: M5 (phần đệm không đổi điểm)**

```python
@pytest.mark.parametrize("backbone", ["bert", "roberta", "distilbert"])
@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_m5_padding_invariance(backbone, pooling):
    torch.manual_seed(0)
    model = build_model(backbone, pooling, pretrained=False,
                        backbone_config=tiny_config(backbone)).eval()
    ids8, mask8 = make_batch(backbone, batch_size=2, seq_len=8)
    ids16, mask16 = pad_batch(backbone, ids8, mask8, to_len=16)

    with torch.no_grad():
        a = model(ids8, mask8)
        b = model(ids16, mask16)

    assert torch.allclose(a, b, atol=1e-5)
```

Review ngắn trong batch đệm dài phải cho cùng điểm như khi đứng một mình. Nếu Mean pooling lấy trung bình cả phần đệm thì `a` và `b` khác nhau và test thất bại; đây là lỗi âm thầm nhất của loại mô hình này vì điểm sẽ phụ thuộc vào batch mà review rơi vào.

## 5. Cách chạy, ánh xạ tiêu chí chấp nhận và giới hạn

Lệnh chạy:

```bash
pytest tests/test_model_builder.py -v                  # tất cả
pytest tests/test_model_builder.py -v -m "not slow"    # bỏ qua test cần mạng và mô hình thật
```

Marker `slow` đã được đăng ký trong `tests/conftest.py` (dùng chung với `docs/02_evaluation_tests.md`).

Ánh xạ với tiêu chí chấp nhận ở mục 1.5 của `docs/03_model.md`:

| Tiêu chí | Test chứng minh |
| :--- | :--- |
| 1. Shape `[B]` cho ba backbone, hai kiểu gộp; `B = 1` vẫn đúng | M4, M7, S2 |
| 2. Phần đệm không đổi điểm | M1, M2, M5 |
| 3. Head đúng, không pooler, số tham số đúng | M8, M9, M10, S2 |
| 4. Cùng seed cho cùng khởi tạo | M14 |
| 5. Gradient chảy; học được batch nhỏ | M12, M15 |
| 6. Tokenizer đúng cặp, token đầu đúng | S1 |
| 7. Toàn bộ test đạt | Mọi test ở trên |

Giới hạn của bộ test:

* Không kiểm tra chất lượng mô hình sau huấn luyện đầy đủ; phần đó thuộc các lần chạy thật.
* Không kiểm tra `fp16` hay GPU; autocast và việc ép `.float()` được kiểm tra ở `docs/04_engine_and_utils_tests.md`.
* Các khẳng định phụ thuộc hành vi của `transformers` (`pooler is None` sau khi bỏ pooler, `hidden_size` của DistilBERT, tham số `add_pooling_layer`) chưa được chạy thử khi soạn tài liệu; M8 và M9 là nơi kiểm chứng ở lần chạy đầu. Nếu phiên bản `transformers` đổi hành vi thì sửa `docs/03_model.md` mục 2.4.
* Ngưỡng của M15 (loss cuối nhỏ hơn 0.5 lần loss đầu) là đề xuất, cần hiệu chỉnh ở lần chạy đầu để test ổn định.
* Test S1 và S2 cần mạng hoặc bộ nhớ đệm Hugging Face nên chỉ chạy được trên Colab hoặc máy đã tải trước checkpoint.
