# Đặc tả Thiết kế Module Mô hình — `model_builder.py`

> **Tài liệu:** `docs/03_model.md` (Define & Design cho mô hình hồi quy)
> **Module liên quan:** `src/model_builder.py`
> **Kế hoạch kiểm thử:** `docs/03_model_tests.md` (tách riêng)
> **Phụ thuộc:** `docs/define.md` (mục 9), `docs/design.md`; `docs/01_data.md` (chỉ về dạng đầu vào `input_ids`, `attention_mask` shape `[B, 256]`; không import `data_setup.py`)
> **Dùng bởi:** `docs/04_engine_and_utils.md` (`engine.py`, `train.py`, `predict.py`); `docs/02_evaluation.md` nhận số tham số từ `count_parameters` qua nơi gọi (`metrics.py` không import module này)
> **Trạng thái:** Đã hiện thực — `src/model_builder.py` + `tests/test_model_builder.py` (14/14 PASS, CPU tiny config, 2026-11-11). `build_tokenizer`/pretrained path chờ kiểm chứng trên Colab (cần mạng).

---

## PHẦN I: DEFINE — Xác định Bài toán & Yêu cầu

### 1.1. Bối cảnh & Mục tiêu

Module `model_builder.py` chịu trách nhiệm **dựng mô hình hồi quy gồm ba phần nối nhau: Transformer Encoder, bước gộp các vector token thành một vector cho cả review, và lớp Linear ra một số thực**. Module cũng cung cấp tokenizer đúng cặp với backbone và hàm đếm tham số. Module không chứa hàm loss, vòng lặp huấn luyện, việc đặt seed hay DataLoader.

Mục tiêu:

* Một lớp mô hình duy nhất chạy được cho BERT, RoBERTa và DistilBERT. Đổi backbone chỉ cần đổi một tên trong cấu hình, mọi thiết lập khác giữ nguyên (theo `docs/define.md` mục 9: "thay backbone, giữ nguyên mọi thiết lập khác").
* Mean pooling có nhân `attention_mask` là mặc định; `[CLS]` là phương án đối chứng.
* Cung cấp tokenizer cùng checkpoint với backbone, để `create_dataloaders` (xem `docs/01_data.md`) nhận đúng tokenizer.
* Cung cấp số tham số cho bảng `model_summary.csv` (xem `docs/02_evaluation.md`).
* Dựng được mô hình nhỏ khởi tạo ngẫu nhiên, không cần mạng, để kiểm thử trên CPU.

**Ba backbone:**

| Tên trong cấu hình | Checkpoint Hugging Face | Vai trò | Ghi chú |
| :--- | :--- | :--- | :--- |
| `bert` | `bert-base-uncased` | Mô hình chính (theo Define) | Chữ thường; token đầu `[CLS]` |
| `roberta` | `roberta-base` (đề xuất) | Mở rộng | Phân biệt hoa thường; token đầu `<s>`; không dùng `token_type_ids` |
| `distilbert` | `distilbert-base-uncased` (đề xuất) | Mở rộng | 6 lớp; không có `pooler_output`; không dùng `token_type_ids` |

Define chỉ nêu `bert-base-uncased`; checkpoint của RoBERTa và DistilBERT là đề xuất của tài liệu này, cần nhóm duyệt.

**Năm bài toán cốt lõi cần giải quyết:**

1. **Ba backbone khác nhau ở phần đầu ra.** DistilBERT không có `pooler_output`; RoBERTa dùng `<s>` thay cho `[CLS]`. Nếu lấy vector từ chỗ riêng của mỗi mô hình thì phải viết ba đường code và phép so sánh không còn đồng nhất. Vì vậy mọi backbone đều đi qua `last_hidden_state` rồi qua cùng một bước gộp.
2. **Phần đệm không được ảnh hưởng điểm.** Review dài đến 256 token, nhưng phần lớn review ngắn hơn nhiều và được đệm cho đủ batch. Mean pooling phải chỉ lấy trung bình trên token thật; nếu quên nhân `attention_mask`, cùng một review sẽ cho điểm khác nhau tuỳ batch nó rơi vào.
3. **Số tham số phải phản ánh mô hình thực dùng.** Lớp pooler của BERT và RoBERTa (khoảng 0,59 triệu tham số) không tham gia tính điểm khi dùng Mean pooling hoặc `[CLS]` lấy từ `last_hidden_state`. Giữ lại sẽ làm số tham số trong báo cáo lệch và để lại trọng số không bao giờ có gradient.
4. **So sánh công bằng.** Cả ba mô hình phải dùng cùng bước gộp, cùng dropout, cùng head và cùng quy tắc khởi tạo, chỉ khác backbone. Việc khởi tạo head phụ thuộc seed, nên phải đặt seed trước khi dựng mô hình.
5. **Kiểm thử không được tải trọng số pretrained.** Mỗi lần chạy pytest mà tải ba mô hình thì chậm và cần mạng; cần một đường dựng từ cấu hình nhỏ khởi tạo ngẫu nhiên.

### 1.2. Đầu vào / Đầu ra

| | Mô tả |
| :--- | :--- |
| **Đầu vào của `build_model`** | Tên backbone, kiểu gộp (`mean` hoặc `cls`), `dropout`, cờ `pretrained`, cấu hình backbone nhỏ (chỉ cho kiểm thử), giá trị khởi tạo bias của head (tuỳ chọn) |
| **Đầu ra của `build_model`** | Một `TextRegressor` (kế thừa `torch.nn.Module`) |
| **Đầu vào của `TextRegressor.forward`** | `input_ids` và `attention_mask`, mỗi tensor shape `[B, L]` kiểu `long`, đúng như một batch từ `create_dataloaders` |
| **Đầu ra của `TextRegressor.forward`** | Tensor shape `[B]`: điểm thô trên thang sao, **chưa clip**, không qua softmax hay sigmoid |
| **`build_tokenizer`** | Tên backbone → tokenizer Hugging Face cùng checkpoint |
| **`count_parameters`** | Mô hình → số tham số (`int`) |

Việc clip về 1.0-5.0 không nằm trong mô hình: huấn luyện (Huber hoặc MSE) tính trên điểm thô; clip chỉ áp dụng khi dự đoán và khi đánh giá, bằng `clip_scores` của `metrics.py` (theo `docs/define.md` mục 6 và 12).

### 1.3. Yêu cầu chức năng

| Mã | Yêu cầu |
| :--- | :--- |
| FR-1 | Có bảng `BACKBONES` gồm đúng ba khoá `bert`, `roberta`, `distilbert`; tên không có trong bảng thì báo `ValueError` kèm danh sách tên hợp lệ. |
| FR-2 | `build_model` trả về `TextRegressor` cho `pooling` thuộc `{"mean", "cls"}`. |
| FR-3 | `forward` chỉ nhận `input_ids` và `attention_mask` (không nhận `token_type_ids`) và trả về shape `[B]`, kể cả khi `B = 1`. |
| FR-4 | `mean` pooling: trung bình các vector token có `attention_mask = 1`; phần đệm không ảnh hưởng kết quả; hàng toàn đệm cho vector 0, không cho `NaN`. |
| FR-5 | `cls` pooling: lấy vector ở vị trí 0 của `last_hidden_state`, không đi qua pooler. |
| FR-6 | Head gồm `Dropout(p)` rồi `Linear(hidden_size, 1)`; không có hàm kích hoạt ở đầu ra. |
| FR-7 | Khi dựng BERT và RoBERTa, bỏ lớp pooler (`add_pooling_layer=False`); DistilBERT vốn không có. |
| FR-8 | `pretrained=True` nạp trọng số pretrained; `pretrained=False` khởi tạo ngẫu nhiên từ `backbone_config` (dùng cho kiểm thử). Backbone luôn được fine-tune toàn bộ, không đóng băng. |
| FR-9 | `output_bias_init` khác `None` thì đặt bias của head bằng giá trị đó. |
| FR-10 | `build_tokenizer` trả về tokenizer cùng checkpoint với backbone, giữ nguyên việc thêm token đặc biệt mặc định. |
| FR-11 | `count_parameters` đếm toàn bộ tham số, hoặc chỉ tham số có `requires_grad` khi `trainable_only=True`. |
| FR-12 | Kiểm tra đầu vào hợp lệ và báo lỗi rõ ràng. |
| FR-13 | Module không tự đặt seed và không tự chuyển `train()` hay `eval()`; nơi gọi làm việc đó. |

### 1.4. Ràng buộc

* Theo `docs/define.md` mục 9: lớp Linear ra một số thực, không dùng softmax; `[CLS]` và MSE chỉ là phương án đối chứng. Số 768 của Define là `hidden_size` của ba backbone bản `base`; code dùng `backbone.config.hidden_size` để cùng một đoạn code chạy được với cấu hình nhỏ khi kiểm thử.
* Cả ba backbone dùng cùng bước gộp, cùng dropout, cùng head, cùng loss, cùng seed và cùng ngân sách huấn luyện.
* Dropout trước head mặc định `0.1` (đề xuất, bằng dropout nội bộ của BERT); dropout bên trong backbone giữ mặc định của từng checkpoint.
* Đầu vào tối đa 256 token, nhỏ hơn giới hạn vị trí của backbone (512 với BERT và DistilBERT, 514 với RoBERTa).
* `model_builder.py` không import module nào khác trong `src/`. Ngược lại, `metrics.py` không import `model_builder.py` (để dashboard không cần `torch`); số tham số do nơi gọi chuyển sang.
* Phụ thuộc `torch` và `transformers`; cả hai nên được ghi phiên bản cố định trong `requirements.txt` (đề xuất), vì hành vi của `AutoModel` với lớp pooler thay đổi giữa các phiên bản.

### 1.5. Tiêu chí chấp nhận

1. Với cả ba backbone và cả hai kiểu gộp, một batch `[B, L]` cho đầu ra shape `[B]` kiểu số thực, hữu hạn; `B = 1` vẫn cho shape `[1]`.
2. Với Mean pooling và chế độ `eval`, thêm phần đệm vào cuối chuỗi không làm đổi điểm (sai khác chỉ do số học).
3. Head là `Linear(hidden_size, 1)`; BERT và RoBERTa không còn pooler; số tham số bằng số tham số backbone cộng `hidden_size + 1`.
4. Cùng seed đặt trước khi dựng cho cùng giá trị khởi tạo.
5. Gradient chảy tới mọi tham số có `requires_grad`; loss giảm rõ rệt khi học một batch nhỏ cố định.
6. Tokenizer đúng cặp: với cả ba backbone, token đầu chuỗi mã hoá là token đầu của tokenizer (`[CLS]` hoặc `<s>`), là cơ sở của `cls` pooling.
7. Bộ test trong `docs/03_model_tests.md` đều đạt.

---

## PHẦN II: DESIGN — Thiết kế Chi tiết

### 2.1. Vị trí trong kiến trúc hệ thống

```text
configs/*  (backbone, pooling, dropout, ...)
      │
      ▼
src/model_builder.py
      ├── build_tokenizer ──► tokenizer ──► data_setup.create_dataloaders   (01)
      ├── build_model ──► TextRegressor ──► engine.py  (huấn luyện: loss Huber/MSE, fp16)
      │                              └──► predict.py (suy luận, rồi clip_scores của metrics.py)   (04)
      └── count_parameters ──► (nơi gọi) ──► metrics.build_model_summary   (02)
```

### 2.2. Kiến trúc mô hình

```text
input_ids, attention_mask            [B, L]
        │
        ▼
Transformer Encoder (backbone, fine-tune toàn bộ)    ──►  last_hidden_state [B, L, H]
        │
        ▼
Gộp vector:  mean (nhân attention_mask)  hoặc  cls (vị trí 0)    ──►  [B, H]
        ▼
Dropout(p)
        ▼
Linear(H → 1)
        ▼
squeeze(-1)                                          ──►  [B]   điểm thô, chưa clip
```

`H` là `backbone.config.hidden_size`: 768 với ba checkpoint `base`.

### 2.3. Thiết kế tham số — `build_model`

| Tham số | Kiểu | Mặc định | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| `backbone` | `str` | `"bert"` | Tên trong `BACKBONES`. |
| `pooling` | `str` | `"mean"` | `"mean"` hoặc `"cls"`. |
| `dropout` | `float` | `0.1` | Xác suất dropout áp lên vector đã gộp, thuộc `[0, 1)`. |
| `pretrained` | `bool` | `True` | `True` nạp trọng số pretrained; `False` khởi tạo ngẫu nhiên. |
| `backbone_config` | cấu hình Hugging Face hoặc `None` | `None` | Chỉ dùng khi `pretrained=False`. `None` thì lấy bằng `AutoConfig.from_pretrained(hf_name)` (chỉ tải tệp cấu hình nhỏ, không tải trọng số). Kiểm thử truyền cấu hình nhỏ để không cần mạng. |
| `output_bias_init` | `float` hoặc `None` | `None` | Giá trị khởi tạo bias của head (xem 2.7). |

Tên tham số trùng với khoá trong `configs/`. Trong cấu hình mỗi lần chạy, `backbone`, `pooling`, `dropout` và `output_bias_init` thuộc module này; `loss` và `seed` thuộc `docs/04_engine_and_utils.md`.

### 2.4. Bảng backbone và cách dựng

```python
BACKBONES = {
    "bert":       {"hf_name": "bert-base-uncased",       "model_type": "bert",       "drop_pooler": True},
    "roberta":    {"hf_name": "roberta-base",            "model_type": "roberta",    "drop_pooler": True},
    "distilbert": {"hf_name": "distilbert-base-uncased", "model_type": "distilbert", "drop_pooler": False},
}
```

**Cách dựng backbone:**

1. `pretrained=True`: `AutoModel.from_pretrained(hf_name, **extra)`.
2. `pretrained=False`: lấy cấu hình (`backbone_config` hoặc `AutoConfig.from_pretrained(hf_name)`), kiểm tra `config.model_type` khớp `model_type` của bảng, rồi `AutoModel.from_config(config, **extra)`.
3. `extra = {"add_pooling_layer": False}` nếu `drop_pooler` là `True`, ngược lại là `{}`. Lý do: `BertModel` và `RobertaModel` nhận tham số này, còn `DistilBertModel` không nhận nên không được truyền.
4. `hidden_size = backbone.config.hidden_size`. Với DistilBERT, `DistilBertConfig` đặt tên chiều ẩn là `dim` và ánh xạ `hidden_size` sang `dim`, nên cách viết này dùng được cho cả ba. Điểm này phụ thuộc hành vi của `transformers`; test M8 kiểm chứng lần đầu.

Khi nạp trọng số pretrained của BERT và RoBERTa, thư viện có thể in cảnh báo "một số trọng số pooler không được dùng": bình thường, vì pooler cố ý bị bỏ.

### 2.5. Các phép gộp vector

**`mean_pool(last_hidden_state, attention_mask)`** trả về `[B, H]`:

$$\mathbf{v} = \frac{\sum_{t=1}^{L} m_t\,\mathbf{h}_t}{\max\!\left(1,\ \sum_{t=1}^{L} m_t\right)}$$

trong đó $\mathbf{h}_t$ là vector token ở vị trí $t$ và $m_t \in \{0, 1\}$ là `attention_mask`. Các quy tắc:

* Tính trong `float32` (ép `last_hidden_state` và mặt nạ sang `float32`), để không tràn số khi huấn luyện `fp16`.
* Mẫu số chặn dưới là `1`: hàng toàn đệm cho vector 0 thay vì `NaN`.
* Token đặc biệt (`[CLS]`, `[SEP]`, `<s>`, `</s>`) có `attention_mask = 1` nên được tính vào trung bình.

**Ví dụ.** `last_hidden_state` của một review là `[[1, 2], [3, 4], [100, 100]]` và mặt nạ `[1, 1, 0]`: vị trí thứ ba là phần đệm nên bị bỏ, kết quả là `[2, 3]`. Nếu quên nhân mặt nạ, kết quả là trung bình cả ba hàng, khoảng `[34.67, 35.33]`, bị phần đệm kéo lệch hoàn toàn.

**`cls_pool(last_hidden_state)`** trả về `last_hidden_state[:, 0]`. Vị trí 0 là `[CLS]` với BERT và DistilBERT, là `<s>` với RoBERTa, miễn là tokenizer thêm token đặc biệt (mặc định). Không dùng `pooler_output`.

### 2.6. Lớp `TextRegressor`

```python
class TextRegressor(nn.Module):
    def __init__(self, backbone, pooling="mean", dropout=0.1, output_bias_init=None): ...
    def forward(self, input_ids, attention_mask) -> torch.Tensor:   # shape (B,)
```

| Thuộc tính | Ý nghĩa |
| :--- | :--- |
| `backbone` | Mô hình Transformer đã dựng. |
| `pooling` | Chuỗi `"mean"` hoặc `"cls"`. |
| `dropout` | `nn.Dropout`. |
| `head` | `nn.Linear(hidden_size, 1)`. |

Tên `backbone` và `head` là một phần của giao diện: `docs/04_engine_and_utils.md` có thể dùng chúng để chia nhóm tham số, và test kiểm tra theo tên.

**Luồng `forward`:**

```text
1. kiểm tra input_ids và attention_mask: cùng 2 chiều, cùng shape
2. h = backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
3. v = mean_pool(h, attention_mask)  hoặc  cls_pool(h)
4. out = head(dropout(v)).squeeze(-1)
5. return out        # shape (B,)
```

* Dùng `squeeze(-1)` chứ không dùng `squeeze()`: `squeeze()` làm batch một mẫu thành tensor 0 chiều, làm hỏng bước tính loss và bước ghép dự đoán.
* Kiểu của `out` theo chế độ autocast (có thể là `float16` khi huấn luyện `fp16`); `engine.py` ép `.float()` trước khi tính loss.
* `ReviewDataset` không trả `token_type_ids` (xem `docs/01_data.md` mục 2.5). BERT khi đó dùng mặc định toàn 0, tức một đoạn văn bản, đúng với bài toán này; RoBERTa và DistilBERT không dùng.

### 2.7. Khởi tạo head, seed và `output_bias_init`

* Head dùng khởi tạo mặc định của `nn.Linear`, phụ thuộc bộ sinh số ngẫu nhiên toàn cục. Vì vậy `set_seed(42)` (của `utils.py`) phải được gọi ngay trước `build_model`. Tái lập được đảm bảo cho từng backbone; việc ba backbone có chung giá trị khởi tạo head không được yêu cầu (Define chỉ yêu cầu "cùng seed").
* **`output_bias_init`:** nhãn lệch về sao cao (Define mục 8), nên mức trung bình của nhãn xa 0. Nếu bias bắt đầu quanh 0, những bước đầu chủ yếu mất vào việc kéo đầu ra lên vùng 3-4 sao. Đặt bias bằng trung bình nhãn Train để mô hình bắt đầu từ một dự đoán hằng số hợp lý. Mặc định `None` (không can thiệp); `docs/04_engine_and_utils.md` quyết định có truyền trung bình Train hay không, và nếu có thì áp dụng như nhau cho cả ba backbone.

### 2.8. Tokenizer — `build_tokenizer`

```python
build_tokenizer(backbone: str = "bert")   # AutoTokenizer.from_pretrained(BACKBONES[backbone]["hf_name"])
```

* Mỗi backbone dùng đúng tokenizer của checkpoint mình. BERT và DistilBERT bản `uncased` tự hạ chữ thường; RoBERTa phân biệt hoa thường. `data_setup.py` chỉ chuẩn hoá khoảng trắng nên không tác động tới điều này.
* Không tắt token đặc biệt (`add_special_tokens=True` mặc định): `cls` pooling dựa vào vị trí 0.
* Hàm cần mạng ở lần đầu (hoặc bộ nhớ đệm của Hugging Face); test dùng tokenizer thật được đánh dấu `slow`.

### 2.9. Đếm tham số — `count_parameters`

```python
count_parameters(model, trainable_only: bool = False) -> int
# sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
```

Số tham số của mô hình đầy đủ (backbone, head), không gồm pooler. Giá trị ước tính khi dùng ba checkpoint `base` (cần kiểm tra bằng test S2): BERT khoảng 108,9 triệu, RoBERTa khoảng 124,1 triệu, DistilBERT khoảng 66,4 triệu. Con số này là `params_total` trong `model_summary.csv`.

### 2.10. Giao diện hàm và ngoại lệ

| Hàm | Đầu vào | Đầu ra | Ngoại lệ |
| :--- | :--- | :--- | :--- |
| `build_model(...)` | xem mục 2.3 | `TextRegressor` | `ValueError`: `backbone` không có trong `BACKBONES` (kèm tên hợp lệ); `pooling` ngoài `{"mean", "cls"}`; `dropout` ngoài `[0, 1)`; `backbone_config.model_type` không khớp backbone. Lỗi tải của Hugging Face (`OSError`) được để nổi lên, không nuốt. |
| `TextRegressor.forward(input_ids, attention_mask)` | hai tensor `[B, L]` | tensor `[B]` | `ValueError` nếu không phải 2 chiều hoặc hai tensor khác shape |
| `build_tokenizer(backbone)` | `str` | tokenizer | `ValueError` nếu `backbone` không hợp lệ |
| `count_parameters(model, trainable_only)` | `nn.Module`, `bool` | `int` | không |
| `mean_pool`, `cls_pool` | tensor | tensor `[B, H]` | không |

### 2.11. Tích hợp với các module khác

| Module | Cách phối hợp |
| :--- | :--- |
| `data_setup.py` | `build_tokenizer` cung cấp tokenizer cho `create_dataloaders`. Từ batch, chỉ `input_ids` và `attention_mask` đi vào `forward`; `labels` và `review_id` thì không. |
| `engine.py` | Gọi `set_seed` rồi `build_model`; tự đặt `train()` hoặc `eval()`; dùng autocast `fp16`; ép đầu ra `.float()` trước khi tính Huber hoặc MSE trên điểm thô; lưu checkpoint kèm cấu hình (`backbone`, `pooling`, `dropout`) để nạp lại đúng kiến trúc. |
| `predict.py` | Dựng lại bằng `build_model(..., pretrained=False)` rồi nạp `state_dict` từ checkpoint, nên suy luận không phải tải lại trọng số pretrained (chỉ tải tệp cấu hình nhỏ); sau đó clip điểm bằng `clip_scores` của `metrics.py`. |
| `metrics.py` | Không import module này. Nơi gọi truyền `count_parameters(model)` vào `build_model_summary` qua `param_counts`. |
| Dashboard | Không dùng module này (không cài `torch`). |

### 2.12. Quyết định thiết kế và điểm mở

| Hạng mục | Quyết định |
| :--- | :--- |
| Mean pooling mặc định; `[CLS]` chỉ đối chứng; Linear 768 xuống 1, không softmax | Đã chốt (theo `docs/define.md` mục 9) |
| BERT là `bert-base-uncased`; cùng backbone, pooling, loss và seed cho cả ba mô hình | Đã chốt (theo `docs/define.md`) |
| Clip chỉ khi dự đoán và đánh giá, không nằm trong mô hình | Đã chốt (theo `docs/define.md` mục 6 và 12) |
| RoBERTa là `roberta-base`, DistilBERT là `distilbert-base-uncased` | Đề xuất; Define không nêu checkpoint |
| Chọn backbone bằng tên trong bảng `BACKBONES` và trong `configs/` | Đề xuất |
| Bỏ lớp pooler của BERT và RoBERTa | Đề xuất, để số tham số đúng và không còn trọng số không có gradient |
| Dropout `0.1` trước head | Đề xuất |
| `output_bias_init`, mặc định `None` | Đề xuất; `docs/04_engine_and_utils.md` quyết định có dùng |
| Tái lập khởi tạo head theo từng backbone, không bắt đồng nhất giữa các backbone | Đề xuất |
| Ghi phiên bản cố định của `torch` và `transformers` | Đề xuất |
| Tốc độ học riêng cho head, đóng băng một phần backbone | Chưa làm; nếu cần do `docs/04_engine_and_utils.md` quyết định |
| Các giả định về hành vi của `transformers` (`add_pooling_layer`, `hidden_size` của DistilBERT) | Chưa chạy thử khi soạn tài liệu; test M8 và M9 kiểm chứng ở lần chạy đầu |
