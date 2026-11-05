# Kế hoạch kiểm thử Module Huấn luyện, Suy luận và Tiện ích — `tests/test_utils.py`, `tests/test_engine.py`, `tests/test_train.py`, `tests/test_predict.py`

> **Tài liệu:** `docs/04_engine_and_utils_tests.md`
> **Đặc tả được kiểm thử:** `docs/04_engine_and_utils.md` (mục 1.5 Tiêu chí chấp nhận, mục 2.3 đến 2.10)
> **Trạng thái:** Bản nháp trước khi hiện thực

---

## 1. Mục tiêu và nguyên tắc

Bộ test kiểm chứng bốn module **mà không cần dữ liệu thật, không cần mạng, không cần GPU** (trừ nhóm S). Test không đánh giá chất lượng mô hình; test kiểm tra **hành vi của vòng huấn luyện, việc lưu và tiếp tục, quy tắc dùng tập Test một lần và các file kết quả xuất ra**.

1. **Dùng mô hình nhỏ khởi tạo ngẫu nhiên** (`docs/03_model_tests.md`) và DataLoader tổng hợp: một lần `fit` ba epoch chạy trong vài giây trên CPU (ước tính).
2. **Tắt mọi nguồn ngẫu nhiên không cần thiết** khi so sánh hai lần chạy: `dropout = 0` ở head và ở backbone (hàm `no_dropout`), `fp16 = False`, `num_workers = 0`.
3. **Ưu tiên giá trị tính tay** cho loss, lịch tốc độ học và các cột kết quả.
4. **Kiểm tra hành vi quan sát được** (file, cột, giá trị, ngoại lệ), không kiểm tra cách cài đặt bên trong.
5. **Mỗi test một ý chính; test lỗi dùng `pytest.raises` hoặc `pytest.warns`.**
6. **Cố định seed** trong mọi test có ngẫu nhiên.
7. **Test cần mạng hoặc GPU đánh dấu `slow` hoặc `gpu`:** tự bỏ qua khi thiếu điều kiện.
8. **Thư mục kết quả luôn là `tmp_path`:** không ghi vào Drive hay thư mục dự án.

**Hàm hỗ trợ trong `tests/conftest.py`** (dùng chung với `docs/01_data_tests.md`, `docs/02_evaluation_tests.md`, `docs/03_model_tests.md`; cần thống nhất để không định nghĩa trùng):

| Hàm hoặc fixture | Nhiệm vụ |
| :--- | :--- |
| `tiny_config(backbone)` | Cấu hình backbone nhỏ (đã định nghĩa ở `docs/03_model_tests.md`) |
| `no_dropout(config)` | Đặt xác suất dropout nội bộ bằng 0: `hidden_dropout_prob` và `attention_probs_dropout_prob` với BERT và RoBERTa; `dropout` và `attention_dropout` với DistilBERT |
| `tiny_model(backbone="distilbert", **kw)` | `build_model(backbone, pretrained=False, backbone_config=no_dropout(tiny_config(backbone)), dropout=0.0, **kw)` |
| `make_loader(n, batch_size, seq_len=8, seed=0, shuffle=False, nan_label_at=None)` | DataLoader tổng hợp, mỗi batch là `dict` đúng dạng của `create_dataloaders`: `input_ids`, `attention_mask` (`long`), `labels` (`float32`, giá trị nguyên 1 đến 5), `review_id` (danh sách chuỗi `"r0"`, `"r1"`, ...). Khi `shuffle=True` có `generator` riêng để `fit` đặt lại seed. `nan_label_at` đặt một nhãn bằng `NaN` |
| `ExplodingLoader` | Đối tượng có `__iter__` và `__len__` mà việc lặp qua nó ném `AssertionError("test_loader bị chạm")` |
| `FakeTokenizer` | Gọi `tokenizer(texts, padding, truncation, max_length, return_tensors="pt")` trả về `dict` có `input_ids` và `attention_mask` (mỗi từ ánh xạ ổn định sang một id trong `[5, 1000)`, đệm tới chuỗi dài nhất trong lô, cắt ở `max_length`) |
| `base_cfg(tmp_path, **overrides)` | `dict` cấu hình đã hợp lệ: `backbone="distilbert"`, `pooling="mean"`, `dropout=0.0`, `loss="huber"`, `epochs=3`, `lr=1e-3`, `batch_size=4`, `warmup_ratio=0.1`, `grad_clip=1.0`, `fp16=False`, `seed=42`, `output_bias_init=None`, các thư mục trỏ vào `tmp_path` |

## 2. Sáu nhóm khía cạnh kiểm thử

| Nhóm | Khía cạnh | Câu hỏi cần trả lời | Test |
| :--- | :--- | :--- | :--- |
| A | `utils.py` | Seed tái lập? Ghi nguyên tử? Checkpoint nạp lại đủ? | U1, U2, U3, U4 |
| B | Thành phần của `engine.py` | Loss, bộ tối ưu, lịch tốc độ học đúng? Một epoch cập nhật trọng số? Kiểm định đúng thứ tự và không đổi mô hình? `fp16` trên CPU xử lý đúng? | L1, L2, L3, L4, L5, L6, L7 |
| C | `fit`: lịch sử, tiếp tục, ranh giới Test | Checkpoint tốt nhất đúng? Tiếp tục cho kết quả như chạy liền? Chặn ghi đè? Không chạm Test? | F1, F2, F3, F4, F5, F6 |
| D | `train.py` | Cấu hình đọc, ép kiểu, từ chối đúng? Seed đặt trước khi dựng mô hình? Dòng lệnh đúng? | T1, T2, T3, T4, T5 |
| E | `predict.py` | Dự đoán trong 1.0-5.0, không phụ thuộc lô? File xuất đúng cột, không ghi đè? Ghép kiểm tra khớp? | P1, P2, P3, P4, P5, P6, P7, P8 |
| S | Mô hình và phần cứng thật (chậm) | Chạy được từ cấu hình đến file kết quả? `fp16` trên GPU ổn định? | S1, S2 |

## 3. Mô tả chi tiết từng test

**Nhóm A: `utils.py`**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| U1 | `test_u1_set_seed_reproducible` | `set_seed(42)`, lấy một số từ `random`, `numpy` và `torch`; gọi lại `set_seed(42)` lấy lần nữa; rồi `set_seed(7)`. | Hai lần seed 42 giống hệt nhau ở cả ba thư viện; seed 7 khác | Quên đặt seed cho một thư viện. |
| U2 | `test_u2_get_device` | `get_device("cpu")`; `get_device("auto")`; `"cuda"` khi máy không có CUDA; `"tpu"`. | `cpu` cho `cpu`; `auto` cho `cpu` hoặc `cuda`; `RuntimeError` cho `cuda` thiếu GPU (bỏ qua nếu máy có GPU); `ValueError` cho `tpu` | Âm thầm rơi về CPU khi người dùng đã yêu cầu GPU. |
| U3 | `test_u3_checkpoint_roundtrip` | Lưu `dict` gồm tensor, `dict` lồng nhau, danh sách `dict`, `str`, `int`, `float`, `bool`, `None`; nạp lại; kiểm tra thư mục sau khi lưu; thử `save_checkpoint` với đối tượng không phải `dict`; nạp file không tồn tại. | Nội dung nạp lại bằng nội dung đã lưu (tensor bằng `torch.equal`); thư mục chỉ có đúng file đã lưu; `TypeError`; `FileNotFoundError` | Checkpoint không nạp được bằng `weights_only=True`; để sót file tạm. |
| U4 | `test_u4_write_atomic_failure` | Có sẵn file nội dung `"cũ"`. Gọi `write_atomic` với hàm ghi nửa chừng rồi ném lỗi; sau đó gọi với hàm ghi thành công vào thư mục chưa tồn tại. | Lỗi được ném lại; file cũ còn nguyên nội dung `"cũ"`; không còn file tạm; lần thành công tạo thư mục cha và có nội dung mới | Phiên bị ngắt giữa lúc ghi làm hỏng checkpoint đang có. |

**Nhóm B: Thành phần của `engine.py`**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| L1 | `test_l1_build_loss_values` | Một phần tử: dự đoán 3.0 so với nhãn 1.0, và dự đoán 1.5 so với nhãn 1.0; Huber `delta=1`, Huber `delta=0.5`, MSE. | Huber `delta=1`: 1.5 và 0.125; Huber `delta=0.5` với sai số 2: 0.875; MSE: 4.0 và 0.25. Tên lạ và `delta <= 0` cho `ValueError` | Nhầm Huber với MSE; `delta` không có tác dụng. |
| L2 | `test_l2_optimizer_groups` | `build_optimizer_and_scheduler` trên mô hình nhỏ; duyệt các nhóm tham số của `optimizer`. | Là `AdamW`; mỗi tham số của mô hình có mặt đúng một lần; nhóm `weight_decay = 0` gồm đúng các tham số có `ndim <= 1`; nhóm còn lại có `weight_decay` của cấu hình; `lr` đúng | Tham số bị bỏ sót hoặc trùng; áp weight decay lên bias và LayerNorm. |
| L3 | `test_l3_scheduler_warmup_decay` | `total_steps = 10`, `warmup_ratio = 0.2`, `lr = 1.0`; lặp `optimizer.step()` rồi `scheduler.step()`, đọc `scheduler.get_last_lr()[0]` sau 0, 1, 2, 6, 10 bước. | `0.0`, `0.5`, `1.0`, `0.5`, `0.0` (sai số `1e-9`) | Lịch cập nhật theo epoch thay vì theo bước; thiếu warmup; không giảm về 0. |
| L4 | `test_l4_train_one_epoch_updates` | Mô hình nhỏ, `make_loader(8, 4)` (hai batch); chép tham số trước; chạy một epoch. | Trả về `train_loss` hữu hạn, `n_batches == 2`; ít nhất một tham số đổi giá trị; tốc độ học sau epoch khác ban đầu; mô hình còn ở chế độ `train` | Không cập nhật trọng số; không bước lịch; trả về khoá sai. |
| L5 | `test_l5_non_finite_loss` | Loader có một nhãn `NaN`. | `FloatingPointError` | Huấn luyện tiếp với loss `NaN`, hỏng cả checkpoint tốt. |
| L6 | `test_l6_evaluate` | `make_loader(7, 3)` (batch cuối thiếu mẫu); gọi `evaluate` hai lần, có và không có `loss_fn`; gọi khi mô hình đang ở `train`; loader rỗng. | `review_id == ["r0", ..., "r6"]` theo đúng thứ tự; `y_true` bằng nhãn gốc; `y_pred_raw` shape `(7,)`, hữu hạn; hai lần giống hệt nhau; tham số không đổi và không có gradient; chế độ ban đầu được khôi phục; `loss` là `float` khi có `loss_fn`, là `None` khi không có; loader rỗng cho `ValueError` | Thứ tự mẫu bị xáo trộn làm dự đoán ghép sai `review_id`; để lại gradient; mô hình kẹt ở chế độ `eval`. |
| L7 | `test_l7_fp16_on_cpu` | `train_one_epoch(..., fp16=True)` trên thiết bị `cpu`. | Có `UserWarning`; epoch vẫn chạy, loss hữu hạn | `fp16` bật trên CPU làm lỗi hoặc im lặng sai. |

**Nhóm C: `fit`**

Các test nhóm C dùng `tiny_model`, `base_cfg`, `make_loader(24, 4, shuffle=True)` cho Train và `make_loader(8, 4)` cho Val.

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| F1 | `test_f1_fit_history_and_best_checkpoint` | `fit` ba epoch; đọc `history`, `history.csv` và `best_model.pt`; nạp `model_state_dict` vào mô hình mới rồi `evaluate` trên Val. | `history` có 3 dòng với đủ cột `epoch`, `train_loss`, `val_loss`, `val_rmse`, `val_mae`, `val_rmse_raw`, `lr`, `epoch_seconds`, `is_best`; `history.csv` khớp `history`; dòng đầu có `is_best = True`; `val_rmse` trong `best_model.pt` bằng giá trị nhỏ nhất của cột; `epoch` bằng chỉ số của giá trị đó; kiểm định lại bằng trọng số đã lưu cho cùng RMSE (sai số `1e-5`) | Lưu nhầm epoch cuối thay vì epoch tốt nhất; `best_model.pt` không ứng với RMSE đã ghi. |
| F2 | `test_f2_resume_matches_uninterrupted` | Lần A: `fit` ba epoch liền (thư mục A). Lần B: `fit(stop_after_epoch=2)`, rồi dựng mô hình và loader mới (cùng seed) và `fit` lại (`resume=True`) trong cùng thư mục B. Bọc `train_one_epoch` bằng `monkeypatch` để đếm số lần gọi. | Số lần gọi: 3 (lần A), 2 rồi 1 (lần B); `history` của B có 3 dòng và `train_loss`, `val_rmse` từng epoch bằng của A (sai số `1e-6`); trọng số `best_model.pt` của A và B bằng nhau (`atol 1e-6`); `train_seconds` trong `run_meta.json` của B bằng tổng `epoch_seconds` của cả ba epoch | Tiếp tục huấn luyện lại từ đầu; quên khôi phục bộ tối ưu hoặc lịch; thứ tự dữ liệu mỗi epoch phụ thuộc việc có bị ngắt hay không; thời gian không cộng dồn. |
| F3 | `test_f3_resume_and_overwrite_guards` | (a) Sau `stop_after_epoch=1`, tiếp tục với `lr` đổi. (b) Sau `stop_after_epoch=1`, gọi `resume=False` mà không `overwrite`. (c) Chạy xong ba epoch rồi `fit` lần nữa. (d) Như (c) nhưng `overwrite=True`. (e) Tiếp tục với cấu hình chỉ khác `num_workers`. (f) Sau `stop_after_epoch=1`, gọi `resume=True, overwrite=True`. | (a) `ValueError` nêu khoá `lr`. (b) và (c) `FileExistsError`. (d) chạy lại từ đầu, `history` có 3 dòng. (e) tiếp tục bình thường. (f) `overwrite` thắng: chạy lại từ epoch 1 (`train_one_epoch` được gọi 3 lần, `history` có 3 dòng) | Ghi đè kết quả đã hoàn tất âm thầm; tiếp tục với cấu hình khác làm kết quả lẫn lộn; khoá không liên quan chặn việc tiếp tục. |
| F4 | `test_f4_completion_artifacts` | Sau khi hoàn tất `fit`; thêm một lần chạy với `keep_last=True`. | Không còn `last.pt`, còn `best_model.pt`; với `keep_last=True` còn `last.pt`; `run_meta.json` có `status = "completed"`, `epochs_done = 3`, `best_epoch`, `best_val_rmse`, `device = "cpu"`, `n_params` bằng `count_parameters(model)`, `train_seconds` bằng tổng `epoch_seconds` và `epoch_seconds_mean` bằng trung bình, mọi `epoch_seconds > 0` | Drive đầy vì để lại `last.pt`; thiếu số liệu mà `docs/02_evaluation.md` cần. |
| F5 | `test_f5_test_set_never_touched` | `run_training(cfg, model=..., loaders=(train, val, ExplodingLoader()))`; đọc chữ ký của `fit`. | Chạy hoàn tất không có `AssertionError`; chữ ký `fit` không có tham số nào chứa `test` | Test lọt vào huấn luyện hoặc chọn checkpoint. |
| F6 | `test_f6_fit_invalid_arguments` | `epochs = 0`; `train_loader` rỗng; `val_loader` rỗng; `loss = "l1"` trong `run_config`; `stop_after_epoch = 0`; `stop_after_epoch = epochs`. | `ValueError` cho cả sáu | Lỗi muộn khó hiểu giữa chừng huấn luyện. |

**Nhóm D: `train.py`**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| T1 | `test_t1_load_config` | File YAML có `lr: 2e-5` (PyYAML đọc thành chuỗi), chỉ các khoá bắt buộc, kèm `overrides = {"loss": "mse", "epochs": "5", "run_name": "bert_mse"}`. Tham số hoá các trường hợp lỗi: khoá lạ `learning_rate`; thiếu `backbone`; `backbone = "gpt2"`; `loss = "l1"`; `epochs = 0`; `lr = -1`; `warmup_ratio = 1.0`; `output_bias_init = "abc"`. | `cfg["lr"] == 2e-5` kiểu `float`; mặc định được áp (`pooling = "mean"`, `seed = 42`, `batch_size = 16`); `epochs == 5` kiểu `int`; `loss == "mse"`; `run_name == "bert_mse"`; khi không ghi đè, `run_name` bằng `backbone`; mọi trường hợp lỗi cho `ValueError` | `lr` là chuỗi làm AdamW lỗi muộn; gõ sai tên khoá bị bỏ qua lặng lẽ. |
| T2 | `test_t2_resolve_output_bias` | `train.parquet` với `stars = [1, 5, 5, 5]` (chỉ file này có trong thư mục); các giá trị `train_mean`, `3.5`, `None`. | `4.0`; `3.5`; `None`; không đòi `val.parquet` hay `test.parquet` | Dùng nhãn Val hoặc Test để đặt bias (rò rỉ). |
| T3 | `test_t3_run_training_wiring` | `run_training(cfg, model=tiny_model(), loaders=...)`. | Trả về `dict` có `run_name`, `best_epoch`, `best_val_rmse`, `train_seconds`, `epoch_seconds_mean`, `n_params`, `history`; có `config.yaml` (đọc lại bằng `yaml.safe_load` bằng `cfg`), `history.csv`, `run_meta.json` trong `results/runs/<run_name>/` và `best_model.pt` trong `checkpoints/<run_name>/` | Thiếu bản sao cấu hình để tái lập; đặt file sai chỗ. |
| T4 | `test_t4_seed_set_before_build` | `monkeypatch` `train.build_model` bằng hàm ghi lại `torch.rand(1)` và các đối số nhận được, rồi trả mô hình nhỏ; `monkeypatch` `train.build_tokenizer` và `train.create_dataloaders` bằng bản tổng hợp; `output_bias_init = 3.7`. `create_dataloaders` giả **tiêu thụ RNG toàn cục** (gọi `torch.rand(10)`) trước khi trả về. | Số ghi lại bằng số lấy ngay sau `set_seed(cfg["seed"])` (chứng minh `set_seed` được gọi lại **ngay trước** `build_model`, không chỉ ở đầu `run_training`); `output_bias_init` nhận được là `3.7` | Quên `set_seed` trước khi dựng mô hình (khởi tạo head không tái lập); `output_bias_init` không được truyền. |
| T5 | `test_t5_cli_overrides` | `main(["--config", f, "--set", "epochs=1", "--set", "loss=mse"])` với `train.run_training` bị `monkeypatch` để ghi lại `cfg`; sau đó `--set loss=l1`; sau đó `--overwrite` và `--no-resume` (ghi lại đối số `resume`, `overwrite` truyền vào `run_training`); sau đó không có cờ nào. | Mã thoát 0 và `cfg` có `epochs == 1` kiểu `int`, `loss == "mse"`; trường hợp `l1` cho mã thoát khác 0; `--overwrite` cho `overwrite=True`, `--no-resume` cho `resume=False`; mặc định `overwrite=False`, `resume=True` | Giá trị dòng lệnh vẫn là chuỗi; lỗi cấu hình làm chương trình thoát với mã 0. |

**Nhóm E: `predict.py`**

Các test nhóm E dùng `tiny_model()` và `FakeTokenizer`.

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| P1 | `test_p1_predict_single_list_and_clip` | `predict("great food", ...)`; `predict` với danh sách 5 chuỗi; rồi đặt `head.weight` bằng 0 và `head.bias` bằng `10.0` (sau đó `-3.0`) và gọi với `clip=True` và `clip=False`. | Chuỗi đơn trả về `float` trong `[1.0, 5.0]`; danh sách trả về `np.ndarray` shape `(5,)` trong `[1.0, 5.0]`, phần tử đầu bằng kết quả của chuỗi đơn (`1e-5`); bias 10: `clip=True` cho đúng `5.0`, `clip=False` cho khoảng `10.0`; bias -3: `clip=True` cho đúng `1.0` | Quên clip; kiểu trả về sai cho một chuỗi đơn. |
| P2 | `test_p2_predict_batch_size_invariance` | 7 chuỗi độ dài khác nhau; `batch_size` là 1, 3 và 32. | Ba kết quả bằng nhau (`atol 1e-5`) | Điểm phụ thuộc lô vì phần đệm (xem `docs/03_model_tests.md`, M5). |
| P3 | `test_p3_predict_invalid_inputs` | Danh sách rỗng; danh sách có phần tử không phải chuỗi; `None`; `batch_size = 0`; `max_length = 0`. | `ValueError` cho cả năm | Đầu vào sai lọt qua tới tokenizer. |
| P4 | `test_p4_predict_restores_mode` | Mô hình ở `train()` rồi gọi `predict`; làm lại với mô hình ở `eval()`. | Sau khi gọi, mô hình ở đúng chế độ ban đầu; tham số không có gradient | `predict` giữa chừng huấn luyện làm mô hình kẹt ở `eval` và tắt dropout. |
| P5 | `test_p5_load_trained_model_roundtrip` | `fit` hai epoch rồi `load_trained_model(best_model.pt, backbone_config=...)`; thử file không tồn tại; thử checkpoint chỉ có `model_state_dict`; nạp **không** truyền `backbone_config` với `AutoConfig.from_pretrained` bị `monkeypatch` để ném lỗi (mô phỏng không có mạng). | Mô hình ở chế độ `eval`; lần nạp không truyền `backbone_config` vẫn thành công và cho cùng RMSE (cấu hình đọc từ khoá `backbone_config` trong checkpoint); `evaluate` trên Val cho RMSE bằng `val_rmse` trong checkpoint (`1e-5`); `info` có `run_name`, `backbone`, `epoch`, `val_rmse`, `run_config`; `FileNotFoundError`; `ValueError` | Dựng lại sai kiến trúc; nạp trọng số không khớp mà không báo. |
| P6 | `test_p6_export_test_predictions` | `export_test_predictions` trên `make_loader(6, 4)`, `model_name="bert"`; rồi gọi lần hai; rồi gọi `overwrite=True`; thêm một mô hình có bias 10. | `bert_test.csv` có đúng cột `review_id`, `true_stars`, `pred_raw`, `pred` theo thứ tự loader; `pred` bằng `clip(pred_raw, 1, 5)`; `bert_test_meta.json` có `n = 6`, `rmse_raw` bằng căn bậc hai trung bình bình phương của `pred_raw - true_stars` tính độc lập, `inference_seconds > 0`; lần hai `FileExistsError`; `overwrite=True` ghi được; mô hình bias 10 cho `pred_raw` khoảng 10, `pred` bằng 5, `rmse_raw` lớn hơn RMSE đã clip | Mất điểm thô nên không tính được `rmse_raw`; ghi đè kết quả Test đã công bố; thứ tự dòng lệch loader. |
| P7 | `test_p7_merge_test_predictions` | Tạo file `bert_test.csv`, `roberta_test.csv` tổng hợp (nhãn 4, 2, 5; dự đoán đã biết), `DataFrame` baseline, và `test.parquet` có `text`. Gọi với `model_names = ["bert", "roberta"]` có và không có `baseline_preds`. Các trường hợp lỗi: một file đổi thứ tự `review_id`; một file đổi `true_stars`; baseline thiếu một `review_id`; `test.parquet` thiếu một `review_id`; `model_names` rỗng; gọi lần hai không `overwrite`. | Có baseline: cột `review_id`, `true_stars`, `pred_baseline`, `pred_bert`, `pred_roberta`, `abs_err_baseline`, `abs_err_bert`, `abs_err_roberta`; không baseline thì không có hai cột baseline; bản `with_text` có thêm `text` ngay sau `review_id`; `abs_err` bằng `|true_stars - pred|` (ví dụ nhãn 4, dự đoán 3.5 cho `0.5`); `true_stars` kiểu số nguyên; `text` khớp `test.parquet` theo `review_id`; giá trị trả về bằng bản không `text`; mọi trường hợp lỗi cho `ValueError` (riêng lần gọi hai cho `FileExistsError`) | Ghép sai dòng giữa các mô hình (dự đoán gán cho review khác); sai cột hoặc thứ tự cột làm dashboard hỏng. |
| P8 | `test_p8_collect_summary_inputs` | Tạo `run_meta.json` cho `bert_mse` và `roberta`, và file meta dự đoán cho `bert`, `roberta`; gọi với `{"bert": "bert_mse", "roberta": "roberta"}`; xoá một file meta rồi gọi lại. | Bốn từ điển `param_counts`, `train_seconds`, `inference_seconds`, `rmse_raw_by_model`, khoá là tên `model` (`bert`, không phải `bert_mse`), giá trị đúng từng file; `FileNotFoundError` khi thiếu file | Nhầm `run_name` với `model` làm bảng tóm tắt lệch tên; thiếu số liệu mà `build_model_summary` cần. |

**Nhóm S: Mô hình và phần cứng thật (đánh dấu `slow` hoặc `gpu`)**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| S1 | `test_s1_smoke_end_to_end` | Tạo `data_smoke/` tổng hợp (8 dòng Train, 4 Val, 4 Test, đúng cột của `docs/01_data.md`); `train.main(["--config", smoke.yaml])` với DistilBERT thật, 1 epoch, `batch_size = 4`, `max_length = 64`, CPU; rồi `load_trained_model`, tạo loader Test thật và `export_test_predictions`. | Mã thoát 0; `history` một dòng hữu hạn; có `best_model.pt`; file dự đoán có 4 dòng, mọi `pred` trong `[1.0, 5.0]` | Lỗi chỉ lộ ra khi nối đủ chuỗi từ cấu hình đến file kết quả với mô hình thật. |
| S2 | `test_s2_fp16_gpu` | Chỉ chạy khi có CUDA (marker `gpu`): `fit` ba epoch với `fp16 = True` trên mô hình nhỏ; so `evaluate(fp16=True)` với `fp16=False`. | Hoàn tất; mọi giá trị trong `history` hữu hạn; hai kết quả `evaluate` gần nhau (`atol 1e-2`) | `GradScaler` hoặc autocast gây `NaN`; điểm `fp16` lệch hẳn `fp32`. |

Nếu không có mạng hoặc `HF_HUB_OFFLINE=1`, test S1 **bỏ qua**; nếu không có CUDA, test S2 **bỏ qua**.

## 4. Hai ví dụ đọc test chi tiết

**Ví dụ 1: F2 (tiếp tục sau khi bị ngắt cho kết quả như chạy liền)**

```python
def test_f2_resume_matches_uninterrupted(tmp_path, monkeypatch):
    calls = []
    real = engine.train_one_epoch
    monkeypatch.setattr(engine, "train_one_epoch",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    def run(tag, **fit_kw):
        set_seed(42)
        model = tiny_model()                                   # dropout bằng 0 ở mọi nơi
        train = make_loader(24, 4, shuffle=True, seed=1)
        val = make_loader(8, 4)
        cfg = base_cfg(tmp_path / tag, epochs=3)
        return engine.fit(model, train, val, cfg,
                          str(tmp_path / tag / "ckpt"), str(tmp_path / tag / "res"), **fit_kw)

    a = run("A")
    n_a = len(calls); calls.clear()
    run("B", stop_after_epoch=2)                               # như bị ngắt phiên
    b = run("B")                                               # phiên mới, tiếp tục
    assert n_a == 3 and len(calls) == 3                        # hai epoch rồi một epoch
    for col in ("train_loss", "val_rmse"):
        assert b["history"][col].to_numpy() == pytest.approx(a["history"][col].to_numpy(), abs=1e-6)
```

Phần then chốt: lần B dựng lại mô hình và loader như một phiên Colab mới, rồi `fit` phải khôi phục mô hình, bộ tối ưu, lịch tốc độ học và thứ tự dữ liệu (`seed + epoch`) để epoch thứ ba y hệt lần chạy liền. Nếu `fit` đặt lại seed của loader chỉ một lần ở đầu hàm, hoặc quên khôi phục bộ tối ưu, hai chuỗi sẽ lệch và test thất bại.

**Ví dụ 2: F5 (Test không bao giờ bị chạm)**

```python
def test_f5_test_set_never_touched(tmp_path):
    cfg = base_cfg(tmp_path, epochs=2)
    loaders = (make_loader(24, 4, shuffle=True), make_loader(8, 4), ExplodingLoader())

    result = train.run_training(cfg, model=tiny_model(), loaders=loaders)

    assert result["history"].shape[0] == 2                     # chạy xong, không ném AssertionError
    params = inspect.signature(engine.fit).parameters
    assert not any("test" in name for name in params)
```

`ExplodingLoader` ném lỗi nếu ai lặp qua nó. Test chứng minh hai điều: `run_training` nhận cả ba loader nhưng không lặp qua loader Test, và `fit` không có đường nhận tập Test qua chữ ký.

## 5. Cách chạy, ánh xạ tiêu chí chấp nhận và giới hạn

Lệnh chạy:

```bash
pytest tests/test_utils.py tests/test_engine.py tests/test_train.py tests/test_predict.py -v
pytest tests/test_utils.py tests/test_engine.py tests/test_train.py tests/test_predict.py -v -m "not slow and not gpu"
```

Các marker `slow` và `gpu` được đăng ký trong `tests/conftest.py` (dùng chung với các file test khác).

Ánh xạ với tiêu chí chấp nhận ở mục 1.5 của `docs/04_engine_and_utils.md`:

| Tiêu chí | Test chứng minh |
| :--- | :--- |
| 1. Seed tái lập; checkpoint nguyên tử, nạp lại đủ | U1, U3, U4 |
| 2. `fit` đúng số epoch, `history` đủ cột, `best_model.pt` ứng với RMSE Val nhỏ nhất | F1, L4, L6 |
| 3. Tiếp tục như chạy liền; cấu hình khác bị từ chối; không ghi đè âm thầm | F2, F3 |
| 4. Chỉ `export_test_predictions` lặp qua `test_loader` | F5, P6 |
| 5. Loss không hữu hạn dừng huấn luyện; `fp16` trên CPU tự tắt | L5, L7, S2 |
| 6. `predict` trong 1.0-5.0, không đổi theo lô, trả lại chế độ | P1, P2, P3, P4 |
| 7. File xuất đúng cột, không ghi đè, kiểm tra khớp | P6, P7, P8 |
| 8. `run_meta.json` đủ số liệu cho `docs/02_evaluation.md` | F4, P8 |
| 9. Cấu hình đọc đúng và từ chối đúng; seed đặt trước khi dựng mô hình | T1, T2, T3, T4, T5 |
| 10. Toàn bộ test đạt | Mọi test ở trên |

Giới hạn của bộ test:

* Không kiểm tra chất lượng mô hình sau huấn luyện đầy đủ, không kiểm tra thời gian một epoch thật; số liệu đó lấy từ `run_meta.json` của lần chạy Tuần 1.
* F2 chỉ chứng minh tái lập trên CPU với `dropout = 0`; trên GPU và với dropout, tiếp tục không giống từng bit (xem `docs/04_engine_and_utils.md` mục 2.11). S2 chỉ kiểm tra tính hữu hạn và độ gần nhau, không kiểm tra tái lập.
* Chưa chạy thử khi soạn tài liệu: nạp checkpoint bằng `weights_only=True` với trạng thái bộ tối ưu, lịch và `GradScaler` (U3 và F2 kiểm chứng ở lần chạy đầu); ngưỡng `atol` của F1, F2, P2 cần hiệu chỉnh nếu phiên bản `torch` cho sai khác số học lớn hơn dự kiến.
* Bước bị `GradScaler` bỏ (gradient inf, không gọi `scheduler.step()`) chỉ xảy ra trên GPU với fp16 nên không có test CPU; kiểm tra bằng log và `lr` trong `history.csv` ở lần chạy GPU đầu.
* Tên `monkeypatch` dùng trong T4 và T5 (`train.build_model`, `train.build_tokenizer`, `train.create_dataloaders`, `train.run_training`) đòi `train.py` nhập các hàm này theo dạng `from going_modular.model_builder import build_model` ở đầu file; nếu nhập theo dạng khác thì đổi đích `monkeypatch`.
* S1 cần mạng hoặc bộ nhớ đệm Hugging Face và dùng dữ liệu tổng hợp, nên không kiểm tra chất lượng trên dữ liệu Yelp thật.