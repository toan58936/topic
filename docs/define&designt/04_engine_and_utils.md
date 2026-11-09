# Đặc tả Thiết kế Module Huấn luyện, Suy luận và Tiện ích — `utils.py`, `engine.py`, `train.py`, `predict.py`

> **Tài liệu:** `docs/04_engine_and_utils.md` (Define & Design cho huấn luyện, suy luận và tiện ích)
> **Module liên quan:** `src/utils.py`, `src/engine.py`, `src/train.py`, `src/predict.py`
> **Kế hoạch kiểm thử:** `docs/04_engine_and_utils_tests.md` (tách riêng)
> **Phụ thuộc:** `docs/define.md` (mục 9, 11, 12, 14), `docs/design.md`; `docs/01_data.md` (dạng batch, `create_dataloaders`, bộ sinh xáo trộn của `train_loader`); `docs/02_evaluation.md` (`rmse`, `mae`, `clip_scores`, đầu vào của `build_model_summary`); `docs/03_model.md` (`build_model`, `build_tokenizer`, `count_parameters`)
> **Dùng bởi:** `notebooks/colab_runner.ipynb`; dashboard đọc kết quả do module này xuất (`docs/05_dashboard.md`, đề xuất, chưa viết)
> **Trạng thái:** Bản nháp trước khi hiện thực (Pre-implementation Design Specification)

---

## PHẦN I: DEFINE — Xác định Bài toán & Yêu cầu

### 1.1. Bối cảnh & Mục tiêu

Bốn module này là **phần chạy**: đưa mô hình ở `docs/03_model.md` qua huấn luyện trên Train, chọn checkpoint tốt nhất bằng Val, rồi tạo ra các file kết quả mà `docs/02_evaluation.md` và dashboard dùng.

| Module | Nhiệm vụ |
| :--- | :--- |
| `utils.py` | Cố định seed, chọn thiết bị, ghi file nguyên tử, lưu và nạp checkpoint |
| `engine.py` | Hàm loss, bộ tối ưu và lịch tốc độ học, một epoch huấn luyện, kiểm định, và `fit` điều phối nhiều epoch (checkpoint mỗi epoch, tiếp tục sau khi bị ngắt, lịch sử huấn luyện) |
| `train.py` | Đọc file cấu hình YAML, kiểm tra, rồi chạy một lần huấn luyện từ đầu đến cuối |
| `predict.py` | Dự đoán cho một hoặc nhiều review; xuất dự đoán trên tập Test của từng mô hình; ghép thành `test_predictions.csv` |

Mục tiêu:

* Huấn luyện ba backbone **cùng thiết lập, cùng ngân sách**, chỉ khác backbone (theo `docs/define.md` mục 9).
* Sống sót qua việc Colab ngắt phiên: lưu checkpoint mỗi epoch và tiếp tục được (theo `docs/define.md` mục 11).
* Cung cấp số liệu mà Define đòi ở cuối Tuần 1: thời gian thực tế của một epoch, và RMSE trên Val của Huber so với MSE trên BERT (theo `docs/define.md` mục 14).
* Dùng tập Test **một lần**, ở cấu hình cuối, đúng quy tắc của `docs/02_evaluation.md`.
* Mọi lần chạy ghi lại cấu hình, thời gian, số tham số để tái lập và để `model_summary.csv` có đủ đầu vào.

**Bảy bài toán cốt lõi cần giải quyết:**

1. **Phiên Colab bị ngắt.** Huấn luyện có thể mất hàng chục phút đến vài giờ. Phải lưu trạng thái đầy đủ (mô hình, bộ tối ưu, lịch tốc độ học, lịch sử) sau mỗi epoch và tiếp tục đúng chỗ dừng, thay vì chạy lại từ đầu.
2. **So sánh công bằng và tái lập.** Cả ba backbone cần cùng siêu tham số huấn luyện, cùng số epoch, cùng thứ tự dữ liệu theo từng epoch. Thứ tự xáo trộn phải phụ thuộc vào (seed, số epoch), không phụ thuộc việc có bị ngắt giữa chừng hay không.
3. **Test chỉ dùng một lần.** Nếu hàm huấn luyện có thể chạm Test thì việc chọn loss, chọn checkpoint, chọn số epoch dễ bị Test chi phối. Vì vậy `fit` không nhận tập Test, và chỉ `predict.py` đọc Test, ở bước xuất kết quả cuối.
4. **Quyết định loss trước Tuần 2.** Define yêu cầu chạy MSE đối chứng trên BERT và đổi mặc định nếu RMSE Val của MSE thấp hơn đáng kể. Cần lịch sử huấn luyện lưu ra file và một hàm so sánh các lần chạy.
5. **`fp16` an toàn.** Điểm đầu ra có thể là `float16`; loss phải tính trên `float32`; gradient phải được co giãn (`GradScaler`); loss không hữu hạn phải dừng ngay chứ không âm thầm huấn luyện tiếp.
6. **Kết quả đến từ nhiều phiên Colab.** Dự đoán Test của từng mô hình sinh ra ở các phiên khác nhau; phải lưu riêng từng mô hình rồi ghép, kèm kiểm tra các file khớp `review_id` và nhãn.
7. **Dung lượng Drive và thời gian ghi.** Checkpoint đầy đủ (mô hình cộng trạng thái AdamW) lớn gấp khoảng ba lần trọng số riêng; cần quy tắc dọn dẹp để không đầy Drive.

### 1.2. Đầu vào / Đầu ra

| | Mô tả |
| :--- | :--- |
| **Đầu vào của `train.py`** | File cấu hình YAML (mục 2.3), tuỳ chọn ghi đè dòng lệnh |
| **Đầu ra của `train.py` / `fit`** | `checkpoints/<run_name>/best_model.pt` (và `last.pt` khi đang chạy); `results/runs/<run_name>/config.yaml`, `history.csv`, `run_meta.json`; một `dict` kết quả |
| **Đầu vào của `predict.py`** | Mô hình đã huấn luyện, tokenizer hoặc `test_loader`, tên mô hình |
| **Đầu ra của `predict.py`** | `predict`: số thực hoặc mảng đã clip về 1.0-5.0. `export_test_predictions`: `results/preds/<model>_test.csv` và `<model>_test_meta.json`. `merge_test_predictions`: `test_predictions.csv` và `test_predictions_with_text.csv` |

**Hai tên cần phân biệt:** `run_name` là tên một lần huấn luyện (tên thư mục, ví dụ `bert`, `bert_mse`); `model` là tên backbone dùng đặt cột `pred_<model>` và dòng của `model_summary.csv` (`bert`, `roberta`, `distilbert`). Một lần chạy thắng được xuất với `model` bằng backbone của nó, kể cả khi `run_name` là `bert_mse`.

### 1.3. Yêu cầu chức năng

| Mã | Yêu cầu |
| :--- | :--- |
| FR-1 | `set_seed` đặt seed cho `random`, `numpy`, `torch` (CPU và CUDA); tuỳ chọn `deterministic` bật chế độ xác định của cuDNN. |
| FR-2 | `get_device("auto")` chọn CUDA nếu có, không thì CPU; yêu cầu `cuda` khi không có CUDA thì báo lỗi. |
| FR-3 | `write_atomic` ghi vào file tạm cùng thư mục rồi đổi tên; lỗi giữa chừng không làm hỏng file cũ và không để lại file tạm. Mọi file kết quả và checkpoint dùng hàm này. |
| FR-4 | `save_checkpoint` chỉ chấp nhận `dict` gồm tensor và kiểu dữ liệu cơ bản; `load_checkpoint` nạp bằng `weights_only=True`. |
| FR-5 | `build_loss("huber", delta)` và `build_loss("mse")`; loss luôn tính trên `float32`. |
| FR-6 | `build_optimizer_and_scheduler`: AdamW; tham số một chiều (bias, LayerNorm) không áp weight decay; lịch tăng tuyến tính (warmup) rồi giảm tuyến tính về 0, cập nhật theo bước. |
| FR-7 | `train_one_epoch`: `fp16` bằng autocast và `GradScaler` (chỉ trên CUDA), cắt gradient theo chuẩn; loss không hữu hạn thì ném `FloatingPointError`. |
| FR-8 | `evaluate`: chế độ `eval`, không gradient; trả về `review_id`, `y_true`, `y_pred_raw` (chưa clip) theo đúng thứ tự `loader`, và loss trung bình trên mẫu. |
| FR-9 | `fit` chạy đúng `epochs` epoch (không dừng sớm); mỗi epoch: huấn luyện, kiểm định trên Val, tính chỉ số bằng `metrics.py`, ghi `history.csv`, lưu `last.pt`, và lưu `best_model.pt` khi RMSE Val (đã clip) tốt hơn mọi epoch trước. |
| FR-10 | Trước mỗi epoch, `fit` đặt lại bộ sinh xáo trộn của `train_loader` bằng `seed + epoch`. |
| FR-11 | `fit` tiếp tục từ `last.pt` khi cấu hình khớp, và từ chối khi cấu hình khác, khi lần chạy đã hoàn tất hoặc khi đã có kết quả mà không được phép ghi đè (bảng ở mục 2.5). |
| FR-12 | `fit` ghi `run_meta.json` gồm số tham số, thời gian từng epoch, tổng thời gian huấn luyện (cộng dồn qua các lần tiếp tục), epoch tốt nhất, trạng thái. |
| FR-13 | `fit` không nhận `test_loader`: không lặp qua tập Test và không dùng nó để chọn mô hình, loss hay số epoch. (`run_training` vẫn gọi `create_dataloaders` nên đọc `test.parquet` để dựng loader, nhưng không lặp qua.) |
| FR-14 | `compare_runs` đọc `history.csv` của nhiều lần chạy và trả về bảng so sánh RMSE Val tốt nhất. |
| FR-15 | `load_config` áp mặc định, ép kiểu số (kể cả khi YAML đọc `2e-5` thành chuỗi), từ chối khoá lạ và giá trị không hợp lệ. |
| FR-16 | `resolve_output_bias` đổi `output_bias_init: train_mean` thành trung bình nhãn Train (chỉ đọc cột `stars` của `train.parquet`). |
| FR-17 | `run_training` thực hiện: `set_seed` rồi chọn thiết bị, tạo tokenizer và DataLoader, `build_model`, `fit`; ghi bản sao cấu hình. Cho phép truyền sẵn mô hình và DataLoader (dùng để kiểm thử). |
| FR-18 | `train.py` chạy được bằng dòng lệnh: `python -m src.train --config <file>`, có `--set khoá=giá_trị` để ghi đè và `--overwrite`, `--no-resume`. |
| FR-19 | `predict` nhận một chuỗi (trả về `float`) hoặc danh sách chuỗi (trả về mảng `numpy`), điểm đã clip về 1.0-5.0 theo mặc định; không để mô hình ở chế độ khác với lúc nhận. |
| FR-20 | `load_trained_model` dựng lại mô hình từ `best_model.pt` và trả về mô hình ở chế độ `eval` kèm thông tin lần chạy. |
| FR-21 | `export_test_predictions` chạy mô hình trên `test_loader` đúng một lần, ghi điểm thô và điểm đã clip, `rmse_raw`, thời gian suy luận; không ghi đè trừ khi `overwrite=True`. |
| FR-22 | `merge_test_predictions` ghép dự đoán của các mô hình (và baseline nếu có) thành `test_predictions.csv` và `test_predictions_with_text.csv`, kiểm tra các file khớp `review_id` và nhãn. |
| FR-23 | `collect_summary_inputs` gom `param_counts`, `train_seconds`, `inference_seconds`, `rmse_raw_by_model` để truyền vào `build_model_summary`. |
| FR-24 | Kiểm tra đầu vào hợp lệ và báo lỗi rõ ràng. |

### 1.4. Ràng buộc

* `seed = 42`, `fp16` bật, checkpoint mỗi epoch, cấu hình mỗi lần chạy được ghi lại (theo `docs/define.md` mục 11).
* Cả ba backbone dùng cùng siêu tham số huấn luyện, cùng loss, cùng seed (theo `docs/define.md` mục 9). Không dừng sớm, để ngân sách huấn luyện bằng nhau.
* Chọn checkpoint tốt nhất bằng RMSE trên **Val**, tính qua `metrics.rmse` trên điểm đã clip (theo `docs/02_evaluation.md` mục 2.10). Tập Test không bao giờ được dùng để chọn checkpoint, loss hay số epoch.
* Điểm đầu ra của mô hình là điểm thô; clip chỉ áp dụng khi tính chỉ số và khi dự đoán, bằng `clip_scores` của `metrics.py`.
* `engine.py` import `metrics.py` và `model_builder.py`; `predict.py` import thêm `data_setup.py` và `engine.py`; `train.py` import tất cả. Không có import theo chiều ngược lại (số thứ tự file tăng dần).
* `src/__init__.py` để trống (theo `docs/02_evaluation.md`), nên `python -m src.train` chạy được từ thư mục gốc repo.
* Checkpoint chỉ chứa tensor và kiểu cơ bản (`dict`, `list`, `str`, `int`, `float`, `bool`, `None`) để nạp được bằng `weights_only=True`; số `numpy` phải đổi sang kiểu Python trước khi lưu.
* Thư viện: `torch`, `transformers` (lịch tốc độ học), `pyyaml`, `pandas`, `numpy`; ghi trong `requirements.txt`.
* Một seed cho mỗi mô hình; tái lập đến mức từng epoch trên CPU, không đảm bảo từng bit trên GPU hay qua điểm tiếp tục (xem mục 2.11).

### 1.5. Tiêu chí chấp nhận

1. `set_seed` cho cùng chuỗi số ngẫu nhiên khi gọi lại; checkpoint ghi nguyên tử và nạp lại đủ nội dung bằng `weights_only=True`.
2. `fit` chạy đúng số epoch, `history.csv` đủ cột, `best_model.pt` ứng với RMSE Val nhỏ nhất (nạp lại và đánh giá cho đúng giá trị đó).
3. Dừng sau epoch k rồi tiếp tục cho cùng kết quả như chạy liền (trên CPU, dropout bằng 0); cấu hình khác bị từ chối; lần chạy đã hoàn tất không bị ghi đè âm thầm.
4. Toàn bộ quy trình huấn luyện chạy được khi `test_loader` bị cài để lỗi nếu có ai lặp qua nó; chỉ `export_test_predictions` lặp qua `test_loader` (`merge_test_predictions` chỉ đọc cột `text` từ `test.parquet`).
5. Loss không hữu hạn dừng huấn luyện bằng `FloatingPointError`; `fp16` trên CPU tự tắt kèm cảnh báo.
6. `predict` trả về số thực trong 1.0-5.0, không đổi theo `batch_size`, và không để mô hình ở chế độ `eval` nếu lúc nhận nó đang ở `train`.
7. `export_test_predictions` và `merge_test_predictions` ghi đúng cột, không ghi đè âm thầm, và từ chối khi các file không khớp `review_id` hoặc nhãn.
8. `run_meta.json` đủ số liệu cho `docs/02_evaluation.md` (`train_seconds`, số tham số) và cho quyết định quy mô mẫu (thời gian mỗi epoch).
9. Bộ test trong `docs/04_engine_and_utils_tests.md` đều đạt.

---

## PHẦN II: DESIGN — Thiết kế Chi tiết

### 2.1. Vị trí trong kiến trúc hệ thống

```text
configs/<run>.yaml
      │
      ▼
train.py: load_config → run_training
      │   set_seed → get_device → build_tokenizer (03) → create_dataloaders (01)
      │   → build_model (03) → fit
      ▼
engine.py: fit ── train_one_epoch ──► evaluate (val) ──► metrics.rmse / mae (02)
      │                                   │
      ├──► checkpoints/<run_name>/{last.pt, best_model.pt}   (Drive, qua utils.write_atomic)
      └──► results/runs/<run_name>/{config.yaml, history.csv, run_meta.json}

                  (chỉ khi cấu hình đã chốt, mỗi mô hình một lần)
predict.py: load_trained_model → export_test_predictions (đọc test_loader)
      └──► results/preds/<model>_test.csv, <model>_test_meta.json
            │
            ▼
      merge_test_predictions ──► test_predictions.csv, test_predictions_with_text.csv
      collect_summary_inputs ──► build_model_summary (02) ──► model_summary.csv
```

### 2.2. Quy trình chạy

**Tuần 1 (kịp Mốc 1, theo `docs/define.md` mục 14):**

1. Tạo thư mục dữ liệu nhỏ `data_smoke/` (khoảng 8 review lấy từ ba file `parquet`) và chạy `train.py --config configs/smoke.yaml`: bước Smoke test của `docs/design.md` mục 8 (không lỗi, loss hợp lệ, mỗi review ra một số thực). **`data_smoke/` chứa text Yelp thật nên phải nằm trong `.gitignore`** (Design mục 7 cấm đưa cột `text` lên repo); hoặc sinh dữ liệu tổng hợp thay vì lấy review thật.
2. **Đo tốc độ trước khi chạy đầy đủ:** chạy `configs/bert.yaml` vài trăm bước (dòng log mỗi `log_every` bước ghi số bước/giây và thời gian epoch ước tính; có thể dừng bằng ngắt thủ công hoặc `epochs` nhỏ). Dựa vào đó quyết định giữ 60.000 hay tăng lên 100.000 review. Nếu đổi kích thước mẫu thì chạy lại `prepare_splits(overwrite=True)` và baseline **trước** các lần chạy đầy đủ, để không lần chạy cuối nào phải lặp lại.
3. Chạy hai lần huấn luyện **đầy đủ** trên BERT với cùng thiết lập: `configs/bert.yaml` (Huber, `run_name: bert`) và `configs/bert_mse.yaml` (MSE, `run_name: bert_mse`). Dùng đủ thiết lập để lần chạy thắng được dùng làm mô hình cuối, không phải huấn luyện lại.
4. `compare_runs(["bert", "bert_mse"], results_dir)` để so RMSE Val tốt nhất theo ngưỡng "đáng kể" mà nhóm chốt **trước khi xem kết quả** (xem mục 2.11).
5. `epoch_seconds_mean` trong `run_meta.json` của hai lần chạy đầy đủ dùng để xác nhận ngân sách thời gian Tuần 2, không còn dùng để đổi kích thước mẫu (đã quyết ở bước 2).

**Tuần 2:** chạy `roberta` và `distilbert` với loss đã chốt. Nếu MSE thắng thì lần chạy `bert_mse` là `bert` cuối; nếu Huber thắng thì `bert` là bản cuối. Không có lần chạy cuối nào phải lặp lại vì quyết định loss; lần lặp duy nhất có thể xảy ra là khi bước đo tốc độ làm đổi kích thước mẫu, và việc đó được làm trước bước 3.

**Sau khi mọi cấu hình đã chốt (mỗi mô hình một lần, trên Test):**

1. Với mỗi mô hình: `load_trained_model` → `create_dataloaders` → `export_test_predictions(..., model_name=backbone)`.
2. `merge_test_predictions` (kèm dự đoán baseline nếu có) → `collect_summary_inputs` → `build_model_summary` → `write_model_summary` (`docs/02_evaluation.md`).
3. Sao chép `test_predictions.csv` và `model_summary.csv` vào `reports/` (xem `docs/design.md` mục 9.2).

### 2.3. Cấu hình một lần chạy — `train.py`

Cấu hình là một file YAML trong `configs/`. Khoá và mặc định:

| Khoá | Kiểu | Mặc định | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| `run_name` | `str` | bằng `backbone` | Tên lần chạy; tên thư mục kết quả |
| `backbone` | `str` | bắt buộc | `bert`, `roberta`, `distilbert` (`docs/03_model.md`) |
| `pooling` | `str` | `mean` | `mean` hoặc `cls` |
| `dropout` | `float` | `0.1` | Dropout trước head |
| `output_bias_init` | `float`, `train_mean` hoặc `null` | `train_mean` | Khởi tạo bias của head; `train_mean` là trung bình nhãn Train |
| `loss` | `str` | `huber` | `huber` hoặc `mse` |
| `huber_delta` | `float` | `1.0` | Chỉ dùng khi `loss = huber` |
| `seed` | `int` | `42` | Hạt giống |
| `max_length` | `int` | `256` | Số token tối đa |
| `batch_size` | `int` | `16` | Kích thước lô (đề xuất) |
| `epochs` | `int` | `3` | Số epoch cố định (đề xuất) |
| `lr` | `float` | `2e-5` | Tốc độ học tối đa (đề xuất) |
| `weight_decay` | `float` | `0.01` | Áp lên tham số nhiều chiều (đề xuất) |
| `warmup_ratio` | `float` | `0.1` | Tỉ lệ bước tăng tốc độ học (đề xuất) |
| `grad_clip` | `float` hoặc `null` | `1.0` | Ngưỡng cắt chuẩn gradient |
| `fp16` | `bool` | `true` | Tự tắt kèm cảnh báo khi không có CUDA |
| `num_workers` | `int` | `0` | Theo `docs/01_data.md` |
| `device` | `str` | `auto` | `auto`, `cpu`, `cuda` |
| `log_every` | `int` | `100` | Số bước giữa hai dòng log; `0` để tắt |
| `data_dir` | `str` | bắt buộc | Thư mục chứa ba file `parquet` (bản sao cục bộ) |
| `checkpoint_dir` | `str` | bắt buộc | Gốc lưu checkpoint (Drive) |
| `results_dir` | `str` | bắt buộc | Gốc lưu kết quả (Drive) |

Ví dụ `configs/bert.yaml`:

```yaml
backbone: bert
pooling: mean
dropout: 0.1
output_bias_init: train_mean
loss: huber
huber_delta: 1.0
seed: 42
max_length: 256
batch_size: 16
epochs: 3
lr: 2.0e-5          # viết 2.0e-5, không viết 2e-5 (xem mục 2.6)
weight_decay: 0.01
warmup_ratio: 0.1
grad_clip: 1.0
fp16: true
data_dir: /content/data
checkpoint_dir: /content/drive/MyDrive/project/checkpoints
results_dir: /content/drive/MyDrive/project/results
```

`configs/bert_mse.yaml` chỉ khác `run_name: bert_mse` và `loss: mse` (hoặc dùng `--set`). `configs/smoke.yaml` trỏ `data_dir` tới `data_smoke/`, đặt `epochs: 1`, `batch_size: 4`, `max_length: 64`, `backbone: distilbert` (nhỏ nhất) và `run_name: smoke`.

Siêu tham số huấn luyện (`batch_size`, `epochs`, `lr`, `weight_decay`, `warmup_ratio`, `grad_clip`) là đề xuất của tài liệu này: Define không nêu giá trị. Cả ba backbone dùng cùng giá trị.

### 2.4. `utils.py`

```python
set_seed(seed: int = 42, deterministic: bool = False) -> None
get_device(prefer: str = "auto") -> torch.device
write_atomic(path: str, write_fn: Callable[[str], None]) -> None
save_checkpoint(path: str, state: dict) -> None
load_checkpoint(path: str, map_location="cpu") -> dict
```

* **`set_seed`:** đặt `random.seed`, `numpy.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`. Với `deterministic=True`, đặt thêm `torch.backends.cudnn.deterministic = True` và `benchmark = False` (chậm hơn, mặc định tắt). Hàm không động tới bộ sinh riêng của `prepare_splits` (xem `docs/01_data.md` mục 2.8).
* **`write_atomic(path, write_fn)`:** tạo thư mục cha nếu thiếu; gọi `write_fn(tmp_path)` với `tmp_path` cùng thư mục đích; thành công thì `os.replace(tmp_path, path)`; lỗi thì xoá file tạm và ném lại lỗi. Hàm này bao mọi lần ghi lên Drive (checkpoint, `history.csv`, `run_meta.json`, CSV dự đoán).
* **`save_checkpoint(path, state)`:** kiểm tra `state` là `dict`, rồi `write_atomic(path, lambda p: torch.save(state, p))`.
* **`load_checkpoint(path, map_location)`:** `torch.load(path, map_location=map_location, weights_only=True)`; `FileNotFoundError` nếu thiếu file.

**Hai file checkpoint của mỗi lần chạy:**

| File | Nội dung (khoá) | Khi nào | Mục đích |
| :--- | :--- | :--- | :--- |
| `last.pt` | `format_version`, `epoch` (số epoch đã xong), `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict` (hoặc `None`), `history` (danh sách `dict`), `best_val_rmse`, `best_epoch`, `train_seconds`, `run_config` | Ghi đè sau mỗi epoch | Tiếp tục huấn luyện |
| `best_model.pt` | `format_version`, `epoch`, `val_rmse`, `model_state_dict`, `run_config`, `backbone_config` (`model.backbone.config.to_dict()`, chỉ kiểu cơ bản) | Khi RMSE Val tốt hơn mọi epoch trước | Suy luận và xuất kết quả |

**Dung lượng ước tính** (`float32`, từ số tham số ở `docs/03_model.md` mục 2.9): `best_model.pt` khoảng 0,44 GB (BERT), 0,50 GB (RoBERTa), 0,27 GB (DistilBERT); `last.pt` lớn gấp khoảng ba (mô hình cộng hai vector trạng thái của AdamW), khoảng 1,3 GB với BERT. Bốn lần chạy (`bert`, `roberta`, `distilbert`, `bert_mse`) giữ lại khoảng 1,6 GB `best_model.pt`. Vì vậy `last.pt` bị xoá khi lần chạy hoàn tất (trừ khi `keep_last=True`).

### 2.5. `engine.py`

```python
build_loss(name: str, delta: float = 1.0) -> nn.Module
build_optimizer_and_scheduler(model, lr, weight_decay, total_steps, warmup_ratio) -> (optimizer, scheduler)
train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device, *,
                scaler=None, fp16=True, grad_clip=1.0, log_every=100) -> dict
evaluate(model, loader, device, *, loss_fn=None, fp16=None) -> dict   # None = tự động: bật khi và chỉ khi CUDA
fit(model, train_loader, val_loader, run_config, checkpoint_dir, results_dir, *,
    device=None, resume=True, overwrite=False, keep_last=False, stop_after_epoch=None) -> dict
compare_runs(run_names, results_dir) -> pandas.DataFrame
```

**Hàm loss — `build_loss`:** `huber` dùng `torch.nn.HuberLoss(delta=delta)` (trung bình theo mẫu); `mse` dùng `torch.nn.MSELoss()`. Đầu ra mô hình luôn được ép `.float()` trước khi tính loss, nhãn là `float32`. Với `delta = 1`: sai số tuyệt đối 2 cho Huber 1,5 (MSE 4,0); sai số 0,5 cho Huber 0,125 (MSE 0,25).

**Bộ tối ưu và lịch — `build_optimizer_and_scheduler`:**

* AdamW với hai nhóm tham số: tham số một chiều (`p.ndim <= 1`: bias, trọng số LayerNorm) có `weight_decay = 0`; tham số còn lại dùng `weight_decay` của cấu hình. Mỗi tham số thuộc đúng một nhóm. Quy tắc theo số chiều dùng được chung cho ba backbone, vì tên LayerNorm khác nhau giữa chúng.
* Lịch tốc độ học: `get_linear_schedule_with_warmup` của `transformers` với `num_warmup_steps = ceil(warmup_ratio × total_steps)` và `total_steps = epochs × len(train_loader)`. Với `total_steps = 10`, `warmup_ratio = 0,2`: hệ số tốc độ học tại bước 0, 1, 2, 6, 10 lần lượt là 0, 0,5, 1,0, 0,5, 0. Cập nhật lịch sau mỗi bước tối ưu, không phải mỗi epoch.

**Một epoch huấn luyện — `train_one_epoch`:**

```text
model.train()
với mỗi batch:
    chuyển input_ids, attention_mask, labels sang device (review_id giữ nguyên)
    optimizer.zero_grad(set_to_none=True)
    với autocast(fp16, chỉ khi CUDA):  out = model(input_ids, attention_mask)
    loss = loss_fn(out.float(), labels)                  # float32
    nếu loss không hữu hạn: ném FloatingPointError (kèm số bước, epoch)
    nếu có scaler: scaler.scale(loss).backward(); scaler.unscale_(optimizer)
    ngược lại:     loss.backward()
    nếu grad_clip: clip_grad_norm_(model.parameters(), grad_clip)
    nếu có scaler: scale_trước = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                   bước_bị_bỏ = scaler.get_scale() < scale_trước      # gradient inf: GradScaler bỏ bước tối ưu
    ngược lại:     optimizer.step(); bước_bị_bỏ = False
    nếu không bước_bị_bỏ: scheduler.step()                            # tránh cảnh báo và lệch lịch một bước
trả về {"train_loss": trung bình theo mẫu, "n_batches", "lr": tốc độ học cuối}
```

`fp16=True` mà thiết bị không phải CUDA thì tắt `fp16` và `GradScaler`, kèm `UserWarning`. Với `evaluate`, `predict` và `export_test_predictions`, `fp16=None` (mặc định) nghĩa là tự động: bật khi có CUDA, tắt khi không, **không cảnh báo**; chỉ `fp16=True` tường minh trên CPU mới cảnh báo. Ba hàm này dùng chung mặc định để cùng một review ra cùng điểm ở mọi đường suy luận. Gọi `loss.item()` mỗi bước để kiểm tra hữu hạn; chi phí này nhỏ so với một bước huấn luyện mô hình 256 token.

**Kiểm định — `evaluate`:** đặt `model.eval()`, dùng `torch.no_grad()`, autocast như trên; trả về

```python
{"review_id": List[str], "y_true": np.ndarray, "y_pred_raw": np.ndarray, "loss": float | None}
```

theo đúng thứ tự `loader` (với `val_loader` và `test_loader` không xáo trộn). `loss` là trung bình theo mẫu khi truyền `loss_fn`, không thì `None`. `y_pred_raw` là điểm thô chưa clip. `evaluate` là hàm suy luận duy nhất của dự án: `fit` dùng nó trên Val và `predict.py` dùng nó trên Test. Nó trả lại `model` về chế độ ban đầu khi xong.

**Điều phối — `fit`.** `run_config` là bản cấu hình đã kiểm tra (mục 2.3). Các khoá `fit` đọc: `epochs`, `lr`, `weight_decay`, `warmup_ratio`, `grad_clip`, `loss`, `huber_delta`, `fp16`, `seed`, `log_every`, `run_name`.

```text
1. kiểm tra tham số; chọn device; chuyển mô hình lên device
2. thư mục chạy: checkpoint_dir/run_name và results_dir/runs/run_name; áp dụng quy tắc ở bảng bên dưới
3. dựng loss, bộ tối ưu, lịch (total_steps = epochs × len(train_loader)), GradScaler nếu fp16 trên CUDA
4. nếu tiếp tục: nạp last.pt vào mô hình, bộ tối ưu, lịch, scaler; lấy history, best_*, train_seconds
5. với epoch = (epoch đã xong + 1) .. epochs:
     a. nếu train_loader.generator có: generator.manual_seed(seed + epoch)
     b. t0 = thời điểm bắt đầu; train_one_epoch; evaluate(val_loader, loss_fn)
     c. val_rmse = metrics.rmse(y_true, y_pred_raw)            # có clip
        val_mae  = metrics.mae(y_true, y_pred_raw)
        val_rmse_raw = metrics.rmse(y_true, y_pred_raw, clip=False)
     d. epoch_seconds = thời gian b (đồng bộ CUDA trước khi đo); train_seconds += epoch_seconds
     e. is_best = val_rmse < best_val_rmse   (bằng nhau: giữ epoch sớm hơn)
     f. nối một dòng vào history; ghi history.csv và run_meta.json (nguyên tử)
     g. nếu is_best: lưu best_model.pt
     h. lưu last.pt (sau cùng, để last.pt luôn khớp history)
     i. nếu stop_after_epoch == epoch: dừng (mô phỏng ngắt phiên)
6. hoàn tất: run_meta.status = "completed"; xoá last.pt nếu không keep_last
7. trả về dict: run_name, best_epoch, best_val_rmse, train_seconds, epoch_seconds_mean,
   n_params, history (DataFrame), checkpoint_dir, results_dir
```

* **Thời gian một epoch** (`epoch_seconds`) gồm huấn luyện và kiểm định Val, không gồm thời gian ghi checkpoint. Đây là con số quyết định quy mô mẫu ở Tuần 1.
* Khi `stop_after_epoch` dừng chạy, `status` giữ `"running"` để lần sau biết cần tiếp tục.
* Nếu `train_loader.generator` là `None` (loader dựng ngoài `create_dataloaders`), `fit` bỏ qua bước 5a và cảnh báo rằng việc tiếp tục sẽ không tái lập thứ tự dữ liệu.

**Cột `history.csv`** (một dòng mỗi epoch):

| Cột | Ý nghĩa |
| :--- | :--- |
| `epoch` | Số epoch, từ 1 |
| `train_loss`, `val_loss` | Loss trung bình theo mẫu; theo dõi overfit (Define mục 13) |
| `val_rmse`, `val_mae` | Trên điểm đã clip; `val_rmse` là tiêu chí chọn checkpoint |
| `val_rmse_raw` | RMSE Val trên điểm chưa clip, để tham khảo |
| `lr` | Tốc độ học ở cuối epoch |
| `epoch_seconds` | Giây của epoch (xem trên) |
| `is_best` | `True` nếu epoch này tốt hơn mọi epoch trước |

**Nội dung `run_meta.json`:** `run_name`, `backbone`, `loss`, `seed`, `status` (`running` hoặc `completed`), `epochs_done`, `best_epoch`, `best_val_rmse`, `n_params` (từ `count_parameters`), `train_seconds` (cộng dồn), `epoch_seconds_mean`, `device` (tên GPU hoặc `cpu`), `torch_version`, `transformers_version`.

**Quy tắc tiếp tục và ghi đè (trạng thái đã có → hành vi):**

| Tình huống | Hành vi |
| :--- | :--- |
| Chưa có gì của lần chạy này | Chạy mới từ epoch 1 |
| Có `last.pt`, `resume=True`, cấu hình khớp | Tiếp tục từ epoch kế tiếp |
| Có `last.pt`, `resume=True`, cấu hình khác | `ValueError` nêu khoá khác nhau |
| `run_meta.status = "completed"` | `FileExistsError`, trừ khi `overwrite=True` (xoá kết quả cũ, chạy lại) |
| Có kết quả cũ, `resume=False`, `overwrite=False` | `FileExistsError` |
| Có `last.pt`, `resume=True`, `overwrite=True` | `overwrite` thắng: xoá kết quả và checkpoint cũ, chạy lại từ epoch 1 |
| `stop_after_epoch` ngoài `[1, epochs − 1]` | `ValueError` (dừng sau epoch cuối thì lần chạy đã hoàn tất, không còn gì để tiếp tục) |

Hai cấu hình được coi là khớp khi mọi khoá bằng nhau, trừ `num_workers`, `device`, `log_every`, `data_dir`, `checkpoint_dir`, `results_dir`.

**`compare_runs(run_names, results_dir)`** đọc `history.csv` của từng lần chạy và trả về bảng một dòng mỗi lần chạy: `run_name`, `best_epoch`, `best_val_rmse`, `best_val_mae`, `train_seconds`, `epoch_seconds_mean`, cùng cột `rel_diff_pct_vs_first` (chênh lệch tương đối của `best_val_rmse` so với dòng đầu, theo công thức $100 \times (\text{RMSE}_{\text{đầu}} - \text{RMSE}_{\text{này}}) / \text{RMSE}_{\text{đầu}}$, dương nghĩa là tốt hơn).

### 2.6. `train.py`

```python
DEFAULT_CONFIG: dict
load_config(path: str, overrides: Optional[Dict[str, str]] = None) -> dict
resolve_output_bias(cfg: dict) -> Optional[float]
run_training(cfg: dict, *, model=None, loaders=None, resume=True, overwrite=False) -> dict
main(argv=None) -> int
```

**`load_config`:**

1. Đọc YAML bằng `yaml.safe_load`; áp `DEFAULT_CONFIG` cho khoá thiếu; áp `overrides` (giá trị chuỗi dòng lệnh được `yaml.safe_load` đọc để có đúng kiểu).
2. Ép kiểu bằng `int(...)` và `float(...)` cho các khoá số. Bước này bắt buộc vì PyYAML đọc `lr: 2e-5` thành **chuỗi** (chỉ `2.0e-5` mới ra số thực); không ép kiểu thì AdamW nhận chuỗi và lỗi muộn khó hiểu.
3. Kiểm tra: khoá lạ → `ValueError` (bắt lỗi gõ sai tên); thiếu `backbone`, `data_dir`, `checkpoint_dir` hoặc `results_dir` → `ValueError`; `backbone` không thuộc `BACKBONES`, `loss` ngoài `{huber, mse}`, `pooling` ngoài `{mean, cls}`, `epochs < 1`, `batch_size < 1`, `lr <= 0`, `warmup_ratio` ngoài `[0, 1)`, `output_bias_init` không phải số, `train_mean` hay `null` → `ValueError`.
4. `run_name` mặc định bằng `backbone`.

**`resolve_output_bias(cfg)`:** `train_mean` → trung bình cột `stars` của `data_dir/train.parquet` (chỉ đọc cột này); số → số đó; `null` → `None`. Chỉ dùng dữ liệu Train, không dùng Val hay Test.

**`run_training(cfg)`:**

```text
1. set_seed(cfg.seed)                          # seed ban đầu cho các bước dựng loader
2. device = get_device(cfg.device)
3. nếu loaders chưa truyền: tokenizer = build_tokenizer(backbone)
                            train_loader, val_loader, test_loader = create_dataloaders(...)
4. set_seed(cfg.seed)                          # NGAY TRƯỚC build_model (docs/03_model.md mục 2.7),
                                               # để trọng số head không phụ thuộc việc bước 3 có tiêu thụ RNG hay không
   nếu model chưa truyền: model = build_model(backbone, pooling, dropout,
                                              output_bias_init=resolve_output_bias(cfg))
5. ghi bản sao cấu hình: results_dir/runs/run_name/config.yaml
6. result = fit(model, train_loader, val_loader, cfg, checkpoint_dir, results_dir, ...)
7. return result
```

`test_loader` được tạo cùng hai loader kia (theo `create_dataloaders`) nhưng **không bao giờ được lặp** trong `run_training`. Hai tham số `model` và `loaders` cho phép kiểm thử mà không cần mạng.

**Dòng lệnh:**

```bash
python -m src.train --config configs/bert.yaml
python -m src.train --config configs/bert.yaml --set run_name=bert_mse --set loss=mse
```

`main` trả về mã thoát 0 khi chạy xong, khác 0 khi lỗi cấu hình.

### 2.7. `predict.py`

```python
predict(texts, model, tokenizer, *, device=None, max_length=256, batch_size=32,
        fp16=None, clip=True) -> float | np.ndarray
load_trained_model(checkpoint_path, device=None, *, backbone_config=None) -> (model, info)
export_test_predictions(model, test_loader, model_name, out_dir, *, device=None,
                        fp16=None, overwrite=False, run_info=None) -> dict
merge_test_predictions(model_names, preds_dir, test_parquet, out_dir, *,
                       baseline_preds=None, overwrite=False) -> pandas.DataFrame
collect_summary_inputs(model_runs, results_dir) -> dict
```

**`predict`:** nhận một chuỗi (trả về `float`) hoặc danh sách chuỗi (trả về `np.ndarray` shape `(N,)`). Gọi tokenizer theo từng lô `batch_size` với `padding=True` (đệm tới chuỗi dài nhất trong lô), `truncation=True`, `max_length`, `return_tensors="pt"`; chỉ đọc `["input_ids"]` và `["attention_mask"]` từ kết quả (nên tokenizer giả trong test chỉ cần hai khoá này). Kết quả không phụ thuộc `batch_size` và cách đệm (đảm bảo bởi `docs/03_model.md` mục 1.5, tiêu chí 2). Điểm đã clip bằng `clip_scores` khi `clip=True`; `clip=False` trả điểm thô. Hàm tự đặt `eval`, `no_grad`, rồi khôi phục chế độ ban đầu của mô hình.

**`load_trained_model(checkpoint_path)`:** nạp `best_model.pt` bằng `load_checkpoint`; đọc `run_config` để lấy `backbone`, `pooling`, `dropout`; xác định `backbone_config`: tham số truyền vào nếu có, không thì khoá `backbone_config` trong checkpoint (dựng lại bằng `AutoConfig.for_model(model_type, **dict)`, **không cần mạng**); chỉ khi checkpoint cũ không có khoá này mới rơi về `AutoConfig.from_pretrained` (cần mạng hoặc cache, kèm cảnh báo); dựng `build_model(backbone, pooling, dropout, pretrained=False, backbone_config=backbone_config)`; `load_state_dict` (nghiêm ngặt); đặt `eval`. Trả về `(model, info)` với `info = {"run_name", "backbone", "epoch", "val_rmse", "run_config"}`.

**`export_test_predictions(model, test_loader, model_name, out_dir)`** — hàm **duy nhất** đọc tập Test:

1. Nếu `out_dir/<model_name>_test.csv` đã có và `overwrite=False` thì `FileExistsError`.
2. Bắt đầu bấm giờ; `evaluate(model, test_loader, device, fp16=fp16)`; dừng bấm giờ (đồng bộ CUDA). `inference_seconds` là thời gian chạy hết `test_loader`, gồm nạp dữ liệu và token hoá, đo cùng một cách cho mọi mô hình.
3. Ghi `<model_name>_test.csv` với các cột `review_id`, `true_stars`, `pred_raw`, `pred` (`pred = clip_scores(pred_raw)`), đúng thứ tự `test_loader`.
4. Ghi `<model_name>_test_meta.json`: `model`, `run_name`, `checkpoint_epoch` (lấy từ `run_info` nếu có), `n`, `rmse_raw` (`metrics.rmse(y_true, pred_raw, clip=False)`), `inference_seconds`, `fp16`, `device`.
5. Trả về `dict` của meta.

Điểm thô (`pred_raw`) chỉ nằm trong file trung gian trên Drive; `test_predictions.csv` cuối chỉ chứa điểm đã clip để file nhẹ (theo `docs/design.md` mục 4).

**`merge_test_predictions(model_names, preds_dir, test_parquet, out_dir)`:**

1. Đọc `<model>_test.csv` của từng `model` trong `model_names` (theo thứ tự đó); kiểm tra mọi file có cùng chuỗi `review_id` và cùng `true_stars`, không thì `ValueError`.
2. Nếu có `baseline_preds` (đường dẫn hoặc `DataFrame` có cột `review_id`, `pred_baseline`, điểm đã clip): nối theo `review_id`; thiếu hoặc thừa `review_id` thì `ValueError`.
3. Tạo `test_predictions.csv` với các cột: `review_id`, `true_stars` (số nguyên), `pred_baseline` (chỉ khi có `baseline_preds`, đứng đầu), `pred_<model>` cho từng mô hình theo thứ tự `model_names`, rồi `abs_err_baseline` (nếu có) và `abs_err_<model>` (`|true_stars − pred|`) theo cùng thứ tự. `model_names` chỉ gồm các mô hình Transformer; baseline không có file `baseline_test.csv` mà đi vào qua `baseline_preds`.
4. Tạo `test_predictions_with_text.csv`: cùng các cột, thêm cột `text` ngay sau `review_id`, lấy từ `test_parquet` bằng `review_id`; `review_id` thiếu thì `ValueError`.
5. Ghi cả hai bằng `write_atomic`; có file sẵn và `overwrite=False` thì `FileExistsError`. Trả về `DataFrame` của bản không có `text`.

**`collect_summary_inputs(model_runs, results_dir)`:** `model_runs` ánh xạ `model` sang `run_name` (ví dụ `{"bert": "bert_mse", "roberta": "roberta", "distilbert": "distilbert"}`). Đọc `runs/<run_name>/run_meta.json` và `preds/<model>_test_meta.json`, trả về

```python
{"param_counts": {model: n_params}, "train_seconds": {model: ...},
 "inference_seconds": {model: ...}, "rmse_raw_by_model": {model: ...},
 "seconds_per_epoch": {model: ...}, "split_id": str}   # lấy từ meta dự đoán và run_meta
```

đúng các tham số của `build_model_summary` (`docs/02_evaluation.md` mục 2.7). Dòng baseline do notebook tự bổ sung vào các từ điển này (`fit_seconds` của baseline làm `train_seconds`, `seconds_per_epoch` và `params_total` để trống, `rmse_raw` tính từ điểm Ridge chưa clip hoặc để trống, `split_id` từ `run_meta`).

### 2.8. Cấu trúc file trên Drive

```
drive/project/
├── checkpoints/
│   └── <run_name>/            # bert, roberta, distilbert, bert_mse
│       ├── best_model.pt
│       └── last.pt            # chỉ tồn tại khi đang chạy dở
└── results/
    ├── runs/
    │   └── <run_name>/        # config.yaml, history.csv, run_meta.json
    ├── preds/                 # <model>_test.csv, <model>_test_meta.json
    ├── test_predictions.csv   # không text
    ├── test_predictions_with_text.csv
    └── model_summary.csv      # do docs/02_evaluation.md ghi
```

### 2.9. Ngoại lệ

| Hàm | Ngoại lệ |
| :--- | :--- |
| `get_device` | `ValueError` nếu `prefer` ngoài `{auto, cpu, cuda}`; `RuntimeError` nếu yêu cầu `cuda` mà không có |
| `save_checkpoint` | `TypeError` nếu `state` không phải `dict` |
| `load_checkpoint` | `FileNotFoundError` nếu thiếu file |
| `build_loss` | `ValueError` nếu tên ngoài `{huber, mse}` hoặc `delta <= 0` |
| `train_one_epoch` | `FloatingPointError` nếu loss không hữu hạn |
| `evaluate` | `ValueError` nếu `loader` rỗng |
| `fit` | `ValueError`: `epochs < 1`, `train_loader` hoặc `val_loader` rỗng, cấu hình khác khi tiếp tục; `FileExistsError` (bảng mục 2.5) |
| `compare_runs` | `FileNotFoundError` nếu thiếu `history.csv` của một lần chạy |
| `load_config`, `resolve_output_bias` | `ValueError`, `FileNotFoundError` (xem mục 2.6) |
| `predict` | `ValueError` nếu danh sách rỗng, phần tử không phải chuỗi, `max_length < 1` hoặc `batch_size < 1` |
| `load_trained_model` | `FileNotFoundError`; `ValueError` nếu checkpoint thiếu `run_config` hoặc không khớp mô hình |
| `export_test_predictions`, `merge_test_predictions` | `FileExistsError`; `ValueError` khi các file không khớp `review_id` hoặc nhãn, khi `baseline_preds` hoặc `test_parquet` thiếu `review_id`, hoặc khi `model_names` rỗng |
| `collect_summary_inputs` | `FileNotFoundError` nếu thiếu `run_meta.json` hoặc file meta dự đoán |

### 2.10. Tích hợp với các module khác

| Module | Cách phối hợp |
| :--- | :--- |
| `data_setup.py` (`01`) | `run_training` gọi `create_dataloaders`. `fit` dựa vào `train_loader.generator` (mục 2.6 của `01`: `torch.Generator().manual_seed(seed)`) để đặt lại thứ tự xáo trộn mỗi epoch. Batch gồm `input_ids`, `attention_mask`, `labels`, `review_id`; `review_id` không đi vào mô hình. |
| `metrics.py` (`02`) | `fit` dùng `rmse` và `mae` (có clip) để chọn checkpoint và ghi lịch sử; `predict.py` dùng `clip_scores` và `rmse(clip=False)`; đầu ra của `collect_summary_inputs` đưa vào `build_model_summary`. |
| `baseline.py` (`02`) | `merge_test_predictions` nhận dự đoán baseline qua `baseline_preds` (cột `review_id`, `pred_baseline`). `save_baseline` và `load_baseline` (`docs/02_evaluation.md` mục 2.8) lưu và nạp pipeline để tạo file này ở Tuần 2 mà không phải huấn luyện lại; nơi gọi lấy `pred_baseline` bằng `predict_baseline`. |
| `model_builder.py` (`03`) | `run_training` gọi `set_seed` rồi `build_model`; `count_parameters` cho `n_params`; `build_tokenizer` cho tokenizer; `load_trained_model` dựng lại bằng `pretrained=False`. Đầu ra mô hình được ép `.float()` trước loss, đúng ghi chú ở `03` mục 2.6. |
| Dashboard | Đọc `test_predictions.csv` và `model_summary.csv` đã commit vào `reports/`; không dùng module nào ở đây. |

### 2.11. Quyết định thiết kế và điểm mở

| Hạng mục | Quyết định |
| :--- | :--- |
| `seed = 42`, `fp16`, checkpoint mỗi epoch, cùng thiết lập cho ba backbone, MSE đối chứng trên BERT | Đã chốt (theo `docs/define.md`) |
| Chọn checkpoint bằng RMSE Val trên điểm đã clip; Test không dùng để chọn | Đã chốt (theo `docs/02_evaluation.md`) |
| `batch_size = 16`, `epochs = 3`, `lr = 2e-5`, `weight_decay = 0.01`, `warmup_ratio = 0.1`, `grad_clip = 1.0` | Đề xuất; Define không nêu. Xác nhận bằng thời gian epoch ở Tuần 1 |
| Không dừng sớm; số epoch cố định cho cả ba backbone | Đề xuất, để ngân sách huấn luyện bằng nhau |
| AdamW, không weight decay cho tham số một chiều; lịch tuyến tính có warmup | Đề xuất |
| Cấu hình bằng YAML; `--set` để ghi đè; từ chối khoá lạ | Đề xuất |
| `output_bias_init = train_mean` làm mặc định, như nhau cho ba backbone | Đề xuất (quyết định mà `docs/03_model.md` mục 2.7 giao cho tài liệu này) |
| Lần chạy đối chứng `bert_mse` dùng đủ thiết lập để làm mô hình cuối nếu thắng | Đề xuất |
| Ngưỡng "thấp hơn đáng kể" để đổi sang MSE | Chưa chốt: nhóm cần đặt con số trước khi xem kết quả (đề xuất: RMSE Val tốt nhất của MSE thấp hơn của Huber ít nhất 1% tương đối, tức `rel_diff_pct_vs_first` từ 1 trở lên). Lưu ý: chỉ có một seed nên 1% có thể nằm trong nhiễu giữa các lần chạy; và RMSE vốn thiên về MSE (cùng hàm mục tiêu) nên nên xét thêm `best_val_mae` trước khi quyết định |
| Chỉ số chi phí không cùng cơ sở | `train_seconds` của Transformer gồm cả kiểm định Val mỗi epoch; `fit_seconds` của baseline chỉ tính tổ hợp thắng, không tính cả lưới; suy luận baseline chạy CPU còn Transformer chạy GPU và gồm token hoá. Báo cáo phải ghi rõ định nghĩa từng cột và thiết bị (`device` trong `run_meta.json` và meta dự đoán) khi so sánh |
| `os.replace` trên Drive FUSE có thể không nguyên tử thật sự | Chưa kiểm chứng; nếu nghi ngờ, ghi cục bộ rồi sao chép; `load_checkpoint` lỗi trên tệp hỏng thì chạy lại epoch cuối |
| `last.pt` xoá khi hoàn tất; ghi thẳng lên Drive mỗi epoch | Đề xuất. Nếu thời gian ghi `last.pt` (khoảng 1,3 GB với BERT) chiếm phần đáng kể của một epoch thì ghi cục bộ rồi sao chép sang Drive; quyết định sau khi đo ở Tuần 1 |
| Tiếp tục tái lập đến mức từng epoch, không từng bit | Hạn chế đã biết: trạng thái ngẫu nhiên của dropout không được lưu, nên dropout sau điểm tiếp tục khác chạy liền; cuDNN trên GPU không đảm bảo xác định từng bit. Chạy tái lập chính xác chỉ được đảm bảo trên CPU với dropout bằng 0 |
| Mỗi mô hình huấn luyện một lần (một seed) | Hạn chế đã biết, nêu trong báo cáo (`docs/02_evaluation.md` mục 2.6) |
| `inference_seconds` đo cả nạp dữ liệu và token hoá | Đề xuất, đo giống nhau cho mọi mô hình |
| Gradient accumulation, tốc độ học riêng cho head, đóng băng một phần backbone | Chưa làm |
| Các giả định về hành vi của `torch`/`transformers` (nạp checkpoint bằng `weights_only=True`, trạng thái của lịch và `GradScaler` chỉ gồm kiểu cơ bản, `AutoConfig.for_model` dựng lại được cấu hình từ `dict`, `get_scale()` giảm khi bước bị bỏ) | Chưa chạy thử khi soạn tài liệu; test U3 và F2 kiểm chứng ở lần chạy đầu |
