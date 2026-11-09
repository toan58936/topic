# Tài liệu Design – Đề tài 7: Text Regression (Yelp)

Nov 8, 2026 · Bản chốt

Tài liệu này mô tả cách xây dựng hệ thống: kiến trúc, module, cấu trúc thư mục, luồng dữ liệu, dashboard, mốc thời gian, rủi ro kỹ thuật và quy trình làm việc với AI. Dự án làm gì, dữ liệu và tiêu chí nằm ở tài liệu Define. Đây là bản chốt: mọi mục là quyết định đã áp dụng, không còn mục chờ duyệt. Sau bản chốt chỉ sửa khi một lần chạy thật lộ ra vấn đề, và lý do phải ghi vào `docs/ai_log.md`.

## 1. Kiến trúc hệ thống

| Lớp | Thành phần | Vai trò |
| --- | --- | --- |
| Mã nguồn | GitHub, repo riêng tư (`src/`) | Nguồn duy nhất của code và của các file kết quả nhẹ không chứa nội dung review; nhánh riêng và Pull Request (theo Define); Colab kéo về bằng `git clone`/`git pull` |
| Tài liệu và kiểm thử | `docs/` và `tests/` | Đặc tả viết trước khi code (đưa cho AI), pytest chạy trên CPU với dữ liệu tổng hợp |
| Dữ liệu và trọng số | Google Drive (chế độ hạn chế, không liên kết công khai) | Ba file dữ liệu đã chia và `split_stats.json`, checkpoint, trọng số cuối, dự đoán Test từng mô hình, bảng kết quả, bản `test_predictions` đầy đủ có cột `text` |
| Tính toán | Colab GPU (notebook mỏng) | Mount Drive, kéo code, huấn luyện, đánh giá và xuất file kết quả |
| Giao diện | Dashboard kết quả (Streamlit) | Chạy trên máy cá nhân, đọc `reports/test_predictions.csv` và `reports/model_summary.csv`, không cần nạp mô hình |

Mô hình: review qua tokenizer, Transformer Encoder (BERT, RoBERTa hoặc DistilBERT), Mean pooling (có nhân `attention_mask`) trên `last_hidden_state`, dropout 0.1, qua lớp Linear ra một số thực. BERT là `bert-base-uncased`, RoBERTa là `roberta-base`, DistilBERT là `distilbert-base-uncased`, khai báo trong bảng `BACKBONES` của `model_builder.py`. Bảng này quy định tham số riêng từng backbone: BERT và RoBERTa nạp với `add_pooling_layer=False`; DistilBERT không có pooler nên không nhận tham số này. Hàm loss mặc định là Huber (`delta = 1.0`); MSE chỉ chạy đối chứng trên BERT. Quy tắc đổi loss nằm ở Define mục 9, xong ở cuối Tuần 1 và áp dụng cho cả ba mô hình.

Lý do chọn Mean pooling: dùng thông tin của mọi token thật thay vì phụ thuộc vào một vị trí duy nhất, và không phụ thuộc vào việc token đầu là `[CLS]` hay `<s>`. Không dùng `pooler_output` vì DistilBERT không có; cả Mean và `[CLS]` đều lấy từ `last_hidden_state` nên chung một đường code cho ba backbone. `[CLS]` có trong cấu hình `pooling` nhưng dự án chỉ chạy `mean`. Lý do chọn Huber: nhãn có nhiễu (ví dụ chấm sao không khớp nội dung).

## 2. Các module

| Module | Nhiệm vụ |
| --- | --- |
| `data_setup.py` | Hai hàm tách riêng. `prepare_splits()` chạy một lần: đọc `review.json` theo khối, làm sạch (bỏ review rỗng, chuẩn hoá khoảng trắng), lấy mẫu phân tầng theo sao 60.000 review giữ phân phối sao tự nhiên, chia phân tầng 70/15/15, loại khỏi val và test review có văn bản trùng khớp với train, lưu `train.parquet`, `val.parquet`, `test.parquet` và `split_stats.json` lên Drive. `split_stats.json` gồm `n_samples`, `seed`, `stars_per_split`, tỉ lệ review Test có `user_id` hoặc `business_id` xuất hiện trong Train, số review bị loại do trùng văn bản, và `split_id` (12 ký tự đầu của sha256 trên danh sách `review_id` Test đã sắp xếp). `create_dataloaders()` chạy ở mọi phiên huấn luyện: đọc ba file đã chia, nhận tokenizer từ bên ngoài, tokenize với cắt và đệm cố định ở 256 token, tạo Dataset và DataLoader (nhãn float32, mỗi mẫu mang `review_id`) |
| `metrics.py` | Nguồn duy nhất của phép clip về 1.0-5.0 (`clip_scores`) và của mọi công thức chỉ số: MAE, RMSE, R², Pearson, Spearman toàn cục; MAE/RMSE theo từng mức sao kèm số mẫu n và Macro-MAE; `rmse_raw`; mốc dự đoán trung bình Train; lập và ghi `model_summary.csv`. Mở rộng tuỳ chọn: `paired_bootstrap` ghi `paired_comparison.csv`. Không import `torch`, và `src/__init__.py` để trống, để dashboard chạy được trên máy không cài PyTorch |
| `baseline.py` | TF-IDF + Ridge; chỉ học trên Train, chọn siêu tham số (`max_features`, `ngram_range`, `alpha`) bằng RMSE Val, hàm huấn luyện không nhận tập Test; lưu và nạp pipeline bằng `joblib`; ghi dự đoán Val và Test của baseline vào `results/preds/` |
| `model_builder.py` | Dựng backbone theo bảng `BACKBONES` (bỏ lớp pooler), Mean pooling có `attention_mask`, dropout 0.1, Linear head (768 xuống 1) có bias khởi tạo bằng trung bình nhãn Train (`output_bias_init`, như nhau cho ba backbone), đầu ra squeeze về `(batch,)`; `build_tokenizer` (tokenizer đúng cặp backbone); `count_parameters`; hai tên `model.backbone` và `model.head` là một phần giao diện |
| `utils.py` | Cố định seed; lưu checkpoint nguyên tử và nạp bằng `weights_only=True` |
| `engine.py` | Vòng huấn luyện và kiểm định, Huber (mặc định) hoặc MSE, `fp16`, ép đầu ra `.float()` trước khi tính loss; lưu checkpoint mỗi epoch kèm cấu hình (`backbone`, `pooling`, `dropout`); ghi lịch sử huấn luyện mỗi lần chạy (epoch, loss train, RMSE và MAE Val, giây mỗi epoch) |
| `train.py` | Chạy huấn luyện từ file cấu hình; gọi `set_seed` ngay trước `build_model`; ghi cấu hình lần chạy (siêu tham số, phiên bản `torch` và `transformers`, loại GPU, commit hash, `split_id`) |
| `predict.py` | Nạp trọng số (dựng lại bằng `build_model(..., pretrained=False)` rồi nạp `state_dict`), dự đoán cho một hoặc nhiều review, trả số thực clip trong 1.0-5.0; lưu dự đoán Test riêng từng mô hình ở `results/preds/<model>_test.csv` (`review_id`, `true_stars`, `pred`, `split_id`); tính `n_tokens` bằng tokenizer của BERT (không cắt); gộp và xuất `test_predictions` thành hai bản (mục 4) |

Mỗi nhóm module có một file đặc tả riêng trong `docs/`, kèm file kiểm thử tách riêng (khuôn mẫu là `docs/01_data.md` và `docs/01_data_tests.md`). Các file này ghi chữ ký đầu vào/đầu ra của từng hàm và là căn cứ để cả nhóm và AI viết code theo cùng một giao diện.

| Thứ tự viết | File đặc tả | Module | File kiểm thử |
| --- | --- | --- | --- |
| 1 | `docs/01_data.md` | `data_setup.py` | `docs/01_data_tests.md` |
| 2 | `docs/02_evaluation.md` | `metrics.py`, `baseline.py` | `docs/02_evaluation_tests.md` |
| 3 | `docs/03_model.md` | `model_builder.py` | `docs/03_model_tests.md` |
| 4 | `docs/04_engine_and_utils.md` | `utils.py`, `engine.py`, `train.py`, `predict.py` | `docs/04_engine_and_utils_tests.md` |
| 5 | `docs/05_dashboard.md` | `app/streamlit_dashboard.py` | `docs/05_dashboard_tests.md` |

Quy tắc đi kèm: số thứ tự file là thứ tự viết, và một file đặc tả chỉ được phụ thuộc vào chữ ký hàm (import) của các file có số nhỏ hơn, nên không có tham chiếu vòng. Ngoại lệ chỉ ở mức tham số truyền vào: `create_dataloaders` nhận tokenizer từ bên ngoài và `utils.set_seed` được gọi từ notebook, nên `01_data.md` không import `model_builder.py` hay `utils.py`; các mục "Phụ thuộc" trong `01_data.md` chỉ là ngữ cảnh tham khảo.

## 3. Cấu trúc thư mục

Trên GitHub (repo riêng tư):

```
project/
├── docs/               # define.md, design.md, 01_data.md ... 05_dashboard.md, các file *_tests.md, ai_log.md
├── src/      # 8 module ở mục 2; __init__.py để trống (để metrics.py nhập được khi không có torch)
├── tests/              # pytest chạy trên CPU (mục 8); fixtures/ chứa dữ liệu tổng hợp, không có review thật
├── configs/            # cấu hình mỗi lần chạy (backbone, pooling, loss, seed, siêu tham số) và cấu hình smoke test
├── notebooks/          # colab_runner.ipynb (huấn luyện, đánh giá, xuất kết quả)
├── app/                # streamlit_dashboard.py, requirements.txt (streamlit, pandas, numpy, scipy; không cài torch)
├── reports/            # test_predictions.csv (KHÔNG có cột text, được commit), model_summary.csv (được commit),
│                       # test_predictions_with_text.csv (sao chép từ Drive, bị .gitignore), biểu đồ
├── requirements.txt    # phiên bản cố định
├── README.md           # tên người phụ trách từng mảng, phiên bản dữ liệu đã tải
└── .gitignore          # bỏ qua data/, *.pt, __pycache__/, reports/*_with_text.csv
```

Trên Drive:

```
drive/project/
├── data/               # train.parquet, val.parquet, test.parquet, split_stats.json
├── checkpoints/        # mỗi mô hình một thư mục (bert, roberta, distilbert), có best_model.pt
└── results/            # preds/ (dự đoán Test từng mô hình và baseline), bảng so sánh, biểu đồ, model_summary.csv,
                        # test_predictions.csv (không text), test_predictions_with_text.csv (đầy đủ)
```

## 4. Luồng dữ liệu

1. Tải Yelp Dataset, lấy file `review.json`; ghi phiên bản dữ liệu đã tải vào README; Leader đối chiếu điều khoản của Yelp trước lần chia sẻ dữ liệu đầu tiên.
2. Chạy `prepare_splits()` một lần (mô tả ở mục 2) với `seed = 42`; đọc `split_stats.json`.
3. Mỗi phiên Colab: mount Drive, `git pull`, copy ba file parquet về ổ cục bộ để đọc nhanh hơn.
4. `create_dataloaders()` đọc ba file, tokenize, cắt và đệm cố định ở 256 token.
5. Huấn luyện từng mô hình, lưu checkpoint lên Drive sau mỗi epoch và ghi lịch sử huấn luyện. Mọi so sánh và quyết định trong Tuần 1 và Tuần 2 dùng Val.
6. Sau lần chạy cuối của mọi mô hình, `predict.py` dự đoán Test một lần cho từng mô hình và baseline, lưu vào `results/preds/`. Khi gộp, ghép theo `review_id` và dừng nếu `split_id`, tập `review_id` hoặc `true_stars` không khớp giữa các file. Xuất lên `results/` trên Drive:
   - `test_predictions.csv`: `review_id`, `true_stars`, `n_tokens`, các cột `pred_baseline`, `pred_bert`, `pred_distilbert`, `pred_roberta` (mô hình nào có kết quả thì có cột), rồi các cột `abs_err_*` tương ứng; không có `text`. Các cột `pred_*` đã clip về 1.0-5.0 nên `abs_err_*` tính trên điểm đã clip.
   - `test_predictions_with_text.csv`: các cột trên cộng thêm `text`, lấy bằng cách nối `review_id` với `test.parquet`.
   - `model_summary.csv`: một dòng mỗi mô hình theo thứ tự `mean_predictor`, `baseline`, `bert`, `distilbert`, `roberta`; cột `model`, `mae`, `rmse`, `rmse_raw`, `r2`, `pearson`, `spearman`, `macro_mae`, `train_seconds`, `seconds_per_epoch`, `params_total`, `rmse_reduction_vs_baseline_pct`, `split_id`. Ô không áp dụng để trống (ví dụ `train_seconds` và hai chỉ số tương quan của `mean_predictor`). Điểm thô không lưu trong CSV dự đoán; `rmse_raw` được tính ngay lúc đánh giá.
   - Kết quả Val (so Huber với MSE, giây mỗi epoch) nằm trong lịch sử huấn luyện do `engine.py` ghi, không nằm trong `model_summary.csv`.
7. Một lần sau lần chạy cuối: tải `test_predictions.csv` và `model_summary.csv` từ Drive, commit vào `reports/` của repo riêng tư. Dashboard đọc hai file này, không cần GPU hay nạp mô hình. Bản `test_predictions_with_text.csv` sao chép thêm từ Drive khi cần xem nội dung review; file này bị `.gitignore`, không bao giờ lên GitHub.

Không trích xuất vector 768 chiều của cả tập dữ liệu về máy: Demo không cần và nó tốn dung lượng.

## 5. Dashboard kết quả (Demo)

| Thành phần | Mô tả |
| --- | --- |
| Mục đích | Trình bày kết quả bằng trực quan; không cần nạp mô hình |
| Dữ liệu vào | `reports/test_predictions.csv` (dashboard dò mọi cột `pred_*` và `abs_err_*` có mặt; mô hình thiếu thì bỏ qua mà không lỗi) và `reports/model_summary.csv`. Nếu có `reports/test_predictions_with_text.csv` thì dashboard dùng bản này và hiện thêm nội dung review |
| Nội dung | Bảng so sánh mốc trung bình, baseline và các mô hình có mặt (MAE, RMSE, R², Pearson/Spearman, `train_seconds`, `seconds_per_epoch`, `params_total`, `rmse_reduction_vs_baseline_pct`, lấy từ `model_summary.csv`); biểu đồ điểm dự đoán so với điểm thật; phân phối sai số; MAE/RMSE theo từng mức sao kèm n và Macro-MAE (tính từ CSV bằng `src.metrics`); bảng ca đoán sai điển hình: 10 review có sai số lớn nhất của mô hình đã chọn, trong mức sao đang lọc nếu có |
| Thao tác | Chọn mô hình, lọc theo số sao, sắp xếp theo sai số của mô hình đã chọn, xem chi tiết một review (có nội dung nếu có bản `with_text`) |
| Công cụ | Streamlit, chạy trên máy cá nhân |
| Ngoài phạm vi | Ô nhập review mới (cần nạp trọng số) |

Cách vận hành:

- Máy chạy dashboard chỉ cần `git clone` repo, cài `app/requirements.txt` rồi chạy `streamlit run app/streamlit_dashboard.py`; không cần GPU, Colab hay Drive. Vì lệnh này chỉ thêm thư mục `app/` vào đường dẫn nhập, app tự thêm thư mục gốc repo để `import src.metrics`.
- Dashboard tự dò file: có `test_predictions_with_text.csv` thì dùng bản đầy đủ, không thì dùng `test_predictions.csv`; hai chế độ dùng chung một đoạn code.
- Khi thiếu cột `text`, mục ca đoán sai điển hình và xem chi tiết review hiển thị `review_id`, số sao thật, dự đoán và sai số.
- Trước buổi báo cáo: máy trình chiếu có sẵn `test_predictions_with_text.csv`, cùng ảnh chụp màn hình dự phòng.

## 6. Mốc thời gian

1. **Mốc 1 (hết Tuần 1):** chốt mẫu đăng ký, xong Define và Design, khung code GitHub (repo riêng tư), `split_stats.json`, kết quả Baseline và mốc trung bình trên Val, ngưỡng mục tiêu (0,85 × RMSE Val của baseline), `seconds_per_epoch` của BERT, kết quả Huber so với MSE và loss đã chốt theo quy tắc ở Define mục 9.
2. **Mốc 2 (hết Tuần 2):** chạy xong BERT, DistilBERT, RoBERTa với cùng thiết lập và loss đã chốt; dự đoán Test xuất một lần; bảng so sánh MAE, RMSE, R² và tương quan trên Test; kiểm tra tiêu chí đạt ở Define mục 4.
3. **Mốc 3 (hết Tuần 3):** dashboard hoàn thiện, phân tích lỗi, nhật ký dùng AI, báo cáo; chừa 2 ngày đệm.

Lịch Tuần 1 theo ngày (mảng theo Define mục 13):

| Ngày | Việc | Mảng |
| --- | --- | --- |
| 1–2 | Tải `review.json`, ghi phiên bản; chạy `prepare_splits` (60.000, `seed = 42`) và đọc `split_stats.json`; dựng khung GitHub; xong đặc tả và code của `data_setup.py`, `metrics.py`, `baseline.py` | Leader, A, B |
| 3 | Baseline và mốc trung bình trên Val, tính ngưỡng mục tiêu; xong đặc tả và code của `model_builder.py`, `utils.py`, `engine.py`, `train.py`; bắt đầu dashboard bằng CSV giả đúng schema | B, C, D, E |
| 4 | Smoke test và pytest xanh; chạy ngắn vài trăm bước với cấu hình cuối (đệm 256, batch 16, `fp16`) để ước lượng tổng thời gian bốn lần huấn luyện so với hạn mức GPU | C, D |
| 5–6 | Huấn luyện đầy đủ BERT với Huber, rồi BERT với MSE | D |
| 7 | Áp dụng quy tắc chọn loss, chốt loss cho Tuần 2, ghi số liệu Mốc 1 | Leader, D |

## 7. Rủi ro kỹ thuật

| Rủi ro | Cách xử lý |
| --- | --- |
| Mất GPU hoặc ngắt phiên Colab | Bật `fp16`, lưu checkpoint lên Drive mỗi epoch, hàm tiếp tục từ checkpoint |
| Hết hạn mức GPU | Thứ tự chạy BERT, DistilBERT, RoBERTa; nếu không đủ hạn mức thì bỏ trọn RoBERTa, không giảm riêng ngân sách của mô hình nào |
| Đọc dữ liệu từ Drive chậm | Copy dữ liệu về ổ cục bộ của Colab ở đầu mỗi phiên |
| File dữ liệu lớn (vài GB) | Tải một lần, đọc theo khối, lấy mẫu một lần bằng `prepare_splits()`, chỉ lưu parquet lên Drive |
| Overfit | Theo dõi loss trên tập val |
| Mô hình dồn dự đoán về giá trị trung bình | Xem MAE/RMSE theo từng mức sao và Macro-MAE, kiểm tra độ lệch chuẩn của dự đoán (bước 4 mục 8) |
| Rò rỉ dữ liệu giữa train và test | Chia ngẫu nhiên phân tầng, loại review val/test trùng văn bản với train; mức trùng `user_id` và `business_id` đọc từ `split_stats.json` và ghi vào báo cáo như giới hạn; không chuyển sang chia theo nhóm |
| So sánh ba mô hình không công bằng | Cùng dữ liệu, seed, pooling, loss và ngân sách huấn luyện (epoch, batch, tốc độ học, warmup, weight decay, grad clip, dropout, `fp16`) |
| Trộn kết quả giữa các phiên chia hoặc phiên chạy | `split_id` ghi trong cấu hình mỗi lần chạy và trong file dự đoán; khi gộp, dừng nếu không khớp |
| Kết quả không giống hệt giữa các lần chạy | Ghi phiên bản `torch`, `transformers`, loại GPU, commit hash; cố định phiên bản trong `requirements.txt`; báo cáo dùng "tái lập gần đúng" |
| Các backbone khác nhau về API | Bảng `BACKBONES` quy định tham số riêng từng backbone; kiểm thử bằng cấu hình nhỏ cho cả ba |
| Các mảng việc lệch giao diện module | Chốt hàm và tham số trong các file đặc tả `docs/NN_*.md` ngay Tuần 1, viết theo thứ tự ở mục 2 |
| Đưa nội dung review (cột `text`) lên GitHub hoặc chia sẻ công khai | Có thể vướng điều khoản của Yelp. Repo riêng tư; bản đưa lên repo luôn bỏ cột `text`; bản có `text` chỉ nằm trên Drive (chế độ hạn chế) hoặc máy cá nhân và bị `.gitignore` |
| Test pytest chậm vì tải trọng số pretrained | Dùng cấu hình mô hình nhỏ khởi tạo ngẫu nhiên khi kiểm thử |

## 8. Quy trình làm việc với AI và kiểm thử

Theo quy trình giảng viên nêu, mọi code do AI sinh ra đi qua bốn bước: AI gen code, Smoke test, Run model, Check description. Mục đích là không đem code AI viết chạy thẳng trên toàn bộ dữ liệu lớn rồi sập giữa chừng hoặc ra kết quả lệch so với thiết kế.

| Bước | Làm gì | Tiêu chí qua bước |
| --- | --- | --- |
| 1. AI gen code | Đưa đặc tả (`docs/define.md`, `docs/design.md` và file đặc tả của module, ví dụ `docs/01_data.md`) cho AI để viết từng module | Code đúng chữ ký hàm trong file đặc tả của module |
| 2. Smoke test | Chạy pytest trên CPU và một vòng huấn luyện thử với khoảng 8 review | Không lỗi; mỗi review ra một số thực; loss hợp lệ (không NaN). Nếu lỗi thì yêu cầu AI sửa |
| 3. Run model | Kéo code lên Colab, chạy huấn luyện thật trên toàn bộ mẫu | Chạy hoàn tất, có checkpoint mỗi epoch |
| 4. Check description | Đối chiếu kết quả với tài liệu Define (input/output, ràng buộc, tiêu chí đánh giá) | Điểm dự đoán nằm trong 1.0-5.0; đúng `seed = 42`, 256 token, checkpoint mỗi epoch, `split_id` khớp; đủ chỉ số, có mốc trung bình và baseline; độ lệch chuẩn dự đoán Val không nhỏ hơn 0,5 lần độ lệch chuẩn nhãn Val; loss Val giảm sau epoch 1; RMSE Val tốt hơn mốc trung bình và baseline |

**Kiểm thử bằng pytest (chạy trên CPU; kế hoạch chi tiết ở các file `docs/NN_*_tests.md`):**

| File | Kiểm tra | Kế hoạch ở |
| --- | --- | --- |
| `test_data_setup.py` | Dataset với khoảng 5 dòng giả: `input_ids` và `attention_mask` có kích thước 256, nhãn kiểu float32; review val/test trùng văn bản với train bị loại; `split_id` ổn định khi chạy lại cùng seed | `01_data_tests.md` |
| `test_metrics.py` | `y_true` và `y_pred` cùng bằng `[1.0, 5.0]`: MAE = 0, RMSE = 0, R² = 1; số mẫu n theo mức sao; mốc trung bình | `02_evaluation_tests.md` |
| `test_baseline.py` | Baseline trên dữ liệu giả: đầu ra shape `(N,)`, nằm trong 1.0-5.0, chữ ký hàm huấn luyện không nhận tập Test | `02_evaluation_tests.md` |
| `test_model_builder.py` | Batch giả (`batch_size=2`, `seq_len=16`) qua cấu hình nhỏ của cả ba backbone (DistilBERT không nhận `add_pooling_layer`): đầu ra shape `(2,)`; bias của head bằng trung bình nhãn Train | `03_model_tests.md` |
| `test_utils.py` | `set_seed` tái lập; checkpoint ghi nguyên tử và nạp lại bằng `weights_only=True` | `04_engine_and_utils_tests.md` |
| `test_engine.py` | Loss, lịch tốc độ học, `fit` (lịch sử, checkpoint tốt nhất, tiếp tục bằng chạy liền, Test không bị lặp qua) | `04_engine_and_utils_tests.md` |
| `test_train.py` | Đọc cấu hình (ép kiểu `2e-5`, từ chối khoá lạ), `set_seed` ngay trước `build_model`, dòng lệnh | `04_engine_and_utils_tests.md` |
| `test_predict.py` | `predict(text)` trả về số thực nằm trong 1.0-5.0; xuất và gộp `test_predictions.csv`; dừng khi `split_id` hoặc `review_id` không khớp | `04_engine_and_utils_tests.md` |

`test_model_builder.py`, `test_engine.py` và `test_predict.py` dùng cấu hình mô hình nhỏ khởi tạo ngẫu nhiên thay vì tải trọng số pretrained, vì tải ba mô hình pretrained mỗi lần chạy sẽ chậm và cần mạng.

**Nhật ký AI (`docs/ai_log.md`):** mỗi dòng ghi theo đúng chuỗi trên: (1) prompt và module AI sinh ra, (2) lỗi phát hiện khi smoke test, (3) kết quả run model, (4) kết luận check description (đạt, hoặc cần chỉnh gì). Mỗi thành viên ghi cho mảng của mình, Leader tổng hợp; nhật ký dùng thẳng cho mục làm việc với AI trong báo cáo.

## 9. Các quyết định đã chốt

### 9.1 Công cụ dựng dashboard: Streamlit

Viết bằng Python thuần nên cả nhóm và AI cùng làm được, đọc CSV trực tiếp, có sẵn bảng, bộ lọc và biểu đồ. Đầu vào chỉ là các file CSV nhẹ nên phần xuất kết quả độc lập với công cụ dựng dashboard.

### 9.2 Cách đưa file kết quả tới máy chạy dashboard

Kết hợp GitHub riêng tư và sao chép từ Drive. Quy ước:

- Repo GitHub để riêng tư; file đưa lên chỉ gồm `reports/test_predictions.csv` (không có `text`, nặng khoảng vài trăm KB đến 1 MB với 9.000 review) và `reports/model_summary.csv`. Nhờ vậy `git pull` cho đúng phiên bản kết quả cùng code, không đưa nội dung review lên GitHub.
- Bản đầy đủ `test_predictions_with_text.csv` chỉ nằm trên Drive và máy cá nhân; `.gitignore` chặn `reports/*_with_text.csv`.
- Chỉ commit CSV sau lần chạy cuối của mọi mô hình, để lịch sử không có nhiều phiên bản kết quả lệch nhau.
- Triển khai công khai không thuộc phạm vi.
- Cột sai số tách theo mô hình (`abs_err_bert`, `abs_err_distilbert`, `abs_err_roberta`, `abs_err_baseline`) vì dashboard cho chọn mô hình và sắp xếp theo sai số của mô hình đó; `model_summary.csv` giữ thời gian chạy và số tham số vì không suy ra được từ file dự đoán, và giúp số trên dashboard khớp với báo cáo.

### 9.3 Thiết lập huấn luyện (áp dụng như nhau cho ba backbone)

| Tham số | Giá trị |
| --- | --- |
| Bộ tối ưu và lịch | AdamW, lịch tuyến tính có warmup (`warmup_ratio = 0.1`) |
| `lr`, `weight_decay` | `2e-5`, `0.01` |
| `batch_size`, `epochs` | `16`, `3` |
| `grad_clip`, `dropout` | `1.0`, `0.1` (trước Linear head) |
| Đệm và cắt | Cố định ở 256 token |
| Độ chính xác | `fp16` (autocast), đầu ra ép `.float()` trước loss |
| Chọn checkpoint | RMSE Val thấp nhất (điểm đã clip) |
| Seed | `42`, gọi `set_seed` ngay trước `build_model` |
| Khởi tạo bias của head | Trung bình nhãn Train |

### 9.4 Thứ tự viết đặc tả và phân việc

Thứ tự 01 → 02 → 03 → 04 → 05 theo mục 2, mỗi file ứng với một thành viên A–E (Define mục 13). `01` và `02` hoàn thành trước để kịp kết quả Baseline ở ngày 3; `03` và `04` hoàn thành ở ngày 3 để kịp huấn luyện BERT ở ngày 5; `05` do thành viên E viết và làm song song bằng CSV giả đúng schema nên không chặn mốc nào. Ranh giới đã chốt: `baseline.py` thuộc `02_evaluation.md` cùng `metrics.py` vì baseline phải gọi `metrics.py` và cần có kết quả ở Mốc 1; `predict.py` thuộc `04_engine_and_utils.md` vì cần cả `model_builder.py` và hàm nạp checkpoint trong `utils.py`.

### 9.5 Định nghĩa dùng chung

- **Ngân sách huấn luyện:** cùng epoch, batch, tốc độ học, warmup, weight decay, grad clip, dropout và `fp16`; không phải cùng thời gian chạy.
- **Cột chi phí:** `train_seconds` là tổng thời gian huấn luyện gồm lượt kiểm định Val, không gồm nạp dữ liệu và tokenize; `seconds_per_epoch` là trung bình mỗi epoch; `params_total` do `count_parameters` tính. Baseline đo trên CPU, `train_seconds` là thời gian fit của tổ hợp thắng và số tham số là số hệ số Ridge.
- **`rmse_reduction_vs_baseline_pct`:** 100 × (RMSE baseline − RMSE mô hình) / RMSE baseline, tính trên Test.
- **`n_tokens`:** số token của review khi không cắt, theo tokenizer của BERT, dùng chung làm số đo độ dài cho ba backbone; review bị cắt khi `n_tokens` lớn hơn 254.
