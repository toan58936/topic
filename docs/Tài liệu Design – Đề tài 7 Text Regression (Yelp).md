# Tài liệu Design – Đề tài 7: Text Regression (Yelp)

2026-11-04 (cập nhật từ bản 2026-11-02: đã chốt các điểm ở mục 9, đối chiếu với `docs/01_data.md`, `docs/define.md` và `docs/02_evaluation.md`)

Tài liệu này mô tả cách xây dựng hệ thống: kiến trúc, module, cấu trúc thư mục, luồng dữ liệu, giao diện Demo, mốc thời gian, rủi ro kỹ thuật và quy trình làm việc với AI. Phần Define (dự án làm gì, dữ liệu, tiêu chí) nằm ở tài liệu Define riêng. Các mục ghi "đề xuất" hoặc "dự kiến" cần nhóm duyệt; các quyết định đã chốt được ghi ở mục 9.

## 1. Kiến trúc hệ thống

| Lớp | Thành phần | Vai trò |
| --- | --- | --- |
| Mã nguồn | GitHub, repo riêng tư (`going_modular/`) | Nguồn duy nhất của code và của các file kết quả nhẹ không chứa nội dung review; làm việc theo nhánh riêng và Pull Request (theo Define); Colab kéo về bằng `git clone`/`git pull` |
| Tài liệu và kiểm thử | `docs/` và `tests/` | Đặc tả viết trước khi code (đưa cho AI), pytest chạy trên CPU |
| Dữ liệu và trọng số | Google Drive | Ba file dữ liệu đã chia và `split_stats.json`, checkpoint, trọng số cuối, bảng kết quả, bản `test_predictions` đầy đủ có cột `text` |
| Tính toán | Colab GPU (notebook mỏng) | Mount Drive, kéo code, huấn luyện, đánh giá và xuất file kết quả |
| Giao diện | Dashboard kết quả (Streamlit, đã chốt) | Chạy trên máy cá nhân, đọc `reports/test_predictions.csv` và `reports/model_summary.csv`, không cần nạp mô hình |

Mô hình: review qua tokenizer, Transformer Encoder (BERT, RoBERTa hoặc DistilBERT), Mean pooling (có nhân `attention_mask`) trên `last_hidden_state`, qua lớp Linear ra một số thực. BERT là `bert-base-uncased` (theo Define); RoBERTa là `roberta-base` và DistilBERT là `distilbert-base-uncased` (đề xuất, ghi trong bảng `BACKBONES` của `model_builder.py`, xem `docs/03_model.md`). Hàm loss mặc định là Huber (`delta = 1.0`); MSE chỉ chạy đối chứng trên BERT. Quy tắc trong Define (mục 9): nếu RMSE trên tập val của MSE thấp hơn đáng kể thì đổi mặc định sang MSE cho cả ba mô hình, và quyết định này phải xong trước khi chạy Tuần 2.

Lý do chọn Mean pooling làm mặc định: cùng một đoạn code chạy được cho cả ba backbone, vì DistilBERT không có `pooler_output` và RoBERTa dùng token `<s>` thay cho `[CLS]`. Lý do chọn Huber: nhãn có nhiễu (ví dụ chấm sao không khớp nội dung). Lưu ý: RMSE là số đo dùng để so Huber với MSE và để đặt ngưỡng mục tiêu, vốn khớp với MSE, nên kết quả đối chứng cần đọc theo RMSE chứ không giả định Huber luôn tốt hơn.

## 2. Các module (đã chốt tên và ranh giới)

| Module | Nhiệm vụ |
| --- | --- |
| `data_setup.py` | Hai hàm tách riêng. `prepare_splits()` chạy một lần: đọc `review.json` theo khối, lấy mẫu phân tầng theo sao 60.000 review (tối đa 100.000) giữ phân phối sao tự nhiên, chia phân tầng 70/15/15, lưu `train.parquet`, `val.parquet`, `test.parquet` và `split_stats.json` lên Drive. `create_dataloaders()` chạy ở mọi phiên huấn luyện: đọc ba file đã chia, nhận tokenizer từ bên ngoài, tokenize, tạo Dataset và DataLoader (nhãn kiểu float32, mỗi mẫu mang `review_id`) |
| `metrics.py` | Nguồn duy nhất của phép clip về 1.0-5.0 (`clip_scores`) và của mọi công thức chỉ số: MAE, RMSE, R², Pearson, Spearman toàn cục; MAE/RMSE theo từng mức sao và Macro-MAE; `rmse_raw` (RMSE trên giá trị thô, tham khảo); khoảng tin cậy bootstrap (đề xuất); lập và ghi `model_summary.csv` (một dòng mỗi mô hình, gồm cả baseline). Không import `torch` và `going_modular/__init__.py` để trống, để dashboard chạy được trên máy không cài PyTorch |
| `baseline.py` | TF-IDF + Ridge; gọi `metrics.py` để chọn siêu tham số nên được đặc tả cùng `metrics.py`; chỉ học trên Train, chọn siêu tham số bằng RMSE Val, hàm huấn luyện không nhận tập Test |
| `model_builder.py` | Dựng backbone (bỏ lớp pooler không dùng), Mean pooling có `attention_mask`, Linear head (768 xuống 1), đầu ra squeeze về shape `(batch,)`; chọn backbone qua tên trong cấu hình; `build_tokenizer` (tokenizer đúng cặp backbone); `count_parameters` (số tham số cho `model_summary.csv`) |
| `utils.py` | Cố định seed, lưu và nạp checkpoint |
| `engine.py` | Vòng huấn luyện và kiểm định, Huber (mặc định) hoặc MSE, `fp16`, lưu checkpoint |
| `train.py` | Chạy huấn luyện từ file cấu hình |
| `predict.py` | Nạp trọng số, dự đoán cho một hoặc nhiều review, trả về số thực giới hạn (clip) trong khoảng 1.0-5.0; xuất `test_predictions` thành hai bản (bản không có `text` và bản đầy đủ có `text`, xem mục 4) |

Mỗi nhóm module có một file đặc tả riêng trong `docs/`, kèm file kiểm thử tách riêng (khuôn mẫu là `docs/01_data.md` và `docs/01_data_tests.md`). Các file này ghi chữ ký đầu vào/đầu ra của từng hàm và là căn cứ để cả nhóm và AI viết code theo cùng một giao diện.

| Thứ tự viết | File đặc tả | Module | File kiểm thử |
| --- | --- | --- | --- |
| 1 | `docs/01_data.md` (đã có bản nháp; phần "Phụ thuộc" cần đổi `02_model.md` thành `03_model.md` và `03_engine_and_utils.md` thành `04_engine_and_utils.md`) | `data_setup.py` | `docs/01_data_tests.md` |
| 2 | `docs/02_evaluation.md` (đã có bản nháp; cần sửa theo mục 9.7) | `metrics.py`, `baseline.py` | `docs/02_evaluation_tests.md` |
| 3 | `docs/03_model.md` (đã có bản nháp; xem mục 9.8) | `model_builder.py` | `docs/03_model_tests.md` |
| 4 | `docs/04_engine_and_utils.md` (đã chốt, chưa viết) | `utils.py`, `engine.py`, `train.py`, `predict.py` | `docs/04_engine_and_utils_tests.md` |
| 5 | `docs/05_dashboard.md` (đề xuất theo `02_evaluation.md`, chưa viết) | `app/streamlit_dashboard.py` | `docs/05_dashboard_tests.md` |

Quy tắc đi kèm: số thứ tự file chính là thứ tự viết, và một file đặc tả chỉ được phụ thuộc vào chữ ký hàm (import) của các file có số nhỏ hơn. Nhờ vậy không có tham chiếu vòng giữa các đặc tả. Ngoại lệ chỉ ở mức tham số truyền vào: `create_dataloaders` nhận tokenizer từ bên ngoài và `utils.set_seed` được gọi từ notebook, nên `01_data.md` không import `model_builder.py` hay `utils.py`; các mục "Phụ thuộc" trong `01_data.md` chỉ mang tính tham khảo ngữ cảnh.

## 3. Cấu trúc thư mục

Trên GitHub (repo riêng tư):

```
project/
├── docs/               # define.md, design.md, 01_data.md ... 04_engine_and_utils.md, các file *_tests.md, ai_log.md
├── going_modular/      # 8 module ở mục 2; __init__.py để trống (để metrics.py nhập được khi không có torch)
├── tests/              # pytest chạy trên CPU (mục 8)
├── configs/            # cấu hình mỗi lần chạy (backbone, pooling, loss, seed) và cấu hình smoke test
├── notebooks/          # colab_runner.ipynb (huấn luyện, đánh giá, xuất kết quả)
├── app/                # streamlit_dashboard.py, requirements.txt (streamlit, pandas, numpy, scipy; không cài torch)
├── reports/            # test_predictions.csv (KHÔNG có cột text, được commit), model_summary.csv (được commit),
│                       # test_predictions_with_text.csv (tuỳ chọn, sao chép từ Drive, bị .gitignore), biểu đồ
├── requirements.txt
├── README.md
└── .gitignore          # bỏ qua data/, *.pt, __pycache__/, reports/*_with_text.csv
```

Trên Drive:

```
drive/project/
├── data/               # train.parquet, val.parquet, test.parquet, split_stats.json
├── checkpoints/        # mỗi mô hình một thư mục (bert, roberta, distilbert), có best_model.pt
└── results/            # bảng so sánh, biểu đồ, model_summary.csv,
                        # test_predictions.csv (không text), test_predictions_with_text.csv (đầy đủ)
```

## 4. Luồng dữ liệu

1. Tải Yelp Dataset, lấy file `review.json`; ghi lại phiên bản dữ liệu đã tải (Define mục 8 yêu cầu đối chiếu giấy phép và ghi phiên bản).
2. Chạy `prepare_splits()` một lần: đọc theo khối, làm sạch, lấy mẫu phân tầng 60.000 review, chia phân tầng 70/15/15 với `seed = 42` giữ phân phối sao tự nhiên, lưu ba file parquet và `split_stats.json` lên Drive.
3. Mỗi phiên Colab: mount Drive, `git pull`, và (đề xuất) copy ba file parquet về ổ cục bộ để đọc nhanh hơn.
4. `create_dataloaders()` đọc ba file, tokenize, cắt ở 256 token.
5. Huấn luyện từng mô hình, lưu checkpoint lên Drive sau mỗi epoch.
6. Đánh giá trên tập test (`test_loader` giữ thứ tự cố định và mang `review_id`): chỉ số toàn cục, theo từng mức sao và Macro-MAE. Xuất lên `results/` trên Drive:
   - `test_predictions.csv`: `review_id`, `true_stars`, `pred_bert`, `pred_roberta`, `pred_distilbert`, `abs_err_bert`, `abs_err_roberta`, `abs_err_distilbert` (không có `text`). Các cột `pred_*` đã clip về 1.0-5.0, nên `abs_err_*` tính trên điểm đã clip.
   - `test_predictions_with_text.csv`: các cột trên cộng thêm `text`, lấy bằng cách nối `review_id` với `test.parquet`.
   - `model_summary.csv`: một dòng mỗi mô hình (baseline ở dòng đầu), chỉ số trên tập test: `mae`, `rmse`, `rmse_raw`, `r2`, `pearson`, `spearman`, `macro_mae`, `train_seconds`, `params_total`, và (đề xuất) `rmse_ci_low`, `rmse_ci_high`, `inference_seconds_per_1000`, `rmse_reduction_vs_baseline_pct`. Điểm thô không lưu trong CSV dự đoán để file nhẹ; `rmse_raw` được tính ngay lúc đánh giá. File này lập một lần ở cấu hình cuối từ bảng dự đoán đủ các mô hình, nên mỗi phiên Colab phải lưu dự đoán test của riêng từng mô hình (cách lưu do `04_engine_and_utils.md` quy định).
   - Kết quả trên tập val (so Huber với MSE, thời gian từng epoch) không nằm trong `model_summary.csv` mà nằm trong lịch sử huấn luyện do `engine.py` ghi (do `04_engine_and_utils.md` quy định).
7. Chỉ làm một lần sau lần chạy cuối: tải `test_predictions.csv` và `model_summary.csv` từ Drive, commit vào `reports/` của repo riêng tư. Dashboard đọc hai file này; không cần GPU hay nạp mô hình. Nếu muốn xem nội dung review, sao chép thêm `test_predictions_with_text.csv` từ Drive vào `reports/` (file này bị `.gitignore`, không bao giờ lên GitHub).

Không trích xuất vector 768 chiều của cả tập dữ liệu về máy: Demo không cần và nó tốn dung lượng.

## 5. Dashboard kết quả (Demo)

| Thành phần | Mô tả |
| --- | --- |
| Mục đích | Trình bày kết quả bằng trực quan; không cần nạp mô hình |
| Dữ liệu vào | `reports/test_predictions.csv` (`review_id`, `true_stars`, `pred_*`, `abs_err_*`) và `reports/model_summary.csv`. Nếu có `reports/test_predictions_with_text.csv` thì dashboard tự dùng bản này và hiện thêm nội dung review |
| Nội dung | Bảng so sánh ba mô hình và baseline (MAE, RMSE, R², Pearson/Spearman, thời gian chạy, số tham số, lấy từ `model_summary.csv`); biểu đồ điểm dự đoán so với điểm thật; phân phối sai số; MAE/RMSE theo từng mức sao và Macro-MAE; bảng các ca đoán sai điển hình |
| Thao tác | Chọn mô hình, lọc theo số sao, sắp xếp theo sai số của mô hình đã chọn, xem chi tiết một review (có nội dung nếu có bản `with_text`) |
| Công cụ | Streamlit (đã chốt), chạy trên máy cá nhân |
| Mở rộng | Ô nhập review mới (cần nạp trọng số); không thuộc phạm vi hiện tại |

Cách vận hành dashboard:

- Máy chạy dashboard chỉ cần `git clone` repo, cài `app/requirements.txt` rồi chạy `streamlit run app/streamlit_dashboard.py`; không cần GPU, Colab hay Drive, nên không bị ảnh hưởng khi Colab ngắt phiên.
- Dashboard tự dò file: có `test_predictions_with_text.csv` thì dùng bản đầy đủ, không thì dùng `test_predictions.csv`. Hai chế độ dùng chung một đoạn code, không phải sửa gì khi đổi máy.
- Khi thiếu cột `text`, mục "ca đoán sai điển hình" và "xem chi tiết review" hiển thị `review_id`, số sao thật, dự đoán và sai số; nhóm đối chiếu nội dung trên máy có bản `with_text` khi cần phân tích lỗi.
- Chuẩn bị sẵn ảnh chụp màn hình làm dự phòng khi báo cáo.
- Nếu sau này làm ô nhập review mới thì cần trọng số của mô hình tốt nhất (khoảng 260-500 MB tuỳ mô hình, ước tính) và phải bàn lại nơi chạy; chưa thuộc phạm vi hiện tại.

## 6. Mốc thời gian

1. Mốc 1 (hết Tuần 1): chốt mẫu đăng ký, xong tài liệu Define và Design, dựng khung code GitHub, có dữ liệu mẫu trên Drive và kết quả Baseline. Define (mục 14) còn yêu cầu năm số liệu chốt ở cuối Tuần 1: (a) thời gian thực tế của một epoch, để quyết định giữ 60.000 hay tăng lên 100.000; (b) số review ở mức sao hiếm nhất trong tập test, đọc từ `stars_per_split` trong `split_stats.json`; (c) tỉ lệ trùng người dùng và doanh nghiệp giữa train và test, cũng từ `split_stats.json`; (d) RMSE val của MSE đối chứng so với Huber trên BERT, để xác nhận loss mặc định (đọc từ lịch sử huấn luyện của `engine.py`, cùng nguồn với thời gian epoch ở (a), không phải từ `model_summary.csv`); (e) ngưỡng mục tiêu so với baseline. Vì (a) và (d) cần chạy BERT thật, cả bốn file đặc tả và code của `model_builder.py`, `engine.py`, `train.py`, `utils.py` phải sẵn sàng để chạy thử BERT cuối Tuần 1, không chờ sang Tuần 2.
2. Mốc 2 (hết Tuần 2): chạy xong BERT, RoBERTa, DistilBERT trên Colab với cùng thiết lập, có bảng so sánh MAE, RMSE, R² và tương quan. Loss mặc định (Huber hoặc MSE) đã chốt từ Mốc 1 và áp dụng cho cả ba mô hình.
3. Mốc 3 (hết Tuần 3): dựng dashboard kết quả, phân tích lỗi, tổng hợp nhật ký dùng AI, hoàn thiện báo cáo.

## 7. Rủi ro kỹ thuật

| Rủi ro | Cách xử lý |
| --- | --- |
| Mất GPU hoặc ngắt phiên Colab | Bật `fp16`, lưu checkpoint lên Drive mỗi epoch, viết hàm tiếp tục từ checkpoint |
| Đọc dữ liệu từ Drive chậm | Copy dữ liệu về ổ cục bộ của Colab ở đầu mỗi phiên |
| File dữ liệu lớn | Đọc theo khối, lấy mẫu một lần bằng `prepare_splits()` |
| Overfit | Theo dõi loss trên tập val |
| Mô hình dồn dự đoán về giá trị trung bình | Xem MAE/RMSE theo từng mức sao và Macro-MAE, kiểm tra phân phối dự đoán |
| Rò rỉ dữ liệu giữa train và test | Chia ngẫu nhiên phân tầng (đã chốt); đọc tỉ lệ `user_id` và `business_id` của Test xuất hiện trong Train từ `split_stats.json` ở Tuần 1 và ghi vào báo cáo; chỉ chia theo nhóm nếu tỉ lệ trùng cao |
| So sánh ba mô hình không công bằng | Cùng dữ liệu, seed, pooling, loss và ngân sách huấn luyện |
| Các mảng việc lệch giao diện module | Chốt hàm và tham số trong các file đặc tả `docs/NN_*.md` ngay Tuần 1, viết theo thứ tự ở mục 2 |
| Đổi quy mô mẫu sau Tuần 1 (60.000 lên 100.000) | Phép chia chỉ chạy một lần nên phải quyết định ở Mốc 1: chạy lại `prepare_splits(overwrite=True)` và baseline (nhẹ, không cần GPU), bỏ kết quả cũ, không trộn kết quả giữa hai phiên bản chia |
| Chốt loss mặc định muộn | Chạy BERT thử với Huber và MSE trong Tuần 1; quyết định trước khi chạy Tuần 2 |
| Đưa nội dung review (cột `text`) lên GitHub | Có thể vướng điều khoản dữ liệu của Yelp. Bản đưa lên repo luôn bỏ cột `text`; bản có `text` chỉ nằm trên Drive hoặc trên máy cá nhân và bị `.gitignore`; repo để riêng tư |
| Test pytest chậm vì tải trọng số pretrained | Dùng cấu hình mô hình nhỏ khởi tạo ngẫu nhiên khi kiểm thử |

## 8. Quy trình làm việc với AI và kiểm thử

Theo quy trình giảng viên nêu, mọi code do AI sinh ra đi qua bốn bước: AI gen code, Smoke test, Run model, Check description. Mục đích là không đem code AI viết chạy thẳng trên toàn bộ dữ liệu lớn rồi sập giữa chừng hoặc ra kết quả lệch so với thiết kế.

| Bước | Làm gì | Tiêu chí qua bước |
| --- | --- | --- |
| 1. AI gen code | Đưa đặc tả (`docs/define.md`, `docs/design.md` và file đặc tả của module, ví dụ `docs/01_data.md`) cho AI để viết từng module | Code đúng chữ ký hàm trong file đặc tả của module |
| 2. Smoke test | Chạy pytest trên CPU và một vòng huấn luyện thử với khoảng 8 review | Không lỗi; mỗi review ra một số thực; loss hợp lệ (không NaN). Nếu lỗi thì yêu cầu AI sửa |
| 3. Run model | Kéo code lên Colab, chạy huấn luyện thật trên toàn bộ mẫu | Chạy hoàn tất, có checkpoint mỗi epoch |
| 4. Check description | Đối chiếu kết quả với tài liệu Define (input/output, ràng buộc, tiêu chí đánh giá) | Điểm dự đoán nằm trong 1.0-5.0; đúng `seed = 42`, 256 token và checkpoint mỗi epoch; đủ chỉ số và có so sánh với baseline |

**Kiểm thử bằng pytest (chạy trên CPU; kế hoạch chi tiết ở các file `docs/NN_*_tests.md`):**

| File | Kiểm tra | Kế hoạch ở |
| --- | --- | --- |
| `test_data_setup.py` | Dataset với khoảng 5 dòng giả: `input_ids` và `attention_mask` có kích thước 256, nhãn kiểu float32 | `01_data_tests.md` |
| `test_metrics.py` | `y_true` và `y_pred` cùng bằng `[1.0, 5.0]`: MAE = 0, RMSE = 0, R² = 1 | `02_evaluation_tests.md` |
| `test_baseline.py` | Baseline trên dữ liệu giả: đầu ra shape `(N,)`, nằm trong 1.0-5.0, chữ ký hàm huấn luyện không nhận tập Test | `02_evaluation_tests.md` |
| `test_model_builder.py` | Batch giả (`batch_size=2`, `seq_len=16`) qua cấu hình của cả ba backbone: đầu ra có shape `(2,)` | `03_model_tests.md` |
| `test_predict.py` | `predict(text)` trả về số thực nằm trong khoảng 1.0-5.0 | `04_engine_and_utils_tests.md` |

Để test chạy nhanh trên CPU (vài giây, ước tính), `test_model_builder.py` và `test_predict.py` nên dùng cấu hình mô hình nhỏ khởi tạo ngẫu nhiên thay vì tải trọng số pretrained, vì tải ba mô hình pretrained mỗi lần chạy sẽ chậm và cần mạng.

**Nhật ký AI (`docs/ai_log.md`):** mỗi dòng ghi theo đúng chuỗi trên: (1) prompt và module AI sinh ra, (2) lỗi phát hiện khi smoke test, (3) kết quả run model, (4) kết luận check description (đạt, hoặc cần chỉnh gì). Nhật ký này dùng thẳng cho mục làm việc với AI trong báo cáo.

## 9. Các điểm đã chốt

### 9.1 Công cụ dựng dashboard: Streamlit

Đã chốt. Lý do: viết bằng Python thuần nên cả nhóm và AI cùng làm được, đọc CSV trực tiếp, có sẵn bảng, bộ lọc và biểu đồ. Vì đầu vào chỉ là các file CSV nhẹ nên nếu sau này muốn đổi công cụ vẫn không phải làm lại phần xuất kết quả.

### 9.2 Cách đưa file kết quả tới máy chạy dashboard

Phân tích ba hướng với Streamlit:

| Hướng | Cách làm | Ưu điểm | Nhược điểm |
| --- | --- | --- | --- |
| A. Sao chép thủ công từ Drive | Mỗi máy tải CSV từ Drive, đặt cạnh app | Không đụng tới GitHub, giữ được cột `text` | Mỗi thành viên tự sao chép; dễ lệch phiên bản giữa dashboard và báo cáo; không có lịch sử thay đổi |
| B. Đưa lên GitHub repo riêng tư, bỏ cột `text` | Commit CSV vào `reports/` | `git pull` là có đúng phiên bản kết quả cùng code; ai cũng chạy được; file chỉ khoảng vài trăm KB đến khoảng 1 MB (tập test 9.000 review, ước tính) | Không có nội dung review nên mất phần xem chi tiết review |
| C. Kết hợp B và A (đã chọn) | Mặc định commit bản không có `text`; bản có `text` sao chép thủ công từ Drive, bị `.gitignore` | Có phiên bản thống nhất cho cả nhóm, vẫn xem được nội dung review khi cần phân tích lỗi, không đưa nội dung review lên GitHub | Thêm một quy ước tên file và một đoạn code dò file |

**Chọn hướng C.** Quy ước:

- Repo GitHub để riêng tư; file đưa lên chỉ gồm `reports/test_predictions.csv` (không có `text`) và `reports/model_summary.csv`.
- Bản đầy đủ `test_predictions_with_text.csv` chỉ nằm trên Drive và, nếu cần, trên máy cá nhân; `.gitignore` chặn `reports/*_with_text.csv`.
- Dashboard tự dò: có bản `with_text` thì dùng, không thì dùng bản thường (chi tiết ở mục 5).
- Chỉ commit CSV sau lần chạy cuối của cả ba mô hình, để tránh nhiều phiên bản kết quả lệch nhau trong lịch sử.
- Triển khai công khai (ví dụ Streamlit Community Cloud) không thuộc phạm vi hiện tại; nếu sau này làm thì chỉ dùng bản không có `text`.

Hai điều chỉnh đi kèm đã được cập nhật trong tài liệu:

- Cột sai số tách theo mô hình (`abs_err_bert`, `abs_err_roberta`, `abs_err_distilbert`) thay cho một cột `abs_error` duy nhất, vì dashboard cho chọn mô hình và sắp xếp theo sai số của mô hình đó.
- Thêm `model_summary.csv` vì thời gian chạy và số tham số không suy ra được từ file dự đoán; file này cũng giúp số trên dashboard khớp với số trong báo cáo.

### 9.3 Module, file đặc tả và thứ tự viết

Đã chốt ở mục 2: giữ nguyên tên 8 module; điều chỉnh ranh giới và thứ tự file đặc tả như sau.

| Thay đổi | Lý do |
| --- | --- |
| `baseline.py` chuyển từ `02_model.md` sang `02_evaluation.md`, ghép với `metrics.py` | Baseline bắt buộc phải gọi `metrics.py` và cần có kết quả ở Mốc 1, nên hai module này viết và kiểm thử cùng nhau |
| `model_builder.py` đứng riêng ở `03_model.md` | Chỉ phụ thuộc cấu hình, không phụ thuộc module khác |
| `predict.py` chuyển sang `04_engine_and_utils.md` | `predict.py` cần cả `model_builder.py` và hàm nạp checkpoint trong `utils.py`; đặt ở file cuối thì không tạo tham chiếu vòng |
| Đổi tên `04_evaluation.md` thành `02_evaluation.md` | Số thứ tự file bằng thứ tự viết |

Thứ tự viết: `01_data` (đã có bản nháp) → `02_evaluation` → `03_model` → `04_engine_and_utils`. Cả bốn file hoàn thành trong Tuần 1 (chốt chữ ký ngay Tuần 1). `01_data` và `02_evaluation` ưu tiên trước để kịp kết quả Baseline; `03_model` và `04_engine_and_utils` ngay sau đó để kịp chạy thử BERT cuối Tuần 1 (mục 6).

### 9.4 Đối chiếu với `docs/01_data.md`

Các điểm đã khớp, không cần sửa: `n_samples = 60.000` (tối đa 100.000), `ratios = 70/15/15`, `seed = 42`, `max_length = 256`; lấy mẫu và chia phân tầng theo sao giữ phân phối tự nhiên; nhãn `float32` giữ thang 1.0-5.0; ba file parquet nằm trên Drive, không lên GitHub; `test_loader` không xáo trộn và mang `review_id`.

Các chỗ đã chỉnh trong Design để khớp với `01_data.md`:

| Chỗ chỉnh | Lý do |
| --- | --- |
| Thêm `split_stats.json` vào mô tả `data_setup.py`, luồng dữ liệu và thư mục Drive | `01_data.md` (FR-5) ghi file này; nó chứa số liệu đếm trùng `user_id`, `business_id` và số mẫu theo từng mức sao mà Mốc 1 cần |
| Rủi ro rò rỉ đọc số liệu từ `split_stats.json` thay vì "đếm riêng" | Tránh làm hai lần cùng một việc |
| Bản `test_predictions_with_text.csv` lấy `text` bằng cách nối `review_id` với `test.parquet` | `01_data.md` (mục 2.8) đã quy định cách nối này; `test.parquet` giữ cột `text` ở Drive |
| Quy tắc phụ thuộc giữa các đặc tả có ngoại lệ ở mức tham số truyền vào | `01_data.md` ghi phụ thuộc vào tài liệu tokenizer và seed (số cao hơn) nhưng chỉ nhận tokenizer qua tham số, không import |

Việc nhóm cần làm ở `01_data.md`: đổi hai tham chiếu trong dòng "Phụ thuộc" (`02_model.md` thành `03_model.md`, `03_engine_and_utils.md` thành `04_engine_and_utils.md`) và đổi `02_model.md` ở các chỗ khác nếu có.

### 9.5 Đối chiếu với `docs/define.md`

Các điểm đã khớp, không cần sửa: phạm vi (Yelp, tiếng Anh, BERT là `bert-base-uncased`, mở rộng RoBERTa và DistilBERT); input cắt ở 256 token, output là số thực clip về 1.0-5.0; Mean pooling và Huber (`delta = 1.0`) là mặc định, MSE đối chứng trên BERT; cùng dữ liệu, seed, pooling, loss, ngân sách huấn luyện và `fp16` cho cả ba backbone; danh sách chỉ số (MAE, RMSE, R², Pearson/Spearman, MAE/RMSE theo mức sao, Macro-MAE, thời gian chạy, số tham số); Colab GPU với checkpoint mỗi epoch; dashboard gồm bảng so sánh, biểu đồ và ca đoán sai điển hình, ô nhập review mới nằm ngoài phạm vi.

Các chỗ đã chỉnh trong Design:

| Chỗ chỉnh | Lý do |
| --- | --- |
| Bỏ cách nói "RMSE là chỉ số chính" | Define không chỉ định chỉ số chính; RMSE chỉ xuất hiện ở quy tắc đổi sang MSE và ngưỡng mục tiêu, nên Design ghi đúng vai trò đó |
| Thêm quy tắc đổi mặc định sang MSE nếu RMSE val của MSE thấp hơn đáng kể (mục 1) | Define mục 9 và mục 14 quy định; quyết định phải xong trước Tuần 2 |
| Mốc 1 thêm năm số liệu cần chốt cuối Tuần 1 và yêu cầu chạy thử BERT trong Tuần 1 (mục 6) | Define mục 14: thời gian một epoch và kết quả MSE đối chứng cần có ở cuối Tuần 1, nên không thể đợi đến Tuần 2 mới chạy BERT |
| Thêm `rmse_raw` vào `model_summary.csv` và nói rõ `pred_*` đã clip (mục 2 và 4) | Define mục 12 yêu cầu clip trước khi tính chỉ số và ghi thêm RMSE trên giá trị thô |
| Ghi phiên bản dữ liệu đã tải (mục 4) | Define mục 8 yêu cầu đối chiếu giấy phép và ghi lại phiên bản |
| Thêm rủi ro đổi quy mô mẫu và chốt loss muộn (mục 7) | Define mục 14: quy mô mẫu và loss đều xác nhận bằng số liệu ở cuối Tuần 1, trong khi phép chia chỉ chạy một lần |
| Ghi nhánh riêng và Pull Request vào dòng GitHub (mục 1) | Define mục 11 |

Đề nghị sửa Define để hai tài liệu không lệch nhau:

- Mục 10 nói dashboard đọc từ `test_predictions.csv`; Design đọc thêm `model_summary.csv`. Nên ghi cả hai file.
- Mục 12 nói clip áp dụng cho "cả ba mô hình". Design đề xuất clip cả baseline để so sánh cùng cơ sở (nhóm cần duyệt); nếu đồng ý, ghi rõ trong Define.
- Mục 13 chưa chốt cách chia 5 mảng việc cho 5 thành viên. Bốn file đặc tả ở mục 2 có thể làm ranh giới chia mảng (đề xuất, chưa quyết định).

### 9.6 Điểm còn mở

- Nhóm xác nhận repo GitHub được tạo ở chế độ riêng tư trước khi commit bất kỳ file kết quả nào.
- Nhóm duyệt việc clip cả dự đoán của baseline về 1.0-5.0 (mục 9.5).
- Ô nhập review mới vẫn ngoài phạm vi; nếu muốn làm thì họp lại để chọn nơi chạy mô hình.
- Nhóm duyệt các cột và hàm mở rộng của `02_evaluation.md` ngoài Define: khoảng tin cậy bootstrap, `inference_seconds_per_1000`, `rmse_reduction_vs_baseline_pct`; nếu giữ thì ghi một dòng vào Define mục 12.
- Nhóm duyệt `docs/05_dashboard.md` làm file đặc tả thứ năm (mục 2).
- Nhóm duyệt các đề xuất của `docs/03_model.md` (mục 9.8): `roberta-base` và `distilbert-base-uncased`, bỏ lớp pooler, dropout 0.1 trước head, `output_bias_init`.

### 9.7 Đối chiếu với `docs/02_evaluation.md`

Các điểm đã khớp: clip là nguồn duy nhất ở `metrics.py`, `rmse_raw` do nơi gọi truyền vào, `metrics.py` không import `torch`, baseline chỉ học trên Train và chọn siêu tham số bằng Val, `model_summary.csv` có `pred_<mô hình>` làm đầu vào, đánh giá test một lần ở cấu hình cuối.

Design đã ghi nhận phần mở rộng của `02_evaluation.md` ở mục 2, 3, 4 và 8. Bản 02 hiện còn các chỗ cần sửa trước khi dùng làm căn cứ cho AI viết code:

| # | Chỗ cần sửa trong `02_evaluation.md` và `02_evaluation_tests.md` | Lý do |
| --- | --- | --- |
| 1 | Test D4 sai: dự đoán bình phương như `[1, 9, 25]` bị clip thành `[1, 5, 5]`, Spearman còn 0.866 chứ không phải 1. Dùng dự đoán đơn điệu nằm trong 1-5 (ví dụ `[1, 1.2, 2, 3.5, 5]` cho Spearman = 1, Pearson ≈ 0.963) hoặc `clip=False` | Test thất bại ngay lần chạy đầu dù hàm đúng |
| 2 | Thêm trường hợp nhãn là hằng số (R², Pearson, Spearman trả `NaN` kèm cảnh báo) vào mục 2.4, 2.9 và thêm một test | Dashboard lọc theo một mức sao nên nhãn là hằng số; R² chia cho 0 |
| 3 | Thêm lưu và nạp pipeline baseline (ví dụ `save_baseline`, `load_baseline` bằng `joblib`) và quy định nơi ghi dự đoán val và test của baseline | `predict.py` chạy ở Tuần 2 trong phiên khác cần `pred_baseline`; không có cách lưu thì phải huấn luyện lại baseline |
| 4 | Ghi rõ tên cột của `grid_results` (`max_features`, `ngram_range`, `alpha`, `val_rmse`, `fit_seconds`) và nói `fit_seconds` là thời gian của tổ hợp thắng, ghi ngay trong vòng lưới | Test D18 đã dùng các cột này; "lần fit cuối" mơ hồ vì pipeline thắng có thể không phải lần fit cuối |
| 5 | `per_star_metrics` báo `ValueError` nếu nhãn ngoài `{1,2,3,4,5}` | Hiện nhãn lạ bị bỏ qua âm thầm |
| 6 | Tiêu đề "Sáu nhóm khía cạnh" ở file kiểm thử thực có bảy nhóm (A-F và S) | Lỗi chữ |

Hai điều 04 phải đặc tả vì 02 dựa vào: (i) `engine.py` ghi lịch sử huấn luyện mỗi lần chạy (epoch, loss train, RMSE và MAE val, giây mỗi epoch) để quyết định Huber hay MSE; (ii) `predict.py` lưu dự đoán test riêng của từng mô hình để `build_model_summary` ghép về sau.

Một điều 05 phải đặc tả: `streamlit run app/streamlit_dashboard.py` chỉ thêm thư mục `app/` vào đường dẫn nhập, không thêm thư mục gốc repo, nên `import going_modular.metrics` cần thêm đường dẫn thủ công.

### 9.8 Đối chiếu với `docs/03_model.md`

Các điểm đã khớp: Mean pooling có `attention_mask` và Linear 768 xuống 1 không softmax (Define mục 9); `[CLS]` chỉ đối chứng; `model_builder.py` không import `metrics.py` và ngược lại; `count_parameters` nằm ở `model_builder.py` đúng như bảng tích hợp của `02_evaluation.md`; tên `bert`, `roberta`, `distilbert` trùng tên mô hình trong `model_summary.csv` và các cột `pred_*`; `ReviewDataset` chỉ trả `input_ids` và `attention_mask`, đúng đầu vào của `forward`.

Design đã ghi nhận ở mục 1, 2 và 9.6: bảng checkpoint, `build_tokenizer`, `count_parameters`, việc bỏ lớp pooler.

Việc `03` đẩy sang `04_engine_and_utils.md`:

| # | Việc | Lý do |
| --- | --- | --- |
| 1 | `engine.py` ép đầu ra `.float()` trước khi tính loss; gọi `set_seed` ngay trước `build_model` | Đầu ra theo autocast có thể là `float16`; khởi tạo head phụ thuộc seed toàn cục |
| 2 | Quyết định có truyền trung bình nhãn Train vào `output_bias_init` hay không, và áp dụng như nhau cho cả ba backbone | Bias bắt đầu quanh 0 trong khi nhãn trung bình gần 4 làm lãng phí các bước đầu |
| 3 | Checkpoint lưu kèm cấu hình `backbone`, `pooling`, `dropout` | `predict.py` dựng lại bằng `build_model(..., pretrained=False)` rồi nạp `state_dict`, nên cần cấu hình đúng |
| 4 | Dùng tên `model.backbone` và `model.head` nếu chia nhóm tham số | Hai tên này là một phần giao diện của `TextRegressor` |

Các giả định về hành vi của `transformers` (`add_pooling_layer`, `hidden_size` của DistilBERT) chưa được chạy thử khi soạn tài liệu; test M8 và M9 kiểm chứng ở lần chạy đầu.