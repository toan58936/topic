# Tài liệu Define – Đề tài 7: Text Regression (Yelp)

Nov 8, 2026 · @Toan · Bản chốt

Tài liệu này chốt dự án làm gì, cho ai, với dữ liệu và tiêu chí nào. Đây là bản chốt: mọi mục là quyết định đã áp dụng, không còn mục chờ duyệt. Cách xây dựng (kiến trúc, module, thư mục, luồng dữ liệu, kiểm thử) nằm ở tài liệu Design riêng. Sau bản chốt chỉ sửa khi một lần chạy thật lộ ra vấn đề, và lý do phải ghi vào `docs/ai_log.md`.

## 1. Tên dự án

Dự đoán điểm cảm xúc liên tục (Sentiment Score) từ văn bản đánh giá khách hàng bằng Transformer Encoder (BERT) kết hợp Linear Regression Head.

- Công nghệ: Transformer Encoder (BERT) + Linear regression head.
- Hành động: dự đoán bằng hồi quy.
- Bài toán: ước lượng điểm cảm xúc từ văn bản review.
- Phạm vi: review trong Yelp Dataset (chủ yếu tiếng Anh, không lọc ngôn ngữ); so sánh BERT với DistilBERT và RoBERTa.

## 2. Hướng chính

Chọn một hướng duy nhất: **AI dự đoán**. Dự án dự đoán một điểm số liên tục (1.0-5.0) từ văn bản review bằng hồi quy trên Transformer Encoder.

Ba hướng còn lại không được chọn: Realtime (nhóm không cam kết yêu cầu về độ trễ), Tối ưu toàn cục (đề tài không giải bài toán tối ưu có ràng buộc), Tự động hoá (nhóm chỉ làm Demo minh hoạ, không tích hợp vào quy trình thật).

## 3. Vấn đề và người dùng

- **Người dùng:** nền tảng đánh giá dịch vụ/nhà hàng và nhóm phân tích dữ liệu của họ.
- **Vấn đề:** điểm sao 1-5 là thang rời rạc và dồn về sao cao, nên khó so sánh mức hài lòng giữa các review.
- **Giá trị mang lại:** ước lượng một điểm số liên tục (1.0-5.0) trực tiếp từ nội dung văn bản. Điểm lẻ là kỳ vọng của mô hình khi nội dung chưa đủ rõ để rơi vào một mức sao. Vì nhãn huấn luyện là sao nguyên, dự án đo mức độ tái tạo số sao và không khẳng định điểm lẻ phản ánh mức hài lòng tinh hơn thang sao; điều này được nêu trong mục Giới hạn của báo cáo.

## 4. Mục tiêu

- Xây dựng mô hình hồi quy dự đoán điểm cảm xúc liên tục từ văn bản review, và cải thiện rõ rệt so với Baseline (TF-IDF + Ridge) trên cùng dữ liệu và cùng tập test cố định.
- **Tiêu chí đạt (đã chốt, tính trên Val):** RMSE Val của BERT thấp hơn RMSE Val của baseline ít nhất 15% (RMSE BERT ≤ 0,85 × RMSE baseline), và Macro-MAE Val của BERT không cao hơn baseline. Mốc 1 chỉ dùng Val; Test chỉ được tính một lần, ở cuối Tuần 2, và báo cáo cùng hai chỉ số này. Nếu không đạt thì báo cáo ghi đúng kết quả, không đổi ngưỡng.
- So sánh công bằng BERT, DistilBERT và RoBERTa về sai số (MAE, RMSE, R², tương quan) và chi phí (thời gian huấn luyện, số tham số).

## 5. Phạm vi

- **Bắt buộc:** dữ liệu và chia dữ liệu, Baseline (TF-IDF + Ridge) và mốc dự đoán trung bình, huấn luyện BERT + Linear regression head, đánh giá hồi quy, phân tích lỗi, dashboard trình bày kết quả, báo cáo.
- **Mở rộng theo đề bài, có trong kế hoạch:** so sánh công bằng với DistilBERT rồi RoBERTa (Tuần 2). Nếu hết hạn mức GPU thì bỏ trọn RoBERTa, không giảm ngân sách riêng của mô hình nào; báo cáo và dashboard vẫn đủ với hai mô hình.
- **Mở rộng tuỳ chọn, không chặn mốc nào:** paired bootstrap cho chênh lệch giữa hai mô hình (sau Mốc 2).
- **Ngoài phạm vi:** triển khai hoặc tích hợp vào hệ thống thật, yêu cầu realtime, ô nhập review mới, lọc ngôn ngữ, chạy nhiều seed, xử lý review thiếu sao hoặc chấm sao lệch nội dung.

## 6. Input / Output

| Thành phần | Mô tả |
| --- | --- |
| Input | Một review dạng chuỗi văn bản (chủ yếu tiếng Anh), cắt ở 256 token |
| Output | Một số thực là điểm dự đoán trên thang sao (1.0-5.0); điểm được giới hạn (clip) về khoảng này khi dự đoán và khi đánh giá |
| Nhãn huấn luyện | Số sao nguyên 1-5 của review, coi như giá trị liên tục |

## 7. Tính năng cốt lõi

1. Đọc, làm sạch, lấy mẫu phân tầng và chia dữ liệu Yelp (70/15/15, `seed = 42`).
2. Baseline TF-IDF + Ridge và mốc dự đoán trung bình.
3. Fine-tune BERT + Linear regression head.
4. So sánh công bằng với DistilBERT và RoBERTa (mở rộng theo đề bài).
5. Đánh giá hồi quy và phân tích lỗi.
6. Dashboard trực quan: bảng so sánh các mô hình, biểu đồ và các ca đoán sai điển hình.

## 8. Dữ liệu

| Hạng mục | Mô tả |
| --- | --- |
| Nguồn | [Yelp Dataset trên Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset), theo gợi ý của giảng viên |
| File sử dụng | `review.json`: văn bản, số sao (1-5), `user_id`, `business_id` |
| Quy mô | 60.000 review, lấy mẫu ngẫu nhiên phân tầng theo số sao, giữ tỉ lệ sao như dữ liệu gốc. Quy mô cố định, không đổi theo thời gian huấn luyện |
| Chia dữ liệu | Train/val/test 70/15/15 (42.000 / 9.000 / 9.000), phân tầng theo số sao, `seed = 42`, chia một lần |
| Tiền xử lý | Bỏ review rỗng; chuẩn hoá khoảng trắng; loại khỏi val và test mọi review có văn bản trùng khớp (sau chuẩn hoá) với một review trong train; cắt ở 256 token khi tokenize |
| Nhãn | Số sao nguyên 1-5, coi như giá trị liên tục; giả định này được nêu rõ trong báo cáo |
| Giới hạn | Phân phối gốc lệch về sao cao; nhãn có thể nhiễu (chấm sao không khớp nội dung); có thể trùng `user_id` hoặc `business_id` giữa train và test; một phần nhỏ review không phải tiếng Anh vì không lọc ngôn ngữ |
| Giấy phép | Theo điều khoản của Yelp. Leader đối chiếu điều khoản trước lần chia sẻ dữ liệu đầu tiên và ghi phiên bản dữ liệu đã tải vào README; dữ liệu trên Drive để chế độ hạn chế, không đặt liên kết công khai |
| Dữ liệu dự phòng | Chỉ kích hoạt khi Yelp không tải được hoặc điều khoản không cho phép dùng: chuyển sang Large Movie Review Dataset (IMDb, Stanford), có điểm 1-10 nhưng chỉ gồm review điểm ≤4 hoặc ≥7. Giữ nguyên phương pháp, nhãn đổi sang thang 1-10, clip về 1.0-10.0, chỉ số tính trên thang đó, và báo cáo nêu rõ phân phối hai đỉnh. Đối chiếu điều khoản của bộ này trước khi dùng |

Tập test giữ phân phối sao tự nhiên; mọi chỉ số theo từng mức sao luôn kèm số mẫu n. Mức trùng `user_id` và `business_id` giữa train và test được đo, ghi vào `split_stats.json` và báo cáo như một giới hạn; dự án không chuyển sang chia theo nhóm.

## 9. Phương pháp

**Baseline:** TF-IDF + Ridge Regression, chạy nhanh, không cần GPU, chọn siêu tham số bằng RMSE trên Val, làm mốc so sánh. **Mốc dự đoán trung bình:** dự đoán mọi review bằng trung bình nhãn Train, để thấy các mô hình tốt hơn mức đoán không cần đọc nội dung bao nhiêu.

**Mô hình chính:** fine-tune `bert-base-uncased` theo các bước sau.

1. Tách review thành token, cắt ở 256 token.
2. Đưa qua Transformer Encoder để mỗi token có vector mang ngữ cảnh.
3. Gộp thành một vector cho cả review bằng Mean pooling (có nhân `attention_mask`); không dùng `[CLS]` hay `pooler_output`.
4. Qua lớp Linear (768 chiều xuống 1) ra một số thực, không dùng softmax.
5. Tối ưu bằng Huber Loss (`delta = 1.0`).

**Quy tắc chọn loss (đã chốt):** BERT chạy Huber và MSE trong Tuần 1 với cùng thiết lập. Chỉ đổi mặc định sang MSE cho cả ba mô hình khi MSE tốt hơn Huber ít nhất 1% (tương đối) ở RMSE Val và không tệ hơn quá 1% (tương đối) ở MAE Val; ngược lại giữ Huber. Quy tắc áp dụng trên Val, xong ở cuối Tuần 1, không đổi sau khi thấy kết quả.

**Ngân sách huấn luyện:** cùng số epoch, batch, tốc độ học, warmup, weight decay, grad clip, dropout và `fp16` cho cả ba backbone, không phải cùng thời gian chạy. Giá trị cụ thể ở tài liệu Design.

**Mở rộng:** thay backbone, giữ nguyên mọi thiết lập khác.

| Mô hình | Vai trò | Ghi chú |
| --- | --- | --- |
| Dự đoán trung bình Train | Mốc tham chiếu | Không dùng nội dung review |
| TF-IDF + Ridge | Baseline | Có n-gram cục bộ, không nắm ngữ nghĩa và ngữ cảnh dài |
| BERT (`bert-base-uncased`) | Mô hình chính | Kiến trúc gốc |
| DistilBERT (`distilbert-base-uncased`) | Mở rộng, chạy trước | Bản rút gọn, nhẹ hơn; so sánh thêm thời gian huấn luyện và số tham số |
| RoBERTa (`roberta-base`) | Mở rộng, chạy sau | Phân biệt hoa-thường, khác tokenizer và dữ liệu tiền huấn luyện |

Ba checkpoint khác nhau về tiền huấn luyện và tokenizer, nên kết luận so sánh là về checkpoint chứ không thuần về kiến trúc.

## 10. Sản phẩm

- Mã nguồn dạng module (`src/`) trên GitHub riêng tư, kèm README (có tên người phụ trách từng mảng và phiên bản dữ liệu) và hướng dẫn cài đặt/chạy (`requirements.txt`).
- Dữ liệu đã chia cố định và `split_stats.json` trên Drive.
- Trọng số mô hình (`.pt`/`.pth`) lưu trên Drive.
- Notebook chạy trên Colab.
- Dashboard trực quan trình bày kết quả: bảng so sánh các mô hình, biểu đồ (điểm dự đoán so với điểm thật, phân phối sai số, sai số theo từng mức sao) và các ca đoán sai điển hình, đọc từ `test_predictions.csv` và `model_summary.csv`. Công cụ dựng dashboard nằm ở tài liệu Design.
- Báo cáo (có mục Giới hạn), kèm nhật ký dùng AI.

## 11. Ràng buộc công nghệ

| Ràng buộc | Nội dung |
| --- | --- |
| Thời gian và nhân sự | 2-3 tuần, 6 thành viên (1 Leader, 5 thành viên phụ trách 5 mảng, xem mục 13) |
| Quản lý mã nguồn | VS Code và GitHub (repo riêng tư), mã dạng module (`src/`), nhánh riêng và Pull Request |
| Lưu trữ | Google Drive chung cho dữ liệu đã chia và trọng số (`.pt`/`.pth`) |
| Tính toán | Google Colab GPU; phiên có thể bị ngắt nên bật `fp16` và lưu checkpoint mỗi epoch |
| Tái lập | Cố định `seed = 42`; mỗi lần chạy ghi phiên bản `torch` và `transformers`, loại GPU, commit hash và `split_id`. Do `fp16` và GPU không tất định, kết quả tái lập gần đúng chứ không giống hệt từng chữ số |
| Thư viện | Python; PyTorch, Hugging Face Transformers, scikit-learn, pandas, Streamlit; phiên bản cố định trong `requirements.txt` |

## 12. Tiêu chí đánh giá

Tất cả mô hình được đo trên cùng một tập test cố định (`seed = 42`), giữ phân phối sao tự nhiên. Điểm dự đoán của mọi mô hình, gồm cả baseline, được giới hạn (clip) về 1.0-5.0 trước khi tính chỉ số; ghi thêm RMSE trên giá trị thô (`rmse_raw`) để tham khảo. Việc clip không làm tăng sai số vì nhãn thật luôn nằm trong khoảng này. Test chỉ được tính một lần, ở cuối Tuần 2; trước đó mọi so sánh và quyết định dùng Val.

| Chỉ số | Ý nghĩa |
| --- | --- |
| MAE | Sai số tuyệt đối trung bình, tính trên thang sao |
| RMSE | Căn bậc hai sai số bình phương trung bình, phạt nặng sai số lớn |
| R² | Phần biến thiên của điểm số mà mô hình giải thích được |
| Pearson / Spearman | Mức tương quan giữa điểm dự đoán và điểm thật |
| MAE/RMSE theo từng mức sao (kèm n), Macro-MAE | Kiểm tra mô hình có dồn dự đoán về giá trị trung bình ở các mức 2-3 sao hay không; Macro-MAE là trung bình MAE của 5 mức sao |
| `train_seconds`, `seconds_per_epoch`, `params_total` | Chi phí: tổng thời gian huấn luyện (gồm lượt kiểm định Val, không gồm nạp dữ liệu và tokenize), thời gian trung bình mỗi epoch, tổng số tham số. Baseline đo trên CPU, `train_seconds` là thời gian fit của tổ hợp thắng, số tham số là số hệ số Ridge |
| `rmse_reduction_vs_baseline_pct` | Tỉ lệ giảm RMSE so với baseline, tính trên Test |

- **Biểu đồ:** điểm dự đoán so với điểm thật, phân phối sai số, sai số theo từng mức sao; cùng bảng so sánh và các ca đoán sai điển hình, chúng tạo thành dashboard trình bày kết quả.
- **Phân tích lỗi:** đọc một mẫu cố định 40 review từ bản có nội dung: 20 review có sai số lớn nhất của mô hình tốt nhất và 20 review ngẫu nhiên từ Test. Mỗi review gán một nhóm: mỉa mai, cảm xúc lẫn lộn, bị cắt ở 256 token, nhãn nhiễu, hoặc khác; báo cáo tỉ lệ từng nhóm. Tỉ lệ review Test bị cắt tính từ cột `n_tokens` (lớn hơn 254).
- **Mục tiêu:** theo mục 4.
- **Giới hạn trong báo cáo:** nhãn là sao nguyên; một seed và ba checkpoint khác loại; phân phối lệch sao cao; nhãn nhiễu; trùng `user_id`/`business_id`; review không phải tiếng Anh; Mean pooling chọn theo lập luận kỹ thuật, không theo thực nghiệm; tái lập gần đúng.
- **Đối chiếu:** kết quả huấn luyện thật được kiểm tra lại với tài liệu này (input/output, ràng buộc, tiêu chí) ở bước Check description trong tài liệu Design.

## 13. Kế hoạch, phân công và rủi ro

| Vai trò | Mảng phụ trách |
| --- | --- |
| Leader | Kiến trúc, review code và Pull Request, đối chiếu giấy phép Yelp, tổng hợp báo cáo |
| Thành viên A | Dữ liệu: `data_setup.py`, `split_stats.json`, đặc tả `01` |
| Thành viên B | Đánh giá và baseline: `metrics.py`, `baseline.py`, đặc tả `02` |
| Thành viên C | Mô hình: `model_builder.py`, đặc tả `03` |
| Thành viên D | Huấn luyện: `utils.py`, `engine.py`, `train.py`, `predict.py`, đặc tả `04`; chạy huấn luyện trên Colab |
| Thành viên E | Dashboard và phân tích lỗi: `app/streamlit_dashboard.py`, đặc tả `05` |

Mỗi người ghi `docs/ai_log.md` cho mảng của mình; Leader tổng hợp. Tên từng thành viên ghi trong README của repo.

| Thời gian | Công việc | Mốc kiểm tra |
| --- | --- | --- |
| Tuần 1 | Dữ liệu và Baseline; Define, Design và khung code GitHub; chạy ngắn ước lượng tốc độ; huấn luyện đầy đủ BERT với Huber rồi MSE; áp dụng quy tắc chọn loss | Mốc 1: `split_stats.json`, kết quả Baseline và mốc trung bình trên Val, ngưỡng mục tiêu, `seconds_per_epoch`, kết quả Huber so với MSE và loss đã chốt |
| Tuần 2 | Huấn luyện DistilBERT rồi RoBERTa với cùng thiết lập và loss đã chốt (BERT đã chạy ở Tuần 1); sau lần chạy cuối, xuất dự đoán Test một lần cho mọi mô hình | Mốc 2: bảng so sánh MAE, RMSE, R², tương quan trên Test và kiểm tra tiêu chí đạt |
| Tuần 3 | Hoàn thiện dashboard, phân tích lỗi, tổng hợp nhật ký dùng AI, báo cáo; chừa 2 ngày đệm | Mốc 3: Dashboard và báo cáo |

| Rủi ro | Cách xử lý |
| --- | --- |
| Mất GPU hoặc ngắt phiên Colab | Bật `fp16`, lưu checkpoint lên Drive mỗi epoch |
| Hết hạn mức GPU | Thứ tự chạy BERT, DistilBERT, RoBERTa; nếu không đủ thì bỏ trọn RoBERTa |
| Dữ liệu lớn | Tải một lần, đọc theo khối, lấy mẫu phân tầng một lần |
| Overfit | Theo dõi loss trên tập val |
| Mô hình dồn dự đoán về giá trị trung bình do lệch nhãn | Xem chỉ số theo từng mức sao và độ lệch chuẩn của dự đoán |
| Rò rỉ giữa train và test | Chia ngẫu nhiên phân tầng, loại review trùng văn bản, báo cáo mức trùng người dùng và doanh nghiệp |
| So sánh không công bằng giữa các mô hình | Cùng dữ liệu, seed, thiết lập và ngân sách huấn luyện |
| Trộn kết quả giữa các phiên chia hoặc phiên chạy | `split_id` ghi trong mỗi lần chạy; khi ghép dự đoán, dừng nếu `split_id` hoặc `review_id` không khớp |

## 14. Các quyết định đã chốt

- Tập test giữ phân phối sao tự nhiên (chia phân tầng 70/15/15); mọi chỉ số theo mức sao kèm số mẫu n.
- Demo: chỉ trình bày kết quả bằng trực quan (bảng so sánh, biểu đồ, ca đoán sai); ô nhập review mới nằm ngoài phạm vi.
- Clip điểm dự đoán về 1.0-5.0 khi dự đoán và khi đánh giá, áp dụng cho mọi mô hình kể cả baseline; ghi thêm `rmse_raw`.
- Mean pooling và Huber (`delta = 1.0`) là mặc định; MSE chạy đối chứng trên BERT; quy tắc đổi loss ở mục 9.
- Quy mô mẫu cố định 60.000 review.
- Chia ngẫu nhiên phân tầng theo số sao; loại review val/test trùng văn bản với train; mức trùng người dùng và doanh nghiệp là giới hạn của báo cáo, không chuyển sang chia theo nhóm.
- Tiêu chí đạt: RMSE Val giảm ít nhất 15% so với baseline và Macro-MAE Val không cao hơn baseline.
- Thứ tự chạy mở rộng: DistilBERT rồi RoBERTa; thiếu hạn mức GPU thì bỏ trọn RoBERTa.
- Một seed; không chạy thêm seed; paired bootstrap là mở rộng tuỳ chọn.
- Không lọc ngôn ngữ; ghi vào Giới hạn.
- Phân công theo mục 13; dữ liệu dự phòng và giấy phép theo mục 8.

**Số liệu ghi lại ở Mốc 1 (chỉ ghi nhận, không còn quyết định chờ):** `seconds_per_epoch` của BERT; số review từng mức sao trong Test; mức trùng `user_id` và `business_id`; số review bị loại do trùng văn bản; RMSE và MAE Val của Huber và MSE (áp dụng quy tắc mục 9); RMSE Val của baseline (tính ngưỡng 0,85 ×).
