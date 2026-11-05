# Tài liệu Define – Đề tài 7: Text Regression (Yelp)

Nov 2, 2026 · @hi

Tài liệu này chốt dự án làm gì, cho ai, với dữ liệu và tiêu chí nào. Nó hợp nhất bản đăng ký đề tài 10 mục và phần Define trước đây. Cách xây dựng (kiến trúc, module, thư mục, luồng dữ liệu, kiểm thử) nằm ở tài liệu Design riêng. Mục chưa chốt được đánh dấu "dự kiến" hoặc gom ở mục 14.

## 1. Tên dự án

Dự đoán điểm cảm xúc liên tục (Sentiment Score) từ văn bản đánh giá khách hàng bằng Transformer Encoder (BERT) kết hợp Linear Regression Head.

- Công nghệ: Transformer Encoder (BERT) + Linear regression head.
- Hành động: dự đoán bằng hồi quy.
- Bài toán: ước lượng điểm cảm xúc từ văn bản review.
- Phạm vi: review tiếng Anh trong Yelp Dataset; mở rộng so sánh với RoBERTa và DistilBERT.

## 2. Hướng chính

Chọn một hướng duy nhất: **AI dự đoán**. Dự án dự đoán một điểm số liên tục (1.0-5.0) từ văn bản review bằng hồi quy trên Transformer Encoder.

Ba hướng còn lại không được chọn: Realtime (nhóm không cam kết yêu cầu về độ trễ), Tối ưu toàn cục (đề tài không giải bài toán tối ưu có ràng buộc), Tự động hoá (nhóm chỉ làm Demo minh hoạ, không tích hợp vào quy trình thật).

## 3. Vấn đề và người dùng

- **Người dùng:** nền tảng đánh giá dịch vụ/nhà hàng và nhóm phân tích dữ liệu của họ.
- **Vấn đề:** điểm sao 1-5 là thang rời rạc, trong khi nội dung review thể hiện mức hài lòng tinh hơn nhiều.
- **Giá trị mang lại:** ước lượng một điểm số liên tục (1.0-5.0) trực tiếp từ nội dung văn bản, để so sánh và phân tích mức hài lòng chi tiết hơn.

## 4. Mục tiêu

- Xây dựng mô hình hồi quy dự đoán điểm cảm xúc liên tục từ văn bản review, và cải thiện rõ rệt so với Baseline (TF-IDF + Ridge) trên cùng tập test cố định.
- So sánh công bằng BERT, RoBERTa và DistilBERT về sai số (MAE, RMSE, R², tương quan) và chi phí (thời gian chạy, số tham số).
- Ngưỡng cụ thể (RMSE tuyệt đối, tỉ lệ giảm sai số so với baseline) sẽ xác định sau khi có kết quả baseline ở cuối Tuần 1.

Mốc so sánh là baseline chạy trên cùng dữ liệu và cùng tập test, nên mục tiêu đo được mà chưa cần cam kết con số khi chưa có dữ liệu.

## 5. Phạm vi

- **Bắt buộc:** Baseline, huấn luyện BERT + Linear regression head, đánh giá hồi quy.
- **Mở rộng:** so sánh công bằng với RoBERTa và DistilBERT; ô nhập review mới để mô hình chấm điểm trực tiếp (nếu còn thời gian, không nằm trong phạm vi hiện tại).
- **Ngoài phạm vi:** triển khai hoặc tích hợp vào hệ thống thật, yêu cầu realtime, xử lý review thiếu sao hoặc chấm sao lệch nội dung (có thể là hướng phát triển sau).

## 6. Input / Output

| Thành phần | Mô tả |
| --- | --- |
| Input | Một review tiếng Anh dạng chuỗi văn bản, cắt ở 256 token |
| Output | Một số thực là điểm dự đoán trên thang sao (khoảng 1.0-5.0); điểm được giới hạn (clip) về khoảng này khi dự đoán và khi đánh giá |
| Nhãn huấn luyện | Số sao nguyên 1-5 của review, coi như giá trị liên tục |

## 7. Tính năng cốt lõi

1. Đọc, lấy mẫu phân tầng và chia dữ liệu Yelp (70/15/15, `seed = 42`).
2. Baseline TF-IDF + Ridge.
3. Fine-tune BERT + Linear regression head.
4. So sánh công bằng với RoBERTa và DistilBERT (mở rộng).
5. Đánh giá hồi quy và phân tích lỗi.
6. Dashboard trực quan: bảng so sánh ba mô hình, biểu đồ và các ca đoán sai điển hình (ô nhập review mới là hướng mở rộng).

## 8. Dữ liệu

| Hạng mục | Mô tả |
| --- | --- |
| Nguồn | [Yelp Dataset trên Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset), theo gợi ý của giảng viên |
| File sử dụng | `review.json`: văn bản review và số sao (1-5) |
| Quy mô | Lấy mẫu ngẫu nhiên phân tầng theo số sao, giữ nguyên tỉ lệ sao như dữ liệu gốc. Quy mô chốt 60.000 review, có thể tăng tới 100.000 nếu thời gian huấn luyện còn dư (xác nhận bằng số liệu ở cuối Tuần 1) |
| Chia dữ liệu | Train/val/test theo tỉ lệ 70/15/15, cố định bằng `seed = 42`; chia ngẫu nhiên phân tầng theo số sao, và đếm tỉ lệ trùng người dùng hoặc doanh nghiệp giữa train và test ở Tuần 1 |
| Tiền xử lý | Cắt ở 256 token; làm sạch (dự kiến): bỏ review rỗng, chuẩn hoá khoảng trắng |
| Nhãn | Số sao nguyên 1-5, coi như giá trị liên tục; giả định này được nêu rõ trong báo cáo |
| Giới hạn | Phân phối gốc lệch về sao cao; nhãn có thể nhiễu (review chấm sao không khớp nội dung); có thể trùng người dùng hoặc doanh nghiệp giữa train và test (đo ở Tuần 1 và ghi vào báo cáo) |
| Giấy phép | Theo điều khoản của Yelp; cần đối chiếu và ghi lại phiên bản dữ liệu đã tải |

Tập test giữ phân phối sao tự nhiên; ngoài chỉ số toàn cục, báo thêm MAE/RMSE theo từng mức sao và Macro-MAE.

## 9. Phương pháp

**Baseline:** TF-IDF + Ridge Regression, chạy nhanh, không cần GPU, làm mốc so sánh.

**Mô hình chính:** fine-tune `bert-base-uncased` theo các bước sau.

1. Tách review thành token, cắt ở 256 token.
2. Đưa qua Transformer Encoder để mỗi token có vector mang ngữ cảnh.
3. Gộp thành một vector cho cả review bằng `[CLS]` hoặc Mean pooling.
4. Qua lớp Linear (768 chiều xuống 1) ra một số thực, không dùng softmax.
5. Tối ưu bằng MSE hoặc Huber Loss.

**Mặc định đã chốt:** Mean pooling và Huber (`delta = 1.0`); `[CLS]` và MSE chỉ là phương án đối chứng. MSE chạy đối chứng trên BERT; nếu RMSE trên tập val của MSE thấp hơn đáng kể thì đổi mặc định sang MSE cho cả ba mô hình trước khi chạy Tuần 2. Lý do kỹ thuật nằm ở tài liệu Design.

**Mở rộng:** thay backbone, giữ nguyên mọi thiết lập khác.

| Mô hình | Vai trò | Ghi chú |
| --- | --- | --- |
| TF-IDF + Ridge | Baseline | Chỉ xét từ nào có mặt, không hiểu ngữ cảnh |
| BERT (`bert-base-uncased`) | Mô hình chính | Kiến trúc gốc |
| RoBERTa | Mở rộng | So sánh về sai số |
| DistilBERT | Mở rộng | Bản rút gọn, nhẹ hơn; so sánh thêm thời gian chạy và số tham số |

Cả ba backbone dùng cùng dữ liệu, cùng seed, cùng cách gộp vector, cùng loss và cùng ngân sách huấn luyện, bật `fp16`.

## 10. Sản phẩm

- Mã nguồn dạng module (`going_modular/`) trên GitHub, kèm README và hướng dẫn cài đặt/chạy (`requirements.txt`).
- Tập dữ liệu mẫu và dữ liệu đã chia cố định trên Drive.
- Trọng số mô hình (`.pt`/`.pth`) lưu trên Drive.
- Notebook chạy trên Colab.
- Dashboard trực quan trình bày kết quả: bảng so sánh ba mô hình, biểu đồ (điểm dự đoán so với điểm thật, phân phối sai số, sai số theo từng mức sao) và các ca đoán sai điển hình, đọc từ `test_predictions.csv`. Ô nhập review mới là hướng mở rộng, chưa nằm trong phạm vi hiện tại. Công cụ dựng dashboard nằm ở tài liệu Design.
- Báo cáo, kèm nhật ký dùng AI.

## 11. Ràng buộc công nghệ

| Ràng buộc | Nội dung |
| --- | --- |
| Thời gian và nhân sự | 2-3 tuần, 6 thành viên (1 Leader, 5 thành viên phụ trách các mảng độc lập) |
| Quản lý mã nguồn | VS Code và GitHub, mã dạng module (`going_modular/`), nhánh riêng và Pull Request |
| Lưu trữ | Google Drive chung cho dữ liệu đã chia và trọng số (`.pt`/`.pth`) |
| Tính toán | Google Colab GPU; phiên có thể bị ngắt nên bật `fp16` và lưu checkpoint mỗi epoch |
| Tái lập | Cố định `seed = 42`, cấu hình mỗi lần chạy được ghi lại |
| Thư viện | Python; PyTorch và Hugging Face Transformers (dự kiến) |

## 12. Tiêu chí đánh giá

Tất cả mô hình được đo trên cùng một tập test cố định (`seed = 42`), giữ phân phối sao tự nhiên. Điểm dự đoán được giới hạn (clip) về 1.0-5.0 trước khi tính chỉ số, áp dụng đồng nhất cho cả ba mô hình; ghi thêm RMSE trên giá trị thô để tham khảo.

| Chỉ số | Ý nghĩa |
| --- | --- |
| MAE | Sai số tuyệt đối trung bình, tính trên thang sao |
| RMSE | Căn bậc hai sai số bình phương trung bình, phạt nặng sai số lớn |
| R² | Phần biến thiên của điểm số mà mô hình giải thích được |
| Pearson / Spearman | Mức tương quan giữa điểm dự đoán và điểm thật |
| MAE/RMSE theo từng mức sao, Macro-MAE | Kiểm tra mô hình có dồn dự đoán về giá trị trung bình ở các mức 2-3 sao hay không; Macro-MAE là trung bình MAE của 5 mức sao |
| Thời gian chạy, số tham số | Chi phí của từng mô hình |

- **Biểu đồ:** điểm dự đoán so với điểm thật, phân phối sai số, sai số theo từng mức sao; cùng bảng so sánh ba mô hình và các ca đoán sai điển hình, chúng tạo thành dashboard trình bày kết quả.
- **Phân tích lỗi:** các nhóm lỗi dự kiến gồm review mỉa mai, cảm xúc lẫn lộn, review quá dài bị cắt ở 256 token, và nhãn nhiễu.
- **Mục tiêu:** cải thiện rõ rệt so với Baseline; ngưỡng cụ thể xác định sau khi có kết quả baseline ở cuối Tuần 1.
- **Đối chiếu:** kết quả huấn luyện thật được kiểm tra lại với tài liệu này (input/output, ràng buộc, tiêu chí) ở bước Check description trong tài liệu Design.

## 13. Kế hoạch và rủi ro

Nhóm 6 thành viên: Leader quản lý kiến trúc và review code, 5 thành viên phụ trách các mảng độc lập (cách chia mảng chưa chốt).

| Thời gian | Công việc | Mốc kiểm tra |
| --- | --- | --- |
| Tuần 1 | Data và Baseline; chốt tài liệu Define và Design, khung code GitHub, dữ liệu mẫu trên Drive | Mốc 1: có kết quả Baseline |
| Tuần 2 | Train BERT, RoBERTa, DistilBERT trên Colab với cùng thiết lập | Mốc 2: bảng so sánh MAE, RMSE, R², tương quan |
| Tuần 3 | Dựng dashboard kết quả, phân tích lỗi, tổng hợp nhật ký dùng AI, hoàn thiện báo cáo | Mốc 3: Dashboard và báo cáo |

| Rủi ro | Cách xử lý |
| --- | --- |
| Mất GPU hoặc ngắt phiên Colab | Bật `fp16`, lưu checkpoint lên Drive mỗi epoch |
| Dữ liệu lớn | Lấy mẫu phân tầng, đọc dữ liệu theo khối |
| Overfit | Theo dõi loss trên tập val |
| Mô hình dồn dự đoán về giá trị trung bình do lệch nhãn | Kiểm tra phân phối dự đoán, xem chỉ số theo từng mức sao |
| So sánh không công bằng giữa các mô hình | Cùng dữ liệu, seed, thiết lập và ngân sách huấn luyện |

## 14. Điểm cần nhóm thống nhất

**Đã chốt**

- Tập test giữ phân phối sao tự nhiên (chia phân tầng 70/15/15); ngoài chỉ số toàn cục, báo thêm MAE/RMSE theo từng mức sao và Macro-MAE.
- Hình thức Demo: chỉ cần trình bày kết quả bằng trực quan (bảng so sánh, biểu đồ, ca đoán sai). Ô nhập review mới là hướng mở rộng, không thuộc phạm vi hiện tại.
- Giới hạn điểm dự đoán (clip) về 1.0-5.0 khi dự đoán và khi đánh giá, áp dụng cho cả ba mô hình; ghi thêm RMSE trên giá trị thô để tham khảo. Việc clip không làm tăng sai số vì nhãn thật luôn nằm trong khoảng này.
- Mặc định Mean pooling và Huber (`delta = 1.0`), MSE đối chứng trên BERT; có quy tắc đổi sang MSE ở mục 9.
- Quy mô mẫu 60.000 review (tối đa 100.000 nếu còn thời gian huấn luyện).
- Chia ngẫu nhiên phân tầng theo số sao, kèm đếm tỉ lệ trùng người dùng hoặc doanh nghiệp; chỉ chuyển sang chia theo nhóm nếu tỉ lệ trùng cao.

**Xác nhận bằng số liệu ở cuối Tuần 1**

- Thời gian huấn luyện thực tế của một epoch, để xác nhận quy mô mẫu (tăng lên 100.000 hoặc giảm).
- Số review ở mức sao hiếm nhất trong tập test, đủ lớn để chỉ số theo từng mức sao có ý nghĩa.
- Tỉ lệ trùng người dùng hoặc doanh nghiệp giữa train và test.
- Kết quả MSE đối chứng trên BERT, để xác nhận loss mặc định.
- Ngưỡng mục tiêu (RMSE tuyệt đối, tỉ lệ giảm sai số so với baseline) sau khi có kết quả baseline.
