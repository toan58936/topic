\# AI Log



\## Phase 2 – Model Building (Member C)



\*\*Thành viên:\*\* C – Model  

\*\*Module:\*\* `src/model\_builder.py`  

\*\*Ngày thực hiện:\*\* 14/11/2026  

\*\*Git branch:\*\* `feature/model-builder`  

\*\*Pull Request:\*\* \[PR #7](https://github.com/toan58936/topic/pull/7)



\### 1. Mục tiêu



Sử dụng AI hỗ trợ xây dựng module Transformer Model Builder phục vụ bài toán dự đoán điểm đánh giá (Text Regression).



Các yêu cầu chính:

\- Hỗ trợ ba backbone: BERT, RoBERTa và DistilBERT.

\- Xây dựng `TextRegressor` với Mean Pooling hoặc CLS Pooling.

\- Sử dụng Dropout và Linear Regression Head.

\- Xây dựng các hàm `build\_model()`, `build\_tokenizer()` và `count\_parameters()`.

\- Cho phép khởi tạo model bằng Tiny Random Config để kiểm thử trên CPU.

\- Không sử dụng `token\_type\_ids` trong forward.



\### 2. AI hỗ trợ thực hiện



\*\*Prompt summary (tóm tắt yêu cầu đã trao đổi với AI):\*\*



Yêu cầu AI hướng dẫn xây dựng `src/model\_builder.py` theo tài liệu thiết kế của dự án, hỗ trợ BERT, RoBERTa và DistilBERT; triển khai pooling, regression head, tokenizer, đếm tham số và giải thích từng phần code.



\*\*Các nội dung AI hỗ trợ:\*\*

\- Đề xuất cấu trúc module và cách khởi tạo ba loại Transformer backbone.

\- Hướng dẫn triển khai Mean Pooling có sử dụng `attention\_mask` để bỏ qua padding.

\- Hướng dẫn triển khai CLS Pooling bằng hidden state tại vị trí đầu tiên.

\- Xây dựng Regression Head gồm Dropout và Linear.

\- Hướng dẫn khởi tạo model bằng Tiny Config để kiểm thử nhanh trên CPU.

\- Hướng dẫn viết test cho output shape, pooling, parameter count, tokenizer và các trường hợp kiểm tra khác.



\### 3. Kiểm thử và xác minh kết quả



\*\*Tiny Config Test:\*\*



Sử dụng cấu hình Transformer nhỏ với:

\- `hidden\_size = 32`

\- `num\_hidden\_layers = 2`

\- `num\_attention\_heads = 2`

\- `intermediate\_size = 64` (BERT/RoBERTa)

\- Cấu hình tương ứng của DistilBERT.



Kiểm tra cả ba backbone có thể khởi tạo và thực hiện forward trên dữ liệu giả lập.



\*\*Các nội dung đã kiểm tra:\*\*

\- Input batch có kích thước `\[2, 16]` cho output `\[2]`.

\- Mean Pooling không bị ảnh hưởng bởi padding.

\- CLS Pooling lấy hidden state tại vị trí 0.

\- Regression Head hỗ trợ khởi tạo bias theo giá trị trung bình nhãn train khi được cung cấp.

\- Parameter count bao gồm backbone và head, không tính pooler đã loại bỏ.

\- Forward không sử dụng `token\_type\_ids`.



\*\*Lệnh kiểm thử:\*\*



```powershell

pytest -v

```



\*\*Kết quả thực tế:\*\*



`49 passed in 25.35s`



Tất cả 49 bài kiểm thử hiện có trong bộ test Model Builder đều thành công trên môi trường CPU.



\### 4. Đánh giá và kiểm tra của thành viên



Sau khi nhận code hỗ trợ từ AI, thành viên đã:

\- Đọc và kiểm tra lại nội dung `src/model\_builder.py`.

\- Đối chiếu implementation với tài liệu thiết kế và tài liệu kiểm thử của Model Builder.

\- Chạy bộ test bằng pytest và kiểm tra kết quả thực tế.

\- Kiểm tra các file cần commit trước khi đưa lên GitHub.

\- Thực hiện commit và push code lên nhánh riêng.



AI được sử dụng như công cụ hỗ trợ viết code, giải thích và hướng dẫn kiểm thử. Kết quả cuối cùng được xác nhận bằng việc chạy test thực tế và kiểm tra Pull Request trên GitHub.



\### 5. GitHub và kết quả bàn giao



\*\*Commit:\*\*

\- Commit ID: `9959764`

\- Message: `feat: add transformer model builder`



\*\*Các file đã commit:\*\*

\- `src/model\_builder.py`

\- `tests/conftest.py`

\- `tests/test\_model\_builder.py`



\*\*Pull Request:\*\*

\- PR #7

\- Branch: `feature/model-builder`

\- Target branch: `develop`

\- Trạng thái: Merged



\### 6. Kết luận



Thành viên C đã hoàn thành phần xây dựng Model Builder, kiểm thử CPU với 49 bài test thành công và hợp nhất Pull Request vào nhánh `develop`.



Phần cần tiếp tục phối hợp là xác nhận giao diện đếm tham số với module `metrics.py` của thành viên B và kiểm tra tích hợp trong toàn bộ pipeline của dự án.

