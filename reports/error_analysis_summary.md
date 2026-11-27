# Phân tích lỗi tự động — mẫu 40 review

Mô hình: **roberta**, chọn theo RMSE Test thấp nhất trong `model_summary.csv`.
Split: `f0e6776f8a1a`. 20 ca sai lớn nhất + 20 ca ngẫu nhiên phân tầng (4/sao), `seed=42`, không trùng review.
Phương pháp gán nhãn: **Tự động hóa (Rule-based NLP Heuristic)** dựa trên độ dài token, từ khóa cảm xúc và cấu trúc tương phản.

| Nhóm | Số review | % mẫu 40 | Worst (20) | Random (20) |
|---|---:|---:|---:|---:|
| Nhãn nhiễu | 0 | 0.0% | 0 | 0 |
| Cảm xúc lẫn lộn | 11 | 27.5% | 2 | 9 |
| Bị cắt 256 token | 8 | 20.0% | 6 | 2 |
| Mỉa mai | 1 | 2.5% | 1 | 0 |
| Khác | 20 | 50.0% | 11 | 9 |

Tỉ lệ Test có `n_tokens > 254`: **11.55%** trên 8999 review.

Tỉ lệ nhóm lỗi trên mẫu có chủ đích này không đại diện cho toàn bộ Test. Cờ `n_tokens` dùng tokenizer BERT theo đặc tả; không tự chứng minh việc cắt gây ra lỗi, đặc biệt với RoBERTa.

## Ví dụ theo nhóm (không sao chép nội dung review)
- **Cảm xúc lẫn lộn**: `r7my8f5IfzUClhXpDnr6Fg`, thật 1, dự đoán 4.117, sai số 3.117. Review pha trộn khen (['great']) và chê ([]) kèm từ nối tương phản.
- **Bị cắt 256 token**: `Drdrx5eWHCR-ZLP4Pl8iew`, thật 5, dự đoán 1.000, sai số 4.000. Văn bản dài 462 token (>254), phần nội dung cuối bị cắt mất dẫn đến sai số 4.00.
- **Mỉa mai**: `45AIVHHzJMrPg2qLLQieMw`, thật 1, dự đoán 3.938, sai số 2.938. Biểu hiện mỉa mai/từ tích cực ngược ngữ cảnh (love, perfect) khiến mô hình dự đoán 3.94 sao.
- **Khác**: `9B-89poaShREU2_aqK4Twg`, thật 5, dự đoán 1.000, sai số 4.000. Review ngắn/trung tính (89 token) hoặc sắc thái mơ hồ; dự đoán 1.00 so với thật 5.
