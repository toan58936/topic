# Đề tài 7: Text Regression (Yelp)

Dự đoán điểm cảm xúc liên tục (1.0-5.0) từ review bằng Transformer Encoder + Linear regression head.
Tài liệu: thư mục `docs/`.


## Project Structure

The project is organized into several folders, each serving a specific purpose:

- `src/`: Contains the main source code of the project, including data processing, model building, training, evaluation, and prediction.
- `app/`: Contains the Streamlit application used to visualize model results and interact with the system.
- `docs/`: Stores project documentation, technical specifications, and design information.
- `plans/`: Contains project plans, task assignments, and development progress.
- `notebooks/`: Stores Jupyter notebooks used for experiments and model development.
- `reports/`: Stores experimental results, evaluation reports, and analysis.
- `tests/`: Contains test files used to verify the functionality of project modules.

## Documentation Update

This section provides an overview of the project structure to help contributors understand the organization of the source code and supporting resources.
  ## Ghi chú tìm hiểu dự án
- Tìm hiểu bài toán Text Regression trên dữ liệu Yelp.
- Mục tiêu là dự đoán điểm đánh giá từ 1.0 đến 5.0.
- Tìm hiểu kiến trúc Transformer Encoder và Regression Head.
- Tìm hiểu luồng xử lý dữ liệu đầu vào và đầu ra của mô hình.
## Thành viên E — Dashboard & Error Analysis

Hướng dẫn đầy đủ (cài đặt Windows, demo, CSV đầu vào, gán nhãn, cross-check và kiểm thử): [docs/member_e_guide.md](docs/member_e_guide.md).

```bash
python -m pip install -r app/requirements.txt
python -m streamlit run app/streamlit_dashboard.py
```

Chưa có dữ liệu training? Chạy `python -m scripts.create_dashboard_demo`, rồi nhập `demo_reports` ở ô **Thư mục CSV**. Dữ liệu demo được ghi rõ là giả lập, không dùng cho báo cáo kết quả.
