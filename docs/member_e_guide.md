# Bàn giao Thành viên E — Dashboard & Error Analysis

## 1. Trạng thái và phạm vi

Đã hiện thực dashboard CSV-only, kiểm thử CPU không cần PyTorch, công cụ lấy mẫu và kiểm tra/tổng hợp Error Analysis, bộ dữ liệu demo sinh được bằng lệnh. Đã cập nhật AI log.

Bản ZIP đầu vào không có `model_summary.csv`, `test_predictions.csv`, `test_predictions_with_text.csv` hoặc checkpoint/data thật. Vì vậy **chưa có kết quả phân tích 40 review thật, nhãn của E hay xác nhận của Leader**. Không suy diễn kết quả training từ dữ liệu giả. PR/merge và xác nhận F/Leader chưa thực hiện trong bản bàn giao ZIP.

## 2. Cài và chạy trên Windows (PowerShell)

Mở terminal tại thư mục `topic-main` vừa giải nén (nơi có README.md):

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r app/requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_dashboard.py
```

Không cần activate venv, GPU, torch hay transformers. Linux/macOS thay `.\.venv\Scripts\python.exe` bằng `.venv/bin/python` (tạo venv với `python3 -m venv .venv`).

Dữ liệu mặc định là `reports/` trong repo, kể cả khi chạy từ thư mục `app/`. Có thể chọn đường dẫn khác bằng ô **Thư mục CSV** trong sidebar hoặc biến môi trường `YELP_REPORTS_DIR`.

**Xem demo ngay khi chưa có kết quả training:**

```powershell
.\.venv\Scripts\python.exe -m scripts.create_dashboard_demo
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_dashboard.py
```

Đổi ô **Thư mục CSV** thành `demo_reports`. Banner ghi rõ **DỮ LIỆU GIẢ LẬP**. Dữ liệu demo gồm 200 review và bảng 5 mô hình, được tạo bằng seed=42; chi phí training để trống vì không có training thật. Script từ chối ghi đè thư mục đã có; dùng `--output-dir demo_reports_2` nếu cần tạo bản khác. Không copy demo sang `reports/` để nộp kết quả.

## 3. Các chức năng dashboard

- Tự ưu tiên `test_predictions_with_text.csv`, fallback về `test_predictions.csv`.
- Bảng toàn bộ mô hình có trong summary (bao gồm mean predictor), giữ ô thiếu là trống; ẩn cột toàn null.
- Scatter dự đoán/nhãn thật kèm đường y=x; histogram sai số tuyệt đối; bảng và biểu đồ MAE/RMSE từng sao, số mẫu n, Macro-MAE.
- Chọn model, lọc sao, sắp xếp; bảng Top-10 luôn là sai số lớn nhất trong bộ lọc.
- Click chọn dòng để mở chi tiết; dropdown là cách chọn thay thế. Text hiển thị đầy đủ khi có.
- Bảng tổng hợp giữ phạm vi toàn Test; per-star và các biểu đồ tính theo bộ lọc.
- Không chạy inference, không import module huấn luyện, không ghi file từ dashboard.
- Dữ liệu lỗi/split mismatch/ID trùng được báo rõ; bộ lọc rỗng không crash. Review thiếu dự đoán của model đang chọn được thông báo và bỏ khỏi phần tính chỉ số đó.

## 4. Giao diện dữ liệu cần nhận từ Leader/F

| File | Nội dung |
|---|---|
| `reports/model_summary.csv` | `model`, các metrics và cost theo spec, `split_id`; RMSE Test hợp lệ để chọn model Error Analysis |
| `reports/test_predictions.csv` | `review_id,true_stars,n_tokens,split_id,pred_<model>,abs_err_<model>` |
| `reports/test_predictions_with_text.csv` | Cùng dữ liệu và thêm `text`; chỉ local/Drive |

Dự đoán đã clip [1,5], abs_err khớp sai số tuyệt đối, `review_id` duy nhất. `n_tokens` là độ dài BERT chưa cắt theo quy ước spec; `split_id` phải thống nhất.

**Điểm cần phối hợp với D/B/F:** `src/predict.py::merge_test_predictions` trong ZIP hiện chưa xuất `n_tokens` và `split_id` vào CSV tổng hợp. Dashboard chấp nhận bản cũ nhưng hiện cảnh báo; Error Analysis yêu cầu bản xuất đầy đủ. Không tự dùng số từ thay cho token count, không tự bịa split_id. `src/metrics.py` mới có metrics lõi và `src/baseline.py` còn rỗng, vì vậy còn chờ summary thật từ B/Leader.

Tài liệu thiết kế có ví dụ `per_star_metrics(..., clip=True)` nhưng implementation hiện tại chỉ nhận hai đối số và tự clip. Dashboard gọi đúng implementation, không sửa module B.

## 5. Hoàn tất Error Analysis sau khi có dữ liệu thật

Chỉ làm sau khi Leader training và F xác nhận đầu ra hợp lệ.

### Bước 1 — Lấy mẫu cố định

```powershell
.\.venv\Scripts\python.exe -m scripts.error_analysis prepare --reports-dir reports
```

Script chọn model có RMSE Test thấp nhất trong summary; hòa điểm thì chọn theo tên. Nếu model tốt nhất không có cột dự đoán (kể cả mean_predictor) thì báo lỗi, không âm thầm chọn model khác. Nhờ Leader xuất cột dự đoán tương ứng nếu gặp trường hợp này.

Lấy top-20 theo abs_err và 4 review/mức sao từ phần Test còn lại, seed=42: đủ **40 review không trùng nhau**. Sắp ID trước khi lấy mẫu để không phụ thuộc thứ tự CSV. Thiếu mẫu ở một mức sao thì dừng. Đây là quy ước làm rõ phần spec chưa chỉ định xử lý trùng giữa worst và random.

Các file tạo ra:

| File | Cách dùng |
|---|---|
| `error_sample.csv` | 5 cột ID/model/sai số/nhãn/dự đoán theo Phase 6 |
| `error_sample_with_text.csv` | Đọc 40 review, loại mẫu, n_tokens và cờ cắt |
| `error_analysis.csv` | E điền `error_group` và `notes`, giữ nguyên 5 cột dữ liệu gốc |
| `leader_crosscheck_with_text.csv` | 4 review cố định cho Leader, chưa chứa nhãn E |
| `error_analysis_manifest.json` | Ghi model/split/seed/ID để kiểm tra tính nhất quán |

Script không ghi đè worksheet đã có. Nếu cần một vòng mới, dùng `--output-dir reports/error_round2` ở **cả hai** lệnh prepare/summarize. Giữ lại vòng cũ để đối chiếu.

### Bước 2 — Đọc và gán nhãn

E đọc `error_sample_with_text.csv`; điền trực tiếp `error_analysis.csv` bằng Excel/CSV editor và lưu UTF-8, giữ `review_id` ở kiểu chuỗi. Mỗi review một nhóm, notes nêu lý do cụ thể; không thay đổi dự đoán hay sai số.

| Giá trị chính xác của error_group | Tiêu chí |
|---|---|
| Nhãn nhiễu | Nội dung tích cực nhưng sao ≤2 hoặc tiêu cực nhưng sao ≥4; đọc toàn văn để loại bỏ mỉa mai/cảm xúc lẫn lộn |
| Cảm xúc lẫn lộn | Cùng có khen và chê; but/however/although chỉ là dấu hiệu gợi ý |
| Bị cắt 256 token | n_tokens >254 theo spec; nên ghi phần thông tin có thể mất, không tự suy ra nguyên nhân chỉ từ độ dài |
| Mỉa mai | Ý nghĩa đảo ngược từ ngữ bề mặt, cần giải thích ngữ cảnh |
| Khác | Không thuộc các nhóm trên; bắt buộc lý do cụ thể |

Nếu nhiều nhóm cùng phù hợp, chọn nguyên nhân chính có bằng chứng rõ nhất, ghi nguyên nhân phụ trong notes. Cờ cắt dùng độ dài BERT, không chứng minh cùng số token ở RoBERTa. Không gán nhóm tự động bằng từ khóa.

### Bước 3 — Leader kiểm tra độc lập

Leader đọc file 4 review và điền `leader_error_group` (cùng 5 nhãn), có thể thêm `leader_notes`, trước khi xem nhãn E. Không thay đổi 4 ID đã chọn. Cần đồng ý ít nhất 3/4 (75%). Nếu thấp hơn, thảo luận bằng chứng, lưu bản cũ, gán lại và kiểm tra lại. Script chỉ tính agreement từ các nhãn được cung cấp, không thể xác thực người nhập là Leader.

### Bước 4 — Xuất tổng hợp

```powershell
.\.venv\Scripts\python.exe -m scripts.error_analysis summarize --reports-dir reports
```

Chỉ xuất khi đủ 40 nhãn, notes hợp lệ và cross-check ≥75%:

- `error_group_summary.csv`: số lượng và % cả 5 nhóm (gồm nhóm 0), tách worst/random.
- `error_analysis_summary.md`: đoạn tổng hợp cho báo cáo, agreement, ví dụ ID và tỉ lệ Test có n_tokens >254.

Tỉ lệ nhóm trên mẫu chủ đích 40 review **không đại diện cho toàn bộ Test**. Kiểm tra và biên tập notes trước khi đưa vào báo cáo; tránh chép nguyên review vào file được commit.

## 6. Kiểm thử và bàn giao GitHub

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest tests/test_dashboard.py tests/test_error_analysis.py -v
```

Test gồm kiểm tra dữ liệu/biểu đồ, AppTest với/không text, đổi model/lọc/sắp xếp/chi tiết, không import torch, sampling tái lập, nhãn và agreement. Các fixture tạo trong thư mục tạm, không thay CSV thật.

`tests/conftest.py` chuyển import torch sang bên trong helper huấn luyện để `pytest tests/test_dashboard.py` không đòi cài torch. `_SynthDataset` vẫn cung cấp `__len__`/`__getitem__` theo protocol của DataLoader. Chưa chạy lại các test training A–D trong môi trường không có training dependencies.

Các thay đổi để review:

- `app/streamlit_dashboard.py`, `app/requirements.txt`
- `scripts/create_dashboard_demo.py`, `scripts/error_analysis.py`
- `tests/test_dashboard.py`, `tests/test_error_analysis.py`, sửa nhỏ `tests/conftest.py`
- `docs/member_e_guide.md`, `docs/ai_log.md`, `reports/error_analysis_status.md`
- README và `.gitignore`

Bản này là ZIP, không có remote GitHub đã kết nối. Sau khi chép thay đổi vào clone của nhóm: tạo branch `feature/dashboard`, chạy test, commit/push, mở PR vào develop, nhờ Leader + một người cross-module review. Chưa đánh dấu PR merged hay Phase 6 done. Chụp ảnh dashboard với dữ liệu thật để Leader dùng làm bản dự phòng khi demo.
