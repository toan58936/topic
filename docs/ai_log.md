# AI Log — Đề tài 7 Text Regression (Yelp)

> Running log theo `plans/workflow.md` mục 6. Mỗi entry: Phase - ngày - owner, Prompt, Smoke Test, Run Model, Check Description.

---

## Phase 1 - 2026-11-09 - A (Data)

### Prompt
> Viết `src/data_setup.py` (`prepare_splits`, `create_dataloaders`, `ReviewDataset`, `_allocate_counts`) theo spec `docs/define&designt/01_data.md` và test plan `docs/tests/01_data_tests.md`. Kèm `tests/conftest.py` + `tests/test_data_setup.py` (D1–D15, S1–S2 slow).

### Smoke Test
- `pytest tests/test_data_setup.py -m "not slow"`: **PASS 15/15** (0.70s, CPU synthetic)
- S1/S2 (`slow`): skip — chưa có `review.json` thật trên máy local.

### Run Model
- Chưa chạy `prepare_splits()` trên `review.json` thật (chờ Leader/Colab, Phase 1.3).

### Check Description
- [x] Đọc file theo khối, không nạp RAM (2 lượt đọc + hash sha256 stream)
- [x] Lấy mẫu phân tầng giữ phân phối tự nhiên (`_allocate_counts` + rng riêng `default_rng(seed)`)
- [x] Chia 70/15/15 phân tầng theo sao, tái lập theo seed (D4)
- [x] Làm sạch NFC/strip/lower/collapse whitespace; loại text rỗng (D6)
- [x] Ghi `train/val/test.parquet` + `split_stats.json` đủ schema (+ khoá `quota` phục vụ test D2)
- [x] Không ghi đè trừ khi `overwrite=True` (D8)
- [x] `ReviewDataset`/`create_dataloaders`: shape `[B,256]`, labels `float32`, review_id list[str], shuffle chỉ Train (D10–D13)
- [x] Val/Test không trùng text với Train sau chuẩn hoá (D15, `duplicate_text_removed`)

### Vấn đề môi trường (local Windows)
- **Smart App Control = On** chặn DLL của pandas/pyarrow/torch bản mới trong `site-packages` (`WinError 4551 / DLL load failed`).
- Workaround đã dùng để chạy test: cài bản cũ hơn vào `%TEMP%` và set `PYTHONPATH`:
  ```powershell
  pip install "pandas==2.2.3" --target $env:TEMP\pdtest
  pip install "pyarrow==18.1.0" --target $env:TEMP\lab2deps
  pip install "torch==2.8.0+cpu" --index-url https://download.pytorch.org/whl/cpu --target $env:TEMP\lab2deps
  $env:PYTHONPATH = "$env:TEMP\pdtest;$env:TEMP\lab2deps;."
  pytest tests/test_data_setup.py -m "not slow"
  ```
- Trên Colab/Linux không gặp vấn đề này. Cân nhắc tắt Smart App Control hoặc chuẩn hoá mọi người dev trên Colab.

### Ghi chú quyết định
- `split_stats.json` có thêm khoá `quota` (`{split: {star: count}}`) ngoài schema gốc — cần cho test D2 đối chiếu hạn ngạch.
- `data_version` = `yelp_YYYY_MM_<sha8>` với YYYY_MM theo thời điểm chạy (UTC), sha8 = 8 ký tự đầu sha256 của `review.json`.
- Phát hiện thêm: tên file `docs/define&designt/Tài liệu Define.md` / `Tài liệu Design.md` bị lỗi encoding trên Windows ("T�i li?u") — cần rename.

---

## Phase 2 - 2026-11-11 - C (Model) + D (Training Infra)

### Prompt
> Viết `src/model_builder.py`, `src/utils.py`, `src/engine.py`, `src/train.py`, `src/predict.py` theo spec `docs/03_model.md`, `docs/04_engine_and_utils.md`; kèm `tests/test_model_builder.py`, `test_utils.py`, `test_engine.py`, `test_train.py`, `test_predict.py`. `src/metrics.py` viết phần lõi (clip_scores, rmse, mae, compute_metrics, per_star_metrics) làm dependency — phần bootstrap/summary/baseline của member B chưa làm.

### Smoke Test
- `pytest tests/ -m "not slow"`: **PASS 60/60** (10.2s, CPU, mô hình tiny random config)
- Gồm: 15 test data_setup + 14 model_builder + 4 utils + 13 engine + 5 train + 9 predict
- S1/S2 (`slow`/`gpu`): skip — chờ Colab/GPU thật.

### Run Model
- Chưa chạy trên `review.json`/checkpoint thật (Phase 3 trở đi, Colab).

### Check Description
- [x] `fit` chạy đúng epochs, history.csv đủ cột, best_model.pt = epoch RMSE Val thấp nhất (F1)
- [x] Resume sau `stop_after_epoch` cho kết quả y hệt chạy liền trên CPU/dropout=0 (F2)
- [x] Guard resume/overwrite: config khác -> ValueError nêu khoá; completed -> FileExistsError; overwrite thắng resume (F3)
- [x] `fit` không nhận test_loader; `run_training` với ExplodingLoader hoàn tất (F5)
- [x] fp16 trên CPU -> UserWarning + tự tắt; loss không hữu hạn -> FloatingPointError (L5, L7)
- [x] `predict` clip [1,5], không phụ thuộc batch_size, khôi phục chế độ train/eval (P1-P4)
- [x] `load_trained_model` đọc backbone_config từ checkpoint, không cần mạng (P5)
- [x] `export/merge_test_predictions` đúng cột, chặn ghi đè, kiểm tra khớp review_id/true_stars (P6, P7)
- [x] `set_seed` gọi lại ngay trước `build_model` (T4)

### Ghi chú quyết định / điểm lệch spec nhỏ
- `fit` nhận `epoch` làm keyword nội bộ của `train_one_epoch` (mặc định 0) để thông báo số epoch trong FloatingPointError.
- `history.csv`/`run_meta.json`/checkpoint ghi **mỗi epoch** bằng `write_atomic` (os.replace).
- `best_model.pt` lưu `model_state_dict` đã `.cpu()`; `last.pt` xoá khi hoàn tất trừ khi `keep_last=True`.
- `predict.py` `AutoConfig.for_model` cần bỏ khoá `model_type` khỏi config dict (transformers 5.x báo trùng tham số).
- `metrics.py` hiện chỉ có phần lõi; `bootstrap_ci`, `paired_bootstrap_diff`, `mean_predictor`, `build_model_summary`, `write_model_summary` và toàn bộ `baseline.py` còn lại của member B.
- `test_model_builder.py` mới viết theo checklist phase2 (spec `03_model_tests.md` có thêm nhóm slow/gpu chưa cover).

---

## Phase 3 - 2026-11-13 - B (Eval)

### Prompt
> Viết toàn bộ `src/metrics.py` và `src/baseline.py` kèm `tests/test_metrics.py`, `tests/test_baseline.py` theo spec `docs/define&designt/02_evaluation.md` và `docs/tests/02_evaluation_tests.md`. `metrics.py`: clip_scores, rmse, mae, compute_metrics (có cờ clip), per_star_metrics (có cờ clip), bootstrap_ci, paired_bootstrap_diff, mean_predictor_metrics, threshold_moc1, build_model_summary, write_model_summary. `baseline.py`: build_baseline, fit_baseline (lưới 84 tổ hợp TF-IDF + Ridge, chọn theo Val RMSE, n_jobs=-1), predict_baseline, save_baseline/load_baseline (joblib, atomic write). Lưu artifact `baseline_val.pkl` và `baseline_test.pkl`.

### Smoke Test
- `pytest tests/test_metrics.py tests/test_baseline.py -m "not slow"`: **PASS 30/30** (~8s, CPU synthetic)
- `pytest tests/test_metrics.py tests/test_baseline.py tests/test_data_setup.py tests/test_utils.py -m "not slow"`: **PASS 49/49** (~9s, CPU)
- Các test cần `transformers` (engine, model_builder, predict, train) bị skip do chưa cài thư viện (môn C/D), không liên quan đến mảng B.

### Run Model
- Chưa chạy trên `review.json` thật (chờ dữ liệu trên Drive, Colab). Test chạy trên dữ liệu tổng hợp synthetic.

### Check Description
- [x] `clip_scores`: đúng giá trị, giữ kiểu (list/ndarray/tensor), không sửa tại chỗ, `low > high` -> ValueError (D8)
- [x] `compute_metrics`: MAE/RMSE/R²/Pearson/Spearman đúng trên ví dụ tính tay; R² có thể âm; nhãn hằng số -> NaN cho tương quan (D1–D5, D5b)
- [x] `per_star_metrics`: MAE/RMSE theo từng mức sao + macro_mae; thiếu mức sao không lỗi; nhãn ngoài 1-5 -> ValueError (D6, D7, D7b)
- [x] Cờ `clip`: clip=True tính trên điểm đã clip, clip=False = rmse_raw; clip không làm tăng RMSE (D9, D10)
- [x] `bootstrap_ci` / `paired_bootstrap_diff`: tái lập theo seed, khoảng chứa điểm ước lượng, tham số sai -> ValueError (D11, D12)
- [x] `build_model_summary`: đủ cột theo thứ tự schema, công thức mức giảm so với baseline, thiếu cột pred / baseline_name không trong models -> ValueError (D13, D14)
- [x] `write_model_summary`: không ghi đè trừ khi `overwrite=True`, ghi nguyên atomic, NaN -> "" (D15)
- [x] `import src.metrics` không kéo theo `torch` (D16)
- [x] `baseline.py`: shape (N,), clip [1,5], chữ ký không có Test, chọn theo Val RMSE, tái lập, học được tín hiệu (RMSE < dự đoán trung bình) (D17–D20)
- [x] Trả về đủ khoá pipeline/best_params/grid_results/fit_seconds/grid_seconds; grid_results 84 dòng, đủ cột (D21)
- [x] Đầu vào sai -> ValueError; save/load joblib -> dự đoán giống hệt, FileExistsError khi không overwrite, FileNotFoundError khi thiếu file (D22, D23)
- [x] `mean_predictor_metrics` + `threshold_moc1` (0.85 × RMSE_baseline_val) đúng công thức

### Ghi chú quyết định
- `compute_metrics` và `per_star_metrics` bổ sung tham số `clip: bool = True` theo spec mục 2.3.
- `build_model_summary` raises `ValueError` nếu thiếu cột `pred_<model>` (theo spec mục 2.9 và test D14).
- `fit_baseline` dùng `joblib.Parallel(n_jobs=-1)` để chạy song song 84 tổ hợp trên CPU; pipeline winning được refit trên Train, `fit_seconds` đo thời gian fit tổ hợp thắng, `grid_seconds` tổng thời gian lưới.
- `save_baseline`/`load_baseline` ghi/nạp bằng `joblib` với atomic write (`os.replace`), kiểm tra đủ khoá bắt buộc.
- `metrics.py` chỉ import `numpy`, `pandas`, `scipy`, không import `torch` (thỏa mãn FR-7, D16).

---

## Phase 6 - 2026-11-13 - E (Dashboard/Error Analysis)

### Smoke Test
- `python -m pytest tests/test_dashboard.py tests/test_error_analysis.py -q`: **39 passed**, 8.53s, Python 3.12, Streamlit 1.67.0; môi trường **không cài torch**.
- AppTest kiểm tra bản có/không text, đổi model, lọc sao 1/5 và rỗng, sắp xếp, dropdown xem chi tiết; test render riêng kiểm tra click chọn dòng.
- Test chuẩn bị/tổng hợp Error Analysis bằng dữ liệu giả: 40 ID duy nhất, 4 random/sao, seed cố định, worksheet Leader không lộ nhãn E, chặn nhãn thiếu/đổi dữ liệu gốc, chặn agreement 2/4, chấp nhận 3/4.
- Một warning từ plugin anyio của môi trường về API pytest bị deprecated, không phải lỗi ứng dụng.
- Không chạy lại suite training A–D (môi trường không cài torch/transformers).

### Run Model
- Không chạy production training hoặc inference; dashboard chỉ đọc CSV.
- Chưa có kết quả training thật trong ZIP. Chưa gán nhãn 40 review thật, chưa Leader cross-check, chưa PR/merge.

### Check Description
- [x] Đọc CSV ưu tiên with_text, bảng summary giữ mọi model có mặt, blank khi thiếu metrics.
- [x] Scatter + identity line, histogram, per-star MAE/RMSE/n + Macro-MAE qua src.metrics.
- [x] Filter/sort, Top-10, detail text; không import torch.
- [x] Hàm/công cụ sampling + annotation validation + summary, tài liệu hướng dẫn Windows.
- [x] Demo riêng có banner SYNTHETIC, không tạo báo cáo thật giả mạo.
- [ ] Nhận CSV thật đầy đủ sau Leader/F validation.
- [ ] E gán nhãn 40 review, Leader kiểm tra 4 review ≥75%, tạo báo cáo cuối.
- [ ] Review/merge PR, ảnh demo với dữ liệu thật.

### Quyết định và điểm tích hợp
- Gọi `per_star_metrics(y_true, y_pred)` theo code hiện có (tự clip), không truyền keyword `clip` mà spec minh họa.
- Chuyển torch import trong tests/conftest.py sang lazy import ở helper để test E chạy độc lập; Dataset giả dùng __len__/__getitem__ protocol.
- Bổ sung scripts/error_analysis.py thay vì sửa module metrics của B. Các CLI `src.metrics --sample-errors` trong design chưa được hiện thực; guide đưa lệnh thực chạy được.
- Loại top-20 khỏi random pool để có đúng 40 review duy nhất; sort ID trước random, seed=42, thiếu 4/sao thì báo lỗi.
- merge_test_predictions hiện thiếu n_tokens/split_id: dashboard chấp nhận kèm cảnh báo; Error Analysis cần đầy đủ, chờ D/F cung cấp đúng schema.

---

## Phase 6 - 2026-11-17 - E (Xử lý feedback Dashboard/Error Analysis)

### Prompt
> Đối chiếu feedback E với assignments, spec/test plan và code Dashboard/Error Analysis; cập nhật tài liệu đã hiện thực, kiểm tra blocker n_tokens/split_id thuộc F, chỉ sửa phạm vi E, không tạo Markdown mới ngoài docs/analysts, không bịa training/annotation/cross-check và không commit/push/PR. Chạy `python -m pytest tests/test_dashboard.py tests/test_error_analysis.py -q`.

### Thay đổi thực hiện
- Sửa tại chỗ `docs/define&designt/05_dashboard.md`: Đã hiện thực; đường dẫn, API/helper, CSV validation, UI, histogram/Vega-Lite, metrics theo bộ lọc và quy trình Error Analysis đúng code hiện tại.
- `docs/tests/05_dashboard_tests.md` đã tồn tại, trái nhận định trong feedback cũ. Đồng bộ fixture 200 review/5 model trong tmp_path, tên test thực, giá trị trả về, AppTest/assertion/version, coverage và lệnh chạy; không tạo file test plan mới.
- Mở rộng test metadata Dashboard thành 3 trường hợp thiếu riêng/tất cả: loader/UI vẫn hiển thị và cảnh báo. Thêm 3 case Error Analysis `prepare` phải dừng trước khi tạo worksheet khi thiếu metadata.
- Sửa test workflow đọc summary bằng `read_text(encoding='utf-8')` để tránh lỗi cp1252 trên Windows; script đã ghi UTF-8 đúng nên không đổi.
- Sửa duy nhất comment lỗi thời trong `app/streamlit_dashboard.py::compute_per_star`, giữ lời gọi clipping mặc định và API. Không cần đổi `scripts/error_analysis.py` vì code hiện có đã chặn metadata thiếu.
- Cập nhật `docs/analysts/status_thanh_vien_e.md` theo bằng chứng hiện tại, bỏ kết luận stale về test plan và không xác nhận lại lịch sử PR/merge chưa kiểm chứng.

### Smoke Test
- Python mặc định thiếu pytest/dependency. Tạo `.venv` được gitignore; cài dependency từ `app/requirements.txt`, pytest và torch CPU do `tests/conftest.py` hiện import torch khi collection. Không sửa fixture chung. Một lần collection trước lúc cài torch hoàn tất báo thiếu `torch.utils`; chạy lại sau khi cài đủ.
- Lần chạy suite trước sửa encoding: **1 failed, 43 passed in 49.88s**; lỗi `UnicodeDecodeError` ở test đọc summary UTF-8 bằng encoding mặc định Windows.
- Sau sửa encoding, chạy lại từ repo root, đặt `.venv/Scripts` lên đầu PATH: `python -m pytest tests/test_dashboard.py tests/test_error_analysis.py -q` → **44 passed in 17.50s**, exit code 0 (26 Dashboard + 18 Error Analysis).
- Môi trường kiểm chứng: Python 3.12.10, pytest 9.2.0, Streamlit 1.67.0, torch 2.15.0+cpu. Test subprocess xác nhận import Dashboard không nạp torch; không tuyên bố môi trường không cài torch.
- Kiểm tra bằng AST: tên test ghi trong test plan khớp hai file test; link tương đối trong spec/test plan tồn tại.
- Không chạy suite module khác; mọi dữ liệu/annotation/cross-check trong test là synthetic ở thư mục tạm.

### Run Model
- Không chạy training hoặc inference thật. `reports/` hiện chỉ có `.gitkeep`; chưa có CSV production để phân tích.
- Chưa gán nhãn người 40 review, chưa Leader cross-check và chưa tạo báo cáo Error Analysis thật.

### Check Description / Blocker
- [x] Đối chiếu code trước khi sửa; spec/test plan đồng bộ hành vi thực tế; không thêm Markdown hoặc khôi phục guide đã bị main xóa.
- [x] Giữ API và phạm vi E; không sửa `src/predict.py` hoặc module thành viên khác; không commit/push/tạo PR.
- [x] Đọc `src/predict.py`: export CSV từng model chưa có n_tokens/split_id; merge chỉ thêm ID/nhãn/dự đoán/sai số/text nên hai CSV cuối cũng thiếu metadata. `sample_reviews` của E bắt buộc cả hai, không được thay bằng dữ liệu giả.
- [ ] F bổ sung n_tokens BERT trước cắt (quy ước ngưỡng >254, không tính hai token đặc biệt), truyền/validate split_id đúng lần chia và xuất đủ hai bản CSV.
- [ ] Leader training/xuất dữ liệu thật → C validate schema/ID/nhãn/split → E prepare và gán nhãn 40 review → Leader gán độc lập 4 review → E summarize sau agreement ≥75%.
