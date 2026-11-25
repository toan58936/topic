# Kế hoạch kiểm thử Module Đánh giá — `tests/test_metrics.py`, `tests/test_baseline.py`

> **Tài liệu:** `docs/02_evaluation_tests.md`
> **Đặc tả được kiểm thử:** `docs/02_evaluation.md` (mục 1.5 Tiêu chí chấp nhận, mục 2.3 đến 2.10)
> **Trạng thái:** Bản nháp trước khi hiện thực

---

## 1. Mục tiêu và nguyên tắc

Bộ test kiểm chứng `metrics.py` và `baseline.py` **độc lập** trước khi lập bảng kết quả. Test không đánh giá chất lượng mô hình; test kiểm tra **công thức chỉ số, phép clip, bảng tóm tắt và các quy tắc của baseline**.

1. **Ưu tiên ví dụ tính tay:** mỗi test của `metrics.py` dùng mảng nhỏ mà kết quả tính được bằng tay (ghi sẵn trong test), để công thức sai là thấy ngay.
2. **Nhanh, chạy trên CPU, không cần mạng, không cần GPU.** Dữ liệu cho baseline là văn bản giả nhỏ trong `tests/conftest.py`.
3. **Kiểm tra hành vi quan sát được:** giá trị, cột, kiểu; không kiểm tra cách cài đặt.
4. **Mỗi test một ý chính.**
5. **Test lỗi dùng `pytest.raises`.**
6. **Cố định seed** trong mọi test có ngẫu nhiên.
7. **Test cần dữ liệu thật đánh dấu `slow`:** tự bỏ qua nếu không có file.

## 2. Bảy nhóm khía cạnh kiểm thử

| Nhóm | Khía cạnh | Câu hỏi cần trả lời | Test |
| :--- | :--- | :--- | :--- |
| A | Công thức chỉ số toàn cục | Đúng giá trị tính tay? Trường hợp biên xử lý đúng? | D1, D2, D3, D4, D5, D5b |
| B | Theo từng mức sao và Macro-MAE | Đúng giá trị tính tay? Thiếu mức sao có sao không? | D6, D7, D7b |
| C | Phép clip | `clip_scores` đúng giá trị, giữ kiểu? Clip có làm tăng RMSE không? | D8, D9, D10 |
| D | Bootstrap | Tái lập theo seed? Khoảng tin cậy hợp lý? Phép ghép cặp đúng? | D11, D12 |
| E | Bảng tóm tắt và tính nhẹ của `metrics.py` | Đúng cột, đúng công thức, không ghi đè? Không kéo theo `torch`? | D13, D14, D15, D16 |
| F | Baseline | Đúng shape, có clip, chỉ dùng Val, tái lập, học được tín hiệu? | D17, D18, D19, D20, D21, D22, D23 |
| S | Dữ liệu thật (chậm) | Baseline chạy được trên `parquet` thật? | S1 |

## 3. Mô tả chi tiết từng test

**Nhóm A: Công thức chỉ số toàn cục**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| D1 | `test_d1_perfect_prediction` | `y_true = y_pred = [1.0, 3.0, 5.0]`. | MAE $= 0$, RMSE $= 0$, R² $= 1$, Pearson $= 1$, Spearman $= 1$ | Công thức sai ở trường hợp đơn giản nhất. |
| D2 | `test_d2_known_values` | `y_true = [1, 3, 5]`, `y_pred = [2, 3, 3]`. | MAE $= 1.0$; RMSE $= \sqrt{5/3} \approx 1.2910$; R² $= 0.375$; hàm `rmse` và `mae` cho cùng giá trị với `compute_metrics` | Nhầm MAE với RMSE; thiếu căn bậc hai; sai mẫu số của R². |
| D3 | `test_d3_r2_can_be_negative` | `y_true = [1, 3, 5]`, `y_pred = [5, 3, 1]`. | R² nhỏ hơn $0$ | Cắt R² về khoảng $[0, 1]$. |
| D4 | `test_d4_correlation_cases` | `y_true = [1, 2, 3, 4, 5]`; dự đoán đơn điệu, phi tuyến và **nằm trong 1-5** `[1.0, 1.2, 2.0, 3.5, 5.0]` (không dùng bình phương: giá trị lớn bị clip thành 5 tạo hạng bằng nhau, Spearman chỉ còn khoảng $0.87$); dự đoán hằng số `[3, 3, 3, 3, 3]`. | Spearman $= 1$ và Pearson $< 1$ cho trường hợp đầu; hai tương quan là `NaN` (không lỗi) cho trường hợp hằng số | Nhầm Pearson với Spearman; chia cho $0$ khi dự đoán hằng số. |
| D5 | `test_d5_invalid_inputs` | Độ dài khác nhau; mảng rỗng; có `NaN` trong dự đoán. | Cả ba `ValueError` | Âm thầm tính trên dữ liệu hỏng. |
| D5b | `test_d5b_constant_true_labels` | `y_true = [3, 3, 3, 3]`, `y_pred = [2, 3, 4, 3]`. | `mae` $= 0.5$ và `rmse` $\approx 0.7071$ tính bình thường; `r2`, `pearson`, `spearman` là `NaN`; không lỗi, không cảnh báo chia cho 0 làm test thất bại | Chia cho 0 khi lọc dashboard ra một mức sao duy nhất. |

**Nhóm B: Theo từng mức sao và Macro-MAE**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| D6 | `test_d6_per_star_known_values` | `y_true = [1,1,3,5,5]`, `y_pred = [2,1,3,4,5]`. | MAE theo sao: $0.5$ (sao 1), $0$ (sao 3), $0.5$ (sao 5); RMSE theo sao $0.7071$, $0$, $0.7071$; `macro_mae` $\approx 0.3333$ trong khi MAE toàn cục là $0.4$ | Nhầm Macro-MAE với MAE toàn cục; nhóm sao sai. |
| D7 | `test_d7_missing_star_levels` | Dữ liệu chỉ có sao 2 và sao 4. | Chỉ trả về hai mức sao có mặt; `macro_mae` là trung bình hai mức; không lỗi | Chia cho $0$ hoặc báo lỗi khi dashboard lọc ra một nhóm con. |
| D7b | `test_d7b_per_star_invalid_labels` | `y_true` chứa `0`; chứa `6`; chứa `2.5`; thêm một ca `y_true = [1.0, 3.0]` (số nguyên dạng thực). | Ba ca đầu `ValueError`; ca cuối chạy bình thường với khoá `1` và `3` kiểu `int` | Nhãn hỏng bị bỏ qua âm thầm khỏi Macro-MAE; khoá kiểu `float`. |

**Nhóm C: Phép clip**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| D8 | `test_d8_clip_scores_values_and_types` | Mảng `[0.2, 1.0, 3.3, 5.0, 7.5]` dưới dạng danh sách, `numpy.ndarray` và `torch.Tensor` (tensor bỏ qua nếu không cài `torch`); kiểm tra đầu vào gốc sau khi gọi. | Giá trị `[1.0, 1.0, 3.3, 5.0, 5.0]`; mỗi loại trả về đúng loại của nó; đầu vào gốc không bị sửa; `low > high` thì `ValueError` | Chặn sai biên; đổi kiểu âm thầm; sửa tại chỗ làm hỏng dữ liệu gốc. |
| D9 | `test_d9_clip_flag_and_rmse_raw` | `compute_metrics` với `clip=True` và `clip=False` trên dự đoán có giá trị ngoài khoảng. | `clip=True` cho `rmse` bằng RMSE của điểm đã clip; `clip=False` cho `rmse` bằng `rmse_raw`; `rmse_raw` luôn tính trên điểm chưa clip | Cờ `clip` không có tác dụng; `rmse_raw` bị clip nhầm. |
| D10 | `test_d10_clip_never_increases_rmse` | Nhãn ngẫu nhiên thuộc $\{1,\dots,5\}$; dự đoán ngẫu nhiên trong khoảng $[-2, 8]$. | `rmse` (đã clip) nhỏ hơn hoặc bằng `rmse_raw` | Clip làm tăng sai số, hoặc không được áp dụng. |

**Nhóm D: Bootstrap**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| D11 | `test_d11_bootstrap_ci` | `bootstrap_ci` trên $200$ mẫu giả, `seed=42` hai lần, thêm `seed=7`. | Cận dưới $\le$ điểm ước lượng $\le$ cận trên; hai lần `seed=42` giống hệt; `seed=7` khác; `n_boot <= 0` và `alpha` ngoài $(0, 1)$ đều `ValueError` | Khoảng không chứa điểm ước lượng; không tái lập; tham số sai lọt qua. |
| D12 | `test_d12_paired_bootstrap_diff` | (a) Hai dự đoán giống hệt nhau; (b) mô hình A sai ít hơn B rõ rệt trên mọi mẫu. | (a) hiệu bằng $0$, khoảng chứa $0$; (b) khoảng tin cậy của hiệu RMSE (A trừ B) nằm hoàn toàn dưới $0$ | Rút chỉ số riêng cho hai mô hình thay vì ghép cặp; dấu của hiệu ngược. |

**Nhóm E: Bảng tóm tắt và tính nhẹ của `metrics.py`**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| D13 | `test_d13_build_model_summary` | `preds_df` giả gồm `true_stars`, `pred_baseline`, `pred_bert`; RMSE baseline $= 1.0$, RMSE của `bert` $= 0.8$; `param_counts`, `train_seconds`, `rmse_raw_by_model` và `inference_seconds` cho trước. | Dòng `bert` có `rmse_reduction_vs_baseline_pct` $= 20$; `params_total`, `train_seconds`, `rmse_raw` đúng giá trị truyền vào; `inference_seconds_per_1000` đúng theo $\text{seconds} / N \times 1000$; dòng baseline có `params_total` và mức giảm để trống; có đủ các cột ở mục 2.7 theo đúng thứ tự, kể cả `device` khi truyền `device_by_model` (để trống khi không truyền) | Công thức mức giảm sai (nhầm mẫu số); đơn vị thời gian sai; thiếu hoặc sai thứ tự cột. |
| D14 | `test_d14_summary_errors` | Thiếu cột `pred_bert` trong `preds_df`; `baseline_name` không nằm trong `models`. | `ValueError` cho cả hai | Lập bảng thiếu mô hình mà không báo. |
| D15 | `test_d15_write_summary_guard` | `write_model_summary` hai lần cùng đường dẫn; lần ba với `overwrite=True`. | Lần hai `FileExistsError`; lần ba ghi được; file đọc lại bằng `pandas` ra đúng bảng | Âm thầm ghi đè bảng kết quả đã công bố. |
| D16 | `test_d16_metrics_has_no_torch` | Chạy một tiến trình Python mới: `import src.metrics`, rồi kiểm tra `"torch"` không nằm trong `sys.modules`. | `torch` không được nạp | `import torch` trong `metrics.py`, hoặc `src/__init__.py` import module khác, làm dashboard đòi PyTorch. |

**Nhóm F: Baseline**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| D17 | `test_d17_baseline_shape_and_clip` | `fit_baseline` trên dữ liệu giả; `predict_baseline` trên văn bản cực đoan. | Đầu ra shape `(N,)`; mọi giá trị nằm trong 1.0-5.0 | Quên clip, đầu ra ngoài khoảng. |
| D18 | `test_d18_baseline_selection_uses_val_only` | Kiểm tra chữ ký `fit_baseline`; lưới hai tổ hợp `alpha` (một rất lớn làm dự đoán gần như hằng số, một nhỏ). | Chữ ký không có tham số Test; `best_params` là tổ hợp có RMSE Val thấp nhất; `grid_results` có RMSE Val của cả hai tổ hợp | Test lọt vào việc chọn siêu tham số. |
| D19 | `test_d19_baseline_reproducible` | Chạy `fit_baseline` hai lần với cùng dữ liệu và `seed`. | Dự đoán giống hệt nhau | Có yếu tố ngẫu nhiên không cố định. |
| D20 | `test_d20_baseline_learns_signal` | Dữ liệu giả: review chứa từ khen gắn $5$ sao, từ chê gắn $1$ sao. | RMSE Val thấp hơn RMSE của cách luôn đoán trung bình Train | Baseline không học được gì, làm mốc vô nghĩa. |
| D21 | `test_d21_baseline_returns_timing_and_keys` | Gọi `fit_baseline` trên dữ liệu giả. | Từ điển trả về có đủ `pipeline`, `best_params`, `grid_results`, `fit_seconds`, `grid_seconds`; `fit_seconds` lớn hơn $0$ và `grid_seconds` $\ge$ `fit_seconds`; `grid_results` có các cột `max_features`, `ngram_range`, `alpha`, `val_rmse` | Thiếu thời gian huấn luyện cho `model_summary.csv`. |
| D22 | `test_d22_baseline_invalid_inputs` | Thiếu cột `stars`; lưới rỗng; danh sách rỗng cho `predict_baseline`. | `ValueError` cho cả ba | Đầu vào sai lọt qua và lỗi muộn khó hiểu. |
| D23 | `test_d23_baseline_save_load` | `fit_baseline` trên dữ liệu giả; `save_baseline` vào `tmp_path`; `load_baseline`; `predict_baseline` trước và sau; lưu lần hai không `overwrite`; nạp file không có. | Dự đoán trước và sau giống hệt; `best_params`, `fit_seconds`, `grid_seconds` và `grid_results` bằng nhau; lần lưu thứ hai `FileExistsError`; nạp file không có `FileNotFoundError` | Phải huấn luyện lại baseline ở Tuần 2; mất `fit_seconds` cho `model_summary.csv`; ghi đè âm thầm. |

**Nhóm S: Dữ liệu thật (đánh dấu `slow`)**

| Mã | Hàm test (dự kiến) | Cách kiểm thử (ví dụ) | Kỳ vọng | Lỗi test này bắt được |
| :--- | :--- | :--- | :--- | :--- |
| S1 | `test_s1_baseline_on_real_parquet` | Đọc `train.parquet` và `val.parquet` thật; `fit_baseline` với lưới thu nhỏ (một hoặc hai tổ hợp). | Chạy hoàn tất; RMSE Val hữu hạn và thấp hơn độ lệch chuẩn của nhãn Val (tức là tốt hơn cách luôn đoán trung bình) | Lỗi chỉ lộ ra trên dữ liệu thật (kiểu cột, văn bản lạ, bộ nhớ). |

Nếu không có file `parquet` thật, test nhóm S **bỏ qua** thay vì báo lỗi.

## 4. Hai ví dụ đọc test chi tiết

**Ví dụ 1: D2 (giá trị tính tay)**

```python
def test_d2_known_values():
    y_true = [1, 3, 5]
    y_pred = [2, 3, 3]

    m = compute_metrics(y_true, y_pred)

    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx((5 / 3) ** 0.5)     # ≈ 1.2910
    assert m["r2"] == pytest.approx(0.375)                 # 1 - 5/8
    assert rmse(y_true, y_pred) == pytest.approx(m["rmse"])
```

Sai số là $[1, 0, -2]$ nên tổng bình phương $= 5$; tổng bình phương quanh trung bình $= 8$. Nếu thiếu căn bậc hai, `rmse` thành $1.667$ và test thất bại ngay. Dòng cuối bảo đảm hàm `rmse` mà `engine.py` và `baseline.py` dùng trùng với `compute_metrics`.

**Ví dụ 2: D18 (baseline chỉ dùng Val để chọn siêu tham số)**

```python
def test_d18_baseline_selection_uses_val_only(fake_train_df, fake_val_df):
    import inspect
    assert "test" not in "".join(inspect.signature(fit_baseline).parameters)

    grid = {"max_features": [5000], "ngram_range": [(1, 1)], "alpha": [0.1, 1e6]}
    out = fit_baseline(fake_train_df, fake_val_df, param_grid=grid, seed=42)

    results = out["grid_results"]                           # mỗi dòng một tổ hợp, có cột val_rmse
    best_row = results.loc[results["val_rmse"].idxmin()]
    assert out["best_params"]["alpha"] == best_row["alpha"]
    assert out["best_params"]["alpha"] == 0.1               # alpha rất lớn làm dự đoán gần hằng số
```

Hai khẳng định cùng chứng minh: Test không có đường vào hàm, và tổ hợp được chọn đúng là tổ hợp có RMSE Val thấp nhất.

## 5. Cách chạy, ánh xạ tiêu chí chấp nhận và giới hạn

Lệnh chạy:

```bash
pytest tests/test_metrics.py tests/test_baseline.py -v                 # tất cả
pytest tests/test_metrics.py tests/test_baseline.py -v -m "not slow"   # bỏ qua test cần dữ liệu thật
```

Marker `slow` được đăng ký trong `tests/conftest.py`.

Ánh xạ với tiêu chí chấp nhận ở mục 1.5 của `docs/02_evaluation.md`:

| Tiêu chí | Test chứng minh |
| :--- | :--- |
| 1. Chỉ số đúng trên ví dụ tính tay; nhãn hằng số cho `NaN` | D1, D2, D3, D4, D5, D5b |
| 2. Theo từng mức sao và Macro-MAE đúng; thiếu mức sao không lỗi; nhãn lạ bị từ chối | D6, D7, D7b |
| 3. `clip_scores` đúng, giữ kiểu; clip không tăng RMSE | D8, D9, D10 |
| 4. `model_summary.csv` đúng cột, đúng công thức, không ghi đè | D13, D14, D15 |
| 5. `metrics.py` không kéo theo `torch` | D16 |
| 6. Baseline: lưới theo Val, chữ ký không Test, clip, tái lập, học được, lưu và nạp lại | D17, D18, D19, D20, D21, D22, D23, S1 |
| 7. Toàn bộ test đạt | Mọi test ở trên |

Giới hạn của bộ test:

* Không kiểm tra chất lượng mô hình Transformer; phần đó thuộc `docs/04_engine_and_utils_tests.md` và các lần chạy thật.
* Khoảng tin cậy bootstrap chỉ được kiểm tra về tính hợp lệ và tái lập, không kiểm tra độ phủ thống kê.
* Việc `engine.py` thực sự gọi `metrics.rmse` được kiểm tra ở `docs/04_engine_and_utils_tests.md`, không ở đây.
* Nhóm S cần file `parquet` thật nên chỉ chạy được sau khi đã chạy `prepare_splits`.
