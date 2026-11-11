# Đặc tả Thiết kế Module Đánh giá — `metrics.py` và `baseline.py`

> **Tài liệu:** `docs/02_evaluation.md` (Define & Design cho chỉ số đánh giá và baseline)
> **Module liên quan:** `src/metrics.py`, `src/baseline.py`
> **Kế hoạch kiểm thử:** `docs/02_evaluation_tests.md` (tách riêng)
> **Phụ thuộc:** `docs/define.md`, `docs/design.md`; `docs/01_data.md` (chỉ về định dạng ba file `parquet`, không import `data_setup.py`)
> **Dùng bởi:** `docs/03_model.md` (không), `docs/04_engine_and_utils.md` (`clip_scores`, `rmse`, `build_model_summary`), dashboard (`docs/05_dashboard.md` — đề xuất, chưa viết)
> **Trạng thái:** Hiện thực một phần (2026-11-11) — `src/metrics.py` đã có `clip_scores`, `rmse`, `mae`, `compute_metrics`, `per_star_metrics` (đủ cho engine/predict). Còn thiếu: `bootstrap_ci`, `paired_bootstrap_diff`, `mean_predictor`, `build_model_summary`, `write_model_summary` và toàn bộ `baseline.py` (member B).

---

## PHẦN I: DEFINE — Xác định Bài toán & Yêu cầu

### 1.1. Bối cảnh & Mục tiêu

Phần này trả lời hai câu hỏi: **đo mô hình bằng thước nào để con số đáng tin?** và **mô hình mốc để so sánh là gì?** `metrics.py` chứa phép giới hạn điểm, mọi công thức chỉ số và việc lập `model_summary.csv`. `baseline.py` dựng và huấn luyện mô hình mốc TF-IDF + Ridge. Hai module được đặc tả cùng nhau vì baseline bắt buộc gọi `metrics.py` để chọn siêu tham số và cần có kết quả ở Mốc 1, trước khi có bất kỳ mô hình Transformer nào.

Mục tiêu:

* Một nguồn duy nhất cho phép clip và mọi công thức chỉ số, dùng chung cho baseline và ba mô hình.
* Chỉ số theo từng mức sao và Macro-MAE để lộ hiện tượng dồn dự đoán về giá trị trung bình.
* Bảng `model_summary.csv` một dòng mỗi mô hình (cả baseline) cho dashboard và báo cáo.
* Baseline công bằng: huấn luyện chỉ trên Train, chọn siêu tham số bằng Val, không chạm Test.

**Bộ chỉ số theo `docs/define.md`:**

| Nhóm | Chỉ số |
| :--- | :--- |
| Toàn cục | MAE, RMSE, R², tương quan Pearson, tương quan Spearman; `rmse_raw` (RMSE trên điểm chưa clip, để tham khảo) |
| Theo từng mức sao | MAE và RMSE cho sao 1 đến 5; Macro-MAE (trung bình MAE của các mức sao) |
| Chi phí | Thời gian huấn luyện, số tham số (và thời gian suy luận, đề xuất) |
| So với baseline | Mức giảm RMSE tương đối (đề xuất) |

**Bốn bài toán cốt lõi cần giải quyết:**

1. **Một nguồn duy nhất cho mọi con số.** Nếu baseline và ba mô hình mỗi nơi tính một kiểu (có clip, không clip, công thức khác nhau), phép so sánh không đáng tin. Mọi số báo cáo đi qua `metrics.py`.
2. **Tập Test giữ phân phối tự nhiên nên RMSE toàn cục bị mức sao phổ biến chi phối.** Một mô hình dồn dự đoán về khoảng 4 sao vẫn có RMSE toàn cục dễ nhìn. Cần chỉ số theo từng mức sao và Macro-MAE.
3. **Baseline phải công bằng thì mới là mốc có giá trị.** Nếu baseline dùng dữ liệu khác, chọn siêu tham số bằng Test hoặc không clip điểm, tuyên bố "cải thiện rõ rệt" mất ý nghĩa.
4. **Kết luận từ khác biệt nhỏ.** Số đo trên khoảng 9.000 mẫu Test có sai số lấy mẫu; khoảng tin cậy giúp biết khác biệt giữa hai mô hình có đáng tin không.

### 1.2. Đầu vào / Đầu ra

| | Mô tả |
| :--- | :--- |
| **Đầu vào của `metrics.py`** | Mảng nhãn thật và điểm dự đoán; bảng dự đoán Test (`true_stars`, `pred_<mô hình>`); số tham số, thời gian huấn luyện và `rmse_raw` do nơi gọi truyền vào |
| **Đầu ra của `metrics.py`** | Từ điển chỉ số; `DataFrame` và file `model_summary.csv` |
| **Đầu vào của `baseline.py`** | `DataFrame` Train và Val đọc từ `parquet` (cột `text`, `stars`), lưới siêu tham số, `seed` |
| **Đầu ra của `baseline.py`** | Từ điển gồm pipeline đã huấn luyện, bảng kết quả lưới trên Val (`grid_results` có cột `max_features`, `ngram_range`, `min_df`, `alpha`, `val_rmse`), siêu tham số tốt nhất (`best_params`), thời gian fit tổ hợp thắng (`fit_seconds`), tổng thời gian lưới (`grid_seconds`); dự đoán là mảng số thực đã clip; file `baseline.joblib` do `save_baseline` ghi |

### 1.3. Yêu cầu chức năng

| Mã | Yêu cầu |
| :--- | :--- |
| FR-1 | `clip_scores` giới hạn điểm về 1.0-5.0, nhận mảng `numpy`, danh sách hoặc tensor và trả về cùng kiểu; là nguồn duy nhất của phép clip trong toàn dự án. |
| FR-2 | `compute_metrics` trả về MAE, RMSE, R², Pearson, Spearman tính trên điểm đã clip, kèm `rmse_raw` tính trên điểm chưa clip. |
| FR-3 | `rmse` và `mae` là hàm đơn giản dùng chung (cho `engine.py` chọn checkpoint, cho `baseline.py` chọn siêu tham số). |
| FR-4 | `per_star_metrics` trả về MAE và RMSE theo từng mức sao có mặt và `macro_mae`. |
| FR-5 | `bootstrap_ci` và `paired_bootstrap_diff` trả về khoảng tin cậy, tái lập theo `seed`; tham số `metric` thuộc `{"rmse", "mae"}`. |
| FR-6 | `build_model_summary` lập bảng một dòng mỗi mô hình; `write_model_summary` ghi `model_summary.csv`, không ghi đè trừ khi `overwrite=True`. |
| FR-7 | `metrics.py` chỉ dùng `numpy`, `pandas`, `scipy`; **không import `torch`**. |
| FR-8 | `fit_baseline` chỉ huấn luyện trên Train, chọn siêu tham số trên Val bằng RMSE (qua `metrics.rmse`), **không nhận tập Test**. Lưới mặc định: `max_features` ∈ {5000, 10000, 20000}, `ngram_range` ∈ {(1,1), (1,2)}, `min_df` ∈ {2, 5}, `max_df` = 0.95, `sublinear_tf` = True, `stop_words` = None, `alpha` ∈ logspace(-3, 3, 7) = {0.001, 0.01, 0.1, 1, 10, 100, 1000}, `solver` = "auto"; tổng 84 tổ hợp. Val set cố định từ `data_setup.py` (không k-fold), `random_state` = 42. |
| FR-9 | `predict_baseline` trả về điểm đã clip bằng `clip_scores`. |
| FR-10 | `fit_baseline` đo thời gian huấn luyện lần fit cuối để đưa vào `model_summary.csv`. |
| FR-11 | Kiểm tra đầu vào hợp lệ và báo `ValueError` rõ ràng. |
| FR-12 | `save_baseline` và `load_baseline` lưu và nạp kết quả `fit_baseline` (pipeline, siêu tham số, thời gian), để tạo dự đoán Test của baseline ở Tuần 2 mà không phải huấn luyện lại. |

### 1.4. Ràng buộc

* Điểm dự đoán luôn được clip về 1.0-5.0 trước khi tính chỉ số, áp dụng đồng nhất cho baseline và ba mô hình; `rmse_raw` được tính riêng từ điểm chưa clip để tham khảo.
* `metrics.py` phải nhập được trên máy không có PyTorch (dashboard chạy trên máy cá nhân, `app/requirements.txt` không có `torch`). Vì vậy `src/__init__.py` phải để trống hoặc rất nhẹ, không import các module khác.
* `metrics.py` không biết gì về mô hình: số tham số và thời gian do nơi gọi truyền vào.
* Baseline dùng `scikit-learn` (`TfidfVectorizer`, `Ridge`); các thư viện `scikit-learn`, `scipy`, `pandas`, `pyarrow` cần ghi trong `requirements.txt`.
* Baseline đọc cùng ba file `parquet` với mô hình chính (xem `docs/01_data.md`).
* Tập Test chỉ được dùng để báo cáo: baseline và ba mô hình được đo trên cùng tập Test, cùng thứ tự `review_id`, **một lần** ở cấu hình cuối (đề xuất, xem mục 2.10).
* Bootstrap mặc định `n_boot = 1000`, `seed = 42` (đề xuất).

### 1.5. Tiêu chí chấp nhận

1. `compute_metrics` cho kết quả đúng trên các ví dụ tính tay (dự đoán hoàn hảo cho MAE và RMSE bằng 0, R² bằng 1).
2. `per_star_metrics` và Macro-MAE đúng trên ví dụ tính tay; mức sao vắng mặt không làm hàm lỗi.
3. `clip_scores` đúng giá trị và giữ nguyên kiểu; clip không làm tăng RMSE.
4. `model_summary.csv` có đủ cột, đúng công thức (đặc biệt mức giảm so với baseline), không ghi đè âm thầm.
5. `import src.metrics` không kéo theo `torch`.
6. Baseline: lưới chọn theo RMSE Val thấp nhất, chữ ký không có Test, dự đoán nằm trong 1.0-5.0, tái lập theo `seed`, và học được tín hiệu (tốt hơn việc luôn đoán trung bình trên dữ liệu có quy luật).
7. Bộ test trong `docs/02_evaluation_tests.md` đều đạt.

---

## PHẦN II: DESIGN — Thiết kế Chi tiết

### 2.1. Vị trí trong kiến trúc hệ thống

```text
train.parquet, val.parquet ──► baseline.py ──► fit_baseline ──► pipeline ──► predict_baseline (đã clip)
                                    │ chọn siêu tham số bằng metrics.rmse trên Val
                                    ▼
                               metrics.py  ◄──────────────── engine.py (chọn checkpoint: metrics.rmse)
                                    │  ◄──────────────────── predict.py (clip_scores)
                                    ▼
          compute_metrics, per_star_metrics, bootstrap_*  ──► build_model_summary ──► model_summary.csv
                                    │
                                    └──► dashboard (đọc model_summary.csv; tính theo bộ lọc từ test_predictions.csv)
```

### 2.2. Kiến trúc pipeline đánh giá

```text
[1. Baseline: fit_baseline(train, val) → pipeline + siêu tham số tốt nhất + thời gian fit]
[2. Ba mô hình: huấn luyện xong (04_engine_and_utils)]
                 │   (từ đây không đổi cấu hình dựa trên kết quả Test)
                 ▼
[3. Dự đoán Test một lần cho baseline và ba mô hình; ghi test_predictions.csv (04_engine_and_utils)]
                 ▼
[4. build_model_summary(preds_df, ...): chỉ số toàn cục + theo mức sao + khoảng tin cậy
                         + chi phí + mức giảm so với baseline]
                 ▼
[5. write_model_summary → results/model_summary.csv  (sau đó commit vào reports/)]
```

### 2.3. Giao diện hàm

**`metrics.py`:**

```python
clip_scores(x, low: float = 1.0, high: float = 5.0)       # trả về cùng kiểu với x

rmse(y_true, y_pred, clip: bool = True) -> float
mae(y_true, y_pred, clip: bool = True) -> float

compute_metrics(y_true, y_pred, clip: bool = True) -> dict
# {"mae", "rmse", "r2", "pearson", "spearman", "rmse_raw", "n"}

per_star_metrics(y_true, y_pred, clip: bool = True) -> dict
# {"per_star": {1: {"mae", "rmse", "n"}, ...}, "macro_mae": float}

bootstrap_ci(y_true, y_pred, metric: str = "rmse", n_boot: int = 1000,
             alpha: float = 0.05, seed: int = 42) -> tuple
# (điểm ước lượng, cận dưới, cận trên)

paired_bootstrap_diff(y_true, pred_a, pred_b, metric: str = "rmse", n_boot: int = 1000,
                      alpha: float = 0.05, seed: int = 42) -> tuple
# (hiệu a trừ b, cận dưới, cận trên)

build_model_summary(preds_df, models: list, rmse_raw_by_model: dict, param_counts: dict,
                    train_seconds: dict, inference_seconds: dict = None,
                    baseline_name: str = "baseline", n_boot: int = 1000, seed: int = 42,
                    seconds_per_epoch: dict = None, split_id: str = None) -> DataFrame

write_model_summary(summary_df, path, overwrite: bool = False) -> None
```

**`baseline.py`:**

```python
build_baseline(max_features: int = 50000, ngram_range: tuple = (1, 2),
               min_df: int = 2, max_df: float = 0.95, sublinear_tf: bool = True,
               stop_words: Optional[str] = None, alpha: float = 1.0, 
               solver: str = "auto", seed: int = 42)         # sklearn Pipeline

fit_baseline(train_df, val_df, param_grid: dict = None, seed: int = 42) -> dict
# {"pipeline", "best_params", "grid_results" (DataFrame), "fit_seconds", "grid_seconds"}

save_baseline(result: dict, path: str, overwrite: bool = False) -> None   # joblib
load_baseline(path: str) -> dict                                            # cùng khoá với fit_baseline

predict_baseline(pipeline, texts) -> numpy.ndarray          # shape (N,), đã clip
```

### 2.4. Công thức các chỉ số

Với $y_i$ là số sao thật và $\hat{y}_i$ là điểm dự đoán **đã giới hạn** về $[1, 5]$, $N$ mẫu:

$$\text{MAE} = \frac{1}{N}\sum_{i} |\hat{y}_i - y_i| \qquad \text{RMSE} = \sqrt{\frac{1}{N}\sum_{i} (\hat{y}_i - y_i)^2}$$

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

* **R²** có thể âm: khi mô hình tệ hơn việc luôn đoán giá trị trung bình $\bar{y}$.
* **Pearson** đo quan hệ tuyến tính giữa $\hat{y}$ và $y$; **Spearman** là Pearson tính trên thứ hạng, đo quan hệ đơn điệu. Cả hai không phạt việc lệch một lượng cố định, nên cần đọc cùng MAE và RMSE.
* Nếu dự đoán là hằng số (độ lệch chuẩn bằng 0), tương quan không xác định: trả về `NaN` thay vì báo lỗi.
* Nếu **nhãn thật** là hằng số (mọi $y_i$ bằng nhau, ví dụ khi dashboard lọc ra một mức sao), $\sum_i (y_i - \bar{y})^2 = 0$: R² và cả hai tương quan không xác định, trả về `NaN` thay vì báo lỗi hay chia cho 0; MAE và RMSE vẫn tính bình thường.
* `rmse_raw` dùng $\hat{y}$ chưa giới hạn, chỉ để tham khảo ảnh hưởng của việc clip.

**Phép clip — `clip_scores`:** giới hạn mỗi giá trị về $[\text{low}, \text{high}]$ và trả về cùng kiểu với đầu vào. Hàm dùng phép chặn có sẵn của kiểu dữ liệu đó (ví dụ `numpy.clip`; nếu đầu vào không phải `numpy.ndarray`, danh sách hay `tuple` mà có thuộc tính `clamp` thì coi là tensor và gọi `.clamp(low, high)`) nên **không cần import `torch`**. Vì nhãn thật luôn nằm trong $[1, 5]$, clip không bao giờ làm tăng sai số của một mẫu.

### 2.5. Chỉ số theo từng mức sao và Macro-MAE

Với mỗi mức sao $c \in \{1,\dots,5\}$ có mặt trong dữ liệu, tính MAE và RMSE trên các mẫu có $y_i = c$. Macro-MAE là trung bình cộng MAE của các mức sao có mặt:

$$\text{Macro-MAE} = \frac{1}{|C|}\sum_{c \in C} \text{MAE}_c$$

**Vì sao cần Macro-MAE bên cạnh MAE toàn cục:** MAE toàn cục là trung bình theo số mẫu nên bị sao 4-5 và sao 1 chi phối. Macro-MAE cho mỗi mức sao trọng số như nhau; nếu mô hình dồn dự đoán về giữa thì MAE ở sao 1 và sao 5 lớn, Macro-MAE xấu hơn MAE toàn cục.

**Ví dụ.** $y = [1,1,3,5,5]$, $\hat{y} = [2,1,3,4,5]$: MAE theo sao là $0.5$ (sao 1), $0$ (sao 3), $0.5$ (sao 5); Macro-MAE $= 1/3 \approx 0.333$; MAE toàn cục $= 0.4$.

Khi dữ liệu thiếu một mức sao (ví dụ dashboard lọc theo bộ lọc), hàm chỉ trả về các mức sao có mặt và Macro-MAE tính trên các mức đó. Khoá của `per_star` là `int` 1-5. Nhãn thật phải là số nguyên trong $\{1,\dots,5\}$ (chấp nhận `3.0`, không chấp nhận `0`, `6`, `2.5`); ngược lại `ValueError`, để nhãn hỏng không bị bỏ qua âm thầm khỏi Macro-MAE.

### 2.6. Khoảng tin cậy bằng bootstrap

**Mục tiêu:** ước lượng độ bất định của một chỉ số do việc **lấy mẫu tập Test**.

**Thuật toán — `bootstrap_ci`:**

1. Dùng `numpy.random.default_rng(seed)`.
2. Lặp `n_boot` lần: rút $N$ chỉ số **có hoàn lại**, tính chỉ số trên mẫu rút.
3. Lấy phân vị $\alpha/2$ và $1 - \alpha/2$ của các giá trị thu được làm cận dưới và cận trên.

**`paired_bootstrap_diff`:** rút **cùng một tập chỉ số** cho cả hai mô hình trong mỗi lần lặp (ghép cặp) rồi lấy hiệu chỉ số. Nếu khoảng tin cậy của hiệu không chứa 0 thì khác biệt giữa hai mô hình không chỉ do may rủi lấy mẫu.

**Hạn chế phải nêu trong báo cáo:** khoảng tin cậy này chỉ phản ánh sai số lấy mẫu của tập Test. Mỗi mô hình chỉ được huấn luyện **một lần** (`seed = 42`), nên nó không phản ánh dao động do khởi tạo và thứ tự dữ liệu khi huấn luyện lại.

### 2.7. Bảng tóm tắt — `build_model_summary` và `model_summary.csv`

**Đầu vào của `build_model_summary`:**

| Tham số | Ý nghĩa |
| :--- | :--- |
| `preds_df` | Bảng dự đoán Test, có cột `true_stars` và `pred_<mô hình>` (đã clip) cho mỗi mô hình, kể cả `pred_baseline` |
| `models` | Danh sách tên theo thứ tự dòng, ví dụ `["baseline", "bert", "roberta", "distilbert"]` |
| `rmse_raw_by_model` | `rmse_raw` của từng mô hình, tính lúc đánh giá từ điểm chưa clip (điểm thô không lưu trong CSV dự đoán) |
| `param_counts` | Số tham số từng mô hình (baseline = số hệ số Ridge) |
| `train_seconds` | Thời gian huấn luyện từng mô hình (tổng `epoch_seconds`, đã gồm kiểm định Val mỗi epoch; với baseline là `fit_seconds` của tổ hợp thắng, không gồm cả lưới). Hai số này **không cùng cơ sở**; báo cáo phải nêu rõ, và `grid_seconds` của baseline là số tham khảo cho chi phí chọn siêu tham số |
| `inference_seconds` | Tuỳ chọn: thời gian suy luận trên cả tập Test của từng mô hình |
| `seconds_per_epoch` | Tuỳ chọn: trung bình thời gian mỗi epoch của từng mô hình Transformer |
| `split_id` | Tuỳ chọn: ID phép chia dữ liệu (từ `split_stats.json`) để ghi vào cột `split_id` |

**Các cột của `model_summary.csv`** (mỗi dòng một mô hình, baseline ở dòng đầu). **Thứ tự cột** đúng như sau: `model`, `mae`, `rmse`, `rmse_raw`, `r2`, `pearson`, `spearman`, `macro_mae`, `train_seconds`, `seconds_per_epoch`, `params_total`, `rmse_reduction_vs_baseline_pct`, `inference_seconds`, `split_id`.

| Cột | Ý nghĩa |
| :--- | :--- |
| `model` | `baseline`, `bert`, `roberta`, `distilbert` |
| `mae`, `rmse`, `rmse_raw`, `r2`, `pearson`, `spearman`, `macro_mae` | Chỉ số trên Test |
| `train_seconds`, `seconds_per_epoch` | Thời gian huấn luyện tổng và trung bình mỗi epoch (baseline: `fit_seconds` và rỗng) |
| `params_total` | Số tham số mô hình (baseline = số hệ số Ridge) |
| `rmse_reduction_vs_baseline_pct` | $100 \times (\text{RMSE}_{\text{baseline}} - \text{RMSE}_{\text{mô hình}}) / \text{RMSE}_{\text{baseline}}$; dòng baseline và mean_predictor để trống |
| `inference_seconds` | Thời gian suy luận trên cả tập Test (gồm nạp dữ liệu và token hoá) |
| `split_id` | ID phép chia dữ liệu (từ `split_stats.json`) để kiểm tra trộn kết quả |

Các cột ghi "đề xuất" là phần mở rộng so với danh sách trong `docs/design.md` mục 4; Design cần ghi nhận nếu nhóm giữ.

### 2.8. Baseline — `baseline.py`

**Mục tiêu:** dựng mốc so sánh TF-IDF + Ridge, tuân thủ FR-8 đến FR-10.

**Thuật toán:**

1. **Dựng pipeline:** `TfidfVectorizer(max_features, ngram_range, min_df, max_df, sublinear_tf, lowercase=True, stop_words=None)` nối với `Ridge(alpha, random_state=seed, solver="auto")`.
2. **Lưới siêu tham số (mặc định):** `max_features` ∈ {5000, 10000, 20000}, `ngram_range` ∈ {(1,1), (1,2)}, `min_df` ∈ {2, 5}, `max_df` = 0.95, `sublinear_tf` = True, `stop_words` = None, `alpha` ∈ logspace(-3, 3, 7) = {0.001, 0.01, 0.1, 1, 10, 100, 1000}, `solver` = "auto"; tổng **84 tổ hợp**.
3. **Chọn trên Val:** với từng tổ hợp, huấn luyện **chỉ trên Train**, dự đoán Val, clip, rồi tính `metrics.rmse`. Chọn tổ hợp có RMSE Val thấp nhất.
4. **Kết quả:** pipeline đã huấn luyện trên Train với tổ hợp tốt nhất (không huấn luyện lại trên Train gộp Val, để giữ nguyên điều kiện so sánh với mô hình chính), bảng `grid_results` (mỗi dòng một tổ hợp, cột `max_features`, `ngram_range`, `min_df`, `alpha`, `val_rmse`), siêu tham số tốt nhất `best_params` (cùng ba khoá: `max_features`, `ngram_range`, `min_df`, `alpha`), `fit_seconds` (thời gian fit của chính tổ hợp thắng) và `grid_seconds` (tổng thời gian fit mọi tổ hợp, gồm dự đoán Val). `fit_seconds` là số đưa vào `train_seconds` của baseline; `grid_seconds` chỉ để báo cáo chi phí thật của việc chọn siêu tham số.
5. **Dự đoán:** `predict_baseline` trả về mảng số thực đã clip.

**Vì sao không dùng Test để chọn siêu tham số:** nếu chọn theo điểm trên Test thì thông tin của Test lọt vào việc chọn mô hình (rò rỉ), điểm của baseline thành lạc quan giả, và phép so sánh với mô hình chính (chọn checkpoint bằng Val) không còn cùng điều kiện. Chữ ký `fit_baseline` không có tham số Test, nên điều này được đảm bảo ngay trong chữ ký.

**Lưu và nạp — `save_baseline`, `load_baseline`:** `save_baseline` ghi bằng `joblib.dump` một `dict` gồm `pipeline`, `best_params`, `grid_results` (DataFrame), `fit_seconds`, `grid_seconds`; ghi nguyên tử (tệp tạm rồi đổi tên). `load_baseline` trả về cùng `dict`, nên `predict_baseline(load_baseline(p)["pipeline"], texts)` cho dự đoán giống hệt trước khi lưu. Tệp `joblib` là định dạng `pickle`: chỉ nạp tệp do chính nhóm tạo, không nạp tệp nhận từ nguồn lạ. Tệp nằm trên Drive, không đưa lên GitHub (thêm `*.joblib` vào `.gitignore`).

**Tính tái lập:** cùng dữ liệu và cùng `seed` cho cùng kết quả.

### 2.9. Ngoại lệ

| Hàm | Ngoại lệ |
| :--- | :--- |
| `clip_scores` | `ValueError` nếu `low > high` |
| `compute_metrics`, `per_star_metrics`, `rmse`, `mae`, `bootstrap_*` | `ValueError` nếu độ dài `y_true` và `y_pred` khác nhau, mảng rỗng, hoặc có `NaN` trong đầu vào |
| `per_star_metrics` | Thêm: `ValueError` nếu nhãn thật không phải số nguyên trong 1-5 |
| `bootstrap_ci`, `paired_bootstrap_diff` | `ValueError` nếu `n_boot <= 0`, `alpha` ngoài $(0, 1)$, hoặc tên chỉ số không hỗ trợ |
| `build_model_summary` | `ValueError` nếu thiếu cột `true_stars` hoặc `pred_<mô hình>`; hoặc `baseline_name` không nằm trong `models` |
| `write_model_summary` | `FileExistsError` nếu file đã có và `overwrite=False` |
| `fit_baseline` | `ValueError` nếu thiếu cột `text` hoặc `stars`, hoặc lưới rỗng |
| `save_baseline`, `load_baseline` | `FileExistsError` nếu file đã có và `overwrite=False`; `FileNotFoundError` khi nạp file không có; `ValueError` nếu file nạp lên thiếu khoá của `fit_baseline` |
| `predict_baseline` | `ValueError` nếu danh sách rỗng hoặc phần tử không phải chuỗi |

### 2.10. Tích hợp với các module khác

| Module | Cách phối hợp |
| :--- | :--- |
| `data_setup.py` | `baseline.py` không import `data_setup.py`; nơi gọi đọc `train.parquet`, `val.parquet` bằng `pandas` và truyền `DataFrame` vào. `review_id` và `true_stars` của `test_predictions.csv` lấy từ `test_loader`. |
| `engine.py` | Dùng `metrics.rmse` (và `clip_scores`) để chọn checkpoint tốt nhất bằng Val, thay vì tự tính công thức. |
| `predict.py` | Dùng `clip_scores` để giới hạn điểm; xuất `test_predictions.csv` có `pred_baseline` cùng các cột `pred_*`. |
| `model_builder.py` | `count_parameters` cung cấp `param_counts` do nơi gọi truyền vào `build_model_summary`; `metrics.py` không biết mô hình. |
| Dashboard | Đọc `model_summary.csv` và `test_predictions.csv`; gọi `per_star_metrics` và `compute_metrics` theo bộ lọc. Vì `metrics.py` không phụ thuộc `torch` nên chạy được trên máy không cài PyTorch. |

### 2.11. Quyết định thiết kế và điểm mở

| Hạng mục | Quyết định |
| :--- | :--- |
| Bộ chỉ số, clip trước khi tính, báo thêm theo từng mức sao và Macro-MAE | Đã chốt trong `docs/define.md` |
| `clip_scores` đặt trong `metrics.py` | Đề xuất, để mọi module phụ thuộc theo chiều số thứ tự tăng dần (file 02 không import file 04) |
| Clip cả dự đoán của baseline | Đã chốt: clip điểm trước khi tính chỉ số và xuất dự đoán |
| `metrics.py` không phụ thuộc `torch`; `__init__.py` để trống | Đã chốt, để dashboard nhẹ |
| `rmse_raw` do nơi gọi truyền vào, không lưu điểm thô trong CSV | Theo `docs/design.md` mục 4 |
| Đánh giá Test một lần ở cấu hình cuối; `write_model_summary` không ghi đè mặc định | Đề xuất (quy tắc làm việc của nhóm) |
| Cột `model_summary.csv`: `seconds_per_epoch`, `inference_seconds`, `split_id` (không có `device`, `rmse_ci_*`, `inference_seconds_per_1000`) | Theo `docs/design.md` §9.9.3 |
| Bootstrap `n_boot = 1000`, `seed = 42` | Đề xuất |
| Baseline dùng `scikit-learn`; lưới **84 tổ hợp** (`max_features`×`ngram_range`×`min_df`×`alpha` + fixed `max_df`, `sublinear_tf`, `stop_words`, `solver`) | Theo `docs/design.md` §9.6 |
| `params_total` baseline = số hệ số Ridge | Theo `docs/design.md` §9.9.3 |
| Không huấn luyện lại baseline trên Train gộp Val | Đã chốt, để giữ so sánh công bằng với mô hình chính |
| Mỗi mô hình huấn luyện một lần (một seed) | Hạn chế đã biết; nêu trong báo cáo |
