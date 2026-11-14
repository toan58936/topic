"""Kiểm tra src/metrics.py theo spec docs/define&designt/02_evaluation.md
và kế hoạch kiểm thử docs/tests/02_evaluation_tests.md (D1–D16).

Chạy: pytest tests/test_metrics.py -v -m "not slow"
"""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    bootstrap_ci,
    build_model_summary,
    clip_scores,
    compute_metrics,
    mae,
    mean_predictor_metrics,
    paired_bootstrap_diff,
    per_star_metrics,
    rmse,
    threshold_moc1,
    write_model_summary,
)


# ---------------- Nhóm A: Công thức chỉ số toàn cục ----------------


def test_d1_perfect_prediction():
    y_true = y_pred = [1.0, 3.0, 5.0]
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["pearson"] == pytest.approx(1.0)
    assert m["spearman"] == pytest.approx(1.0)
    assert m["rmse_raw"] == pytest.approx(0.0)
    assert m["n"] == 3


def test_d2_known_values():
    y_true = [1, 3, 5]
    y_pred = [2, 3, 3]
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx((5 / 3) ** 0.5)
    assert m["r2"] == pytest.approx(0.375)
    assert rmse(y_true, y_pred) == pytest.approx(m["rmse"])
    assert mae(y_true, y_pred) == pytest.approx(m["mae"])


def test_d3_r2_can_be_negative():
    y_true = [1, 3, 5]
    y_pred = [5, 3, 1]
    m = compute_metrics(y_true, y_pred)
    assert m["r2"] < 0


def test_d4_correlation_cases():
    y_true = [1, 2, 3, 4, 5]
    y_pred_mono = [1.0, 1.2, 2.0, 3.5, 5.0]
    m = compute_metrics(y_true, y_pred_mono)
    assert m["spearman"] == pytest.approx(1.0)
    assert m["pearson"] < 1.0

    y_pred_const = [3, 3, 3, 3, 3]
    m = compute_metrics(y_true, y_pred_const)
    assert np.isnan(m["pearson"])
    assert np.isnan(m["spearman"])


def test_d5_invalid_inputs():
    with pytest.raises(ValueError):
        compute_metrics([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        compute_metrics([], [])
    with pytest.raises(ValueError):
        compute_metrics([1, 2, 3], [1, float("nan"), 3])
    with pytest.raises(ValueError):
        rmse([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        mae([1], [])


def test_d5b_constant_true_labels():
    y_true = [3, 3, 3, 3]
    y_pred = [2, 3, 4, 3]
    m = compute_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(0.5)
    assert m["rmse"] == pytest.approx(0.5 ** 0.5)
    assert np.isnan(m["r2"])
    assert np.isnan(m["pearson"])
    assert np.isnan(m["spearman"])


# ---------------- Nhóm B: Theo từng mức sao và Macro-MAE ----------------


def test_d6_per_star_known_values():
    y_true = [1, 1, 3, 5, 5]
    y_pred = [2, 1, 3, 4, 5]
    res = per_star_metrics(y_true, y_pred)
    ps = res["per_star"]
    assert ps[1]["mae"] == pytest.approx(0.5)
    assert ps[1]["rmse"] == pytest.approx(0.7071067811865476)
    assert ps[1]["n"] == 2
    assert ps[3]["mae"] == pytest.approx(0.0)
    assert ps[3]["rmse"] == pytest.approx(0.0)
    assert ps[3]["n"] == 1
    assert ps[5]["mae"] == pytest.approx(0.5)
    assert ps[5]["rmse"] == pytest.approx(0.7071067811865476)
    assert ps[5]["n"] == 2
    assert res["macro_mae"] == pytest.approx(1.0 / 3.0)
    # MAE toàn cục là 0.4, khác macro_mae
    assert compute_metrics(y_true, y_pred)["mae"] == pytest.approx(0.4)


def test_d7_missing_star_levels():
    y_true = [2, 2, 4, 4]
    y_pred = [2, 3, 4, 5]
    res = per_star_metrics(y_true, y_pred)
    assert set(res["per_star"].keys()) == {2, 4}
    assert res["macro_mae"] == pytest.approx((0.0 + 1.0) / 2.0)


def test_d7b_per_star_invalid_labels():
    for bad in ([0, 3, 5], [1, 6, 3], [1, 2.5, 3]):
        with pytest.raises(ValueError):
            per_star_metrics(bad, [1, 2, 3])
    # Số nguyên dạng thực (3.0) chấp nhận được, khoá kiểu int
    res = per_star_metrics([1.0, 3.0], [1.0, 3.0])
    assert set(res["per_star"].keys()) == {1, 3}
    assert all(isinstance(k, int) for k in res["per_star"])


# ---------------- Nhóm C: Phép clip ----------------


def test_d8_clip_scores_values_and_types():
    x = [0.2, 1.0, 3.3, 5.0, 7.5]
    assert clip_scores(x) == [1.0, 1.0, 3.3, 5.0, 5.0]
    assert clip_scores(np.array(x)).tolist() == [1.0, 1.0, 3.3, 5.0, 5.0]
    # Đầu vào gốc không bị sửa
    assert x == [0.2, 1.0, 3.3, 5.0, 7.5]
    # Tensor (nếu có torch)
    try:
        import torch  # noqa: F401
        t = torch.tensor(x)
        out = clip_scores(t)
        assert torch.is_tensor(out)
        assert out.tolist() == pytest.approx([1.0, 1.0, 3.3, 5.0, 5.0])
    except ImportError:
        pass
    with pytest.raises(ValueError):
        clip_scores([1.0, 2.0], low=5.0, high=1.0)


def test_d9_clip_flag_and_rmse_raw():
    y_true = [1, 3, 5]
    y_pred = [0.0, 3.0, 8.0]  # có giá trị ngoài [1,5]
    m_clip = compute_metrics(y_true, y_pred, clip=True)
    m_raw = compute_metrics(y_true, y_pred, clip=False)
    assert m_clip["rmse"] == pytest.approx(rmse(y_true, y_pred, clip=True))
    assert m_raw["rmse"] == pytest.approx(m_raw["rmse_raw"])
    assert m_clip["rmse"] != m_raw["rmse_raw"]


def test_d10_clip_never_increases_rmse():
    rng = np.random.default_rng(42)
    y_true = rng.integers(1, 6, size=200).astype(float)
    y_pred = rng.uniform(-2, 8, size=200)
    assert rmse(y_true, y_pred, clip=True) <= rmse(y_true, y_pred, clip=False) + 1e-12


# ---------------- Nhóm D: Bootstrap ----------------


def test_d11_bootstrap_ci():
    rng = np.random.default_rng(0)
    y_true = rng.integers(1, 6, size=200).astype(float)
    y_pred = y_true + rng.normal(0, 0.5, size=200)

    point, lo, hi = bootstrap_ci(y_true, y_pred, metric="rmse", n_boot=200, seed=42)
    assert lo <= point <= hi

    point2, lo2, hi2 = bootstrap_ci(y_true, y_pred, metric="rmse", n_boot=200, seed=42)
    assert (point, lo, hi) == (point2, lo2, hi2)

    point3, lo3, hi3 = bootstrap_ci(y_true, y_pred, metric="rmse", n_boot=200, seed=7)
    # Điểm ước lượng tính trên toàn bộ dữ liệu, không phụ thuộc seed
    assert point3 == point
    # Cận dưới/cận trên có thể khác nhau do mẫu bootstrap khác theo seed
    assert (lo3, hi3) != (lo, hi)

    with pytest.raises(ValueError):
        bootstrap_ci(y_true, y_pred, n_boot=0)
    with pytest.raises(ValueError):
        bootstrap_ci(y_true, y_pred, alpha=1.5)
    with pytest.raises(ValueError):
        bootstrap_ci(y_true, y_pred, metric="accuracy")


def test_d12_paired_bootstrap_diff():
    rng = np.random.default_rng(1)
    y_true = rng.integers(1, 6, size=150).astype(float)
    pred_a = y_true + rng.normal(0, 0.2, size=150)
    pred_b = y_true + rng.normal(0, 0.8, size=150)

    # (a) Hai dự đoán giống hệt -> hiệu 0, khoảng chứa 0
    diff, lo, hi = paired_bootstrap_diff(y_true, pred_a, pred_a, n_boot=200, seed=42)
    assert diff == pytest.approx(0.0)
    assert lo <= 0 <= hi

    # (b) A sai ít hơn B rõ rệt -> khoảng diff (A-B) hoàn toàn dưới 0
    diff, lo, hi = paired_bootstrap_diff(y_true, pred_a, pred_b, n_boot=200, seed=42)
    assert diff < 0
    assert hi < 0

    with pytest.raises(ValueError):
        paired_bootstrap_diff(y_true, pred_a, pred_b, n_boot=-5)
    with pytest.raises(ValueError):
        paired_bootstrap_diff(y_true, pred_a, pred_b, alpha=0.0)


# ---------------- Nhóm E: Bảng tóm tắt và tính nhẹ ----------------


def _make_preds_df():
    return pd.DataFrame(
        {
            "true_stars": [1, 2, 3, 4, 5, 1, 5],
            "pred_baseline": [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 5.0],
            "pred_bert": [1.2, 2.1, 2.9, 4.1, 4.8, 1.1, 4.9],
        }
    )


def test_d13_build_model_summary():
    preds_df = _make_preds_df()
    models = ["baseline", "bert"]
    rmse_raw_by_model = {"baseline": 0.1, "bert": 0.08}
    param_counts = {"baseline": 100, "bert": 100000}
    train_seconds = {"baseline": 10.0, "bert": 100.0}
    inference_seconds = {"baseline": 0.5, "bert": 2.0}

    df = build_model_summary(
        preds_df,
        models,
        rmse_raw_by_model,
        param_counts,
        train_seconds,
        inference_seconds=inference_seconds,
        n_boot=100,
        seed=42,
    )

    expected_cols = [
        "model", "mae", "rmse", "rmse_raw", "r2", "pearson", "spearman",
        "macro_mae", "train_seconds", "seconds_per_epoch", "params_total",
        "rmse_reduction_vs_baseline_pct", "inference_seconds", "split_id",
    ]
    assert list(df.columns) == expected_cols
    assert list(df["model"]) == models

    bert_row = df[df["model"] == "bert"].iloc[0]
    # RMSE baseline = 0 (dự đoán hoàn hảo), rmse_reduction để trống vì chia 0
    base_row = df[df["model"] == "baseline"].iloc[0]
    assert base_row["rmse"] == pytest.approx(0.0)
    assert np.isnan(base_row["rmse_reduction_vs_baseline_pct"])
    assert base_row["params_total"] == pytest.approx(100)
    assert base_row["train_seconds"] == pytest.approx(10.0)
    assert bert_row["params_total"] == pytest.approx(100000)
    assert bert_row["train_seconds"] == pytest.approx(100.0)
    assert bert_row["inference_seconds"] == pytest.approx(2.0)
    assert bert_row["split_id"] is None


def test_d13b_model_summary_with_reduction():
    # RMSE baseline = 1.0, RMSE bert = 0.8 -> giảm 20%
    preds_df = pd.DataFrame(
        {
            "true_stars": [1.0, 3.0, 5.0],
            "pred_baseline": [2.0, 3.0, 4.0],  # RMSE = sqrt(1+0+1)/sqrt(3)... ≈ 0.816
            "pred_bert": [1.0, 3.0, 5.0],  # RMSE = 0
        }
    )
    # Để đảm bảo RMSE baseline = 1.0, pred_baseline = [2,3,4] cho y=[1,3,5]:
    # err = [1,0,-1], rmse = sqrt(2/3) ≠ 1.0. Dùng pred_baseline = [1,3,5] -> rmse=0 không được.
    # Thay vào đó tạo y_true sao cho baseline rmse = 1.0: y=[1,3,5], pred=[2,3,3] -> rmse=sqrt(5/3)
    # Tính giảm theo công thức thực tế, chỉ kiểm tra cột có giá trị hợp lệ.
    models = ["baseline", "bert"]
    df = build_model_summary(
        preds_df,
        models,
        {"baseline": 0.0, "bert": 0.0},
        {"baseline": 10, "bert": 100},
        {"baseline": 5.0, "bert": 50.0},
        n_boot=50,
        seed=42,
    )
    bert_row = df[df["model"] == "bert"].iloc[0]
    base_row = df[df["model"] == "baseline"].iloc[0]
    # BERT dự đoán hoàn hảo -> rmse = 0, giảm tối đa
    assert bert_row["rmse"] == pytest.approx(0.0)
    assert base_row["rmse"] > 0
    assert bert_row["rmse_reduction_vs_baseline_pct"] == pytest.approx(100.0)
    assert np.isnan(base_row["rmse_reduction_vs_baseline_pct"])


def test_d14_summary_errors():
    preds_df = pd.DataFrame(
        {"true_stars": [1, 2, 3], "pred_baseline": [1, 2, 3]}
    )
    with pytest.raises(ValueError):
        build_model_summary(
            preds_df,
            ["baseline", "bert"],
            {},
            {},
            {},
            baseline_name="baseline",
        )
    with pytest.raises(ValueError):
        build_model_summary(
            pd.DataFrame({"true_stars": [1, 2, 3], "pred_baseline": [1, 2, 3]}),
            ["bert"],
            {},
            {},
            {},
            baseline_name="baseline",
        )


def test_d15_write_summary_guard(tmp_path):
    df = pd.DataFrame({"model": ["baseline"], "rmse": [1.0]})
    path = tmp_path / "model_summary.csv"
    write_model_summary(df, path)
    assert path.exists()

    with pytest.raises(FileExistsError):
        write_model_summary(df, path)

    write_model_summary(df, path, overwrite=True)
    assert path.exists()

    again = pd.read_csv(path)
    assert again["model"].iloc[0] == "baseline"


def test_d16_metrics_has_no_torch():
    code = (
        "import sys; "
        "import src.metrics; "
        "print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=None,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


# ---------------- Mean predictor & Mốc 1 threshold ----------------


def test_mean_predictor_metrics():
    y_true = [1, 3, 5]
    train_mean = 3.0
    m = mean_predictor_metrics(y_true, train_mean)
    assert m["predictor"] == "mean"
    # Dự đoán hằng 3.0 cho y=[1,3,5]: MAE = (2+0+2)/3 = 4/3
    assert m["mae"] == pytest.approx(4 / 3.0)
    assert m["n"] == 3


def test_threshold_moc1():
    assert threshold_moc1(1.0) == pytest.approx(0.85)
    assert threshold_moc1(2.0) == pytest.approx(1.7)
    with pytest.raises(ValueError):
        threshold_moc1(float("nan"))
    with pytest.raises(ValueError):
        threshold_moc1(None)
