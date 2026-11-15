"""Metrics cho Text Regression (Yelp).

Spec: docs/define&designt/02_evaluation.md
Module này là nguồn duy nhất của phép clip (clip_scores) và mọi công thức chỉ số:
MAE, RMSE, R², Pearson, Spearman toàn cục; MAE/RMSE theo từng mức sao kèm số mẫu n
và Macro-MAE; rmse_raw; khoảng tin cậy bootstrap; bảng model_summary.csv.
Không import torch, để dashboard chạy được trên máy không cài PyTorch.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def clip_scores(x, low: float = 1.0, high: float = 5.0):
    """Giới hạn điểm về [low, high], trả về cùng kiểu với đầu vào.

    Không cần torch: dùng np.clip cho ndarray, .clamp cho tensor,
    min/max cho list/tuple/scalar. Không sửa tại chỗ đầu vào.
    """
    if low > high:
        raise ValueError(f"low ({low}) không được lớn hơn high ({high})")
    if isinstance(x, np.ndarray):
        return np.clip(x, low, high)
    if hasattr(x, "clamp"):
        return x.clamp(low, high)
    if isinstance(x, (list, tuple)):
        return type(x)(min(max(float(v), low), high) for v in x)
    return min(max(float(x), low), high)


def _as_float_arrays(y_true, y_pred):
    """Chuyển hai mảng về float64 1-D, kiểm tra shape khớp và không chứa NaN."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true và y_pred phải cùng shape, nhận {y_true.shape} vs {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("y_true và y_pred không được rỗng")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValueError("y_true và y_pred không được chứa NaN")
    return y_true, y_pred


def rmse(y_true, y_pred, clip: bool = True) -> float:
    """RMSE giữa nhãn thật và dự đoán; clip=True thì clip dự đoán về [1,5] trước."""
    y_true, y_pred = _as_float_arrays(y_true, y_pred)
    if clip:
        y_pred = np.clip(y_pred, 1.0, 5.0)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred, clip: bool = True) -> float:
    """MAE giữa nhãn thật và dự đoán; clip=True thì clip dự đoán về [1,5] trước."""
    y_true, y_pred = _as_float_arrays(y_true, y_pred)
    if clip:
        y_pred = np.clip(y_pred, 1.0, 5.0)
    return float(np.mean(np.abs(y_pred - y_true)))


def compute_metrics(y_true, y_pred, clip: bool = True) -> dict:
    """Bộ chỉ số toàn cục trên điểm đã clip, kèm rmse_raw trên điểm chưa clip.

    Trả về dict có khoá: mae, rmse, r2, pearson, spearman, rmse_raw, n.
    Khi clip=False, rmse bằng rmse_raw (tính trên điểm chưa clip).
    """
    y_true, y_pred = _as_float_arrays(y_true, y_pred)
    n = len(y_true)

    if clip:
        y_pred_eval = np.clip(y_pred, 1.0, 5.0)
    else:
        y_pred_eval = y_pred

    err = y_pred_eval - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    y_true_const = np.all(y_true == y_true[0])
    y_pred_const = np.all(y_pred_eval == y_pred_eval[0])
    if y_true_const or y_pred_const:
        pearson = float("nan")
        spearman = float("nan")
    else:
        pearson = float(scipy_stats.pearsonr(y_true, y_pred_eval)[0])
        spearman = float(scipy_stats.spearmanr(y_true, y_pred_eval)[0])

    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "rmse_raw": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "n": n,
    }


def per_star_metrics(y_true, y_pred, clip: bool = True) -> dict:
    """MAE/RMSE theo từng mức sao có mẫu + macro_mae.

    y_true phải là số nguyên trong {1..5} (chấp nhận 3.0, không chấp nhận 2.5).
    Khoá của per_star là int 1-5. Mức sao vắng mặt không trả về.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true và y_pred phải cùng shape")
    if y_true.size == 0:
        raise ValueError("y_true và y_pred không được rỗng")

    stars = y_true.astype(np.float64)
    if not np.all(np.equal(stars, np.floor(stars))) or not np.all(
        (stars >= 1) & (stars <= 5)
    ):
        raise ValueError("y_true phải là số nguyên trong {1,...,5}")

    per_star = {}
    maes = []
    for c in range(1, 6):
        mask = stars == c
        n = int(mask.sum())
        if n == 0:
            continue
        y_p = y_pred[mask]
        if clip:
            y_p = np.clip(y_p, 1.0, 5.0)
        err = y_p - stars[mask]
        mae_c = float(np.mean(np.abs(err)))
        rmse_c = float(np.sqrt(np.mean(err ** 2)))
        per_star[int(c)] = {"mae": mae_c, "rmse": rmse_c, "n": n}
        maes.append(mae_c)

    return {
        "per_star": per_star,
        "macro_mae": float(np.mean(maes)) if maes else float("nan"),
    }


def bootstrap_ci(
    y_true,
    y_pred,
    metric: str = "rmse",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple:
    """Khoảng tin cậy bootstrap cho một chỉ số (rmse hoặc mae).

    Thuật toán: rút N chỉ số có hoàn lại n_boot lần, tính chỉ số trên mẫu rút,
    lấy phân vị alpha/2 và 1-alpha/2 làm cận dưới/cận trên.

    Trả về (điểm ước lượng trên toàn bộ dữ liệu, cận dưới, cận trên).
    """
    if metric not in ("rmse", "mae"):
        raise ValueError(f"metric không hỗ trợ: {metric}, chỉ chấp nhận 'rmse' hoặc 'mae'")
    if n_boot <= 0:
        raise ValueError(f"n_boot phải > 0, nhận {n_boot}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha phải nằm trong (0, 1), nhận {alpha}")

    y_true, y_pred = _as_float_arrays(y_true, y_pred)
    n = len(y_true)

    if metric == "rmse":
        def stat(yt, yp):
            return float(np.sqrt(np.mean((yp - yt) ** 2)))
    else:
        def stat(yt, yp):
            return float(np.mean(np.abs(yp - yt)))

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_stats[i] = stat(y_true[idx], y_pred[idx])

    point = stat(y_true, y_pred)
    lower = float(np.quantile(boot_stats, alpha / 2.0))
    upper = float(np.quantile(boot_stats, 1.0 - alpha / 2.0))
    return (point, lower, upper)


def paired_bootstrap_diff(
    y_true,
    pred_a,
    pred_b,
    metric: str = "rmse",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple:
    """Khoảng tin cậy bootstrap cho hiệu chênh lệch hai mô hình (ghép cặp).

    Rút cùng một tập chỉ số cho cả hai mô hình trong mỗi lần lặp, rồi lấy hiệu
    chỉ số (a trừ b). Nếu khoảng không chứa 0 thì chênh lệch không chỉ do may
    rủi lấy mẫu.

    Trả về (hiệu quan sát a-b, cận dưới, cận trên).
    """
    if metric not in ("rmse", "mae"):
        raise ValueError(f"metric không hỗ trợ: {metric}, chỉ chấp nhận 'rmse' hoặc 'mae'")
    if n_boot <= 0:
        raise ValueError(f"n_boot phải > 0, nhận {n_boot}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha phải nằm trong (0, 1), nhận {alpha}")

    y_true, pred_a = _as_float_arrays(y_true, pred_a)
    _, pred_b = _as_float_arrays(y_true, pred_b)
    n = len(y_true)

    if metric == "rmse":
        def stat(yt, yp):
            return float(np.sqrt(np.mean((yp - yt) ** 2)))
    else:
        def stat(yt, yp):
            return float(np.mean(np.abs(yp - yt)))

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = stat(y_true[idx], pred_a[idx]) - stat(y_true[idx], pred_b[idx])

    diff_obs = stat(y_true, pred_a) - stat(y_true, pred_b)
    lower = float(np.quantile(boot_diffs, alpha / 2.0))
    upper = float(np.quantile(boot_diffs, 1.0 - alpha / 2.0))
    return (diff_obs, lower, upper)


def mean_predictor_metrics(y_true, train_mean: float) -> dict:
    """Chỉ số của mốc dự đoán trung bình Train trên tập đánh giá.

    Dự đoán mọi mẫu bằng train_mean (đã clip về [1,5]), rồi tính compute_metrics.
    Trả về dict cùng cấu trúc với compute_metrics, thêm khoá "predictor" = "mean".
    """
    y_true_arr, _ = _as_float_arrays(y_true, y_true)
    n = len(y_true_arr)
    y_pred = np.full(n, float(train_mean), dtype=np.float64)
    metrics = compute_metrics(y_true_arr, y_pred, clip=True)
    metrics["predictor"] = "mean"
    return metrics


def threshold_moc1(rmse_baseline_val: float) -> float:
    """Ngưỡng Mốc 1: Threshold = 0.85 × RMSE_baseline_val.

    Mô hình chính đạt Mốc 1 nếu RMSE Val <= ngưỡng này (giảm ít nhất 15%
    so với baseline).
    """
    if rmse_baseline_val is None or not np.isfinite(rmse_baseline_val):
        raise ValueError(f"rmse_baseline_val phải là số hữu hạn, nhận {rmse_baseline_val}")
    return 0.85 * float(rmse_baseline_val)


def build_model_summary(
    preds_df: pd.DataFrame,
    models: list,
    rmse_raw_by_model: dict,
    param_counts: dict,
    train_seconds: dict,
    inference_seconds: Optional[dict] = None,
    baseline_name: str = "baseline",
    n_boot: int = 1000,
    seed: int = 42,
    seconds_per_epoch: Optional[dict] = None,
    split_id: Optional[str] = None,
) -> pd.DataFrame:
    """Lập bảng tóm tắt một dòng mỗi mô hình theo schema model_summary.csv.

    Cột theo thứ tự: model, mae, rmse, rmse_raw, r2, pearson, spearman,
    macro_mae, train_seconds, seconds_per_epoch, params_total,
    rmse_reduction_vs_baseline_pct, inference_seconds, split_id.

    baseline_name phải nằm trong models. Nếu baseline_name là "mean_predictor"
    hoặc không thể tính (thiếu cột), mức giảm để trống (NaN).
    """
    if baseline_name not in models:
        raise ValueError(
            f"baseline_name '{baseline_name}' không nằm trong models: {models}"
        )
    if "true_stars" not in preds_df.columns:
        raise ValueError("preds_df thiếu cột bắt buộc: true_stars")

    y_true = preds_df["true_stars"].to_numpy(dtype=np.float64)
    n = len(y_true)

    rows = []
    baseline_rmse = None
    for model in models:
        col = f"pred_{model}"
        if col not in preds_df.columns:
            raise ValueError(
                f"preds_df thiếu cột bắt buộc: {col} cho mô hình '{model}'"
            )

        y_pred = preds_df[col].to_numpy(dtype=np.float64)
        m = compute_metrics(y_true, y_pred, clip=True)
        ps = per_star_metrics(y_true, y_pred, clip=True)

        row = {
            "model": model,
            "mae": m["mae"],
            "rmse": m["rmse"],
            "rmse_raw": rmse_raw_by_model.get(model, np.nan),
            "r2": m["r2"],
            "pearson": m["pearson"],
            "spearman": m["spearman"],
            "macro_mae": ps["macro_mae"],
            "train_seconds": train_seconds.get(model, np.nan),
            "seconds_per_epoch": (
                seconds_per_epoch.get(model, np.nan) if seconds_per_epoch else np.nan
            ),
            "params_total": param_counts.get(model, np.nan),
            "rmse_reduction_vs_baseline_pct": np.nan,
            "inference_seconds": (
                inference_seconds.get(model, np.nan) if inference_seconds else np.nan
            ),
            "split_id": split_id,
        }
        rows.append(row)

        if model == baseline_name:
            baseline_rmse = m["rmse"]

    df = pd.DataFrame(
        rows,
        columns=[
            "model",
            "mae",
            "rmse",
            "rmse_raw",
            "r2",
            "pearson",
            "spearman",
            "macro_mae",
            "train_seconds",
            "seconds_per_epoch",
            "params_total",
            "rmse_reduction_vs_baseline_pct",
            "inference_seconds",
            "split_id",
        ],
    )

    # Tính mức giảm so với baseline (bỏ qua dòng baseline và mean_predictor).
    if baseline_rmse is not None and baseline_rmse > 0:
        for i, model in enumerate(models):
            if model == baseline_name:
                continue
            if model == "mean_predictor":
                continue
            model_rmse = df.loc[i, "rmse"]
            if not np.isnan(model_rmse):
                df.loc[i, "rmse_reduction_vs_baseline_pct"] = float(
                    100.0 * (baseline_rmse - model_rmse) / baseline_rmse
                )

    return df


def write_model_summary(summary_df: pd.DataFrame, path, overwrite: bool = False) -> None:
    """Ghi model_summary.csv, không ghi đè trừ khi overwrite=True.

    Định dạng: float %.6f, không index, encoding utf-8, NaN -> chuỗi rỗng.
    Ghi nguyên tử (tệp tạm rồi os.replace).
    """
    path = os.path.abspath(path)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"File đã tồn tại: {path}. Truyền overwrite=True để ghi đè."
        )

    tmp_path = path + ".tmp"
    summary_df.to_csv(
        tmp_path,
        index=False,
        encoding="utf-8",
        float_format="%.6f",
        na_rep="",
    )
    os.replace(tmp_path, path)
