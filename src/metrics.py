"""Metrics cho Text Regression (Yelp).

Spec: docs/define&designt/02_evaluation.md
Phần lõi (clip_scores, rmse, mae, compute_metrics, per_star_metrics) đã hiện thực
cho engine.py/predict.py; bootstrap_ci, paired_bootstrap_diff,
build_model_summary, write_model_summary, mean_predictor sẽ bổ sung (member B).
"""

import numpy as np


def clip_scores(x, low: float = 1.0, high: float = 5.0):
    """Giới hạn điểm về [low, high], trả về cùng kiểu với đầu vào.

    Không cần torch: dùng np.clip cho ndarray, .clamp cho tensor,
    min/max cho list/tuple/scalar.
    """
    if isinstance(x, np.ndarray):
        return np.clip(x, low, high)
    if hasattr(x, "clamp"):
        return x.clamp(low, high)
    if isinstance(x, (list, tuple)):
        return type(x)(min(max(float(v), low), high) for v in x)
    return min(max(float(x), low), high)


def _as_float_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true và y_pred phải cùng shape, nhận {y_true.shape} vs {y_pred.shape}"
        )
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


def compute_metrics(y_true, y_pred) -> dict:
    """Bộ chỉ số toàn cục trên điểm đã clip, kèm rmse_raw trên điểm chưa clip."""
    from scipy import stats as scipy_stats

    y_true, y_pred = _as_float_arrays(y_true, y_pred)
    y_pred_c = np.clip(y_pred, 1.0, 5.0)
    err = y_pred_c - y_true
    n = len(y_true)

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if np.all(y_true == y_true[0]) or np.all(y_pred_c == y_pred_c[0]):
        pearson = float("nan")
        spearman = float("nan")
    else:
        pearson = float(scipy_stats.pearsonr(y_true, y_pred_c)[0])
        spearman = float(scipy_stats.spearmanr(y_true, y_pred_c)[0])

    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "rmse_raw": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "n": n,
    }


def per_star_metrics(y_true, y_pred) -> dict:
    """MAE/RMSE theo từng mức sao có mẫu + macro_mae.

    y_true phải là số nguyên trong {1..5} (chấp nhận 3.0, không chấp nhận 2.5).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_true = y_true.ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true và y_pred phải cùng shape")

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
        err = np.clip(y_pred[mask], 1.0, 5.0) - stars[mask]
        mae_c = float(np.mean(np.abs(err)))
        rmse_c = float(np.sqrt(np.mean(err**2)))
        per_star[c] = {"mae": mae_c, "rmse": rmse_c, "n": n}
        maes.append(mae_c)
    return {"per_star": per_star, "macro_mae": float(np.mean(maes)) if maes else float("nan")}
