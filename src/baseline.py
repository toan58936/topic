"""Baseline TF-IDF + Ridge cho Text Regression (Yelp).

Spec: docs/define&designt/02_evaluation.md (mục 2.8)
Chỉ học trên Train, chọn siêu tham số bằng RMSE Val trên lưới cố định 84 tổ hợp,
không nhận tập Test. Lưu/nạp pipeline bằng joblib.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.metrics import clip_scores, rmse


# ---------- Hằng số lưới mặc định (84 tổ hợp) ----------

MAX_FEATURES_GRID = [5000, 10000, 20000]
NGRAM_RANGE_GRID = [(1, 1), (1, 2)]
MIN_DF_GRID = [2, 5]
ALPHA_GRID = list(np.logspace(-3, 3, 7))  # [0.001, 0.01, 0.1, 1, 10, 100, 1000]

MAX_DF = 0.95
SUBLINEAR_TF = True
STOP_WORDS = None
SOLVER = "auto"
LOWERCASE = True

REQUIRED_RESULT_KEYS = ("pipeline", "best_params", "grid_results", "fit_seconds", "grid_seconds")


def build_baseline(
    max_features: int = 50000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
    stop_words: Optional[str] = None,
    alpha: float = 1.0,
    solver: str = "auto",
    seed: int = 42,
) -> Pipeline:
    """Dựng sklearn Pipeline: TfidfVectorizer nối với Ridge.

    Trả về Pipeline chưa fit. Các tham số cụ thể từng tổ hợp sẽ đượcOverride
    khi chạy fit_baseline với param_grid.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        stop_words=stop_words,
        lowercase=LOWERCASE,
    )
    ridge = Ridge(alpha=alpha, random_state=seed, solver=solver)
    return Pipeline([("tfidf", vectorizer), ("ridge", ridge)])


def _default_param_grid() -> dict:
    """Trả về lưới mặc định 84 tổ hợp (max_features × ngram_range × min_df × alpha)."""
    return {
        "max_features": list(MAX_FEATURES_GRID),
        "ngram_range": list(NGRAM_RANGE_GRID),
        "min_df": list(MIN_DF_GRID),
        "alpha": list(ALPHA_GRID),
    }


def _expand_param_grid(param_grid: dict) -> list:
    """Mở rộng lưới thành danh sách các dict tham số (mỗi dict là một tổ hợp).

    Các khoá không phải max_features/ngram_range/min_df/alpha (ví dụ max_df,
    sublinear_tf, stop_words, solver) được giữ nguyên cố định theo spec.
    """
    max_features_list = param_grid.get("max_features", MAX_FEATURES_GRID)
    ngram_range_list = param_grid.get("ngram_range", NGRAM_RANGE_GRID)
    min_df_list = param_grid.get("min_df", MIN_DF_GRID)
    alpha_list = param_grid.get("alpha", ALPHA_GRID)

    combos = []
    for mf in max_features_list:
        for nr in ngram_range_list:
            for md in min_df_list:
                for al in alpha_list:
                    combos.append(
                        {
                            "max_features": mf,
                            "ngram_range": nr,
                            "min_df": md,
                            "alpha": al,
                        }
                    )
    return combos


def _evaluate_combo(combo: dict, X_train, y_train, X_val, y_val) -> dict:
    """Huấn luyện một tổ hợp trên Train, dự đoán Val, trả về val_rmse và thời gian fit."""
    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(
        max_features=combo["max_features"],
        ngram_range=tuple(combo["ngram_range"]),
        min_df=combo["min_df"],
        max_df=MAX_DF,
        sublinear_tf=SUBLINEAR_TF,
        stop_words=STOP_WORDS,
        lowercase=LOWERCASE,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    ridge = Ridge(alpha=combo["alpha"], random_state=42, solver=SOLVER)
    ridge.fit(X_train_vec, y_train)

    y_val_pred = ridge.predict(X_val_vec)
    y_val_pred = clip_scores(y_val_pred)

    val_rmse = rmse(y_val, y_val_pred, clip=True)
    fit_seconds = time.perf_counter() - t0

    return {
        "max_features": combo["max_features"],
        "ngram_range": combo["ngram_range"],
        "min_df": combo["min_df"],
        "alpha": combo["alpha"],
        "val_rmse": float(val_rmse),
        "fit_seconds": float(fit_seconds),
    }


def fit_baseline(train_df: pd.DataFrame, val_df: pd.DataFrame,
                 param_grid: Optional[dict] = None, seed: int = 42) -> dict:
    """Huấn luyện baseline TF-IDF + Ridge, chọn siêu tham số bằng RMSE Val.

    Chỉ huấn luyện trên Train, chọn siêu tham số trên Val (qua metrics.rmse),
    không nhận tập Test. Lưới mặc định: 84 tổ hợp.

    Trả về dict: {"pipeline", "best_params", "grid_results" (DataFrame),
    "fit_seconds", "grid_seconds"}.
    """
    missing = [c for c in ("text", "stars") if c not in train_df.columns]
    if missing:
        raise ValueError(f"train_df thiếu cột: {', '.join(missing)}")
    missing = [c for c in ("text", "stars") if c not in val_df.columns]
    if missing:
        raise ValueError(f"val_df thiếu cột: {', '.join(missing)}")

    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["stars"].to_numpy(dtype=np.float64)
    X_val = val_df["text"].astype(str).tolist()
    y_val = val_df["stars"].to_numpy(dtype=np.float64)

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("train_df và val_df không được rỗng")

    grid = _default_param_grid() if param_grid is None else dict(param_grid)
    combos = _expand_param_grid(grid)
    if len(combos) == 0:
        raise ValueError("param_grid rỗng, không có tổ hợp nào để tìm kiếm")

    # Chạy song song n_jobs=-1 trên CPU.
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_evaluate_combo)(combo, X_train, y_train, X_val, y_val)
        for combo in combos
    )

    grid_results = pd.DataFrame(results, columns=[
        "max_features", "ngram_range", "min_df", "alpha", "val_rmse", "fit_seconds",
    ])

    # Chọn tổ hợp có RMSE Val thấp nhất.
    best_idx = int(grid_results["val_rmse"].idxmin())
    best_row = grid_results.iloc[best_idx]
    best_params = {
        "max_features": best_row["max_features"],
        "ngram_range": best_row["ngram_range"],
        "min_df": best_row["min_df"],
        "alpha": best_row["alpha"],
    }

    grid_seconds = float(grid_results["fit_seconds"].sum())

    # Refit pipeline trên Train với tổ hợp thắng (để có đối tượng pipeline đã fit).
    pipeline = build_baseline(
        max_features=best_params["max_features"],
        ngram_range=best_params["ngram_range"],
        min_df=best_params["min_df"],
        alpha=best_params["alpha"],
        seed=seed,
    )
    t0 = time.perf_counter()
    X_train_vec = pipeline.named_steps["tfidf"].fit_transform(X_train)
    pipeline.named_steps["ridge"].fit(X_train_vec, y_train)
    fit_seconds = time.perf_counter() - t0

    return {
        "pipeline": pipeline,
        "best_params": best_params,
        "grid_results": grid_results,
        "fit_seconds": fit_seconds,
        "grid_seconds": grid_seconds,
    }


def save_baseline(result: dict, path, overwrite: bool = False) -> None:
    """Lưu kết quả fit_baseline bằng joblib (dict: pipeline, best_params, ...).

    Ghi nguyên tử (tệp tạm rồi os.replace). Raises FileExistsError nếu path
    đã tồn tại và overwrite=False.
    """
    path = os.path.abspath(path)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"File đã tồn tại: {path}. Truyền overwrite=True để ghi đè."
        )

    missing = [k for k in REQUIRED_RESULT_KEYS if k not in result]
    if missing:
        raise ValueError(f"result thiếu khoá bắt buộc: {', '.join(missing)}")

    tmp_path = path + ".tmp"
    joblib.dump(result, tmp_path)
    os.replace(tmp_path, path)


def load_baseline(path: str) -> dict:
    """Nạp kết quả fit_baseline từ file joblib, kiểm tra đủ khoá bắt buộc."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    result = joblib.load(path)
    missing = [k for k in REQUIRED_RESULT_KEYS if k not in result]
    if missing:
        raise ValueError(f"File joblib thiếu khoá bắt buộc: {', '.join(missing)}")
    return result


def predict_baseline(pipeline: Pipeline, texts) -> np.ndarray:
    """Dự đoán điểm cho danh sách văn bản, trả về mảng đã clip về [1.0, 5.0]."""
    if texts is None or len(texts) == 0:
        raise ValueError("texts không được rỗng")
    if not all(isinstance(t, str) for t in texts):
        raise ValueError("mọi phần tử trong texts phải là chuỗi")

    y_pred = pipeline.predict(list(texts))
    return clip_scores(y_pred)
