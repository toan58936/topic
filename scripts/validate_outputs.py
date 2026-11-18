#!/usr/bin/env python3
"""Validate integration outputs: merge consistency, schema, metric reconciliation, dashboard readiness.

Usage:
    python -m scripts.validate_outputs --results-dir <path> [--split-stats <path>] [--tolerance <float>]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import compute_metrics, per_star_metrics


REQUIRED_TEST_PREDICTIONS_COLS = [
    "review_id", "true_stars", "n_tokens", "split_id"
]

REQUIRED_MODEL_SUMMARY_COLS = [
    "model", "mae", "rmse", "rmse_raw", "r2", "pearson", "spearman",
    "macro_mae", "train_seconds", "seconds_per_epoch", "params_total",
    "rmse_reduction_vs_baseline_pct", "inference_seconds", "split_id"
]

MODEL_ORDER = ["mean_predictor", "baseline", "bert", "distilbert", "roberta"]
PRED_MODEL_NAMES = ["bert", "distilbert", "roberta"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate integration outputs")
    parser.add_argument("--results-dir", required=True, help="Thư mục chứa kết quả (reports/ hoặc results/)")
    parser.add_argument("--split-stats", help="Đường dẫn split_stats.json (mặc định: <results-dir>/../data/split_stats.json)")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Dung sai tuyệt đối cho metric reconciliation")
    parser.add_argument("--skip-dashboard", action="store_true", help="Bỏ qua check dashboard readiness")
    return parser.parse_args()


def load_all_data(results_dir: Path, split_stats_path: Path | None) -> dict:
    """Đọc tất cả file, trả về dict DataFrames."""
    data = {}
    results_dir = Path(results_dir)

    # split_stats.json
    if split_stats_path is None:
        split_stats_path = results_dir.parent / "data" / "split_stats.json"
    if not split_stats_path.exists():
        raise FileNotFoundError(f"Không tìm thấy split_stats.json: {split_stats_path}")
    data["split_stats_path"] = split_stats_path
    with open(split_stats_path, encoding="utf-8") as f:
        data["split_stats"] = json.load(f)

    # test_predictions.csv
    preds_path = results_dir / "test_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Thiếu test_predictions.csv: {preds_path}")
    data["test_predictions"] = pd.read_csv(preds_path)

    # test_predictions_with_text.csv (tuỳ chọn)
    with_text_path = results_dir / "test_predictions_with_text.csv"
    if with_text_path.exists():
        data["test_predictions_with_text"] = pd.read_csv(with_text_path)

    # model_summary.csv
    summary_path = results_dir / "model_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Thiếu model_summary.csv: {summary_path}")
    data["model_summary"] = pd.read_csv(summary_path)

    # Individual model predictions
    preds_subdir = results_dir / "preds"
    data["individual_preds"] = {}
    data["individual_metas"] = {}
    for model in PRED_MODEL_NAMES:
        csv_path = preds_subdir / f"{model}_test.csv"
        meta_path = preds_subdir / f"{model}_test_meta.json"
        if csv_path.exists() and meta_path.exists():
            data["individual_preds"][model] = pd.read_csv(csv_path)
            with open(meta_path, encoding="utf-8") as f:
                data["individual_metas"][model] = json.load(f)

    return data


def check_merge_consistency(data: dict, split_stats: dict) -> list[str]:
    """Kiểm tra merge consistency. Trả về list lỗi (rỗng = PASS)."""
    errors = []
    preds = data["test_predictions"]

    # 1. split_id khớp
    if "split_id" not in preds.columns:
        errors.append("test_predictions.csv thiếu cột split_id")
    else:
        pred_split_id = preds["split_id"].iloc[0] if len(preds) > 0 else None
        stats_split_id = split_stats.get("split_id")
        if pred_split_id != stats_split_id:
            errors.append(f"split_id mismatch: predictions={pred_split_id}, split_stats={stats_split_id}")

    # 2. review_id unique
    if preds["review_id"].duplicated().any():
        errors.append("Duplicate review_id in test_predictions.csv")

    # 3. true_stars hợp lệ
    invalid_stars = ~preds["true_stars"].isin([1, 2, 3, 4, 5])
    if invalid_stars.any():
        invalid_vals = preds.loc[invalid_stars, "true_stars"].unique()
        errors.append(f"Invalid true_stars values: {invalid_vals}")

    # 4. Nếu có with_text, review_id và true_stars phải khớp
    if "test_predictions_with_text" in data:
        with_text = data["test_predictions_with_text"]
        merged = preds[["review_id", "true_stars"]].merge(
            with_text[["review_id", "true_stars"]], on="review_id", suffixes=("", "_wt")
        )
        if not (merged["true_stars"] == merged["true_stars_wt"]).all():
            errors.append("true_stars mismatch between test_predictions.csv and _with_text version")

    # 5. n_tokens > 0
    if "n_tokens" in preds.columns:
        if (preds["n_tokens"] <= 0).any():
            errors.append("n_tokens có giá trị <= 0")

    return errors


def check_schemas(data: dict) -> list[str]:
    """Kiểm tra schema từng file. Trả về list lỗi."""
    errors = []
    preds = data["test_predictions"]
    summary = data["model_summary"]

    # 2.2.1 test_predictions.csv
    for col in REQUIRED_TEST_PREDICTIONS_COLS:
        if col not in preds.columns:
            errors.append(f"test_predictions.csv thiếu cột bắt buộc: {col}")

    # pred_* columns
    for model in PRED_MODEL_NAMES:
        pred_col = f"pred_{model}"
        abs_col = f"abs_err_{model}"
        if pred_col in preds.columns:
            # pred trong [1, 5]
            out_of_range = ~preds[pred_col].between(1.0, 5.0)
            if out_of_range.any():
                errors.append(f"{pred_col} out of range [1, 5] tại {out_of_range.sum()} dòng")
            # abs_err khớp
            if abs_col in preds.columns:
                expected = (preds["true_stars"] - preds[pred_col]).abs()
                mismatch = ~np.isclose(preds[abs_col], expected, atol=1e-4)
                if mismatch.any():
                    errors.append(f"{abs_col} mismatch tại {mismatch.sum()} dòng")

    # baseline if present
    if "pred_baseline" in preds.columns:
        out_of_range = ~preds["pred_baseline"].between(1.0, 5.0)
        if out_of_range.any():
            errors.append(f"pred_baseline out of range [1, 5] tại {out_of_range.sum()} dòng")
        if "abs_err_baseline" in preds.columns:
            expected = (preds["true_stars"] - preds["pred_baseline"]).abs()
            mismatch = ~np.isclose(preds["abs_err_baseline"], expected, atol=1e-4)
            if mismatch.any():
                errors.append(f"abs_err_baseline mismatch tại {mismatch.sum()} dòng")

    # 2.2.2 model_summary.csv
    for col in REQUIRED_MODEL_SUMMARY_COLS:
        if col not in summary.columns:
            errors.append(f"model_summary.csv thiếu cột bắt buộc: {col}")

    # Kiểm tra từng dòng model
    for _, row in summary.iterrows():
        model = row["model"]
        if model not in MODEL_ORDER:
            errors.append(f"model_summary: model không hợp lệ '{model}'")

        # Các cột metric phải hữu hạn (trừ model thiếu)
        for metric in ["mae", "rmse", "rmse_raw", "r2", "pearson", "spearman", "macro_mae"]:
            val = row[metric]
            if pd.notna(val) and not np.isfinite(val):
                errors.append(f"model_summary.{model}.{metric} không hữu hạn: {val}")

        if pd.notna(row["params_total"]) and row["params_total"] < 0:
            errors.append(f"model_summary.{model}.params_total < 0: {row['params_total']}")
        if pd.notna(row["train_seconds"]) and row["train_seconds"] < 0:
            errors.append(f"model_summary.{model}.train_seconds < 0: {row['train_seconds']}")
        if pd.notna(row["seconds_per_epoch"]) and row["seconds_per_epoch"] < 0:
            errors.append(f"model_summary.{model}.seconds_per_epoch < 0: {row['seconds_per_epoch']}")
        if pd.notna(row["inference_seconds"]) and row["inference_seconds"] < 0:
            errors.append(f"model_summary.{model}.inference_seconds < 0: {row['inference_seconds']}")

        if "split_id" in summary.columns:
            if pd.notna(row["split_id"]) and len(str(row["split_id"])) != 12:
                errors.append(f"model_summary.{model}.split_id độ dài != 12: {row['split_id']}")

    # 2.2.3 <model>_test.csv
    for model, df in data.get("individual_preds", {}).items():
        required_cols = ["review_id", "true_stars", "pred_raw", "pred", "split_id"]
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"{model}_test.csv thiếu cột: {col}")

        # pred đã clip [1,5]
        if "pred" in df.columns:
            out_of_range = ~df["pred"].between(1.0, 5.0)
            if out_of_range.any():
                errors.append(f"{model}_test.csv: pred out of range [1, 5]")

        # pred_raw không bị clip (có thể <1 hoặc >5) - chỉ cảnh báo nếu tất cả đều trong [1,5]
        if "pred_raw" in df.columns:
            all_clipped = df["pred_raw"].between(1.0, 5.0).all()
            if all_clipped and len(df) > 10:
                errors.append(f"{model}_test.csv: pred_raw có vẻ đã bị clip (tất cả trong [1,5])")

    # 2.2.4 _test_meta.json
    for model, meta in data.get("individual_metas", {}).items():
        if meta.get("model") != model:
            errors.append(f"{model}_test_meta.json: model field mismatch")
        if meta.get("rmse_raw") is not None and not np.isfinite(meta["rmse_raw"]):
            errors.append(f"{model}_test_meta.json: rmse_raw không hữu hạn")
        if meta.get("inference_seconds") is not None and meta["inference_seconds"] <= 0:
            errors.append(f"{model}_test_meta.json: inference_seconds <= 0")

    return errors


def reconcile_metrics(data: dict, tolerance: float) -> list[str]:
    """Đối chiếu model_summary.csv với tính toán lại từ test_predictions.csv."""
    errors = []
    summary = data["model_summary"]
    preds = data["test_predictions"]

    for _, row in summary.iterrows():
        model = row["model"]
        if model == "mean_predictor":
            continue

        pred_col = f"pred_{model}"
        if pred_col not in preds.columns:
            errors.append(f"Missing pred column {pred_col} for model {model}")
            continue

        y_true = preds["true_stars"].values
        y_pred = preds[pred_col].values

        # Global metrics
        recalc = compute_metrics(y_true, y_pred, clip=True)
        for metric in ["mae", "rmse", "r2", "pearson", "spearman"]:
            recorded = row[metric]
            if pd.isna(recorded):
                continue
            diff = abs(recalc[metric] - recorded)
            if diff > tolerance:
                errors.append(f"{model}.{metric}: recorded={recorded:.6f}, recalc={recalc[metric]:.6f}, diff={diff:.6f} > {tolerance}")

        # Per-star + macro_mae
        per_star = per_star_metrics(y_true, y_pred, clip=True)
        if pd.notna(row["macro_mae"]):
            diff = abs(per_star["macro_mae"] - row["macro_mae"])
            if diff > tolerance:
                errors.append(f"{model}.macro_mae: recorded={row['macro_mae']:.6f}, recalc={per_star['macro_mae']:.6f}, diff={diff:.6f}")

        # rmse_raw (từ pred_raw trong <model>_test.csv)
        if model in data.get("individual_preds", {}):
            raw_preds = data["individual_preds"][model]["pred_raw"].values
            raw_rmse = np.sqrt(np.mean((raw_preds - y_true) ** 2))
            if pd.notna(row["rmse_raw"]):
                diff = abs(raw_rmse - row["rmse_raw"])
                if diff > tolerance:
                    errors.append(f"{model}.rmse_raw: recorded={row['rmse_raw']:.6f}, recalc={raw_rmse:.6f}, diff={diff:.6f}")

    return errors


def check_dashboard_readiness(data: dict) -> list[str]:
    """Kiểm tra sẵn sàng cho dashboard."""
    errors = []
    summary = data["model_summary"]
    preds = data["test_predictions"]

    # Model summary required columns for dashboard
    dashboard_cols = [
        "model", "mae", "rmse", "r2", "pearson", "spearman", "macro_mae",
        "train_seconds", "seconds_per_epoch", "params_total",
        "rmse_reduction_vs_baseline_pct", "inference_seconds", "split_id"
    ]
    for col in dashboard_cols:
        if col not in summary.columns:
            errors.append(f"model_summary.csv thiếu cột dashboard: {col}")

    # test_predictions có pred_* và abs_err_* cho model có mặt
    for model in PRED_MODEL_NAMES:
        pred_col = f"pred_{model}"
        abs_col = f"abs_err_{model}"
        if pred_col in preds.columns and abs_col not in preds.columns:
            errors.append(f"test_predictions.csv thiếu {abs_col} cho model {model}")

    # Không NaN toàn cột bắt buộc
    for col in REQUIRED_TEST_PREDICTIONS_COLS:
        if col in preds.columns and preds[col].isna().any():
            errors.append(f"test_predictions.csv cột {col} có NaN")

    for col in ["model", "split_id"]:
        if col in summary.columns and summary[col].isna().any():
            errors.append(f"model_summary.csv cột {col} có NaN")

    return errors


def print_summary(checks: dict[str, list[str]]) -> bool:
    """In bảng PASS/FAIL, trả về True nếu all pass."""
    all_pass = True
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for check_name, errors in checks.items():
        status = "PASS" if not errors else "FAIL"
        if errors:
            all_pass = False
        print(f"  [{status}] {check_name}")
        for err in errors:
            print(f"    - {err}")
    print("=" * 60)
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("VALIDATION FAILED")
    print("=" * 60 + "\n")
    return all_pass


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    split_stats_path = Path(args.split_stats) if args.split_stats else None

    try:
        data = load_all_data(results_dir, split_stats_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    split_stats = data["split_stats"]

    all_errors = {}
    all_errors["merge"] = check_merge_consistency(data, split_stats)
    all_errors["schemas"] = check_schemas(data)
    all_errors["metrics"] = reconcile_metrics(data, args.tolerance)
    if not args.skip_dashboard:
        all_errors["dashboard"] = check_dashboard_readiness(data)

    ok = print_summary(all_errors)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())