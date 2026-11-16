import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate integration outputs"
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Thu muc chua ket qua (reports/ hoac results/)",
    )
    parser.add_argument(
        "--split-stats",
        help="Duong dan split_stats.json",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Dung sai cho metric reconciliation",
    )
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Bo qua dashboard readiness",
    )
    return parser.parse_args()


def load_all_data(results_dir: Path, split_stats_path: Path | None) -> dict:
    """Doc cac file ket qua va tra ve du lieu validation."""

    results_dir = Path(results_dir)
    data = {
        "results_dir": results_dir,
        "individual_preds": {},
        "individual_meta": {},
        "load_errors": [],
    }

    if not results_dir.is_dir():
        data["load_errors"].append(
            f"Results directory does not exist: {results_dir}"
        )
        return data

    # Doc test_predictions.csv
    predictions_path = results_dir / "test_predictions.csv"
    if predictions_path.is_file():
        data["test_predictions"] = pd.read_csv(predictions_path)
    else:
        data["load_errors"].append(
            f"Missing file: {predictions_path}"
        )

    # Doc test_predictions_with_text.csv neu co
    with_text_path = results_dir / "test_predictions_with_text.csv"
    if with_text_path.is_file():
        data["test_predictions_with_text"] = pd.read_csv(with_text_path)

    # Doc model_summary.csv
    summary_path = results_dir / "model_summary.csv"
    if summary_path.is_file():
        data["model_summary"] = pd.read_csv(summary_path)
    else:
        data["load_errors"].append(
            f"Missing file: {summary_path}"
        )

    # Tim split_stats.json
    if split_stats_path is None:
        split_stats_path = results_dir.parent / "data" / "split_stats.json"

        if not split_stats_path.is_file():
            alternative_path = results_dir / "data" / "split_stats.json"
            if alternative_path.is_file():
                split_stats_path = alternative_path

    split_stats_path = Path(split_stats_path)
    data["split_stats_path"] = split_stats_path

    if split_stats_path.is_file():
        try:
            with open(split_stats_path, "r", encoding="utf-8") as file:
                data["split_stats"] = json.load(file)
        except (json.JSONDecodeError, OSError) as exc:
            data["load_errors"].append(
                f"Cannot read split_stats.json: {exc}"
            )
    else:
        data["load_errors"].append(
            f"Missing split_stats.json: {split_stats_path}"
        )

    # Doc cac file du doan va metadata cua tung model
    preds_dir = results_dir / "preds"

    if preds_dir.is_dir():
        for csv_path in sorted(preds_dir.glob("*_test.csv")):
            model = csv_path.name.removesuffix("_test.csv")

            try:
                data["individual_preds"][model] = pd.read_csv(csv_path)
            except (pd.errors.ParserError, OSError) as exc:
                data["load_errors"].append(
                    f"Cannot read {csv_path}: {exc}"
                )

            meta_path = preds_dir / f"{model}_test_meta.json"

            if meta_path.is_file():
                try:
                    with open(meta_path, "r", encoding="utf-8") as file:
                        data["individual_meta"][model] = json.load(file)
                except (json.JSONDecodeError, OSError) as exc:
                    data["load_errors"].append(
                        f"Cannot read {meta_path}: {exc}"
                    )
            else:
                data["load_errors"].append(
                    f"Missing metadata: {meta_path}"
                )
    else:
        data["load_errors"].append(
            f"Missing predictions directory: {preds_dir}"
        )

    return data


def check_merge_consistency(data: dict, split_stats: dict) -> list[str]:
    """Kiem tra tinh nhat quan cua du lieu sau khi merge."""
    errors = []

    preds = data.get("test_predictions")

    if preds is None:
        errors.append("Missing test_predictions data")
        return errors

    required_columns = ["review_id", "true_stars", "split_id", "n_tokens"]
    missing_columns = [
        col for col in required_columns if col not in preds.columns
    ]

    if missing_columns:
        errors.append(
            f"Missing required columns in test_predictions.csv: {missing_columns}"
        )
        return errors

    if preds.empty:
        errors.append("test_predictions.csv is empty")
        return errors

    # 1. Kiem tra gia tri thieu
    if preds[required_columns].isna().any().any():
        errors.append("NaN values found in required columns")

    # 2. Kiem tra split_id
    expected_split_id = split_stats.get("split_id")

    if not (preds["split_id"] == expected_split_id).all():
        actual_split_ids = preds["split_id"].unique()
        errors.append(
            f"split_id mismatch: predictions={actual_split_ids}, "
            f"split_stats={expected_split_id}"
        )

    # 3. review_id phai duy nhat
    if preds["review_id"].duplicated().any():
        errors.append("Duplicate review_id in test_predictions.csv")

    # 4. true_stars phai nam trong 1-5
    invalid_stars = ~preds["true_stars"].isin([1, 2, 3, 4, 5])

    if invalid_stars.any():
        errors.append(
            f"Invalid true_stars values: "
            f"{preds.loc[invalid_stars, 'true_stars'].unique()}"
        )

    # 5. n_tokens phai lon hon 0
    invalid_tokens = pd.to_numeric(
        preds["n_tokens"], errors="coerce"
    ).isna() | (pd.to_numeric(
        preds["n_tokens"], errors="coerce"
    ) <= 0)

    if invalid_tokens.any():
        errors.append("Invalid n_tokens values: all values must be > 0")

    # 6. Neu co file with_text, review_id va true_stars phai khop
    if "test_predictions_with_text" in data:
        with_text = data["test_predictions_with_text"]

        required_wt = ["review_id", "true_stars"]
        if not all(col in with_text.columns for col in required_wt):
            errors.append(
                "Missing review_id or true_stars in test_predictions_with_text.csv"
            )
        else:
            merged = preds[["review_id", "true_stars"]].merge(
                with_text[["review_id", "true_stars"]],
                on="review_id",
                how="outer",
                suffixes=("", "_wt"),
                indicator=True,
            )

            if not (merged["_merge"] == "both").all():
                errors.append(
                    "review_id mismatch between test_predictions.csv "
                    "and _with_text version"
                )

            matched = merged[merged["_merge"] == "both"]

            if not (
                matched["true_stars"] == matched["true_stars_wt"]
            ).all():
                errors.append(
                    "true_stars mismatch between test_predictions.csv "
                    "and _with_text version"
                )

    return errors


def check_schemas(data: dict) -> list[str]:
    """Kiem tra schema va gia tri cua cac file ket qua."""
    errors = []

    preds = data.get("test_predictions")
    summary = data.get("model_summary")

    if preds is None:
        errors.append("Missing test_predictions data")
        return errors

    if summary is None:
        errors.append("Missing model_summary data")
        return errors

    # 1. Kiem tra test_predictions.csv
    required_preds = [
        "review_id", "true_stars", "n_tokens", "split_id"
    ]

    missing = [col for col in required_preds if col not in preds.columns]

    if missing:
        errors.append(
            f"Missing required columns in test_predictions.csv: {missing}"
        )

    if not missing:
        if preds[required_preds].isna().any().any():
            errors.append("NaN values in required test_predictions columns")

        if preds["review_id"].astype(str).str.strip().eq("").any():
            errors.append("Empty review_id in test_predictions.csv")

        if not preds["true_stars"].isin([1, 2, 3, 4, 5]).all():
            errors.append("Invalid true_stars values in test_predictions.csv")

        tokens = pd.to_numeric(preds["n_tokens"], errors="coerce")

        if tokens.isna().any() or (tokens <= 0).any() or (tokens > 512).any():
            errors.append("Invalid n_tokens: values must be > 0 and <= 512")

        if preds["split_id"].astype(str).str.len().ne(12).any():
            errors.append("Invalid split_id: expected string length 12")

    # Kiem tra cac cot prediction va absolute error
    prediction_columns = [
        col for col in preds.columns
        if col.startswith("pred_")
    ]

    for pred_col in prediction_columns:
        model = pred_col.removeprefix("pred_")
        values = pd.to_numeric(preds[pred_col], errors="coerce")

        if values.isna().any() or (
            (values < 1.0) | (values > 5.0)
        ).any():
            errors.append(
                f"{pred_col} out of range [1,5] or contains invalid values"
            )

        abs_err_col = f"abs_err_{model}"

        if abs_err_col in preds.columns and "true_stars" in preds.columns:
            actual_error = (
                pd.to_numeric(preds["true_stars"], errors="coerce")
                - values
            ).abs()

            recorded_error = pd.to_numeric(
                preds[abs_err_col], errors="coerce"
            )

            if (
                recorded_error.isna().any()
                or (recorded_error < 0).any()
                or not np.allclose(
                    actual_error,
                    recorded_error,
                    atol=1e-6,
                    rtol=0,
                    equal_nan=False,
                )
            ):
                errors.append(f"{abs_err_col} mismatch")

    # 2. Kiem tra model_summary.csv
    required_summary = [
        "model", "mae", "rmse", "rmse_raw", "r2",
        "pearson", "spearman", "macro_mae",
        "train_seconds", "seconds_per_epoch", "params_total",
        "rmse_reduction_vs_baseline_pct",
        "inference_seconds", "split_id",
    ]

    missing = [
        col for col in required_summary
        if col not in summary.columns
    ]

    if missing:
        errors.append(
            f"Missing required columns in model_summary.csv: {missing}"
        )

    if not missing:
        allowed_models = {
            "mean_predictor", "baseline",
            "bert", "distilbert", "roberta",
        }

        if not summary["model"].isin(allowed_models).all():
            errors.append("Invalid model names in model_summary.csv")

        if summary["model"].duplicated().any():
            errors.append("Duplicate model names in model_summary.csv")

        for _, row in summary.iterrows():
            model = row["model"]

            for col in [
                "mae", "rmse", "rmse_raw", "r2",
                "pearson", "spearman", "macro_mae",
            ]:
                value = pd.to_numeric(
                    pd.Series([row[col]]), errors="coerce"
                ).iloc[0]

                if pd.notna(value) and not np.isfinite(value):
                    errors.append(
                        f"Invalid {col} value for model {model}"
                    )

            for col in [
                "train_seconds", "seconds_per_epoch",
                "params_total", "inference_seconds",
            ]:
                value = pd.to_numeric(
                    pd.Series([row[col]]), errors="coerce"
                ).iloc[0]

                if pd.notna(value) and (
                    not np.isfinite(value) or value < 0
                ):
                    errors.append(
                        f"Invalid {col} value for model {model}"
                    )

            if pd.notna(row["split_id"]):
                if len(str(row["split_id"])) != 12:
                    errors.append(
                        f"Invalid split_id length for model {model}"
                    )

    # 3. Kiem tra tung file preds/<model>_test.csv
    individual_preds = data.get("individual_preds", {})
    individual_meta = data.get("individual_meta", {})

    for model, frame in individual_preds.items():
        required_individual = [
            "review_id", "true_stars", "pred_raw", "pred", "split_id"
        ]

        missing = [
            col for col in required_individual
            if col not in frame.columns
        ]

        if missing:
            errors.append(
                f"Missing columns in {model}_test.csv: {missing}"
            )
            continue

        if frame[required_individual].isna().any().any():
            errors.append(
                f"NaN values in required columns of {model}_test.csv"
            )

        if frame["review_id"].duplicated().any():
            errors.append(
                f"Duplicate review_id in {model}_test.csv"
            )

        if not frame["true_stars"].isin([1, 2, 3, 4, 5]).all():
            errors.append(
                f"Invalid true_stars in {model}_test.csv"
            )

        raw_values = pd.to_numeric(frame["pred_raw"], errors="coerce")
        pred_values = pd.to_numeric(frame["pred"], errors="coerce")

        if raw_values.isna().any() or not np.isfinite(raw_values).all():
            errors.append(
                f"Invalid pred_raw values in {model}_test.csv"
            )

        if (
            pred_values.isna().any()
            or not np.isfinite(pred_values).all()
            or ((pred_values < 1.0) | (pred_values > 5.0)).any()
        ):
            errors.append(
                f"Invalid pred values: expected [1,5] in {model}_test.csv"
            )

        if not preds.empty and all(
            col in preds.columns for col in ["review_id", "true_stars", "split_id"]
        ):
            merged = preds[
                ["review_id", "true_stars", "split_id"]
            ].merge(
                frame[
                    ["review_id", "true_stars", "split_id"]
                ],
                on="review_id",
                how="outer",
                suffixes=("_main", "_individual"),
                indicator=True,
            )

            if not (merged["_merge"] == "both").all():
                errors.append(
                    f"review_id mismatch between test_predictions.csv "
                    f"and {model}_test.csv"
                )
            else:
                if not (
                    merged["true_stars_main"]
                    == merged["true_stars_individual"]
                ).all():
                    errors.append(
                        f"true_stars mismatch in {model}_test.csv"
                    )

                if not (
                    merged["split_id_main"]
                    == merged["split_id_individual"]
                ).all():
                    errors.append(
                        f"split_id mismatch in {model}_test.csv"
                    )

        # 4. Kiem tra metadata JSON
        meta = individual_meta.get(model)

        if meta is None:
            errors.append(f"Missing metadata for model {model}")
            continue

        required_meta = [
            "model", "run_name", "checkpoint_epoch", "n",
            "rmse_raw", "inference_seconds", "fp16", "device",
        ]

        missing_meta = [
            key for key in required_meta if key not in meta
        ]

        if missing_meta:
            errors.append(
                f"Missing metadata keys for {model}: {missing_meta}"
            )
            continue

        if meta["model"] != model:
            errors.append(f"Metadata model mismatch for {model}")

        if not isinstance(meta["run_name"], str) or not meta["run_name"]:
            errors.append(f"Invalid run_name for {model}")

        if (
            not isinstance(meta["checkpoint_epoch"], (int, np.integer))
            or isinstance(meta["checkpoint_epoch"], bool)
            or meta["checkpoint_epoch"] < 1
        ):
            errors.append(f"Invalid checkpoint_epoch for {model}")

        if meta["n"] != len(frame):
            errors.append(f"Metadata n mismatch for {model}")

        if (
            not isinstance(meta["fp16"], (bool, np.bool_))
        ):
            errors.append(f"Invalid fp16 for {model}")

        if not isinstance(meta["device"], str) or not meta["device"]:
            errors.append(f"Invalid device for {model}")

        inference_time = pd.to_numeric(
            pd.Series([meta["inference_seconds"]]), errors="coerce"
        ).iloc[0]

        if (
            pd.isna(inference_time)
            or not np.isfinite(inference_time)
            or inference_time <= 0
        ):
            errors.append(f"Invalid inference_seconds for {model}")

        meta_rmse = pd.to_numeric(
            pd.Series([meta["rmse_raw"]]), errors="coerce"
        ).iloc[0]

        if (
            pd.isna(meta_rmse)
            or not np.isfinite(meta_rmse)
            or raw_values.isna().any()
        ):
            errors.append(f"Invalid rmse_raw metadata for {model}")
        else:
            true_values = pd.to_numeric(
                frame["true_stars"], errors="coerce"
            )

            calculated_rmse = np.sqrt(
                np.mean((raw_values.to_numpy() - true_values.to_numpy()) ** 2)
            )

            if abs(calculated_rmse - meta_rmse) > 1e-4:
                errors.append(
                    f"{model}.rmse_raw diff > 1e-4: "
                    f"metadata={meta_rmse:.6f}, "
                    f"recalc={calculated_rmse:.6f}"
                )

    return errors


def reconcile_metrics(data: dict, tolerance: float) -> list[str]:
    """Doi chieu metric trong model_summary voi ket qua tinh lai."""
    from src.metrics import compute_metrics, per_star_metrics

    errors = []

    summary = data.get("model_summary")
    preds = data.get("test_predictions")
    individual_preds = data.get("individual_preds", {})

    if summary is None or preds is None:
        errors.append("Missing model_summary or test_predictions data")
        return errors

    if "true_stars" not in preds.columns:
        errors.append("Missing true_stars column in test_predictions.csv")
        return errors

    y_true = pd.to_numeric(
        preds["true_stars"], errors="coerce"
    ).to_numpy(dtype=float)

    if not np.isfinite(y_true).all():
        errors.append("Invalid true_stars values for metric reconciliation")
        return errors

    for _, row in summary.iterrows():
        model = row["model"]

        if model == "mean_predictor":
            continue

        pred_col = f"pred_{model}"

        if pred_col not in preds.columns:
            errors.append(
                f"Missing pred column {pred_col} for model {model}"
            )
            continue

        y_pred = pd.to_numeric(
            preds[pred_col], errors="coerce"
        ).to_numpy(dtype=float)

        if (
            len(y_pred) != len(y_true)
            or not np.isfinite(y_pred).all()
        ):
            errors.append(
                f"Invalid prediction data for model {model}"
            )
            continue

        # 1. Doi chieu cac metric toan cuc
        try:
            recalc = compute_metrics(y_true, y_pred, clip=True)
        except (ValueError, TypeError) as exc:
            errors.append(
                f"Cannot calculate metrics for {model}: {exc}"
            )
            continue

        for metric in ["mae", "rmse", "r2", "pearson", "spearman"]:
            recorded = pd.to_numeric(
                pd.Series([row[metric]]), errors="coerce"
            ).iloc[0]

            if pd.isna(recorded):
                continue

            calculated = recalc[metric]

            if not np.isfinite(calculated):
                errors.append(
                    f"{model}.{metric}: recalculated value is not finite"
                )
                continue

            metric_tolerance = (
                max(tolerance, 1e-3)
                if metric in ["pearson", "spearman"]
                else tolerance
            )

            diff = abs(calculated - recorded)

            if diff > metric_tolerance:
                errors.append(
                    f"{model}.{metric}: recorded={recorded:.6f}, "
                    f"recalc={calculated:.6f}, diff={diff:.6f} "
                    f"> {metric_tolerance}"
                )

        # 2. Doi chieu macro_mae
        try:
            per_star = per_star_metrics(
                y_true, y_pred, clip=True
            )
        except (ValueError, TypeError) as exc:
            errors.append(
                f"Cannot calculate per-star metrics for {model}: {exc}"
            )
            per_star = None

        if per_star is not None and pd.notna(row["macro_mae"]):
            recorded = float(row["macro_mae"])
            calculated = per_star["macro_mae"]

            if not np.isfinite(calculated):
                errors.append(
                    f"{model}.macro_mae: recalculated value is not finite"
                )
            else:
                diff = abs(calculated - recorded)

                if diff > tolerance:
                    errors.append(
                        f"{model}.macro_mae: recorded={recorded:.6f}, "
                        f"recalc={calculated:.6f}, diff={diff:.6f}"
                    )

        # 3. Doi chieu rmse_raw tu file model_test.csv
        if model not in individual_preds:
            continue

        individual = individual_preds[model]

        if "pred_raw" not in individual.columns:
            errors.append(
                f"Missing pred_raw column in {model}_test.csv"
            )
            continue

        raw_preds = pd.to_numeric(
            individual["pred_raw"], errors="coerce"
        ).to_numpy(dtype=float)

        if (
            len(raw_preds) != len(y_true)
            or not np.isfinite(raw_preds).all()
        ):
            errors.append(
                f"Invalid pred_raw data for {model}"
            )
            continue

        raw_rmse = float(
            np.sqrt(np.mean((raw_preds - y_true) ** 2))
        )

        recorded_raw = pd.to_numeric(
            pd.Series([row["rmse_raw"]]), errors="coerce"
        ).iloc[0]

        if pd.notna(recorded_raw):
            diff = abs(raw_rmse - recorded_raw)

            if diff > tolerance:
                errors.append(
                    f"{model}.rmse_raw: recorded={recorded_raw:.6f}, "
                    f"recalc={raw_rmse:.6f}, diff={diff:.6f}"
                )

    return errors


def check_dashboard_readiness(data: dict) -> list[str]:
    """Kiem tra du lieu co san sang cho Dashboard hay khong."""
    errors = []

    summary = data.get("model_summary")
    preds = data.get("test_predictions")

    if summary is None:
        errors.append("Missing model_summary.csv")
        return errors

    if preds is None:
        errors.append("Missing test_predictions.csv")
        return errors

    # 1. Cac cot bat buoc trong model_summary.csv
    required_summary = [
        "model",
        "mae",
        "rmse",
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
    ]

    missing_summary = [
        col for col in required_summary
        if col not in summary.columns
    ]

    if missing_summary:
        errors.append(
            f"Dashboard model_summary missing columns: {missing_summary}"
        )
    else:
        # 2. Khong de cot bat buoc bi NaN toan bo
        for col in required_summary:
            if summary[col].isna().all():
                errors.append(
                    f"Dashboard model_summary column '{col}' is entirely NaN"
                )

    # 3. Kiem tra pred_* va abs_err_* cho moi model co mat
    if "model" not in summary.columns:
        errors.append(
            "Cannot check prediction columns: missing model column"
        )
        return errors

    for model in summary["model"].dropna().unique():
        if model == "mean_predictor":
            continue

        pred_col = f"pred_{model}"
        abs_err_col = f"abs_err_{model}"

        if pred_col not in preds.columns:
            errors.append(
                f"Dashboard missing prediction column: {pred_col}"
            )

        if abs_err_col not in preds.columns:
            errors.append(
                f"Dashboard missing absolute error column: {abs_err_col}"
            )

    return errors


def print_summary(checks: dict[str, list[str]]) -> bool:
    """In bang PASS/FAIL cho cac hang muc validation."""
    print("\n========== INTEGRATION VALIDATION SUMMARY ==========")

    all_pass = True

    for name, errors in checks.items():
        if errors:
            all_pass = False
            print(f"[FAIL] {name}")

            for error in errors:
                print(f"       - {error}")
        else:
            print(f"[PASS] {name}")

    print("=====================================================")

    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("VALIDATION FAILED")

    return all_pass

def main() -> int:
    """Dieu phoi cac buoc kiem tra integration outputs."""
    args = parse_args()

    results_dir = Path(args.results_dir)
    split_stats_path = (
        Path(args.split_stats) if args.split_stats else None
    )

    data = load_all_data(results_dir, split_stats_path)
    split_stats = data.get("split_stats", {})

    checks = {
        "load_data": data.get("load_errors", []),
        "merge_consistency": check_merge_consistency(
            data, split_stats
        ),
        "schemas": check_schemas(data),
        "metrics": reconcile_metrics(data, args.tolerance),
    }

    if not args.skip_dashboard:
        checks["dashboard"] = check_dashboard_readiness(data)

    all_pass = print_summary(checks)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
