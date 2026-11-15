"""Synthetic fixtures only; never write them into production reports/."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from src.metrics import compute_metrics, per_star_metrics


def create_demo(output_dir="demo_reports"):
    path = Path(output_dir)
    if path.resolve().name == "reports" or path.exists():
        raise FileExistsError("Chọn thư mục mới, khác reports/, để tránh ghi đè dữ liệu thật.")
    path.mkdir(parents=True)
    # Mark immediately, so even an interrupted run cannot look like real results.
    (path / "DEMO_ONLY.txt").write_text("SYNTHETIC DATA — NOT TRAINING RESULTS\n", encoding="utf-8")
    rng = np.random.default_rng(42)
    stars = np.tile(np.arange(1, 6), 40)
    df = pd.DataFrame({"review_id": [f"demo_{i:04d}" for i in range(len(stars))],
                       "true_stars": stars, "n_tokens": rng.integers(10, 600, len(stars)),
                       "split_id": "demo00000042"})
    summaries = []
    for model, noise in [("mean_predictor", 0), ("baseline", 1.3), ("bert", .6), ("distilbert", .8), ("roberta", .7)]:
        pred = np.full(len(stars), 3.0) if model == "mean_predictor" else np.clip(stars + rng.normal(0, noise, len(stars)), 1, 5)
        metrics = compute_metrics(stars, pred)
        metrics.pop("n")
        summaries.append({"model": model, **metrics, "macro_mae": per_star_metrics(stars, pred)["macro_mae"],
                          "train_seconds": np.nan, "seconds_per_epoch": np.nan, "params_total": np.nan,
                          "rmse_reduction_vs_baseline_pct": np.nan, "inference_seconds": np.nan,
                          "split_id": "demo00000042"})
        if model != "mean_predictor":
            df[f"pred_{model}"] = pred
            df[f"abs_err_{model}"] = np.abs(pred - stars)
    pd.DataFrame(summaries).to_csv(path / "model_summary.csv", index=False, float_format="%.6f")
    df.to_csv(path / "test_predictions.csv", index=False, float_format="%.6f")
    df["text"] = [f"SYNTHETIC REVIEW {i}. This is a demo fixture, not a real Yelp review. "
                  "Good food but slow service. " + "Long demo content. " * (i % 8) for i in range(len(df))]
    df.to_csv(path / "test_predictions_with_text.csv", index=False, float_format="%.6f")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="demo_reports")
    print(create_demo(parser.parse_args().output_dir))
