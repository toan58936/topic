#!/usr/bin/env python3
"""Meta-tests for validate_outputs.py: verify the validation script catches corruptions.

These tests use synthetic corrupted data to prove that validate_outputs.py
correctly detects various error types (exit code 1, appropriate error messages).
"""

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.validate_outputs import (
    load_all_data, check_merge_consistency, check_schemas,
    reconcile_metrics, check_dashboard_readiness, main
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="session")
def valid_results_dir(tmp_path_factory):
    """Tạo bộ dữ liệu ĐÚNG đầy đủ cho meta-test."""
    base = tmp_path_factory.mktemp("valid_results")
    
    # Cấu trúc thư mục
    (base / "preds").mkdir(parents=True)
    (base / "data").mkdir(parents=True)
    
    # split_stats.json
    split_stats = {
        "split_id": "a1b2c3d4e5f6",
        "seed": 42,
        "n_samples": 60000,
        "train_size": 42000,
        "val_size": 9000,
        "test_size": 9000,
        "stars_per_split": {
            "train": {"1": 8000, "2": 8000, "3": 10000, "4": 8000, "5": 8000},
            "val": {"1": 1500, "2": 1500, "3": 2000, "4": 1500, "5": 1500},
            "test": {"1": 1500, "2": 1500, "3": 2000, "4": 1500, "5": 1500}
        },
        "stars_distribution_original": {"1": 11000, "2": 11000, "3": 14000, "4": 11000, "5": 11000},
        "duplicate_text_removed": {"val": 0, "test": 0},
        "user_id_overlap_rate": {"train_val": 0.01, "train_test": 0.01},
        "business_id_overlap_rate": {"train_val": 0.02, "train_test": 0.02},
        "created_at": "2026-11-09T03:15:00Z"
    }
    with open(base / "data" / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(split_stats, f)
    
    # Tạo 100 dòng test data giả
    n = 100
    review_ids = [f"rev_{i:05d}" for i in range(n)]
    true_stars = [1, 2, 3, 4, 5] * 20
    
    # Base test_predictions.csv
    test_preds = pd.DataFrame({
        "review_id": review_ids,
        "true_stars": true_stars,
        "n_tokens": [50 + (i % 50) for i in range(n)],
        "split_id": "a1b2c3d4e5f6",
        "pred_baseline": [3.2, 2.8, 3.5, 4.1, 2.5] * 20,
        "pred_bert": [3.1, 2.9, 3.4, 4.0, 2.6] * 20,
        "pred_distilbert": [3.3, 2.7, 3.6, 3.9, 2.7] * 20,
        "pred_roberta": [3.0, 3.0, 3.3, 4.2, 2.4] * 20,
    })
    # abs_err columns
    for model in ["baseline", "bert", "distilbert", "roberta"]:
        test_preds[f"abs_err_{model}"] = (test_preds["true_stars"] - test_preds[f"pred_{model}"]).abs()
    
    test_preds.to_csv(base / "test_predictions.csv", index=False)
    test_preds.to_csv(base / "test_predictions_with_text.csv", index=False)  # copy for test
    
    # model_summary.csv - compute metrics from test_preds
    from src.metrics import compute_metrics, per_star_metrics
    
    summary_rows = []
    
    # mean_predictor
    y_true = test_preds["true_stars"].values
    mean_pred = float(y_true.mean())
    y_mean = np.full_like(y_true, mean_pred)
    m = compute_metrics(y_true, y_mean, clip=True)
    ps = per_star_metrics(y_true, y_mean, clip=True)
    summary_rows.append({
        "model": "mean_predictor",
        "mae": m["mae"], "rmse": m["rmse"], "rmse_raw": "",
        "r2": m["r2"], "pearson": m["pearson"], "spearman": m["spearman"],
        "macro_mae": ps["macro_mae"],
        "train_seconds": "", "seconds_per_epoch": "", "params_total": 0,
        "rmse_reduction_vs_baseline_pct": "", "inference_seconds": 0.0,
        "split_id": "a1b2c3d4e5f6"
    })
    
    # baseline
    y_pred = test_preds["pred_baseline"].values
    m = compute_metrics(y_true, y_pred, clip=True)
    ps = per_star_metrics(y_true, y_pred, clip=True)
    baseline_rmse = m["rmse"]
    summary_rows.append({
        "model": "baseline",
        "mae": m["mae"], "rmse": m["rmse"], "rmse_raw": "",
        "r2": m["r2"], "pearson": m["pearson"], "spearman": m["spearman"],
        "macro_mae": ps["macro_mae"],
        "train_seconds": 10.0, "seconds_per_epoch": "", "params_total": 20000,
        "rmse_reduction_vs_baseline_pct": "", "inference_seconds": 0.5,
        "split_id": "a1b2c3d4e5f6"
    })
    
    # Transformer models
    for model_name, pred_col in [("bert", "pred_bert"), ("distilbert", "pred_distilbert"), ("roberta", "pred_roberta")]:
        y_pred = test_preds[pred_col].values
        m = compute_metrics(y_true, y_pred, clip=True)
        ps = per_star_metrics(y_true, y_pred, clip=True)
        reduction = 100 * (baseline_rmse - m["rmse"]) / baseline_rmse
        
        # Raw RMSE
        raw_preds = y_pred + np.random.normal(0, 0.1, len(y_pred))
        raw_rmse = float(np.sqrt(np.mean((raw_preds - y_true) ** 2)))
        
        summary_rows.append({
            "model": model_name,
            "mae": m["mae"], "rmse": m["rmse"], "rmse_raw": raw_rmse,
            "r2": m["r2"], "pearson": m["pearson"], "spearman": m["spearman"],
            "macro_mae": ps["macro_mae"],
            "train_seconds": 1200.0, "seconds_per_epoch": 400.0, "params_total": 110000000,
            "rmse_reduction_vs_baseline_pct": reduction, "inference_seconds": 30.0,
            "split_id": "a1b2c3d4e5f6"
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(base / "model_summary.csv", index=False)
    
    # Individual model predictions
    for model_name in ["bert", "distilbert", "roberta"]:
        pred_col = f"pred_{model_name}"
        y_pred = test_preds[pred_col].values
        raw_preds = y_pred + np.random.normal(0, 0.1, len(y_pred))
        
        ind_df = pd.DataFrame({
            "review_id": review_ids,
            "true_stars": true_stars,
            "pred_raw": raw_preds,
            "pred": y_pred,
            "split_id": "a1b2c3d4e5f6"
        })
        ind_df.to_csv(base / "preds" / f"{model_name}_test.csv", index=False)
        
        meta = {
            "model": model_name,
            "run_name": model_name,
            "checkpoint_epoch": 3,
            "n": n,
            "rmse_raw": float(np.sqrt(np.mean((raw_preds - y_true) ** 2))),
            "inference_seconds": 30.0,
            "fp16": True,
            "device": "cuda",
            "split_id": "a1b2c3d4e5f6"
        }
        with open(base / "preds" / f"{model_name}_test_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    
    return base


@pytest.fixture(params=[
    "split_id_mismatch",
    "duplicate_review_id",
    "invalid_true_stars",
    "pred_out_of_range",
    "abs_err_mismatch",
    "model_summary_metric_wrong",
    "missing_pred_column",
    "rmse_raw_mismatch",
    "missing_split_id_column",
    "individual_pred_raw_not_match",
    "dashboard_missing_column",
])
def corrupted_results_dir(tmp_path, valid_results_dir, request):
    """Copy valid data and apply one corruption type."""
    corruption_type = request.param
    test_dir = tmp_path / f"corrupted_{corruption_type}"
    shutil.copytree(valid_results_dir, test_dir)
    
    if corruption_type == "split_id_mismatch":
        df = pd.read_csv(test_dir / "test_predictions.csv")
        df["split_id"] = "wrong_split_id"
        df.to_csv(test_dir / "test_predictions.csv", index=False)
    elif corruption_type == "duplicate_review_id":
        df = pd.read_csv(test_dir / "test_predictions.csv")
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        df.to_csv(test_dir / "test_predictions.csv", index=False)
    elif corruption_type == "invalid_true_stars":
        df = pd.read_csv(test_dir / "test_predictions.csv")
        df.loc[0, "true_stars"] = 0
        df.to_csv(test_dir / "test_predictions.csv", index=False)
    elif corruption_type == "pred_out_of_range":
        df = pd.read_csv(test_dir / "test_predictions.csv")
        df.loc[0, "pred_bert"] = 6.0
        df.to_csv(test_dir / "test_predictions.csv", index=False)
    elif corruption_type == "abs_err_mismatch":
        df = pd.read_csv(test_dir / "test_predictions.csv")
        df.loc[0, "abs_err_bert"] = 9.9
        df.to_csv(test_dir / "test_predictions.csv", index=False)
    elif corruption_type == "model_summary_metric_wrong":
        df = pd.read_csv(test_dir / "model_summary.csv")
        idx = df[df["model"] == "bert"].index[0]
        df.loc[idx, "mae"] += 0.1
        df.to_csv(test_dir / "model_summary.csv", index=False)
    elif corruption_type == "missing_pred_column":
        df = pd.read_csv(test_dir / "test_predictions.csv")
        df = df.drop(columns=["pred_distilbert"])
        df.to_csv(test_dir / "test_predictions.csv", index=False)
    elif corruption_type == "rmse_raw_mismatch":
        df = pd.read_csv(test_dir / "model_summary.csv")
        idx = df[df["model"] == "bert"].index[0]
        df.loc[idx, "rmse_raw"] += 0.1
        df.to_csv(test_dir / "model_summary.csv", index=False)
    elif corruption_type == "missing_split_id_column":
        df = pd.read_csv(test_dir / "test_predictions.csv")
        df = df.drop(columns=["split_id"])
        df.to_csv(test_dir / "test_predictions.csv", index=False)
    elif corruption_type == "individual_pred_raw_not_match":
        df = pd.read_csv(test_dir / "preds" / "bert_test.csv")
        df.loc[0, "pred_raw"] += 10.0
        df.to_csv(test_dir / "preds" / "bert_test.csv", index=False)
    elif corruption_type == "dashboard_missing_column":
        df = pd.read_csv(test_dir / "model_summary.csv")
        df = df.drop(columns=["seconds_per_epoch"])
        df.to_csv(test_dir / "model_summary.csv", index=False)
    
    yield test_dir, corruption_type


# ============================================================
# Tests
# ============================================================

def test_valid_data_passes(valid_results_dir):
    """Valid data should pass validation (exit code 0)."""
    with patch.object(sys, 'argv', ['validate_outputs', '--results-dir', str(valid_results_dir)]):
        exit_code = main()
    assert exit_code == 0


@pytest.mark.parametrize("corruption_type,expected_error_substring", [
    ("split_id_mismatch", "split_id mismatch"),
    ("duplicate_review_id", "Duplicate review_id"),
    ("invalid_true_stars", "Invalid true_stars values"),
    ("pred_out_of_range", "out of range"),
    ("abs_err_mismatch", "abs_err.*mismatch"),
    ("model_summary_metric_wrong", "bert.mae.*diff"),
    ("missing_pred_column", "Missing pred column"),
    ("rmse_raw_mismatch", "rmse_raw.*diff"),
    ("missing_split_id_column", "split_id"),
    ("individual_pred_raw_not_match", "rmse_raw.*diff"),
    ("dashboard_missing_column", "seconds_per_epoch"),
])
def test_corruption_detected(corrupted_results_dir, corruption_type, expected_error_substring, capsys):
    """Each corruption should be detected (exit code 1)."""
    results_dir, actual_corruption = corrupted_results_dir
    assert actual_corruption == corruption_type
    
    with patch.object(sys, 'argv', ['validate_outputs', '--results-dir', str(results_dir)]):
        exit_code = main()
    
    assert exit_code == 1, f"Corruption {corruption_type} should fail but exited 0"
    
    captured = capsys.readouterr()
    assert expected_error_substring in captured.out or expected_error_substring in captured.err, \
        f"Expected error substring '{expected_error_substring}' not found in output"


# Unit tests for individual check functions
def test_check_merge_consistency_split_id_mismatch(valid_results_dir):
    data = load_all_data(valid_results_dir, valid_results_dir / "data" / "split_stats.json")
    data["test_predictions"]["split_id"] = "different_split_id"
    split_stats = json.load(open(valid_results_dir / "data" / "split_stats.json"))
    
    errors = check_merge_consistency(data, split_stats)
    assert any("split_id mismatch" in e for e in errors)


def test_check_merge_consistency_duplicate_review_id(valid_results_dir):
    data = load_all_data(valid_results_dir, valid_results_dir / "data" / "split_stats.json")
    df = data["test_predictions"]
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    data["test_predictions"] = df
    split_stats = json.load(open(valid_results_dir / "data" / "split_stats.json"))
    
    errors = check_merge_consistency(data, split_stats)
    assert any("Duplicate review_id" in e for e in errors)


def test_check_merge_consistency_invalid_true_stars(valid_results_dir):
    data = load_all_data(valid_results_dir, valid_results_dir / "data" / "split_stats.json")
    data["test_predictions"].loc[0, "true_stars"] = 0
    split_stats = json.load(open(valid_results_dir / "data" / "split_stats.json"))
    
    errors = check_merge_consistency(data, split_stats)
    assert any("Invalid true_stars values" in e for e in errors)


def test_check_schemas_pred_out_of_range(valid_results_dir):
    data = load_all_data(valid_results_dir, valid_results_dir / "data" / "split_stats.json")
    data["test_predictions"]["pred_bert"] = 6.0
    
    errors = check_schemas(data)
    assert any("pred_bert" in e and "out of range" in e for e in errors)


def test_check_schemas_abs_err_mismatch(valid_results_dir):
    data = load_all_data(valid_results_dir, valid_results_dir / "data" / "split_stats.json")
    data["test_predictions"]["abs_err_bert"] = 9.9
    
    errors = check_schemas(data)
    assert any("abs_err_bert" in e and "mismatch" in e for e in errors)


def test_reconcile_metrics_detects_wrong_mae(valid_results_dir):
    data = load_all_data(valid_results_dir, valid_results_dir / "data" / "split_stats.json")
    idx = data["model_summary"][data["model_summary"]["model"] == "bert"].index[0]
    data["model_summary"].loc[idx, "mae"] += 0.1
    
    errors = reconcile_metrics(data, tolerance=1e-4)
    assert any("bert.mae" in e and "diff" in e for e in errors)


def test_reconcile_metrics_missing_pred_column(valid_results_dir):
    data = load_all_data(valid_results_dir, valid_results_dir / "data" / "split_stats.json")
    data["test_predictions"] = data["test_predictions"].drop(columns=["pred_distilbert"])
    
    errors = reconcile_metrics(data, tolerance=1e-4)
    assert any("Missing pred column pred_distilbert" in e for e in errors)


def test_reconcile_metrics_rmse_raw_mismatch(valid_results_dir):
    data = load_all_data(valid_results_dir, valid_results_dir / "data" / "split_stats.json")
    idx = data["model_summary"][data["model_summary"]["model"] == "bert"].index[0]
    data["model_summary"].loc[idx, "rmse_raw"] += 0.1
    
    errors = reconcile_metrics(data, tolerance=1e-4)
    assert any("bert.rmse_raw" in e and "diff" in e for e in errors)


def test_check_dashboard_readiness_missing_column(valid_results_dir):
    data = load_all_data(valid_results_dir, valid_results_dir / "data" / "split_stats.json")
    data["model_summary"] = data["model_summary"].drop(columns=["seconds_per_epoch"])
    
    errors = check_dashboard_readiness(data)
    assert any("seconds_per_epoch" in e for e in errors)