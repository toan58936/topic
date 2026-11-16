import sys

from scripts.validate_outputs import (
    check_merge_consistency,
    check_schemas,
    load_all_data,
    main,
    reconcile_metrics,
)


def test_valid_data_passes(valid_results_dir, monkeypatch, capsys):
    """Du lieu hop le phai vuot qua validation."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_outputs", "--results-dir", str(valid_results_dir)],
    )

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "ALL CHECKS PASSED" in captured.out


def test_corruption_detected(corrupted_results_dir, monkeypatch):
    """Ca 11 tinh huong du lieu hong deu phai bi phat hien."""
    results_dir, corruption_type = corrupted_results_dir

    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_outputs", "--results-dir", str(results_dir)],
    )

    exit_code = main()

    assert exit_code == 1, (
        f"Corruption {corruption_type} should fail but exited 0"
    )


def test_check_merge_consistency_split_id_mismatch(valid_results_dir):
    """Phat hien split_id khong khop."""
    split_stats_path = valid_results_dir.parent / "data" / "split_stats.json"
    data = load_all_data(valid_results_dir, split_stats_path)

    data["test_predictions"]["split_id"] = "different_split_id"

    errors = check_merge_consistency(data, data["split_stats"])

    assert any("split_id mismatch" in error for error in errors)


def test_reconcile_metrics_detects_wrong_mae(valid_results_dir):
    """Phat hien MAE trong model_summary bi sai."""
    split_stats_path = valid_results_dir.parent / "data" / "split_stats.json"
    data = load_all_data(valid_results_dir, split_stats_path)

    idx = data["model_summary"][
        data["model_summary"]["model"] == "bert"
    ].index[0]
    data["model_summary"].loc[idx, "mae"] += 0.1

    errors = reconcile_metrics(data, tolerance=1e-4)

    assert any(
        "bert.mae" in error and "diff" in error
        for error in errors
    )


def test_check_schemas_pred_out_of_range(valid_results_dir):
    """Phat hien gia tri du doan vuot khoang 1-5."""
    split_stats_path = valid_results_dir.parent / "data" / "split_stats.json"
    data = load_all_data(valid_results_dir, split_stats_path)

    data["test_predictions"]["pred_bert"] = 6.0

    errors = check_schemas(data)

    assert any(
        "pred_bert" in error and "out of range" in error
        for error in errors
    )


def test_check_schemas_abs_err_mismatch(valid_results_dir):
    """Phat hien abs_err khong khop voi true_stars va pred."""
    split_stats_path = valid_results_dir.parent / "data" / "split_stats.json"
    data = load_all_data(valid_results_dir, split_stats_path)

    data["test_predictions"]["abs_err_bert"] = 9.9

    errors = check_schemas(data)

    assert any(
        "abs_err_bert" in error and "mismatch" in error
        for error in errors
    )
