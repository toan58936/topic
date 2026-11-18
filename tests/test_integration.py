#!/usr/bin/env python3
"""Integration tests for end-to-end validation of Yelp review rating outputs."""

import pytest
from pathlib import Path

from scripts.validate_outputs import (
    load_all_data, check_merge_consistency, check_schemas,
    reconcile_metrics, check_dashboard_readiness
)


@pytest.fixture(scope="session")
def results_dir():
    """Thư mục kết quả (local reports/ hoặc mount Drive)."""
    return Path("reports")


@pytest.fixture(scope="session")
def split_stats_path(results_dir):
    """Đường dẫn split_stats.json."""
    return results_dir.parent / "data" / "split_stats.json"


@pytest.fixture(scope="session")
def validation_data(results_dir, split_stats_path):
    """Load all validation data."""
    return load_all_data(results_dir, split_stats_path)


def test_merge_consistency(validation_data, split_stats_path):
    """Test merge consistency: split_id, review_id unique, true_stars valid."""
    import json
    split_stats = json.load(open(split_stats_path))
    errors = check_merge_consistency(validation_data, split_stats)
    assert not errors, f"Merge consistency failed:\n" + "\n".join(errors)


def test_schemas(validation_data):
    """Test schema validation for all output files."""
    errors = check_schemas(validation_data)
    assert not errors, f"Schema validation failed:\n" + "\n".join(errors)


def test_metric_reconciliation(validation_data):
    """Test metric reconciliation against recomputed values."""
    errors = reconcile_metrics(validation_data, tolerance=1e-4)
    assert not errors, f"Metric reconciliation failed:\n" + "\n".join(errors)


def test_dashboard_readiness(validation_data):
    """Test dashboard readiness: required columns, no NaN in required cols."""
    errors = check_dashboard_readiness(validation_data)
    assert not errors, f"Dashboard readiness failed:\n" + "\n".join(errors)