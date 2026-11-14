"""Kiểm tra src/baseline.py theo spec docs/define&designt/02_evaluation.md
và kế hoạch kiểm thử docs/tests/02_evaluation_tests.md (D17–D23).

Chạy: pytest tests/test_baseline.py -v -m "not slow"
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from src.baseline import (
    ALPHA_GRID,
    MAX_FEATURES_GRID,
    MIN_DF_GRID,
    NGRAM_RANGE_GRID,
    build_baseline,
    fit_baseline,
    load_baseline,
    predict_baseline,
    save_baseline,
)
from src.metrics import rmse


# ---------------- Fixtures dữ liệu giả ----------------


@pytest.fixture()
def fake_train_df():
    rows = []
    i = 0
    # Từng từ khen -> 5 sao, từng từ chê -> 1 sao, trung bình -> 3 sao
    for text, stars in [
        ("great place love it amazing", 5),
        ("good food nice service", 5),
        ("awesome experience recommended", 5),
        ("excellent great good", 5),
        ("terrible bad awful hate", 1),
        ("bad terrible worst", 1),
        ("awful hate bad", 1),
        ("poor bad terrible", 1),
        ("okay fine average", 3),
        ("nothing special average", 3),
    ] * 10:
        i += 1
        rows.append({"text": text, "stars": stars})
    return pd.DataFrame(rows)


@pytest.fixture()
def fake_val_df():
    rows = [
        {"text": "great amazing love it", "stars": 5},
        {"text": "terrible bad hate", "stars": 1},
        {"text": "good nice service", "stars": 5},
        {"text": "awful worst bad", "stars": 1},
        {"text": "okay average fine", "stars": 3},
        {"text": "excellent great", "stars": 5},
        {"text": "bad terrible", "stars": 1},
        {"text": "nothing special", "stars": 3},
    ]
    return pd.DataFrame(rows)


# ---------------- Nhóm F: Baseline ----------------


def test_d17_baseline_shape_and_clip(fake_train_df, fake_val_df):
    out = fit_baseline(fake_train_df, fake_val_df, seed=42)
    pipeline = out["pipeline"]
    texts = ["great amazing", "terrible bad", "okay fine"]
    preds = predict_baseline(pipeline, texts)
    assert preds.shape == (3,)
    assert np.all((preds >= 1.0) & (preds <= 5.0))


def test_d18_baseline_selection_uses_val_only(fake_train_df, fake_val_df):
    # Chữ ký không có tham số Test
    sig = inspect.signature(fit_baseline)
    assert "test" not in "".join(sig.parameters)

    grid = {
        "max_features": [5000],
        "ngram_range": [(1, 1)],
        "alpha": [0.1, 1e6],
    }
    out = fit_baseline(fake_train_df, fake_val_df, param_grid=grid, seed=42)

    results = out["grid_results"]
    assert "val_rmse" in results.columns
    best_row = results.loc[results["val_rmse"].idxmin()]
    assert out["best_params"]["alpha"] == best_row["alpha"]
    assert out["best_params"]["alpha"] == 0.1


def test_d19_baseline_reproducible(fake_train_df, fake_val_df):
    out1 = fit_baseline(fake_train_df, fake_val_df, seed=42)
    out2 = fit_baseline(fake_train_df, fake_val_df, seed=42)
    p1 = predict_baseline(out1["pipeline"], fake_val_df["text"].tolist())
    p2 = predict_baseline(out2["pipeline"], fake_val_df["text"].tolist())
    np.testing.assert_array_equal(p1, p2)


def test_d20_baseline_learns_signal(fake_train_df, fake_val_df):
    out = fit_baseline(fake_train_df, fake_val_df, seed=42)
    y_val = fake_val_df["stars"].to_numpy(dtype=np.float64)
    mean_val = float(np.mean(y_val))
    preds_mean = np.full_like(y_val, mean_val)
    rmse_mean = rmse(y_val, preds_mean)

    preds = predict_baseline(out["pipeline"], fake_val_df["text"].tolist())
    rmse_base = rmse(y_val, preds)
    assert rmse_base <= rmse_mean + 1e-6


def test_d21_baseline_returns_timing_and_keys(fake_train_df, fake_val_df):
    out = fit_baseline(fake_train_df, fake_val_df, seed=42)
    for k in ("pipeline", "best_params", "grid_results", "fit_seconds", "grid_seconds"):
        assert k in out, f"thiếu khoá {k}"
    assert out["fit_seconds"] > 0
    assert out["grid_seconds"] >= out["fit_seconds"]
    gr = out["grid_results"]
    for c in ("max_features", "ngram_range", "min_df", "alpha", "val_rmse"):
        assert c in gr.columns
    assert len(gr) == 84  # 3 × 2 × 2 × 7


def test_d22_baseline_invalid_inputs(fake_train_df, fake_val_df):
    bad_train = fake_train_df.drop(columns=["stars"])
    with pytest.raises(ValueError):
        fit_baseline(bad_train, fake_val_df, seed=42)

    bad_val = fake_val_df.drop(columns=["text"])
    with pytest.raises(ValueError):
        fit_baseline(fake_train_df, bad_val, seed=42)

    with pytest.raises(ValueError):
        fit_baseline(fake_train_df, fake_val_df, param_grid={"alpha": []}, seed=42)

    with pytest.raises(ValueError):
        predict_baseline(build_baseline(), [])

    with pytest.raises(ValueError):
        predict_baseline(build_baseline(), ["ok", 123])


def test_d23_baseline_save_load(fake_train_df, fake_val_df, tmp_path):
    out = fit_baseline(fake_train_df, fake_val_df, seed=42)

    # Lưu val artifact
    val_path = tmp_path / "baseline_val.pkl"
    save_baseline(out, val_path)
    assert val_path.exists()

    # Lưu test artifact (cùng pipeline, dùng cho dự đoán Test)
    test_path = tmp_path / "baseline_test.pkl"
    save_baseline(out, test_path, overwrite=True)
    assert test_path.exists()

    # Lặp lại dự đoán trước và sau khi nạp
    texts = fake_val_df["text"].tolist()
    preds_before = predict_baseline(out["pipeline"], texts)

    loaded = load_baseline(val_path)
    preds_after = predict_baseline(loaded["pipeline"], texts)
    np.testing.assert_array_equal(preds_before, preds_after)

    for k in ("best_params", "fit_seconds", "grid_seconds"):
        assert loaded[k] == out[k]
    pd.testing.assert_frame_equal(loaded["grid_results"], out["grid_results"])

    # Lưu lần hai không overwrite -> FileExistsError
    with pytest.raises(FileExistsError):
        save_baseline(out, val_path)

    # Nạp file không có -> FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_baseline(tmp_path / "khong_ton_tai.pkl")


def test_default_grid_84_combinations():
    assert len(MAX_FEATURES_GRID) == 3
    assert len(NGRAM_RANGE_GRID) == 2
    assert len(MIN_DF_GRID) == 2
    assert len(ALPHA_GRID) == 7
    assert 3 * 2 * 2 * 7 == 84
    assert ALPHA_GRID == pytest.approx([0.001, 0.01, 0.1, 1, 10, 100, 1000])


def test_build_baseline_pipeline(fake_train_df, fake_val_df):
    pipe = build_baseline(seed=42)
    pipe.fit(fake_train_df["text"].tolist(), fake_train_df["stars"].to_numpy())
    preds = predict_baseline(pipe, fake_val_df["text"].tolist())
    assert preds.shape == (len(fake_val_df),)
    assert np.all((preds >= 1.0) & (preds <= 5.0))
