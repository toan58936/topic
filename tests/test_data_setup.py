"""Tests cho src/data_setup.py — spec docs/tests/01_data_tests.md."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from src.data_setup import (
    ReviewDataset,
    _allocate_counts,
    create_dataloaders,
    prepare_splits,
)

SPLITS = ("train", "val", "test")
STARS = (1, 2, 3, 4, 5)


def _read_split(output_dir, split):
    return pd.read_parquet(Path(output_dir) / f"{split}.parquet")


def _all_ids(output_dir):
    return {s: set(_read_split(output_dir, s)["review_id"]) for s in SPLITS}


def _run_prepare(review_path, out_dir, **kwargs):
    kwargs.setdefault("n_samples", 100)
    kwargs.setdefault("seed", 42)
    return prepare_splits(str(review_path), str(out_dir), **kwargs)


# ---------------- Nhóm A: Chia phân tầng đúng ----------------

def test_d1_disjoint_and_complete(fake_review_file, tmp_path):
    out = tmp_path / "splits"
    _run_prepare(fake_review_file, out)
    ids = _all_ids(out)
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    union = ids["train"] | ids["val"] | ids["test"]
    assert len(union) == 100


def test_d2_stratified_per_split(fake_review_file, tmp_path):
    out = tmp_path / "splits"
    stats = _run_prepare(fake_review_file, out)
    for split in SPLITS:
        df = _read_split(out, split)
        for star in STARS:
            actual = int((df["stars"] == star).sum())
            expected = stats["quota"][split][str(star)]
            assert abs(actual - expected) <= 1


def test_d3_sample_proportional(fake_review_file, tmp_path):
    out = tmp_path / "splits"
    stats = _run_prepare(fake_review_file, out)
    original = {int(k): v for k, v in stats["stars_distribution_original"].items()}
    total_original = sum(original.values())
    sampled = {star: 0 for star in STARS}
    for split in SPLITS:
        df = _read_split(out, split)
        for star in STARS:
            sampled[star] += int((df["stars"] == star).sum())
    for star in STARS:
        expected = 100 * original[star] / total_original
        assert abs(sampled[star] - expected) <= 1


# ---------------- Nhóm B: Tái lập và phần dư ----------------

def test_d4_reproducible_by_seed(fake_review_file, tmp_path):
    out1, out2, out3 = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    np.random.seed(123)
    _run_prepare(fake_review_file, out1, seed=42)
    np.random.seed(999)
    _run_prepare(fake_review_file, out2, seed=42)
    _run_prepare(fake_review_file, out3, seed=7)
    for split in SPLITS:
        ids1 = _read_split(out1, split)["review_id"].tolist()
        ids2 = _read_split(out2, split)["review_id"].tolist()
        ids3 = _read_split(out3, split)["review_id"].tolist()
        assert ids1 == ids2
        assert ids1 != ids3


def test_d5_remainder_distribution():
    assert _allocate_counts(10, (0.7, 0.15, 0.15)) == [7, 2, 1]
    assert sum(_allocate_counts(10, (0.7, 0.15, 0.15))) == 10
    assert sum(_allocate_counts(7, (0.5, 0.3, 0.2))) == 7


# ---------------- Nhóm C: Làm sạch và kiểm tra đầu vào ----------------

def test_d6_cleaning(tmp_path):
    # File nhỏ: 7 review/sao = 35 dòng, một dòng rỗng (star 1), một dòng
    # whitespace-thừa (star 3). n_samples=34 = số review hợp lệ -> chọn hết.
    rows = []
    i = 0
    for star in STARS:
        for _ in range(7):
            i += 1
            rows.append(
                {
                    "review_id": f"cl_{i:03d}",
                    "user_id": f"u{i}",
                    "business_id": f"b{i}",
                    "stars": star,
                    "text": f"cleaning test review {i} for star {star}",
                }
            )
    rows[4]["text"] = ""                       # star 1, rỗng -> bị loại
    rows[16]["text"] = "  lots    of\n\n   spaces\t\there  "  # star 3
    path = tmp_path / "review_dirty.json"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    out = tmp_path / "splits"
    _run_prepare(path, out, n_samples=34)
    all_df = pd.concat([_read_split(out, s) for s in SPLITS])
    assert len(all_df) == 34
    assert not all_df["text"].isna().any()
    assert (all_df["text"].str.strip() != "").all()
    assert not all_df["text"].str.contains(r"\s{2,}|\n|\t", regex=True).any()
    messy = all_df[all_df["review_id"] == "cl_017"]
    assert len(messy) == 1
    assert messy.iloc[0]["text"] == "lots of spaces here"


def test_d7_invalid_inputs(fake_review_file, tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_splits(str(tmp_path / "nope.json"), str(tmp_path / "o1"))
    with pytest.raises(ValueError):
        _run_prepare(fake_review_file, tmp_path / "o2", ratios=(0.5, 0.3, 0.3))
    with pytest.raises(ValueError):
        _run_prepare(fake_review_file, tmp_path / "o3", ratios=(0.7, 0.3, 0.0))
    with pytest.raises(ValueError):
        _run_prepare(fake_review_file, tmp_path / "o4", n_samples=0)
    with pytest.raises(ValueError):
        _run_prepare(fake_review_file, tmp_path / "o5", n_samples=100000)


def test_d8_no_silent_overwrite(fake_review_file, tmp_path):
    out = tmp_path / "splits"
    _run_prepare(fake_review_file, out)
    with pytest.raises(FileExistsError):
        _run_prepare(fake_review_file, out)
    stats = _run_prepare(fake_review_file, out, overwrite=True)
    assert stats["n_samples"] == 100


def test_d9_stats_file(fake_review_file, tmp_path):
    out = tmp_path / "splits"
    _run_prepare(fake_review_file, out)
    stats = json.loads((out / "split_stats.json").read_text(encoding="utf-8"))
    for key in (
        "split_id", "seed", "data_version", "n_samples",
        "train_size", "val_size", "test_size",
        "stars_per_split", "stars_distribution_original",
        "duplicate_text_removed",
        "user_id_overlap_rate", "business_id_overlap_rate",
        "created_at",
    ):
        assert key in stats, f"thiếu khoá {key}"
    assert len(stats["split_id"]) == 12
    total = sum(stats["stars_per_split"][s][str(st)] for s in SPLITS for st in STARS)
    assert total == stats["train_size"] + stats["val_size"] + stats["test_size"]
    for split in ("val", "test"):
        assert "train_" + split in stats["user_id_overlap_rate"]
        assert "train_" + split in stats["business_id_overlap_rate"]


def test_d15_no_text_overlap_val_test_vs_train(fake_review_file, tmp_path):
    """Val/Test không chứa text trùng với Train (sau chuẩn hoá)."""
    out = tmp_path / "splits"
    _run_prepare(fake_review_file, out)
    train_texts = set(_read_split(out, "train")["text"])
    for split in ("val", "test"):
        texts = set(_read_split(out, split)["text"])
        assert texts.isdisjoint(train_texts)


# ---------------- Nhóm D: Dataset và DataLoader ----------------

def test_d10_item_shapes_dtypes(fake_tokenizer):
    df = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(5)],
            "user_id": [f"u{i}" for i in range(5)],
            "business_id": [f"b{i}" for i in range(5)],
            "text": [f"review text number {i}" for i in range(5)],
            "stars": [1, 3, 5, 2, 4],
        }
    )
    ds = ReviewDataset(df, fake_tokenizer, max_length=256)
    item = ds[0]
    assert item["input_ids"].shape == (256,)
    assert item["input_ids"].dtype == torch.long
    assert item["attention_mask"].shape == (256,)
    assert item["attention_mask"].dtype == torch.long
    assert item["labels"].dtype == torch.float32
    assert float(item["labels"]) == 1.0
    assert isinstance(item["review_id"], str)


def test_d11_truncate_and_pad(fake_tokenizer):
    long_text = " ".join(["word"] * 300)
    df = pd.DataFrame(
        {
            "review_id": ["long", "short"],
            "user_id": ["u1", "u2"],
            "business_id": ["b1", "b2"],
            "text": [long_text, "hi there"],
            "stars": [5, 1],
        }
    )
    ds = ReviewDataset(df, fake_tokenizer, max_length=256)
    long_item = ds[0]
    short_item = ds[1]
    assert long_item["input_ids"].shape == (256,)
    assert long_item["attention_mask"].sum().item() == 256  # toàn token thật
    assert short_item["input_ids"].shape == (256,)
    pad_positions = short_item["attention_mask"] == 0
    assert pad_positions.sum().item() > 0
    assert short_item["input_ids"][pad_positions].eq(0).all()


def test_d12_loader_flags_and_batch(prepared_dir, fake_tokenizer):
    data_dir, _ = prepared_dir
    train_loader, val_loader, test_loader = create_dataloaders(
        str(data_dir), fake_tokenizer, batch_size=16
    )
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)
    batch = next(iter(val_loader))
    assert batch["input_ids"].shape == (16, 256)
    assert batch["attention_mask"].shape == (16, 256)
    assert batch["labels"].shape == (16,)
    assert batch["labels"].dtype == torch.float32
    assert isinstance(batch["review_id"], list)
    assert all(isinstance(r, str) for r in batch["review_id"])


def _epoch_ids(loader):
    ids = []
    for batch in loader:
        ids.extend(batch["review_id"])
    return ids


def test_d13_loader_order_stable(prepared_dir, fake_tokenizer):
    data_dir, _ = prepared_dir
    _, val1, test1 = create_dataloaders(str(data_dir), fake_tokenizer, batch_size=16, seed=42)
    _, val2, test2 = create_dataloaders(str(data_dir), fake_tokenizer, batch_size=16, seed=42)
    _, _, test3 = create_dataloaders(str(data_dir), fake_tokenizer, batch_size=16, seed=42)
    _, _, test_diff = create_dataloaders(str(data_dir), fake_tokenizer, batch_size=16, seed=7)

    # Val/Test: thứ tự ổn định giữa hai lần duyệt
    assert _epoch_ids(val1) == _epoch_ids(val2)
    assert _epoch_ids(val1) == _epoch_ids(val1)
    assert _epoch_ids(test1) == _epoch_ids(test2)
    assert _epoch_ids(test1) == _epoch_ids(test3)

    # Train: hai loader cùng seed -> cùng thứ tự; khác seed -> khác thứ tự
    train_a1, _, _ = create_dataloaders(str(data_dir), fake_tokenizer, batch_size=16, seed=42)
    train_a2, _, _ = create_dataloaders(str(data_dir), fake_tokenizer, batch_size=16, seed=42)
    train_b, _, _ = create_dataloaders(str(data_dir), fake_tokenizer, batch_size=16, seed=7)
    assert _epoch_ids(train_a1) == _epoch_ids(train_a2)
    assert _epoch_ids(train_a1) != _epoch_ids(train_b)

    # Cùng một train_loader duyệt hai lần -> thứ tự khác (xáo trộn mỗi epoch)
    first = _epoch_ids(train_a1)
    second = _epoch_ids(train_a1)
    assert first != second
    assert sorted(first) == sorted(second)


def test_d14_missing_files_or_columns(prepared_dir, fake_tokenizer, tmp_path):
    data_dir, _ = prepared_dir
    # Thiếu một file parquet
    broken = tmp_path / "broken"
    broken.mkdir()
    for split in ("train", "val"):
        (broken / f"{split}.parquet").write_bytes(
            (Path(data_dir) / f"{split}.parquet").read_bytes()
        )
    with pytest.raises(FileNotFoundError):
        create_dataloaders(str(broken), fake_tokenizer)
    # File parquet thiếu cột stars
    bad = tmp_path / "badcols"
    bad.mkdir()
    for split in SPLITS:
        df = _read_split(data_dir, split).drop(columns=["stars"])
        df.to_parquet(bad / f"{split}.parquet", index=False)
    with pytest.raises(ValueError):
        create_dataloaders(str(bad), fake_tokenizer)


# ---------------- Nhóm S: Dữ liệu thật (chậm) ----------------

def _real_review_path():
    env_path = os.environ.get("REVIEW_JSON")
    candidates = [env_path] if env_path else []
    candidates += ["review.json", "data/review.json", "../review.json"]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None


@pytest.mark.slow
def test_s1_real_file_sizes(tmp_path):
    path = _real_review_path()
    if path is None:
        pytest.skip("Không có review.json thật")
    out = tmp_path / "real_splits"
    stats = prepare_splits(str(path), str(out), n_samples=60000, seed=42)
    assert 41500 <= stats["train_size"] <= 42500
    assert 8500 <= stats["val_size"] <= 9500
    assert 8500 <= stats["test_size"] <= 9500
    for split in SPLITS:
        df = _read_split(out, split)
        assert not df["text"].isna().any()
        assert (df["text"].str.strip() != "").all()


@pytest.mark.slow
def test_s2_star_distribution_matches_file(tmp_path):
    path = _real_review_path()
    if path is None:
        pytest.skip("Không có review.json thật")
    out = tmp_path / "real_splits2"
    stats = prepare_splits(str(path), str(out), n_samples=60000, seed=42)
    original = {int(k): v for k, v in stats["stars_distribution_original"].items()}
    total_original = sum(original.values())
    sampled = {star: 0 for star in STARS}
    for split in SPLITS:
        df = _read_split(out, split)
        for star in STARS:
            sampled[star] += int((df["stars"] == star).sum())
    for star in STARS:
        expected = 60000 * original[star] / total_original
        assert abs(sampled[star] - expected) <= 1
