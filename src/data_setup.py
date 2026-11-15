"""Data setup: prepare Train/Val/Test splits và DataLoader cho Text Regression (Yelp).

Spec: docs/define&designt/01_data.md
Tests: docs/tests/01_data_tests.md
"""

import hashlib
import json
import math
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

SPLITS = ("train", "val", "test")
STARS = (1, 2, 3, 4, 5)
REQUIRED_FIELDS = ("review_id", "user_id", "business_id", "stars", "text")
PARQUET_COLUMNS = ["review_id", "user_id", "business_id", "text", "stars"]


def _clean_text(text) -> str:
    """Chuẩn hoá văn bản: strip -> lower -> NFC -> collapse whitespace.

    Trả về chuỗi rỗng nếu text không phải chuỗi hoặc chỉ gồm whitespace.
    """
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFC", text.strip().lower())
    return " ".join(normalized.split())


def _parse_stars(value):
    """Chấp nhận int hoặc float nguyên trong [1, 5]; ngược lại trả về None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        star = value
    elif isinstance(value, float) and value.is_integer():
        star = int(value)
    else:
        return None
    return star if star in STARS else None


def _allocate_counts(total: int, weights) -> list:
    """Phân bổ `total` mẫu theo `weights` (tổng = 1.0), tổng kết quả = total.

    Quy tắc: floor từng phần, phần dư cộng dần cho chỉ số có phần lẻ lớn nhất;
    hoà thì chỉ số nhỏ hơn được ưu tiên.
    """
    weights = [float(w) for w in weights]
    quotas = [total * w for w in weights]
    counts = [math.floor(q) for q in quotas]
    remainder = total - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda i: (-(quotas[i] - counts[i]), i),
    )
    for i in order[:remainder]:
        counts[i] += 1
    return counts


def _iter_clean_rows(review_path: Path, chunk_size: int):
    """Đọc review.json theo khối, trả về generator các dict đã làm sạch.

    Bỏ qua dòng không parse được JSON, thiếu trường bắt buộc, stars không hợp
    lệ, hoặc text rỗng sau chuẩn hoá. Raise ValueError nếu thiếu trường bắt buộc.
    """
    with open(review_path, "r", encoding="utf-8") as f:
        chunk = []
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Dòng {line_no} không phải JSON hợp lệ: {exc}") from exc
            missing = [k for k in REQUIRED_FIELDS if k not in row]
            if missing:
                raise ValueError(
                    f"Dòng {line_no} thiếu trường bắt buộc: {', '.join(missing)}"
                )
            star = _parse_stars(row["stars"])
            text = _clean_text(row["text"])
            if star is None or not text:
                continue
            chunk.append(
                {
                    "review_id": str(row["review_id"]),
                    "user_id": str(row["user_id"]),
                    "business_id": str(row["business_id"]),
                    "text": text,
                    "stars": star,
                }
            )
            if len(chunk) >= chunk_size:
                yield from chunk
                chunk = []
        if chunk:
            yield from chunk


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_outputs(output_dir: Path, overwrite: bool) -> None:
    outputs = [output_dir / f"{split}.parquet" for split in SPLITS]
    outputs.append(output_dir / "split_stats.json")
    existing = [p.name for p in outputs if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"File đã tồn tại trong {output_dir}: {', '.join(existing)}. "
            "Truyền overwrite=True để ghi đè, hoặc chọn output_dir khác."
        )


def prepare_splits(
    review_path: str,
    output_dir: str,
    n_samples: int = 60000,
    ratios=(0.70, 0.15, 0.15),
    seed: int = 42,
    chunk_size: int = 100000,
    overwrite: bool = False,
) -> dict:
    """Tạo ba tập Train/Val/Test phân tầng theo sao từ review.json.

    Trả về dict nội dung split_stats.json. Xem docs/define&designt/01_data.md.
    """
    review_path = Path(review_path)
    output_dir = Path(output_dir)
    if not review_path.exists():
        raise FileNotFoundError(f"Không tìm thấy review.json: {review_path}")
    ratios = tuple(float(r) for r in ratios)
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios phải gồm 3 giá trị dương cộng bằng 1, nhận: {ratios}")
    if n_samples <= 0:
        raise ValueError(f"n_samples phải > 0, nhận: {n_samples}")
    _check_outputs(output_dir, overwrite)

    rng = np.random.default_rng(seed)

    # Lượt 1: đếm số review hợp lệ theo mức sao.
    counts_by_star = {star: 0 for star in STARS}
    total_valid = 0
    for row in _iter_clean_rows(review_path, chunk_size):
        counts_by_star[row["stars"]] += 1
        total_valid += 1
    if total_valid < n_samples:
        raise ValueError(
            f"File chỉ có {total_valid} review hợp lệ < n_samples={n_samples}"
        )

    # Hạn ngạch mẫu mỗi mức sao (tổng = n_samples).
    weights = [counts_by_star[s] / total_valid for s in STARS]
    sample_per_star = dict(zip(STARS, _allocate_counts(n_samples, weights)))

    # Chọn ngẫu nhiên m_c vị trí trong N_c review hợp lệ của mỗi mức sao.
    chosen = {}
    for star in STARS:
        m = sample_per_star[star]
        if m == 0:
            chosen[star] = set()
            continue
        positions = rng.choice(counts_by_star[star], size=m, replace=False)
        chosen[star] = set(int(p) for p in positions)

    # Lượt 2: đọc lại file, giữ các review thuộc tập đã chọn.
    seen = {star: 0 for star in STARS}
    rows_by_star = {star: [] for star in STARS}
    for row in _iter_clean_rows(review_path, chunk_size):
        star = row["stars"]
        idx = seen[star]
        seen[star] += 1
        if idx in chosen[star]:
            rows_by_star[star].append(row)

    # Chia mỗi mức sao thành Train/Val/Test theo ratios, kiểm tra mỗi tập
    # có ít nhất 1 mẫu cho mỗi mức sao.
    split_rows = {split: [] for split in SPLITS}
    quota = {split: {} for split in SPLITS}
    for star in STARS:
        rows = rows_by_star[star]
        order = rng.permutation(len(rows))
        rows = [rows[i] for i in order]
        alloc = _allocate_counts(len(rows), ratios)
        if min(alloc) < 1:
            raise ValueError(
                f"Mức sao {star} chỉ có {len(rows)} mẫu, không đủ để mỗi tập "
                "Train/Val/Test có ít nhất 1 mẫu"
            )
        pos = 0
        for split, count in zip(SPLITS, alloc):
            split_rows[split].extend(rows[pos : pos + count])
            quota[split][star] = count
            pos += count

    # Loại review ở Val/Test có text trùng với Train (sau chuẩn hoá).
    train_texts = {row["text"] for row in split_rows["train"]}
    duplicate_removed = {}
    for split in ("val", "test"):
        kept = [r for r in split_rows[split] if r["text"] not in train_texts]
        duplicate_removed[split] = len(split_rows[split]) - len(kept)
        split_rows[split] = kept

    # Xáo trộn thứ tự dòng trong mỗi tập rồi ghi parquet.
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    for split in SPLITS:
        rows = split_rows[split]
        perm = rng.permutation(len(rows))
        rows = [rows[i] for i in perm]
        df = pd.DataFrame(rows, columns=PARQUET_COLUMNS)
        df["stars"] = df["stars"].astype("int64")
        frames[split] = df
        df.to_parquet(output_dir / f"{split}.parquet", index=False)

    # Thống kê phục vụ split_stats.json.
    def overlap_rate(ids_subset, ids_train) -> float:
        subset_unique = set(ids_subset)
        if not subset_unique:
            return 0.0
        return len(subset_unique & set(ids_train)) / len(subset_unique)

    train_users = frames["train"]["user_id"].tolist()
    train_biz = frames["train"]["business_id"].tolist()
    stars_per_split = {
        split: {str(s): int((frames[split]["stars"] == s).sum()) for s in STARS}
        for split in SPLITS
    }
    test_review_ids = sorted(frames["test"]["review_id"].tolist())
    split_id = hashlib.sha256("\n".join(test_review_ids).encode("utf-8")).hexdigest()[:12]
    file_hash = _file_sha256(review_path)
    now = datetime.now(timezone.utc)

    stats = {
        "split_id": split_id,
        "seed": seed,
        "data_version": f"yelp_{now.year:04d}_{now.month:02d}_{file_hash[:8]}",
        "n_samples": n_samples,
        "train_size": len(frames["train"]),
        "val_size": len(frames["val"]),
        "test_size": len(frames["test"]),
        "stars_per_split": stars_per_split,
        "stars_distribution_original": {
            str(s): counts_by_star[s] for s in STARS
        },
        "duplicate_text_removed": duplicate_removed,
        "user_id_overlap_rate": {
            "train_val": overlap_rate(frames["val"]["user_id"].tolist(), train_users),
            "train_test": overlap_rate(frames["test"]["user_id"].tolist(), train_users),
        },
        "business_id_overlap_rate": {
            "train_val": overlap_rate(frames["val"]["business_id"].tolist(), train_biz),
            "train_test": overlap_rate(frames["test"]["business_id"].tolist(), train_biz),
        },
        "quota": {split: {str(s): quota[split][s] for s in STARS} for split in SPLITS},
        "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(output_dir / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


class ReviewDataset(Dataset):
    """Dataset biến một dòng parquet thành input cho transformer backbone."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        missing = [c for c in PARQUET_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame thiếu cột: {', '.join(missing)}")
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0].long(),
            "attention_mask": enc["attention_mask"][0].long(),
            "labels": torch.tensor(float(row["stars"]), dtype=torch.float32),
            "review_id": str(row["review_id"]),
        }


def create_dataloaders(
    data_dir: str,
    tokenizer,
    batch_size: int = 16,
    max_length: int = 256,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Đọc ba file parquet và trả về (train_loader, val_loader, test_loader)."""
    data_dir = Path(data_dir)
    frames = []
    for split in SPLITS:
        path = data_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Thiếu file dữ liệu: {path}")
        frames.append(pd.read_parquet(path))

    datasets = [ReviewDataset(df, tokenizer, max_length) for df in frames]
    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        datasets[0],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        datasets[1],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        datasets[2],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
