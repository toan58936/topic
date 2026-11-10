import json
import zlib
from pathlib import Path

import pytest
import torch


def _make_review(i: int, stars: int, text: str, user_id: str = None, business_id: str = None):
    return {
        "review_id": f"review_{i:05d}",
        "user_id": user_id or f"user_{i % 97:04d}",
        "business_id": business_id or f"biz_{i % 53:04d}",
        "stars": stars,
        "text": text,
    }


def build_fake_rows():
    """~300 dòng hợp lệ, phân phối lệch về 4-5 sao, kèm vài dòng bẩn.

    Phân phối hợp lệ: {1:20, 2:25, 3:50, 4:95, 5:110} = 300 dòng.
    """
    rows = []
    i = 0
    star_counts = {1: 20, 2: 25, 3: 50, 4: 95, 5: 110}
    for star, count in star_counts.items():
        for _ in range(count):
            i += 1
            rows.append(_make_review(i, star, f"This is review number {i} with some words about the place."))
    # Dòng text rỗng và chỉ whitespace (bị loại khi làm sạch)
    rows.append(_make_review(9001, 5, ""))
    rows.append(_make_review(9002, 4, "   \n\t  "))
    rows.append(_make_review(9003, 3, "   "))
    # Dòng text nhiều khoảng trắng thừa (được chuẩn hoá)
    i += 1
    rows.append(_make_review(i, 2, "  lots    of\n\n   spaces\t\there  "))
    # Dòng stars dạng float nguyên (hợp lệ)
    i += 1
    row = _make_review(i, 5, "float stars row here")
    row["stars"] = 4.0
    rows.append(row)
    # Dòng stars không hợp lệ (bị loại)
    i += 1
    row = _make_review(i, 1, "bad stars row")
    row["stars"] = 6
    rows.append(row)
    # Trộn ngẫu nhiên thứ tự dòng (cố định, không dùng random)
    rows = rows[::2] + rows[1::2]
    return rows


@pytest.fixture()
def fake_review_rows():
    return build_fake_rows()


@pytest.fixture()
def fake_review_file(tmp_path, fake_review_rows):
    path = tmp_path / "review.json"
    with open(path, "w", encoding="utf-8") as f:
        for row in fake_review_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


class FakeTokenizer:
    """Tokenizer giả: mỗi từ -> một id, thêm BOS/EOS, pad bằng 0.

    Trả về dict {"input_ids": [1, L], "attention_mask": [1, L]}.
    """

    bos, eos, pad = 1, 2, 0

    def __call__(self, text, truncation=False, max_length=None, padding=None,
                 return_tensors=None, **kwargs):
        words = text.split()
        ids = [self.bos] + [(zlib.crc32(w.encode("utf-8")) % 500) + 3 for w in words] + [self.eos]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length" and max_length is not None:
            pad_len = max_length - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad] * pad_len
                mask = mask + [0] * pad_len
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([mask], dtype=torch.long),
        }


@pytest.fixture()
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture()
def prepared_dir(tmp_path, fake_review_file):
    """Chạy prepare_splits một lần, trả về (output_dir, stats)."""
    from src.data_setup import prepare_splits

    out = tmp_path / "splits"
    stats = prepare_splits(str(fake_review_file), str(out), n_samples=100, seed=42)
    return out, stats
