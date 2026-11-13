import json
import zlib
from pathlib import Path

import pytest


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
    """Tokenizer giả dùng chung cho mọi test.

    Mỗi từ -> một id ổn định trong [5, 1000), thêm BOS=1/EOS=2, pad=0.
    Hỗ trợ cả hai cách gọi:
      - tokenizer(text, truncation=, max_length=, padding="max_length", return_tensors="pt")
        -> {"input_ids": [1, L], "attention_mask": [1, L]}  (data_setup)
      - tokenizer(list_texts, padding=True, truncation=True, max_length=, return_tensors="pt")
        -> {"input_ids": [B, Lmax], "attention_mask": [B, Lmax]}  (predict)
    """

    bos, eos, pad = 1, 2, 0

    def _encode(self, text, max_length=None, truncation=False):
        words = text.split()
        ids = [self.bos] + [(zlib.crc32(w.encode("utf-8")) % 500) + 3 for w in words] + [self.eos]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return ids

    def __call__(self, text, padding=False, truncation=False, max_length=None,
                 return_tensors=None, **kwargs):
        import torch

        single = isinstance(text, str)
        texts = [text] if single else list(text)
        encs = [self._encode(t, max_length=max_length, truncation=truncation) for t in texts]

        if padding == "max_length" and max_length is not None:
            length = max_length
        else:
            length = max(len(e) for e in encs)

        input_ids = torch.full((len(encs), length), self.pad, dtype=torch.long)
        attention_mask = torch.zeros((len(encs), length), dtype=torch.long)
        for i, ids in enumerate(encs):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


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


# ---------- Helpers dùng chung cho test engine/train/predict/model (04, 03) ----------

def tiny_config(backbone: str):
    """Cấu hình backbone nhỏ khởi tạo ngẫu nhiên, không cần mạng."""
    from transformers import BertConfig, DistilBertConfig, RobertaConfig

    if backbone == "bert":
        return BertConfig(
            vocab_size=1000, hidden_size=32, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=64,
        )
    if backbone == "roberta":
        return RobertaConfig(
            vocab_size=1000, hidden_size=32, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=64,
        )
    if backbone == "distilbert":
        return DistilBertConfig(
            vocab_size=1000, dim=32, n_layers=2, n_heads=2,
        )
    raise ValueError(f"backbone không hỗ trợ trong test: {backbone}")


def no_dropout(config):
    """Đặt dropout nội bộ của backbone về 0 để tái lập chính xác."""
    for attr in ("hidden_dropout_prob", "attention_probs_dropout_prob", "dropout", "attention_dropout"):
        if hasattr(config, attr):
            setattr(config, attr, 0.0)
    return config


def tiny_model(backbone="distilbert", **kw):
    """TextRegressor nhỏ, pretrained=False, không dropout."""
    from src.model_builder import build_model

    return build_model(
        backbone,
        pretrained=False,
        backbone_config=no_dropout(tiny_config(backbone)),
        dropout=0.0,
        **kw,
    )


class _SynthDataset:
    """Dataset tổng hợp đúng dạng batch của create_dataloaders."""

    def __init__(self, n, seq_len, seed, nan_label_at=None):
        import torch

        g = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(3, 1000, (n, seq_len), generator=g)
        self.attention_mask = torch.ones(n, seq_len, dtype=torch.long)
        labels = torch.tensor([1 + (i % 5) for i in range(n)], dtype=torch.float32)
        if nan_label_at is not None:
            labels[nan_label_at] = float("nan")
        self.labels = labels
        self.review_id = [f"r{i}" for i in range(n)]

    def __len__(self):
        return len(self.review_id)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
            "review_id": self.review_id[i],
        }


def make_loader(n, batch_size, seq_len=8, seed=0, shuffle=False, nan_label_at=None):
    """DataLoader tổng hợp; shuffle=True có generator riêng cho fit reseed."""
    import torch
    from torch.utils.data import DataLoader

    ds = _SynthDataset(n, seq_len, seed, nan_label_at=nan_label_at)
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        generator=generator, num_workers=0,
    )


class ExplodingLoader:
    """Loader giả ném lỗi nếu bị lặp qua — dùng kiểm tra Test không bị chạm."""

    def __iter__(self):
        raise AssertionError("test_loader bị chạm")

    def __len__(self):
        return 4


def base_cfg(tmp_path, **overrides):
    """Cấu hình fit/run_training hợp lệ trỏ vào tmp_path."""
    cfg = dict(
        run_name="test_run",
        backbone="distilbert",
        pooling="mean",
        dropout=0.0,
        output_bias_init=None,
        loss="huber",
        huber_delta=1.0,
        seed=42,
        max_length=64,
        batch_size=4,
        epochs=3,
        lr=1e-3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        grad_clip=1.0,
        fp16=False,
        num_workers=0,
        device="cpu",
        log_every=0,
        data_dir=str(tmp_path / "data"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        results_dir=str(tmp_path / "results"),
    )
    cfg.update(overrides)
    return cfg
