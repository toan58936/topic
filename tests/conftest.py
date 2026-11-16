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
from transformers import (
    BertConfig,
    RobertaConfig,
    DistilBertConfig,
)


# Tạo cấu hình model nhỏ để kiểm thử trên CPU
@pytest.fixture
def tiny_config():
    def _make_config(backbone):

        if backbone == "bert":
            return BertConfig(
                vocab_size=1000,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=64,
            )

        elif backbone == "roberta":
            return RobertaConfig(
                vocab_size=1000,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=64,
            )

        elif backbone == "distilbert":
            return DistilBertConfig(
                vocab_size=1000,
                dim=32,
                n_layers=2,
                n_heads=2,
                hidden_dim=64,
            )

        else:
            raise ValueError(f"Backbone không hợp lệ: {backbone}")

    return _make_config


# Tắt dropout để kết quả kiểm thử ổn định
@pytest.fixture
def no_dropout():
    def _disable_dropout(config):

        if config.model_type in ("bert", "roberta"):
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        elif config.model_type == "distilbert":
            config.dropout = 0.0
            config.attention_dropout = 0.0
            config.qa_dropout = 0.0
            config.seq_classif_dropout = 0.0

        return config

    return _disable_dropout


# Tạo dữ liệu giả để kiểm thử
@pytest.fixture
def make_batch():
    def _make_batch(backbone, batch_size=2, seq_len=16):

        input_ids = torch.randint(
            low=5,
            high=1000,
            size=(batch_size, seq_len),
            dtype=torch.long,
        )

        attention_mask = torch.ones(
            (batch_size, seq_len),
            dtype=torch.long,
        )

        return input_ids, attention_mask

    return _make_batch


# Thêm padding vào dữ liệu
@pytest.fixture
def pad_batch():
    def _pad_batch(backbone, input_ids, attention_mask, pad_length=4):

        pad_token_id = 1 if backbone == "roberta" else 0

        batch_size = input_ids.shape[0]

        padding_ids = torch.full(
            (batch_size, pad_length),
            pad_token_id,
            dtype=torch.long,
        )

        padding_mask = torch.zeros(
            (batch_size, pad_length),
            dtype=torch.long,
        )

        padded_ids = torch.cat(
            [input_ids, padding_ids],
            dim=1,
        )

        padded_mask = torch.cat(
            [attention_mask, padding_mask],
            dim=1,
        )

        return padded_ids, padded_mask

    return _pad_batch


@pytest.fixture()
def valid_results_dir(tmp_path):
    """Tạo bộ dữ liệu kết quả giả hợp lệ cho Integration Test."""
    import numpy as np
    import pandas as pd

    from src.metrics import build_model_summary

    results_dir = tmp_path / "results"
    preds_dir = results_dir / "preds"
    data_dir = tmp_path / "data"

    preds_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    split_id = "abc123def456"

    # Nhãn thật của 6 review
    y_true = np.array([1, 2, 3, 4, 5, 4], dtype=float)

    # Dự đoán chưa clip của từng mô hình
    raw_predictions = {
        "bert": np.array([1.2, 2.1, 2.8, 4.2, 4.7, 3.8]),
        "distilbert": np.array([1.1, 2.3, 3.1, 3.7, 4.8, 4.2]),
        "roberta": np.array([1.3, 1.9, 3.2, 4.1, 5.2, 3.9]),
    }

    # Tạo test_predictions.csv
    merged = pd.DataFrame({
        "review_id": [f"review_{i:03d}" for i in range(6)],
        "true_stars": y_true.astype(int),
        "n_tokens": [12, 20, 35, 18, 42, 25],
        "split_id": split_id,
    })

    rmse_raw_by_model = {}

    for model, raw in raw_predictions.items():
        clipped = np.clip(raw, 1.0, 5.0)

        merged[f"pred_{model}"] = clipped
        merged[f"abs_err_{model}"] = np.abs(y_true - clipped)

        rmse_raw_by_model[model] = float(
            np.sqrt(np.mean((raw - y_true) ** 2))
        )

        # File dự đoán riêng của từng mô hình
        individual = pd.DataFrame({
            "review_id": merged["review_id"],
            "true_stars": y_true.astype(int),
            "pred_raw": raw,
            "pred": clipped,
            "split_id": split_id,
        })

        individual.to_csv(
            preds_dir / f"{model}_test.csv",
            index=False,
        )

        # Metadata tương ứng
        meta = {
            "model": model,
            "run_name": f"test_{model}",
            "checkpoint_epoch": 2,
            "n": len(y_true),
            "rmse_raw": rmse_raw_by_model[model],
            "inference_seconds": 0.5,
            "fp16": False,
            "device": "cpu",
        }

        with open(
            preds_dir / f"{model}_test_meta.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    merged.to_csv(
        results_dir / "test_predictions.csv",
        index=False,
    )

    # Tạo model_summary bằng hàm chuẩn của dự án
    models = ["bert", "distilbert", "roberta"]

    summary = build_model_summary(
        preds_df=merged,
        models=models,
        rmse_raw_by_model=rmse_raw_by_model,
        param_counts={
            "bert": 1000,
            "distilbert": 800,
            "roberta": 1200,
        },
        train_seconds={
            "bert": 10.0,
            "distilbert": 8.0,
            "roberta": 12.0,
        },
        inference_seconds={
            "bert": 0.5,
            "distilbert": 0.4,
            "roberta": 0.6,
        },
        seconds_per_epoch={
            "bert": 5.0,
            "distilbert": 4.0,
            "roberta": 6.0,
        },
        baseline_name="bert",
        split_id=split_id,
    )

    summary.to_csv(
        results_dir / "model_summary.csv",
        index=False,
    )

    # Thông tin split dùng để đối chiếu
    split_stats = {
        "split_id": split_id,
    }

    with open(
        data_dir / "split_stats.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(split_stats, f, ensure_ascii=False, indent=2)

    return results_dir

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
    import shutil
    import pandas as pd

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
