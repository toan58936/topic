
import pytest
import torch

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
