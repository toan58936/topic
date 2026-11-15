"""Model builder: Transformer Encoder + pooling + Linear head cho Text Regression.

Spec: docs/define&designt/03_model.md
"""

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

BACKBONES = {
    "bert": {"hf_name": "bert-base-uncased", "model_type": "bert", "drop_pooler": True},
    "roberta": {"hf_name": "roberta-base", "model_type": "roberta", "drop_pooler": True},
    "distilbert": {"hf_name": "distilbert-base-uncased", "model_type": "distilbert", "drop_pooler": False},
}

POOLINGS = ("mean", "cls")


def _check_backbone(backbone: str) -> dict:
    if backbone not in BACKBONES:
        raise ValueError(
            f"backbone không hợp lệ: {backbone!r}. "
            f"Tên hợp lệ: {sorted(BACKBONES)}"
        )
    return BACKBONES[backbone]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Trung bình các vector token có attention_mask=1, trả về [B, H] float32."""
    h = last_hidden_state.float()
    mask = attention_mask.to(h.dtype).unsqueeze(-1)
    summed = (h * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def cls_pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """Lấy vector ở vị trí 0 của last_hidden_state, trả về [B, H]."""
    return last_hidden_state[:, 0]


class TextRegressor(nn.Module):
    """Transformer backbone + mean/cls pooling + Dropout + Linear(H -> 1)."""

    def __init__(self, backbone, pooling="mean", dropout=0.1, output_bias_init=None):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling phải thuộc {POOLINGS}, nhận: {pooling!r}")
        self.backbone = backbone
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        hidden_size = backbone.config.hidden_size
        self.head = nn.Linear(hidden_size, 1)
        if output_bias_init is not None:
            with torch.no_grad():
                self.head.bias.fill_(float(output_bias_init))

    def forward(self, input_ids, attention_mask):
        if input_ids.dim() != 2 or attention_mask.dim() != 2:
            raise ValueError("input_ids và attention_mask phải là tensor 2 chiều [B, L]")
        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                f"input_ids và attention_mask phải cùng shape, nhận "
                f"{tuple(input_ids.shape)} vs {tuple(attention_mask.shape)}"
            )
        h = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        if self.pooling == "mean":
            v = mean_pool(h, attention_mask)
        else:
            v = cls_pool(h)
        return self.head(self.dropout(v)).squeeze(-1)


def build_model(
    backbone: str = "bert",
    pooling: str = "mean",
    dropout: float = 0.1,
    pretrained: bool = True,
    backbone_config=None,
    output_bias_init=None,
) -> TextRegressor:
    """Dựng TextRegressor cho backbone trong BACKBONES."""
    spec = _check_backbone(backbone)
    if pooling not in POOLINGS:
        raise ValueError(f"pooling phải thuộc {POOLINGS}, nhận: {pooling!r}")
    if not (0.0 <= float(dropout) < 1.0):
        raise ValueError(f"dropout phải thuộc [0, 1), nhận: {dropout!r}")

    extra = {"add_pooling_layer": False} if spec["drop_pooler"] else {}
    if pretrained:
        hf_backbone = AutoModel.from_pretrained(spec["hf_name"], **extra)
    else:
        if backbone_config is None:
            config = AutoConfig.from_pretrained(spec["hf_name"])
        else:
            config = backbone_config
        if getattr(config, "model_type", None) != spec["model_type"]:
            raise ValueError(
                f"backbone_config.model_type={getattr(config, 'model_type', None)!r} "
                f"không khớp backbone {backbone!r} (cần {spec['model_type']!r})"
            )
        hf_backbone = AutoModel.from_config(config, **extra)

    return TextRegressor(
        hf_backbone, pooling=pooling, dropout=dropout,
        output_bias_init=output_bias_init,
    )


def build_tokenizer(backbone: str = "bert"):
    """Tokenizer Hugging Face cùng checkpoint với backbone."""
    spec = _check_backbone(backbone)
    return AutoTokenizer.from_pretrained(spec["hf_name"])


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Tổng số tham số (hoặc chỉ tham số requires_grad)."""
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )
