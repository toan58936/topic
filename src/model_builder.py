
from typing import Optional

import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    BertConfig,
    RobertaConfig,
    DistilBertConfig,
    BertModel,
    RobertaModel,
    DistilBertModel,
)


# ============================================================
# 1. BACKBONE REGISTRY
# ============================================================

BACKBONES = {
    "bert": {
        "hf_name": "bert-base-uncased",
        "config_class": BertConfig,
        "model_class": BertModel,
    },
    "roberta": {
        "hf_name": "roberta-base",
        "config_class": RobertaConfig,
        "model_class": RobertaModel,
    },
    "distilbert": {
        "hf_name": "distilbert-base-uncased",
        "config_class": DistilBertConfig,
        "model_class": DistilBertModel,
    },
}


# ============================================================
# 2. POOLING FUNCTIONS
# ============================================================

def mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Tính trung bình vector các token không phải padding."""

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states phải có 3 chiều.")

    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("attention_mask không khớp với hidden_states.")

    mask = attention_mask.unsqueeze(-1).to(
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    masked_hidden = hidden_states * mask

    summed = masked_hidden.sum(dim=1)

    counts = mask.sum(dim=1).clamp(min=1.0)

    return summed / counts


def cls_pool(hidden_states: torch.Tensor) -> torch.Tensor:
    """Lấy vector của token đầu tiên trong chuỗi."""

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states phải có 3 chiều.")

    return hidden_states[:, 0, :]


# ============================================================
# 3. TRANSFORMER REGRESSION MODEL
# ============================================================

class TransformerRegressor(nn.Module):
    """Transformer kết hợp Regression Head."""

    def __init__(
        self,
        backbone: nn.Module,
        pooling: str = "mean",
        dropout: float = 0.1,
        output_bias_init: Optional[float] = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.pooling = pooling

        hidden_size = getattr(
            backbone.config,
            "hidden_size",
            None,
        )

        if hidden_size is None:
            hidden_size = backbone.config.dim
            backbone.config.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Linear(hidden_size, 1)

        if output_bias_init is not None:
            with torch.no_grad():
                self.head.bias.fill_(float(output_bias_init))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Lan truyền dữ liệu qua Transformer và Regression Head."""

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids phải có shape [batch_size, seq_len]."
            )

        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask phải có 2 chiều."
            )

        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                "input_ids và attention_mask phải cùng shape."
            )

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        if self.pooling == "mean":
            pooled = mean_pool(
                hidden_states,
                attention_mask,
            )

        elif self.pooling == "cls":
            pooled = cls_pool(hidden_states)

        else:
            raise ValueError(
                f"Pooling không hợp lệ: {self.pooling}"
            )

        pooled = self.dropout(pooled)

        output = self.head(pooled)

        return output.squeeze(-1)


# ============================================================
# 4. BUILD MODEL
# ============================================================

def build_model(
    backbone: str = "bert",
    pooling: str = "mean",
    dropout: float = 0.1,
    pretrained: bool = True,
    backbone_config=None,
    output_bias_init: Optional[float] = None,
) -> TransformerRegressor:
    """Xây dựng mô hình Transformer Regression."""

    # Kiểm tra backbone
    if backbone not in BACKBONES:
        raise ValueError(
            f"Backbone không hợp lệ: {backbone}"
        )

    # Kiểm tra pooling
    if pooling not in ("mean", "cls"):
        raise ValueError(
            f"Pooling không hợp lệ: {pooling}"
        )

    # Kiểm tra dropout
    if (
        not isinstance(dropout, (int, float))
        or isinstance(dropout, bool)
        or not 0.0 <= dropout < 1.0
    ):
        raise ValueError(
            "dropout phải nằm trong khoảng [0, 1)."
        )

    backbone_info = BACKBONES[backbone]

    config_class = backbone_info["config_class"]
    model_class = backbone_info["model_class"]

    # Kiểm tra cấu hình
    if backbone_config is not None:

        if not isinstance(backbone_config, config_class):
            raise ValueError(
                f"Cấu hình không phù hợp với backbone {backbone}."
            )

    # Khởi tạo Backbone
    if pretrained:

        model_kwargs = {}

        if backbone_config is not None:
            model_kwargs["config"] = backbone_config

        if backbone in ("bert", "roberta"):
            model_kwargs["add_pooling_layer"] = False

        transformer = model_class.from_pretrained(
            backbone_info["hf_name"],
            **model_kwargs,
        )

    else:

        if backbone_config is None:
            backbone_config = config_class()

        if backbone in ("bert", "roberta"):
            transformer = model_class(
                backbone_config,
                add_pooling_layer=False,
            )
        else:
            transformer = model_class(backbone_config)

    # Đảm bảo hidden_size có thể truy cập
    if not hasattr(transformer.config, "hidden_size"):
        transformer.config.hidden_size = transformer.config.dim

    # Xây dựng mô hình hoàn chỉnh
    model = TransformerRegressor(
        backbone=transformer,
        pooling=pooling,
        dropout=float(dropout),
        output_bias_init=output_bias_init,
    )

    return model


# ============================================================
# 5. BUILD TOKENIZER
# ============================================================

def build_tokenizer(backbone: str):
    """Tải tokenizer tương ứng với backbone."""

    if backbone not in BACKBONES:
        raise ValueError(
            f"Backbone không hợp lệ: {backbone}"
        )

    return AutoTokenizer.from_pretrained(
        BACKBONES[backbone]["hf_name"]
    )


# ============================================================
# 6. COUNT PARAMETERS
# ============================================================

def count_parameters(
    model: nn.Module,
    trainable_only: bool = False,
) -> int:
    """Đếm tổng số tham số của mô hình."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )
