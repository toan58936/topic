"""Tests cho src/model_builder.py — spec docs/tests/03_model_tests.md (nhóm CPU)."""

import pytest
import torch

from conftest import no_dropout, tiny_config, tiny_model
from src.model_builder import (
    BACKBONES,
    build_model,
    build_tokenizer,
    cls_pool,
    count_parameters,
    mean_pool,
)


def _batch(model, seq_len=16, batch=2):
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(3, 1000, (batch, seq_len), generator=g)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    return input_ids, attention_mask


@pytest.mark.parametrize("backbone", ["bert", "roberta", "distilbert"])
@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_m1_m2_forward_shapes(backbone, pooling):
    model = tiny_model(backbone, pooling=pooling)
    model.eval()
    input_ids, mask = _batch(model)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=mask)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    # B = 1 vẫn cho shape [1]
    with torch.no_grad():
        out1 = model(input_ids=input_ids[:1], attention_mask=mask[:1])
    assert out1.shape == (1,)


def test_m3_mean_pooling_padding_invariant():
    model = tiny_model("distilbert", pooling="mean")
    model.eval()
    input_ids, mask = _batch(model, seq_len=16)
    # Thêm phần đệm vào cuối: mask = 0 cho 8 vị trí cuối
    pad_ids = torch.cat([input_ids, torch.zeros(2, 8, dtype=torch.long)], dim=1)
    pad_mask = torch.cat([mask, torch.zeros(2, 8, dtype=torch.long)], dim=1)
    with torch.no_grad():
        out_a = model(input_ids=input_ids, attention_mask=mask)
        out_b = model(input_ids=pad_ids, attention_mask=pad_mask)
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_m4_cls_pool_position0():
    h = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    pooled = cls_pool(h)
    assert torch.equal(pooled, h[:, 0])


def test_m5_mean_pool_math():
    h = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = mean_pool(h, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 3.0]]))
    # Hàng toàn đệm -> vector 0, không NaN
    mask_zero = torch.zeros(1, 3, dtype=torch.long)
    pooled_zero = mean_pool(h, mask_zero)
    assert torch.equal(pooled_zero, torch.zeros(1, 2))
    assert torch.isfinite(pooled_zero).all()


def test_m6_head_bias_init():
    torch.manual_seed(0)
    model = tiny_model("distilbert", output_bias_init=3.7)
    assert float(model.head.bias) == pytest.approx(3.7)


def test_m7_param_count_no_pooler():
    for backbone in ("bert", "distilbert"):
        model = tiny_model(backbone)
        n = count_parameters(model)
        head_params = model.head.weight.numel() + model.head.bias.numel()
        backbone_params = count_parameters(model.backbone)
        assert n == backbone_params + head_params
        # BERT không còn pooler khi add_pooling_layer=False
        if backbone == "bert":
            assert not hasattr(model.backbone, "pooler") or model.backbone.pooler is None


def test_m8_invalid_inputs():
    with pytest.raises(ValueError):
        build_model("gpt2", pretrained=False)
    with pytest.raises(ValueError):
        tiny_model("distilbert", pooling="attn")
    with pytest.raises(ValueError):
        build_model(
            "distilbert", pretrained=False,
            backbone_config=no_dropout(tiny_config("distilbert")), dropout=1.5,
        )
    # backbone_config sai model_type
    with pytest.raises(ValueError):
        build_model("bert", pretrained=False, backbone_config=tiny_config("distilbert"))


def test_m9_same_seed_same_init():
    torch.manual_seed(42)
    m1 = tiny_model("distilbert")
    torch.manual_seed(42)
    m2 = tiny_model("distilbert")
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


def test_m10_forward_rejects_bad_input():
    model = tiny_model()
    model.eval()
    with pytest.raises(ValueError):
        model(torch.randint(0, 100, (8,)), torch.ones(8, dtype=torch.long))
    # attention_mask khác shape
    with pytest.raises(ValueError):
        model(
            torch.randint(0, 100, (2, 8)),
            torch.ones(2, 4, dtype=torch.long),
        )
