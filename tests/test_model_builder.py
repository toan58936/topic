
import os
import socket

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig

from src.model_builder import (
    BACKBONES,
    mean_pool,
    cls_pool,
    build_model,
    build_tokenizer,
    count_parameters,
)


BACKBONE_NAMES = ["bert", "roberta", "distilbert"]
POOLING_TYPES = ["mean", "cls"]


# ============================================================
# NHÓM A - PHÉP GỘP VECTOR
# ============================================================

def test_m1_mean_pool_known_values():
    """Kiểm tra mean pooling bằng giá trị tính tay."""

    hidden = torch.tensor(
        [
            [[1., 2.], [3., 4.], [100., 100.]],
            [[5., 6.], [9., 9.], [9., 9.]],
        ]
    )

    mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 0, 0],
        ]
    )

    out = mean_pool(hidden, mask)

    expected = torch.tensor(
        [
            [2., 3.],
            [5., 6.],
        ]
    )

    assert torch.allclose(out, expected)


def test_m2_mean_pool_zero_mask():
    """Hàng có attention_mask toàn 0 phải trả về vector 0."""

    hidden = torch.tensor(
        [
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]],
        ]
    )

    mask = torch.tensor(
        [
            [0, 0],
            [1, 0],
        ]
    )

    out = mean_pool(hidden, mask)

    assert torch.equal(out[0], torch.zeros(2))
    assert torch.equal(out[1], torch.tensor([5., 6.]))
    assert torch.isfinite(out).all()


def test_m3_cls_pool_first_token():
    """CLS pooling phải lấy vector token đầu tiên."""

    torch.manual_seed(42)

    hidden = torch.randn(3, 5, 4)

    out = cls_pool(hidden)

    expected = hidden[:, 0, :]

    assert torch.equal(out, expected)


# ============================================================
# NHÓM B - HÌNH DẠNG VÀ GIAO DIỆN
# ============================================================

@pytest.mark.parametrize("backbone", BACKBONE_NAMES)
@pytest.mark.parametrize("pooling", POOLING_TYPES)
def test_m4_forward_shape(backbone, pooling, tiny_config, make_batch):
    """Kiểm tra shape đầu ra với ba backbone và hai kiểu pooling."""

    torch.manual_seed(42)

    model = build_model(
        backbone=backbone,
        pooling=pooling,
        pretrained=False,
        backbone_config=tiny_config(backbone),
    )

    model.eval()

    input_ids, attention_mask = make_batch(
        backbone,
        batch_size=2,
        seq_len=16,
    )

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    assert output.shape == (2,)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()

    # Kiểm tra batch chỉ có một mẫu
    one_ids, one_mask = make_batch(
        backbone,
        batch_size=1,
        seq_len=16,
    )

    with torch.no_grad():
        one_output = model(one_ids, one_mask)

    assert one_output.shape == (1,)
    assert torch.isfinite(one_output).all()


@pytest.mark.parametrize("backbone", BACKBONE_NAMES)
@pytest.mark.parametrize("pooling", POOLING_TYPES)
def test_m5_padding_invariance(
    backbone,
    pooling,
    tiny_config,
    make_batch,
    pad_batch,
):
    """Thêm padding không được làm thay đổi điểm dự đoán."""

    torch.manual_seed(42)

    model = build_model(
        backbone=backbone,
        pooling=pooling,
        pretrained=False,
        backbone_config=tiny_config(backbone),
    )

    model.eval()

    ids8, mask8 = make_batch(
        backbone,
        batch_size=2,
        seq_len=8,
    )

    # Thêm 8 token padding: độ dài từ 8 thành 16
    ids16, mask16 = pad_batch(
        backbone,
        ids8,
        mask8,
        pad_length=8,
    )

    with torch.no_grad():
        output_original = model(ids8, mask8)
        output_padded = model(ids16, mask16)

    assert torch.allclose(
        output_original,
        output_padded,
        atol=1e-5,
    )


def test_m6_registry_and_invalid_args(tiny_config):
    """Kiểm tra registry và các tham số không hợp lệ."""

    # Registry phải có đúng ba backbone
    assert set(BACKBONES.keys()) == {
        "bert",
        "roberta",
        "distilbert",
    }

    assert BACKBONES["bert"]["hf_name"] == "bert-base-uncased"

    # Backbone không hợp lệ
    with pytest.raises(ValueError):
        build_model(
            backbone="unknown",
            pretrained=False,
        )

    # Pooling không hợp lệ
    with pytest.raises(ValueError):
        build_model(
            backbone="bert",
            pooling="max",
            pretrained=False,
            backbone_config=tiny_config("bert"),
        )

    # Dropout không hợp lệ
    with pytest.raises(ValueError):
        build_model(
            backbone="bert",
            dropout=1.0,
            pretrained=False,
            backbone_config=tiny_config("bert"),
        )

    # Truyền cấu hình BERT cho RoBERTa
    with pytest.raises(ValueError):
        build_model(
            backbone="roberta",
            pretrained=False,
            backbone_config=tiny_config("bert"),
        )

    # Kiểm tra forward với input_ids sai số chiều
    model = build_model(
        backbone="bert",
        pretrained=False,
        backbone_config=tiny_config("bert"),
    )

    wrong_ids = torch.randint(5, 100, (16,))
    wrong_mask = torch.ones(16, dtype=torch.long)

    with pytest.raises(ValueError):
        model(wrong_ids, wrong_mask)

    # Kiểm tra hai tensor có shape không khớp
    input_ids = torch.randint(5, 100, (2, 8))
    attention_mask = torch.ones((2, 7), dtype=torch.long)

    with pytest.raises(ValueError):
        model(input_ids, attention_mask)


@pytest.mark.parametrize("backbone", BACKBONE_NAMES)
def test_m7_batch_format_from_dataloader(
    backbone,
    tiny_config,
    make_batch,
):
    """Kiểm tra batch có shape [4, 256] và kiểu long."""

    torch.manual_seed(42)

    model = build_model(
        backbone=backbone,
        pretrained=False,
        backbone_config=tiny_config(backbone),
    )

    model.eval()

    input_ids, attention_mask = make_batch(
        backbone,
        batch_size=4,
        seq_len=256,
    )

    assert input_ids.shape == (4, 256)
    assert attention_mask.shape == (4, 256)

    assert input_ids.dtype == torch.long
    assert attention_mask.dtype == torch.long

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    assert output.shape == (4,)
    assert torch.isfinite(output).all()


# ============================================================
# NHÓM C - HEAD VÀ THAM SỐ
# ============================================================

@pytest.mark.parametrize("backbone", BACKBONE_NAMES)
def test_m8_head_structure(backbone, tiny_config):
    """Kiểm tra cấu trúc Linear head và hidden_size."""

    torch.manual_seed(42)

    model = build_model(
        backbone=backbone,
        pretrained=False,
        backbone_config=tiny_config(backbone),
    )

    hidden_size = model.backbone.config.hidden_size

    assert hidden_size == 32

    assert isinstance(model.head, nn.Linear)
    assert model.head.in_features == hidden_size
    assert model.head.out_features == 1


@pytest.mark.parametrize("backbone", BACKBONE_NAMES)
def test_m9_pooler_dropped_and_param_count(backbone, tiny_config):
    """Kiểm tra pooler đã được loại bỏ và số tham số đúng."""

    torch.manual_seed(42)

    model = build_model(
        backbone=backbone,
        pretrained=False,
        backbone_config=tiny_config(backbone),
    )

    if backbone in ("bert", "roberta"):
        assert model.backbone.pooler is None

    elif backbone == "distilbert":
        assert not hasattr(model.backbone, "pooler")

    backbone_params = sum(
        parameter.numel()
        for parameter in model.backbone.parameters()
    )

    hidden_size = model.backbone.config.hidden_size

    expected_total = backbone_params + hidden_size + 1

    assert count_parameters(model) == expected_total


@pytest.mark.parametrize("backbone", BACKBONE_NAMES)
def test_m10_count_parameters(backbone, tiny_config):
    """Kiểm tra đếm toàn bộ và tham số có thể huấn luyện."""

    torch.manual_seed(42)

    model = build_model(
        backbone=backbone,
        pretrained=False,
        backbone_config=tiny_config(backbone),
    )

    expected_total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert count_parameters(model) == expected_total

    # Đóng băng các tham số của head
    for parameter in model.head.parameters():
        parameter.requires_grad = False

    expected_trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    actual_trainable = count_parameters(
        model,
        trainable_only=True,
    )

    assert actual_trainable == expected_trainable

    hidden_size = model.backbone.config.hidden_size

    assert actual_trainable == expected_total - (hidden_size + 1)
    assert actual_trainable < expected_total


@pytest.mark.parametrize("backbone", BACKBONE_NAMES)
def test_m11_output_bias_init(backbone, tiny_config):
    """Kiểm tra khả năng khởi tạo bias đầu ra."""

    torch.manual_seed(42)

    model_with_bias = build_model(
        backbone=backbone,
        pretrained=False,
        backbone_config=tiny_config(backbone),
        output_bias_init=3.7,
    )

    assert torch.allclose(
        model_with_bias.head.bias,
        torch.tensor([3.7]),
    )

    torch.manual_seed(42)

    model_without_bias = build_model(
        backbone=backbone,
        pretrained=False,
        backbone_config=tiny_config(backbone),
        output_bias_init=None,
    )

    assert not torch.allclose(
        model_without_bias.head.bias,
        torch.tensor([3.7]),
    )


# ============================================================
# NHÓM D - HUẤN LUYỆN VÀ TÁI LẬP
# ============================================================

@pytest.mark.parametrize("backbone", BACKBONE_NAMES)
def test_m12_gradients_flow(
    backbone,
    tiny_config,
    make_batch,
):
    """Kiểm tra gradient chảy qua các tham số của mô hình."""

    torch.manual_seed(42)

    model = build_model(
        backbone=backbone,
        pretrained=False,
        backbone_config=tiny_config(backbone),
    )

    model.train()

    input_ids, attention_mask = make_batch(
        backbone,
        batch_size=2,
        seq_len=16,
    )

    targets = torch.randn(2)

    predictions = model(input_ids, attention_mask)

    loss = F.mse_loss(predictions, targets)

    loss.backward()

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, (
                f"Tham số {name} không có gradient"
            )

            assert torch.isfinite(parameter.grad).all(), (
                f"Gradient của {name} không hữu hạn"
            )


@pytest.mark.parametrize("backbone", BACKBONE_NAMES)
def test_m13_dropout_train_vs_eval(
    backbone,
    tiny_config,
    make_batch,
):
    """Eval phải ổn định, train phải có tác động của dropout."""

    torch.manual_seed(42)

    model = build_model(
        backbone=backbone,
        dropout=0.5,
        pretrained=False,
        backbone_config=tiny_config(backbone),
    )

    input_ids, attention_mask = make_batch(
        backbone,
        batch_size=4,
        seq_len=16,
    )

    # Chế độ evaluation: hai lần dự đoán phải giống nhau
    model.eval()

    with torch.no_grad():
        eval_output_1 = model(input_ids, attention_mask)
        eval_output_2 = model(input_ids, attention_mask)

    assert torch.allclose(
        eval_output_1,
        eval_output_2,
        atol=1e-7,
    )

    # Chế độ training: dropout hoạt động nên kết quả có thể khác
    model.train()

    with torch.no_grad():
        train_output_1 = model(input_ids, attention_mask)
        train_output_2 = model(input_ids, attention_mask)

    assert not torch.allclose(
        train_output_1,
        train_output_2,
    )


@pytest.mark.parametrize("backbone", BACKBONE_NAMES)
def test_m14_seed_reproducible_init(backbone, tiny_config):
    """Cùng seed phải tạo ra trọng số giống nhau."""

    torch.manual_seed(42)

    model_1 = build_model(
        backbone=backbone,
        pretrained=False,
        backbone_config=tiny_config(backbone),
    )

    torch.manual_seed(42)

    model_2 = build_model(
        backbone=backbone,
        pretrained=False,
        backbone_config=tiny_config(backbone),
    )

    state_1 = model_1.state_dict()
    state_2 = model_2.state_dict()

    assert state_1.keys() == state_2.keys()

    for key in state_1:
        assert torch.equal(state_1[key], state_2[key]), (
            f"Trọng số {key} không giống nhau khi cùng seed"
        )

    # Seed khác phải tạo ra ít nhất một trọng số khác
    torch.manual_seed(7)

    model_3 = build_model(
        backbone=backbone,
        pretrained=False,
        backbone_config=tiny_config(backbone),
    )

    state_3 = model_3.state_dict()

    assert any(
        not torch.equal(state_1[key], state_3[key])
        for key in state_1
    )


@pytest.mark.parametrize("backbone", BACKBONE_NAMES)
def test_m15_overfit_small_batch(
    backbone,
    tiny_config,
    make_batch,
    no_dropout,
):
    """Kiểm tra mô hình có thể học một batch nhỏ hay không."""

    torch.manual_seed(42)

    config = no_dropout(tiny_config(backbone))

    model = build_model(
        backbone=backbone,
        dropout=0.0,
        pretrained=False,
        backbone_config=config,
    )

    model.train()

    input_ids, attention_mask = make_batch(
        backbone,
        batch_size=8,
        seq_len=12,
    )

    targets = torch.tensor(
        [1., 2., 3., 4., 5., 2., 3., 4.]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    # Loss ban đầu
    predictions = model(input_ids, attention_mask)
    initial_loss = F.huber_loss(predictions, targets).item()

    # Huấn luyện 100 bước
    for _ in range(100):
        optimizer.zero_grad()

        predictions = model(input_ids, attention_mask)

        loss = F.huber_loss(predictions, targets)

        loss.backward()
        optimizer.step()

    # Loss sau huấn luyện
    with torch.no_grad():
        final_predictions = model(input_ids, attention_mask)
        final_loss = F.huber_loss(
            final_predictions,
            targets,
        ).item()

    assert final_loss < initial_loss * 0.5, (
        f"{backbone}: Loss ban đầu={initial_loss:.4f}, "
        f"Loss cuối={final_loss:.4f}"
    )


# ============================================================
# NHÓM E - TOKENIZER VÀ MÔ HÌNH PRETRAINED (SLOW)
# ============================================================

def _huggingface_available():
    """Kiểm tra có thể kết nối Hugging Face Hub hay không."""

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return False

    try:
        connection = socket.create_connection(
            ("huggingface.co", 443),
            timeout=3,
        )
        connection.close()
        return True

    except OSError:
        return False


def _skip_if_huggingface_unavailable():
    """Bỏ qua test cần checkpoint thật nếu không có mạng."""

    if not _huggingface_available():
        pytest.skip(
            "Không có kết nối Hugging Face hoặc đang bật HF_HUB_OFFLINE=1"
        )


@pytest.mark.slow
@pytest.mark.parametrize("backbone", BACKBONE_NAMES)
def test_s1_tokenizer_special_tokens(backbone):
    """Kiểm tra tokenizer và các token đặc biệt."""

    _skip_if_huggingface_unavailable()

    tokenizer = build_tokenizer(backbone)

    encoded = tokenizer(
        "great food",
        add_special_tokens=True,
    )

    input_ids = encoded["input_ids"]

    assert input_ids[0] == tokenizer.cls_token_id
    assert input_ids[-1] == tokenizer.sep_token_id


@pytest.mark.slow
@pytest.mark.parametrize(
    "backbone,expected_params",
    [
        ("bert", 108_900_000),
        ("roberta", 124_100_000),
        ("distilbert", 66_400_000),
    ],
)
def test_s2_pretrained_dims_and_param_counts(
    backbone,
    expected_params,
):
    """Kiểm tra kích thước và số tham số của mô hình pretrained."""

    _skip_if_huggingface_unavailable()

    model = build_model(
        backbone=backbone,
        pretrained=True,
    )

    hidden_size = model.backbone.config.hidden_size

    assert hidden_size == 768

    actual_params = count_parameters(model)

    lower_bound = expected_params * 0.99
    upper_bound = expected_params * 1.01

    assert lower_bound <= actual_params <= upper_bound, (
        f"{backbone}: Số tham số thực tế {actual_params:,}, "
        f"ước tính {expected_params:,}"
    )

    tokenizer = build_tokenizer(backbone)

    batch = tokenizer(
        [
            "great food",
            "the service was excellent",
        ],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    model.eval()

    with torch.no_grad():
        output = model(
            batch["input_ids"],
            batch["attention_mask"],
        )

    assert output.shape == (2,)
    assert torch.isfinite(output).all()
