"""Tests cho src/utils.py — spec docs/tests/04_engine_and_utils_tests.md nhóm A."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils import (
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    write_atomic,
)


def test_u1_set_seed_reproducible():
    set_seed(42)
    r1, n1, t1 = random.random(), np.random.rand(), torch.rand(1)
    set_seed(42)
    r2, n2, t2 = random.random(), np.random.rand(), torch.rand(1)
    assert r1 == r2 and n1 == n2 and torch.equal(t1, t2)
    set_seed(7)
    r3, n3, t3 = random.random(), np.random.rand(), torch.rand(1)
    assert not (r1 == r3 and n1 == n3 and torch.equal(t1, t3))


def test_u2_get_device():
    assert get_device("cpu") == torch.device("cpu")
    auto = get_device("auto")
    assert auto.type in ("cpu", "cuda")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            get_device("cuda")
    with pytest.raises(ValueError):
        get_device("tpu")


def test_u3_checkpoint_roundtrip(tmp_path):
    state = {
        "tensor": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "nested": {"a": [1, 2, 3], "b": "text"},
        "list_of_dicts": [{"x": 1}, {"y": None}],
        "s": "hello",
        "i": 3,
        "f": 2.5,
        "b": True,
        "n": None,
    }
    path = tmp_path / "ckpt" / "model.pt"
    save_checkpoint(str(path), state)
    loaded = load_checkpoint(str(path))
    assert torch.equal(loaded["tensor"], state["tensor"])
    assert loaded["nested"] == state["nested"]
    assert loaded["list_of_dicts"] == state["list_of_dicts"]
    assert loaded["s"] == "hello" and loaded["i"] == 3
    assert loaded["f"] == 2.5 and loaded["b"] is True and loaded["n"] is None
    # Thư mục chỉ chứa đúng file checkpoint, không sót file tạm
    assert [p.name for p in path.parent.iterdir()] == ["model.pt"]
    with pytest.raises(TypeError):
        save_checkpoint(str(tmp_path / "bad.pt"), [1, 2, 3])
    with pytest.raises(FileNotFoundError):
        load_checkpoint(str(tmp_path / "missing.pt"))


def test_u4_write_atomic_failure(tmp_path):
    target = tmp_path / "sub" / "out.txt"
    target.parent.mkdir(parents=True)
    target.write_text("cũ", encoding="utf-8")

    def failing_write(p):
        with open(p, "w", encoding="utf-8") as f:
            f.write("nửa chừng")
        raise RuntimeError("mô phỏng ngắt giữa chừng")

    with pytest.raises(RuntimeError):
        write_atomic(str(target), failing_write)
    assert target.read_text(encoding="utf-8") == "cũ"
    assert not (tmp_path / "sub" / "out.txt.tmp").exists()

    write_atomic(str(target), lambda p: Path(p).write_text("mới", encoding="utf-8"))
    assert target.read_text(encoding="utf-8") == "mới"
