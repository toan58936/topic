"""Tiện ích: seed, thiết bị, ghi file nguyên tử, checkpoint.

Spec: docs/define&designt/04_engine_and_utils.md mục 2.4
"""

import os
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Cố định seed cho random, numpy, torch (CPU và CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "auto") -> torch.device:
    """Chọn thiết bị: auto -> CUDA nếu có, không thì CPU."""
    if prefer not in ("auto", "cpu", "cuda"):
        raise ValueError(f"prefer phải thuộc {{auto, cpu, cuda}}, nhận: {prefer!r}")
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Yêu cầu device='cuda' nhưng máy không có CUDA")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_atomic(path: str, write_fn: Callable[[str], None]) -> None:
    """Ghi qua file tạm cùng thư mục rồi os.replace; lỗi thì dọn file tạm."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        write_fn(str(tmp_path))
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def save_checkpoint(path: str, state: dict) -> None:
    """Lưu checkpoint (dict) bằng torch.save qua write_atomic."""
    if not isinstance(state, dict):
        raise TypeError(f"state phải là dict, nhận: {type(state).__name__}")
    write_atomic(path, lambda p: torch.save(state, p))


def load_checkpoint(path: str, map_location="cpu") -> dict:
    """Nạp checkpoint bằng weights_only=True."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
    return torch.load(path, map_location=map_location, weights_only=True)
