"""CLI huấn luyện: đọc YAML config, chạy run_training.

Spec: docs/define&designt/04_engine_and_utils.md mục 2.6
Chạy: python -m src.train --config configs/bert.yaml [--set k=v ...] [--overwrite] [--no-resume]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

from src.data_setup import create_dataloaders
from src.engine import fit
from src.model_builder import BACKBONES, build_model, build_tokenizer
from src.utils import get_device, set_seed, write_atomic

DEFAULT_CONFIG = {
    "run_name": None,
    "backbone": None,
    "pooling": "mean",
    "dropout": 0.1,
    "output_bias_init": "train_mean",
    "loss": "huber",
    "huber_delta": 1.0,
    "seed": 42,
    "max_length": 256,
    "batch_size": 16,
    "epochs": 3,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "grad_clip": 1.0,
    "fp16": True,
    "num_workers": 0,
    "device": "auto",
    "log_every": 100,
    "data_dir": None,
    "checkpoint_dir": None,
    "results_dir": None,
}

INT_KEYS = {"seed", "max_length", "batch_size", "epochs", "num_workers", "log_every"}
FLOAT_KEYS = {
    "dropout", "huber_delta", "lr", "weight_decay", "warmup_ratio", "grad_clip",
}
REQUIRED_KEYS = {"backbone", "data_dir", "checkpoint_dir", "results_dir"}


def load_config(path: str, overrides: Optional[Dict[str, str]] = None) -> dict:
    """Đọc YAML, áp mặc định + overrides, ép kiểu số, kiểm tra."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"File cấu hình phải là mapping YAML, nhận: {type(raw).__name__}")

    cfg = dict(DEFAULT_CONFIG)
    cfg.update(raw)
    if overrides:
        for key, value in overrides.items():
            cfg[key] = yaml.safe_load(value)

    unknown = set(cfg) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Khoá cấu hình không hợp lệ: {sorted(unknown)}")

    for key in INT_KEYS:
        if cfg.get(key) is not None:
            cfg[key] = int(cfg[key])
    for key in FLOAT_KEYS:
        if cfg.get(key) is not None:
            cfg[key] = float(cfg[key])
    if cfg.get("lr") is not None:
        cfg["lr"] = float(cfg["lr"])
    if cfg.get("fp16") is not None:
        cfg["fp16"] = bool(cfg["fp16"])

    missing = [k for k in REQUIRED_KEYS if cfg.get(k) in (None, "")]
    if missing:
        raise ValueError(f"Thiếu khoá bắt buộc trong cấu hình: {missing}")

    if cfg["backbone"] not in BACKBONES:
        raise ValueError(
            f"backbone không hợp lệ: {cfg['backbone']!r}. "
            f"Tên hợp lệ: {sorted(BACKBONES)}"
        )
    if cfg["loss"] not in ("huber", "mse"):
        raise ValueError(f"loss phải thuộc {{huber, mse}}, nhận: {cfg['loss']!r}")
    if cfg["pooling"] not in ("mean", "cls"):
        raise ValueError(f"pooling phải thuộc {{mean, cls}}, nhận: {cfg['pooling']!r}")
    if cfg["epochs"] < 1:
        raise ValueError(f"epochs phải >= 1, nhận: {cfg['epochs']}")
    if cfg["batch_size"] < 1:
        raise ValueError(f"batch_size phải >= 1, nhận: {cfg['batch_size']}")
    if cfg["max_length"] < 1:
        raise ValueError(f"max_length phải >= 1, nhận: {cfg['max_length']}")
    if float(cfg["lr"]) <= 0:
        raise ValueError(f"lr phải > 0, nhận: {cfg['lr']}")
    if not (0.0 <= float(cfg["warmup_ratio"]) < 1.0):
        raise ValueError(f"warmup_ratio phải thuộc [0, 1), nhận: {cfg['warmup_ratio']}")
    obi = cfg["output_bias_init"]
    if obi is not None and obi != "train_mean":
        try:
            cfg["output_bias_init"] = float(obi)
        except (TypeError, ValueError):
            raise ValueError(
                f"output_bias_init phải là số, 'train_mean' hoặc null, nhận: {obi!r}"
            )
    if cfg["dropout"] is not None and not (0.0 <= float(cfg["dropout"]) < 1.0):
        raise ValueError(f"dropout phải thuộc [0, 1), nhận: {cfg['dropout']}")

    if not cfg.get("run_name"):
        cfg["run_name"] = cfg["backbone"]
    return cfg


def resolve_output_bias(cfg: dict):
    """train_mean -> trung bình cột stars của train.parquet; số -> số đó; null -> None."""
    obi = cfg.get("output_bias_init")
    if obi is None:
        return None
    if obi == "train_mean":
        train_path = Path(cfg["data_dir"]) / "train.parquet"
        if not train_path.exists():
            raise FileNotFoundError(f"Thiếu train.parquet để tính output_bias: {train_path}")
        df = pd.read_parquet(train_path, columns=["stars"])
        return float(df["stars"].mean())
    return float(obi)


def run_training(cfg: dict, *, model=None, loaders=None, resume=True, overwrite=False) -> dict:
    """Chạy một lần huấn luyện đầy đủ từ cấu hình."""
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))

    if loaders is None:
        tokenizer = build_tokenizer(cfg["backbone"])
        loaders = create_dataloaders(
            cfg["data_dir"], tokenizer,
            batch_size=int(cfg["batch_size"]),
            max_length=int(cfg["max_length"]),
            num_workers=int(cfg["num_workers"]),
            seed=int(cfg["seed"]),
        )
    train_loader, val_loader, _test_loader = loaders

    # Đặt lại seed NGAY TRƯỚC build_model để khởi tạo head tái lập được
    set_seed(int(cfg["seed"]))
    if model is None:
        model = build_model(
            cfg["backbone"],
            pooling=cfg.get("pooling", "mean"),
            dropout=float(cfg.get("dropout", 0.1)),
            pretrained=True,
            output_bias_init=resolve_output_bias(cfg),
        )

    res_run_dir = Path(cfg["results_dir"]) / "runs" / cfg["run_name"]
    res_run_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(
        str(res_run_dir / "config.yaml"),
        lambda p: Path(p).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"),
    )

    return fit(
        model, train_loader, val_loader, cfg,
        cfg["checkpoint_dir"], cfg["results_dir"],
        device=device, resume=resume, overwrite=overwrite,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train text regression model")
    parser.add_argument("--config", required=True, help="Đường dẫn file YAML cấu hình")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="Ghi đè cấu hình")
    parser.add_argument("--overwrite", action="store_true", help="Chạy lại từ đầu")
    parser.add_argument("--no-resume", action="store_true", help="Không tiếp tục từ last.pt")
    args = parser.parse_args(argv)

    overrides = {}
    for item in args.overrides:
        if "=" not in item:
            print(f"--set không đúng định dạng KEY=VALUE: {item!r}", file=sys.stderr)
            return 1
        key, value = item.split("=", 1)
        overrides[key.strip()] = value

    try:
        cfg = load_config(args.config, overrides=overrides or None)
    except (ValueError, FileNotFoundError, yaml.YAMLError) as exc:
        print(f"Lỗi cấu hình: {exc}", file=sys.stderr)
        return 1

    try:
        run_training(cfg, resume=not args.no_resume, overwrite=args.overwrite)
    except Exception as exc:
        print(f"Lỗi khi huấn luyện: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
