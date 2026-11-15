"""Tests cho src/train.py — spec docs/tests/04_engine_and_utils_tests.md nhóm D."""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

import src.train as train_mod
from conftest import base_cfg, make_loader, tiny_model


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _minimal_yaml(tmp_path, **overrides):
    cfg = dict(
        backbone="distilbert",
        data_dir=str(tmp_path / "data"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        results_dir=str(tmp_path / "results"),
    )
    cfg.update(overrides)
    return _write_yaml(tmp_path / "cfg.yaml", cfg)


def test_t1_load_config(tmp_path):
    # lr: 2e-5 -> PyYAML đọc thành chuỗi, load_config phải ép float
    yaml_text = (
        "backbone: distilbert\n"
        "data_dir: /tmp/data\n"
        "checkpoint_dir: /tmp/ckpt\n"
        "results_dir: /tmp/results\n"
        "lr: 2e-5\n"
    )
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    cfg = train_mod.load_config(
        str(path), overrides={"loss": "mse", "epochs": "5", "run_name": "bert_mse"}
    )
    assert cfg["lr"] == pytest.approx(2e-5) and isinstance(cfg["lr"], float)
    assert cfg["pooling"] == "mean"
    assert cfg["seed"] == 42
    assert cfg["batch_size"] == 16
    assert cfg["epochs"] == 5 and isinstance(cfg["epochs"], int)
    assert cfg["loss"] == "mse"
    assert cfg["run_name"] == "bert_mse"

    # Không ghi đè run_name -> bằng backbone
    cfg2 = train_mod.load_config(str(path))
    assert cfg2["run_name"] == "distilbert"

    # Các trường hợp lỗi
    bad_cases = [
        {"backbone": "distilbert", "data_dir": "d", "checkpoint_dir": "c",
         "results_dir": "r", "learning_rate": 1e-3},          # khoá lạ
        {"data_dir": "d", "checkpoint_dir": "c", "results_dir": "r"},  # thiếu backbone
        {"backbone": "gpt2", "data_dir": "d", "checkpoint_dir": "c", "results_dir": "r"},
        {"backbone": "bert", "data_dir": "d", "checkpoint_dir": "c", "results_dir": "r", "loss": "l1"},
        {"backbone": "bert", "data_dir": "d", "checkpoint_dir": "c", "results_dir": "r", "epochs": 0},
        {"backbone": "bert", "data_dir": "d", "checkpoint_dir": "c", "results_dir": "r", "lr": -1},
        {"backbone": "bert", "data_dir": "d", "checkpoint_dir": "c", "results_dir": "r", "warmup_ratio": 1.0},
        {"backbone": "bert", "data_dir": "d", "checkpoint_dir": "c", "results_dir": "r", "output_bias_init": "abc"},
    ]
    for i, bad in enumerate(bad_cases):
        p = _write_yaml(tmp_path / f"bad{i}.yaml", bad)
        with pytest.raises(ValueError):
            train_mod.load_config(str(p))


def test_t2_resolve_output_bias(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"stars": [1, 5, 5, 5]}).to_parquet(data_dir / "train.parquet", index=False)

    cfg = base_cfg(tmp_path, data_dir=str(data_dir), output_bias_init="train_mean")
    assert train_mod.resolve_output_bias(cfg) == pytest.approx(4.0)

    cfg = base_cfg(tmp_path, output_bias_init=3.5)
    assert train_mod.resolve_output_bias(cfg) == 3.5

    cfg = base_cfg(tmp_path, output_bias_init=None)
    assert train_mod.resolve_output_bias(cfg) is None


def test_t3_run_training_wiring(tmp_path):
    cfg = base_cfg(tmp_path, epochs=2)
    model = tiny_model()
    loaders = (
        make_loader(24, 4, seed=0, shuffle=True),
        make_loader(8, 4, seed=1),
        make_loader(4, 4, seed=2),
    )
    result = train_mod.run_training(cfg, model=model, loaders=loaders)
    assert result["run_name"] == "test_run"
    for key in ("best_epoch", "best_val_rmse", "train_seconds",
                "epoch_seconds_mean", "n_params", "history"):
        assert key in result

    run_dir = Path(cfg["results_dir"]) / "runs" / "test_run"
    saved_cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert saved_cfg["backbone"] == cfg["backbone"]
    assert saved_cfg["epochs"] == cfg["epochs"]
    assert (run_dir / "history.csv").exists()
    assert (run_dir / "run_meta.json").exists()
    assert (Path(cfg["checkpoint_dir"]) / "test_run" / "best_model.pt").exists()


def test_t4_seed_set_before_build(tmp_path, monkeypatch):
    recorded = {}

    def fake_build_model(backbone, pooling="mean", dropout=0.1,
                         pretrained=True, backbone_config=None, output_bias_init=None):
        # Ghi lại torch.rand(1) tại thời điểm build_model được gọi
        recorded["rand"] = torch.rand(1)
        recorded["args"] = dict(backbone=backbone, pooling=pooling,
                                dropout=dropout, pretrained=pretrained,
                                output_bias_init=output_bias_init)
        return tiny_model(backbone)

    monkeypatch.setattr(train_mod, "build_model", fake_build_model)
    monkeypatch.setattr(train_mod, "build_tokenizer", lambda b: object())

    def fake_loaders(data_dir, tokenizer, **kw):
        # Giả lập create_dataloaders tiêu thụ RNG toàn cục
        torch.rand(10)
        return (make_loader(24, 4, seed=0, shuffle=True),
                make_loader(8, 4, seed=1),
                make_loader(4, 4, seed=2))

    monkeypatch.setattr(train_mod, "create_dataloaders", fake_loaders)

    cfg = base_cfg(tmp_path, epochs=1, output_bias_init=3.7)
    train_mod.run_training(cfg)

    # Ngay sau set_seed(seed) mới thì torch.rand(1) phải có giá trị xác định
    torch.manual_seed(cfg["seed"])
    expected = torch.rand(1)
    assert torch.equal(recorded["rand"], expected)
    assert recorded["args"]["output_bias_init"] == pytest.approx(3.7)
    assert recorded["args"]["pretrained"] is True


def test_t5_cli_overrides(tmp_path, monkeypatch):
    calls = []

    def fake_run_training(cfg, **kwargs):
        calls.append((cfg, kwargs))
        return {"run_name": cfg["run_name"]}

    monkeypatch.setattr(train_mod, "run_training", fake_run_training)

    cfg_path = _minimal_yaml(tmp_path)
    rc = train_mod.main([
        "--config", str(cfg_path),
        "--set", "epochs=1", "--set", "loss=mse",
    ])
    assert rc == 0
    cfg, kwargs = calls[-1]
    assert cfg["epochs"] == 1 and isinstance(cfg["epochs"], int)
    assert cfg["loss"] == "mse"
    assert kwargs["resume"] is True and kwargs["overwrite"] is False

    # Lỗi cấu hình -> mã thoát khác 0
    rc = train_mod.main(["--config", str(cfg_path), "--set", "loss=l1"])
    assert rc != 0

    # --overwrite và --no-resume
    rc = train_mod.main([
        "--config", str(cfg_path), "--overwrite", "--no-resume",
    ])
    assert rc == 0
    _, kwargs = calls[-1]
    assert kwargs["overwrite"] is True and kwargs["resume"] is False
