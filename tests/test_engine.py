"""Tests cho src/engine.py — spec docs/tests/04_engine_and_utils_tests.md nhóm B, C."""

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import base_cfg, make_loader, tiny_model
from src.engine import (
    build_loss,
    build_optimizer_and_scheduler,
    evaluate,
    fit,
    train_one_epoch,
)
from src.model_builder import count_parameters
from src.utils import load_checkpoint, set_seed


def _cpu():
    return torch.device("cpu")


# ---------------- Nhóm B: Thành phần engine ----------------

def test_l1_build_loss_values():
    pred = torch.tensor([3.0, 1.5])
    true = torch.tensor([1.0, 1.0])
    # Huber delta=1: |e|=2 -> 1.5 ; |e|=0.5 -> 0.125
    huber = build_loss("huber", delta=1.0)
    vals = huber(pred, true)
    assert vals.item() == pytest.approx((1.5 + 0.125) / 2)
    huber_half = build_loss("huber", delta=0.5)
    # |e|=2 với delta=0.5: 0.5*(4-0.25)+0.5*0.5*0.25 = 1.875+0.0625? tính theo công thức:
    # huber(delta) = 0.5*e^2 nếu |e|<=d else d*(|e|-0.5d)
    # e=2,d=0.5 -> 0.5*(2-0.25)=0.875 ; e=0.5 -> 0.5*0.25=0.125
    vals2 = huber_half(pred, true)
    assert vals2.item() == pytest.approx((0.875 + 0.125) / 2)
    mse = build_loss("mse")
    assert mse(pred, true).item() == pytest.approx((4.0 + 0.25) / 2)
    with pytest.raises(ValueError):
        build_loss("l1")
    with pytest.raises(ValueError):
        build_loss("huber", delta=0.0)


def test_l2_optimizer_groups():
    model = tiny_model()
    opt, sched = build_optimizer_and_scheduler(
        model, lr=1e-3, weight_decay=0.01, total_steps=10, warmup_ratio=0.2
    )
    assert isinstance(opt, torch.optim.AdamW)
    seen = {}
    for group in opt.param_groups:
        for p in group["params"]:
            assert id(p) not in seen, "tham số xuất hiện ở hai nhóm"
            seen[id(p)] = group
    model_params = {id(p): p for p in model.parameters()}
    assert set(seen) == set(model_params), "có tham số bị bỏ sót"
    for pid, group in seen.items():
        p = model_params[pid]
        if p.ndim <= 1:
            assert group["weight_decay"] == 0.0
        else:
            assert group["weight_decay"] == 0.01
    # Scheduler warmup đưa lr về 0 ngay khi khởi tạo; kiểm tra lr gốc đã truyền
    for group in opt.param_groups:
        assert group.get("initial_lr", group["lr"]) == pytest.approx(1e-3)


def test_l3_scheduler_warmup_decay():
    model = tiny_model()
    opt, sched = build_optimizer_and_scheduler(
        model, lr=1.0, weight_decay=0.01, total_steps=10, warmup_ratio=0.2
    )
    # Hệ số tốc độ học tại bước 0, 1, 2, 6, 10 lần lượt: 0, 0.5, 1.0, 0.5, 0.0
    checkpoints = {0, 1, 2, 6, 10}
    observed = {}
    observed[0] = sched.get_last_lr()[0]
    for step in range(1, 11):
        opt.step()
        sched.step()
        if step in checkpoints:
            observed[step] = sched.get_last_lr()[0]
    expected = {0: 0.0, 1: 0.5, 2: 1.0, 6: 0.5, 10: 0.0}
    for step, exp in expected.items():
        assert observed[step] == pytest.approx(exp, abs=1e-9), f"bước {step}"


def test_l4_train_one_epoch_updates():
    model = tiny_model()
    loader = make_loader(8, 4, seed=0)
    opt, sched = build_optimizer_and_scheduler(
        model, lr=1e-2, weight_decay=0.01, total_steps=4, warmup_ratio=0.0
    )
    loss_fn = build_loss("mse")
    before = [p.detach().clone() for p in model.parameters()]
    model.train()
    stats = train_one_epoch(
        model, loader, opt, sched, loss_fn, _cpu(), fp16=False, grad_clip=1.0
    )
    assert np.isfinite(stats["train_loss"])
    assert stats["n_batches"] == 2
    assert any(
        not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters())
    )
    assert model.training


def test_l5_non_finite_loss():
    model = tiny_model()
    loader = make_loader(8, 4, seed=0, nan_label_at=3)
    opt, sched = build_optimizer_and_scheduler(
        model, lr=1e-2, weight_decay=0.01, total_steps=4, warmup_ratio=0.0
    )
    with pytest.raises(FloatingPointError):
        train_one_epoch(
            model, loader, opt, sched, build_loss("mse"), _cpu(), fp16=False
        )


def test_l6_evaluate():
    model = tiny_model()
    loader = make_loader(7, 3, seed=0)
    model.train()
    before = [p.detach().clone() for p in model.parameters()]

    res1 = evaluate(model, loader, _cpu(), loss_fn=build_loss("mse"))
    res2 = evaluate(model, loader, _cpu())

    assert res1["review_id"] == [f"r{i}" for i in range(7)]
    assert res1["y_true"].shape == (7,)
    assert res1["y_pred_raw"].shape == (7,)
    assert np.isfinite(res1["y_pred_raw"]).all()
    assert np.array_equal(res1["y_pred_raw"], res2["y_pred_raw"])
    assert isinstance(res1["loss"], float)
    assert res2["loss"] is None
    assert all(torch.equal(b, p.detach()) for b, p in zip(before, model.parameters()))
    assert model.training  # chế độ ban đầu được khôi phục
    assert not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())

    with pytest.raises(ValueError):
        evaluate(model, make_loader(0, 4), _cpu())


def test_l7_fp16_on_cpu():
    model = tiny_model()
    loader = make_loader(8, 4, seed=0)
    opt, sched = build_optimizer_and_scheduler(
        model, lr=1e-2, weight_decay=0.01, total_steps=4, warmup_ratio=0.0
    )
    with pytest.warns(UserWarning):
        stats = train_one_epoch(
            model, loader, opt, sched, build_loss("mse"), _cpu(), fp16=True
        )
    assert np.isfinite(stats["train_loss"])


# ---------------- Nhóm C: fit ----------------

def _fit_kwargs(model, tmp_path, epochs=3, **cfg_over):
    cfg = base_cfg(tmp_path, epochs=epochs, **cfg_over)
    train_loader = make_loader(24, 4, seed=0, shuffle=True)
    val_loader = make_loader(8, 4, seed=1)
    return model, train_loader, val_loader, cfg


def test_f1_fit_history_and_best_checkpoint(tmp_path):
    model = tiny_model()
    m, tr, va, cfg = _fit_kwargs(model, tmp_path)
    result = fit(m, tr, va, cfg, cfg["checkpoint_dir"], cfg["results_dir"], device=_cpu())

    history = result["history"]
    assert len(history) == 3
    for col in ("epoch", "train_loss", "val_loss", "val_rmse", "val_mae",
                "val_rmse_raw", "lr", "epoch_seconds", "is_best"):
        assert col in history.columns

    hist_csv = pd.read_csv(Path(result["results_dir"]) / "history.csv")
    pd.testing.assert_frame_equal(
        history.reset_index(drop=True), hist_csv.reset_index(drop=True),
        check_dtype=False,
    )
    assert bool(history.iloc[0]["is_best"]) is True

    best_idx = history["val_rmse"].idxmin()
    ckpt = load_checkpoint(str(Path(result["checkpoint_dir"]) / "best_model.pt"))
    assert ckpt["val_rmse"] == pytest.approx(float(history.loc[best_idx, "val_rmse"]))
    assert ckpt["epoch"] == int(history.loc[best_idx, "epoch"])

    # Nạp lại best weights -> evaluate cho cùng RMSE
    model2 = tiny_model()
    model2.load_state_dict(ckpt["model_state_dict"])
    val_res = evaluate(model2, va, _cpu())
    rmse_again = float(np.sqrt(np.mean((np.clip(val_res["y_pred_raw"], 1, 5) - val_res["y_true"]) ** 2)))
    assert rmse_again == pytest.approx(ckpt["val_rmse"], abs=1e-5)


def test_f2_resume_matches_uninterrupted(tmp_path, monkeypatch):
    calls = {"n": 0}
    orig = __import__("src.engine", fromlist=["train_one_epoch"]).train_one_epoch

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    monkeypatch.setattr("src.engine.train_one_epoch", counting)

    # Lần A: chạy liền 3 epoch
    dir_a = tmp_path / "a"
    set_seed(42)
    m_a = tiny_model()
    cfg_a = base_cfg(dir_a, epochs=3)
    tr_a = make_loader(24, 4, seed=0, shuffle=True)
    va_a = make_loader(8, 4, seed=1)
    res_a = fit(m_a, tr_a, va_a, cfg_a, cfg_a["checkpoint_dir"], cfg_a["results_dir"], device=_cpu())
    calls["n"] = 0

    # Lần B: dừng sau epoch 2 rồi tiếp tục
    dir_b = tmp_path / "b"
    cfg_b = base_cfg(dir_b, epochs=3)
    set_seed(42)
    m_b1 = tiny_model()
    tr_b1 = make_loader(24, 4, seed=0, shuffle=True)
    va_b1 = make_loader(8, 4, seed=1)
    fit(m_b1, tr_b1, va_b1, cfg_b, cfg_b["checkpoint_dir"], cfg_b["results_dir"],
        device=_cpu(), stop_after_epoch=2)
    assert calls["n"] == 2

    # Dựng model + loader mới (cùng seed), tiếp tục
    set_seed(42)
    m_b2 = tiny_model()
    tr_b2 = make_loader(24, 4, seed=0, shuffle=True)
    va_b2 = make_loader(8, 4, seed=1)
    res_b = fit(m_b2, tr_b2, va_b2, cfg_b, cfg_b["checkpoint_dir"], cfg_b["results_dir"],
                device=_cpu())
    assert calls["n"] == 3

    hist_a = res_a["history"]
    hist_b = res_b["history"]
    assert len(hist_b) == 3
    for col in ("train_loss", "val_rmse"):
        assert hist_b[col].tolist() == pytest.approx(hist_a[col].tolist(), abs=1e-6)

    ckpt_a = load_checkpoint(str(Path(res_a["checkpoint_dir"]) / "best_model.pt"))
    ckpt_b = load_checkpoint(str(Path(res_b["checkpoint_dir"]) / "best_model.pt"))
    for k in ckpt_a["model_state_dict"]:
        assert torch.allclose(ckpt_a["model_state_dict"][k], ckpt_b["model_state_dict"][k], atol=1e-6)

    meta_b = json.loads((Path(res_b["results_dir"]) / "run_meta.json").read_text())
    assert meta_b["train_seconds"] == pytest.approx(
        float(hist_b["epoch_seconds"].sum()), rel=1e-6
    )


def test_f3_resume_and_overwrite_guards(tmp_path):
    # Chuẩn bị: chạy dừng sau epoch 1
    cfg = base_cfg(tmp_path, epochs=3)
    tr = make_loader(24, 4, seed=0, shuffle=True)
    va = make_loader(8, 4, seed=1)
    m = tiny_model()
    fit(m, tr, va, cfg, cfg["checkpoint_dir"], cfg["results_dir"], device=_cpu(),
        stop_after_epoch=1)

    # (a) tiếp tục với lr khác -> ValueError nêu 'lr'
    cfg_bad = dict(cfg, lr=1e-2)
    m2 = tiny_model()
    tr2 = make_loader(24, 4, seed=0, shuffle=True)
    va2 = make_loader(8, 4, seed=1)
    with pytest.raises(ValueError, match="lr"):
        fit(m2, tr2, va2, cfg_bad, cfg["checkpoint_dir"], cfg["results_dir"], device=_cpu())

    # (b) resume=False -> FileExistsError
    with pytest.raises(FileExistsError):
        fit(tiny_model(), make_loader(24, 4, seed=0, shuffle=True), va2,
            cfg, cfg["checkpoint_dir"], cfg["results_dir"], device=_cpu(), resume=False)

    # (e) chỉ khác num_workers -> tiếp tục bình thường (chạy nốt 2 epoch)
    cfg_nw = dict(cfg, num_workers=2)
    res_e = fit(tiny_model(), make_loader(24, 4, seed=0, shuffle=True), va2,
                cfg_nw, cfg["checkpoint_dir"], cfg["results_dir"], device=_cpu())
    assert res_e["history"] is not None

    # (c) chạy lại trên lần chạy đã hoàn tất -> FileExistsError
    with pytest.raises(FileExistsError):
        fit(tiny_model(), make_loader(24, 4, seed=0, shuffle=True), va2,
            cfg, cfg["checkpoint_dir"], cfg["results_dir"], device=_cpu())

    # (d) overwrite=True -> chạy lại từ đầu, history 3 dòng
    res_d = fit(tiny_model(), make_loader(24, 4, seed=0, shuffle=True), va2,
                cfg, cfg["checkpoint_dir"], cfg["results_dir"], device=_cpu(), overwrite=True)
    assert len(res_d["history"]) == 3


def test_f3b_overwrite_beats_resume(tmp_path, monkeypatch):
    """(f) resume=True + overwrite=True -> overwrite thắng, chạy lại từ epoch 1."""
    cfg = base_cfg(tmp_path, epochs=3)
    tr = make_loader(24, 4, seed=0, shuffle=True)
    va = make_loader(8, 4, seed=1)
    fit(tiny_model(), tr, va, cfg, cfg["checkpoint_dir"], cfg["results_dir"],
        device=_cpu(), stop_after_epoch=1)

    calls = {"n": 0}
    import src.engine as engine
    orig = engine.train_one_epoch

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)
    monkeypatch.setattr(engine, "train_one_epoch", counting)

    res = fit(tiny_model(), make_loader(24, 4, seed=0, shuffle=True), va,
              cfg, cfg["checkpoint_dir"], cfg["results_dir"],
              device=_cpu(), resume=True, overwrite=True)
    assert calls["n"] == 3
    assert len(res["history"]) == 3


def test_f4_completion_artifacts(tmp_path):
    cfg = base_cfg(tmp_path, epochs=3)
    tr = make_loader(24, 4, seed=0, shuffle=True)
    va = make_loader(8, 4, seed=1)
    model = tiny_model()
    result = fit(model, tr, va, cfg, cfg["checkpoint_dir"], cfg["results_dir"], device=_cpu())

    ckpt_dir = Path(result["checkpoint_dir"])
    assert not (ckpt_dir / "last.pt").exists()
    assert (ckpt_dir / "best_model.pt").exists()

    # keep_last=True -> còn last.pt
    cfg2 = base_cfg(tmp_path / "k", epochs=2)
    res2 = fit(tiny_model(), make_loader(24, 4, seed=0, shuffle=True),
               make_loader(8, 4, seed=1), cfg2, cfg2["checkpoint_dir"],
               cfg2["results_dir"], device=_cpu(), keep_last=True)
    assert (Path(res2["checkpoint_dir"]) / "last.pt").exists()

    meta = json.loads((Path(result["results_dir"]) / "run_meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["epochs_done"] == 3
    assert meta["best_epoch"] is not None
    assert meta["device"] == "cpu"
    assert meta["n_params"] == count_parameters(model)
    hist = result["history"]
    assert meta["train_seconds"] == pytest.approx(sum(hist["epoch_seconds"]), rel=1e-6)
    assert meta["epoch_seconds_mean"] == pytest.approx(hist["epoch_seconds"].mean(), rel=1e-6)
    assert (hist["epoch_seconds"] > 0).all()


def test_f5_test_set_never_touched(tmp_path):
    import src.train as train_mod
    from conftest import ExplodingLoader

    cfg = base_cfg(tmp_path, epochs=2)
    model = tiny_model()
    loaders = (
        make_loader(24, 4, seed=0, shuffle=True),
        make_loader(8, 4, seed=1),
        ExplodingLoader(),
    )
    result = train_mod.run_training(cfg, model=model, loaders=loaders)
    assert result["run_name"] == "test_run"

    sig = inspect.signature(fit)
    assert not any("test" in p.lower() for p in sig.parameters)


def test_f6_fit_invalid_arguments(tmp_path):
    cfg = base_cfg(tmp_path, epochs=0)
    tr = make_loader(24, 4, seed=0, shuffle=True)
    va = make_loader(8, 4, seed=1)
    with pytest.raises(ValueError):
        fit(tiny_model(), tr, va, cfg, cfg["checkpoint_dir"], cfg["results_dir"], device=_cpu())

    cfg = base_cfg(tmp_path / "e", epochs=3)
    with pytest.raises(ValueError):
        fit(tiny_model(), make_loader(0, 4), va, cfg, cfg["checkpoint_dir"], cfg["results_dir"], device=_cpu())
    with pytest.raises(ValueError):
        fit(tiny_model(), tr, make_loader(0, 4), cfg, cfg["checkpoint_dir"], cfg["results_dir"], device=_cpu())

    cfg_bad_loss = base_cfg(tmp_path / "l", epochs=3, loss="l1")
    with pytest.raises(ValueError):
        fit(tiny_model(), tr, va, cfg_bad_loss, cfg_bad_loss["checkpoint_dir"], cfg_bad_loss["results_dir"], device=_cpu())

    cfg_s0 = base_cfg(tmp_path / "s0", epochs=3)
    with pytest.raises(ValueError):
        fit(tiny_model(), tr, va, cfg_s0, cfg_s0["checkpoint_dir"], cfg_s0["results_dir"],
            device=_cpu(), stop_after_epoch=0)
    with pytest.raises(ValueError):
        fit(tiny_model(), tr, va, cfg_s0, cfg_s0["checkpoint_dir"], cfg_s0["results_dir"],
            device=_cpu(), stop_after_epoch=3)
