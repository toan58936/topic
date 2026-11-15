"""Training engine: loss, optimizer/scheduler, epoch train/eval, fit, compare_runs.

Spec: docs/define&designt/04_engine_and_utils.md mục 2.5
"""

import contextlib
import csv
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

from src import metrics
from src.model_builder import count_parameters
from src.utils import get_device, load_checkpoint, save_checkpoint, write_atomic

FORMAT_VERSION = 1
RESUME_IGNORE_KEYS = {
    "num_workers", "device", "log_every", "data_dir",
    "checkpoint_dir", "results_dir",
}

HISTORY_COLUMNS = [
    "epoch", "train_loss", "val_loss", "val_rmse", "val_mae",
    "val_rmse_raw", "lr", "epoch_seconds", "is_best",
]


def build_loss(name: str, delta: float = 1.0) -> nn.Module:
    """Huber (delta) hoặc MSE loss."""
    if name == "huber":
        if delta <= 0:
            raise ValueError(f"delta phải > 0, nhận: {delta}")
        return nn.HuberLoss(delta=float(delta))
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"loss phải thuộc {{huber, mse}}, nhận: {name!r}")


def build_optimizer_and_scheduler(
    model, lr, weight_decay, total_steps, warmup_ratio
):
    """AdamW (weight decay tắt cho tham số ndim<=1) + lịch tuyến tính có warmup."""
    decay, no_decay = [], []
    for p in model.parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    num_warmup = math.ceil(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def _resolve_fp16(fp16, device: torch.device):
    """fp16=None -> tự động (chỉ CUDA). fp16=True trên CPU -> cảnh báo, tắt."""
    if fp16 is None:
        return device.type == "cuda"
    if fp16 and device.type != "cuda":
        warnings.warn(
            "fp16=True nhưng device không phải CUDA; tự động tắt fp16.",
            UserWarning,
            stacklevel=3,
        )
        return False
    return bool(fp16)


def train_one_epoch(
    model, loader, optimizer, scheduler, loss_fn, device, *,
    scaler=None, fp16=True, grad_clip=1.0, log_every=100, epoch=0,
) -> dict:
    """Một epoch huấn luyện; trả về train_loss (trung bình theo mẫu), n_batches, lr."""
    model.train()
    if fp16 and device.type != "cuda":
        warnings.warn(
            "fp16=True nhưng device không phải CUDA; tự động tắt fp16.",
            UserWarning,
            stacklevel=2,
        )
        fp16 = False
    total_loss = 0.0
    total_samples = 0
    n_batches = 0
    t0 = time.time()
    use_fp16 = scaler is not None or (fp16 and device.type == "cuda")
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_fp16
        else contextlib.nullcontext()
    )

    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        n = labels.shape[0]

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(out.float(), labels)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError(
                f"Loss không hữu hạn ({loss_value}) tại bước {step}, epoch {epoch}"
            )

        skipped = False
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < scale_before
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if not skipped:
            scheduler.step()

        total_loss += loss_value * n
        total_samples += n
        n_batches += 1
        if log_every and step % log_every == 0:
            elapsed = time.time() - t0
            print(
                f"  epoch {epoch} step {step}/{n_batches}+ "
                f"loss={loss_value:.4f} ({step / elapsed:.1f} it/s)"
            )

    return {
        "train_loss": total_loss / max(total_samples, 1),
        "n_batches": n_batches,
        "lr": scheduler.get_last_lr()[0],
    }


def evaluate(model, loader, device, *, loss_fn=None, fp16=None) -> dict:
    """Đánh giá trên loader (không xáo trộn); trả về review_id, y_true, y_pred_raw, loss."""
    if len(loader) == 0:
        raise ValueError("loader rỗng")
    was_training = model.training
    model.eval()
    use_fp16 = _resolve_fp16(fp16, torch.device(device) if not isinstance(device, torch.device) else device)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_fp16
        else contextlib.nullcontext()
    )
    review_ids, y_true, y_pred = [], [], []
    total_loss = 0.0
    total_samples = 0
    try:
        with torch.no_grad(), autocast_ctx:
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask).float()
                review_ids.extend(batch["review_id"])
                y_true.append(labels.detach().cpu().numpy())
                y_pred.append(out.detach().cpu().numpy())
                if loss_fn is not None:
                    batch_loss = loss_fn(out, labels)
                    total_loss += float(batch_loss.item()) * labels.shape[0]
                    total_samples += labels.shape[0]
    finally:
        if was_training:
            model.train()

    return {
        "review_id": review_ids,
        "y_true": np.concatenate(y_true) if y_true else np.array([]),
        "y_pred_raw": np.concatenate(y_pred) if y_pred else np.array([]),
        "loss": (total_loss / total_samples) if loss_fn is not None else None,
    }


def _atomic_json(path: Path, payload: dict) -> None:
    write_atomic(str(path), lambda p: Path(p).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    ))


def _atomic_csv(path: Path, rows: list, fieldnames: list) -> None:
    def _write(p):
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    write_atomic(str(path), _write)


def _run_meta_payload(run_config, status, epochs_done, best_epoch, best_val_rmse,
                      n_params, train_seconds, epoch_seconds_mean, device):
    import transformers
    import torch as _torch

    return {
        "run_name": run_config.get("run_name"),
        "backbone": run_config.get("backbone"),
        "loss": run_config.get("loss"),
        "seed": run_config.get("seed"),
        "status": status,
        "epochs_done": epochs_done,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "n_params": n_params,
        "train_seconds": train_seconds,
        "epoch_seconds_mean": epoch_seconds_mean,
        "device": str(device),
        "torch_version": _torch.__version__,
        "transformers_version": transformers.__version__,
    }


def _config_diff(stored: dict, current: dict) -> list:
    keys = (set(stored) | set(current)) - RESUME_IGNORE_KEYS
    return sorted(k for k in keys if stored.get(k) != current.get(k))


def fit(
    model, train_loader, val_loader, run_config, checkpoint_dir, results_dir, *,
    device=None, resume=True, overwrite=False, keep_last=False, stop_after_epoch=None,
) -> dict:
    """Huấn luyện `epochs` epoch: checkpoint mỗi epoch, resume, history.csv, run_meta.json."""
    epochs = int(run_config.get("epochs", 3))
    if epochs < 1:
        raise ValueError(f"epochs phải >= 1, nhận: {epochs}")
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("train_loader và val_loader không được rỗng")
    if stop_after_epoch is not None and not (1 <= stop_after_epoch <= epochs - 1):
        raise ValueError(
            f"stop_after_epoch phải thuộc [1, epochs-1]={[1, epochs - 1]}, "
            f"nhận: {stop_after_epoch}"
        )

    device = device or get_device(run_config.get("device", "auto"))
    device = torch.device(device) if not isinstance(device, torch.device) else device
    run_name = run_config.get("run_name") or run_config.get("backbone")
    ckpt_run_dir = Path(checkpoint_dir) / run_name
    res_run_dir = Path(results_dir) / "runs" / run_name
    last_pt = ckpt_run_dir / "last.pt"
    best_pt = ckpt_run_dir / "best_model.pt"
    history_path = res_run_dir / "history.csv"
    meta_path = res_run_dir / "run_meta.json"

    # --- Quy tắc resume / overwrite (bảng mục 2.5 spec) ---
    completed = False
    if meta_path.exists():
        try:
            completed = json.loads(meta_path.read_text(encoding="utf-8")).get("status") == "completed"
        except Exception:
            completed = False

    if overwrite:
        for p in (last_pt, best_pt, history_path, meta_path):
            if p.exists():
                p.unlink()
        ckpt_run_dir.mkdir(parents=True, exist_ok=True)
        res_run_dir.mkdir(parents=True, exist_ok=True)
        start_epoch, history, best_val_rmse, best_epoch = 1, [], float("inf"), None
        train_seconds = 0.0
    elif completed:
        raise FileExistsError(
            f"Lần chạy {run_name!r} đã hoàn tất (run_meta.status=completed). "
            "Dùng overwrite=True để chạy lại."
        )
    elif resume and last_pt.exists():
        state = load_checkpoint(str(last_pt), map_location="cpu")
        stored_cfg = state.get("run_config", {})
        diff = _config_diff(stored_cfg, run_config)
        if diff:
            raise ValueError(
                f"Cấu hình khác với lần chạy đang dở ở {last_pt}: {diff}"
            )
        start_epoch = int(state["epoch"]) + 1
        history = list(state.get("history", []))
        best_val_rmse = state.get("best_val_rmse", float("inf"))
        best_epoch = state.get("best_epoch")
        train_seconds = float(state.get("train_seconds", 0.0))
    elif last_pt.exists() or best_pt.exists() or history_path.exists() or meta_path.exists():
        raise FileExistsError(
            f"Đã có kết quả của lần chạy {run_name!r} nhưng resume=False "
            "và overwrite=False."
        )
    else:
        start_epoch, history, best_val_rmse, best_epoch = 1, [], float("inf"), None
        train_seconds = 0.0

    ckpt_run_dir.mkdir(parents=True, exist_ok=True)
    res_run_dir.mkdir(parents=True, exist_ok=True)

    # --- Dựng loss, optimizer, scheduler, scaler ---
    loss_fn = build_loss(run_config.get("loss", "huber"), run_config.get("huber_delta", 1.0))
    total_steps = epochs * len(train_loader)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        lr=float(run_config.get("lr", 2e-5)),
        weight_decay=float(run_config.get("weight_decay", 0.01)),
        total_steps=total_steps,
        warmup_ratio=float(run_config.get("warmup_ratio", 0.1)),
    )
    fp16_cfg = bool(run_config.get("fp16", True))
    if fp16_cfg and device.type != "cuda":
        warnings.warn(
            "fp16=True nhưng device không phải CUDA; tự động tắt fp16.",
            UserWarning,
            stacklevel=2,
        )
    use_fp16 = fp16_cfg and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    model.to(device)

    # --- Tiếp tục từ last.pt ---
    if resume and start_epoch > 1 and last_pt.exists():
        state = load_checkpoint(str(last_pt), map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])

    seed = int(run_config.get("seed", 42))
    log_every = int(run_config.get("log_every", 100) or 0)
    grad_clip = run_config.get("grad_clip")
    grad_clip = float(grad_clip) if grad_clip is not None else None
    n_params = count_parameters(model)
    if getattr(train_loader, "generator", None) is None:
        warnings.warn(
            "train_loader không có generator; thứ tự xáo trộn sẽ không "
            "tái lập được khi tiếp tục.",
            UserWarning,
            stacklevel=2,
        )
    try:
        backbone_cfg = model.backbone.config.to_dict()
    except Exception:
        backbone_cfg = {}

    def _save_last(epoch_done):
        save_checkpoint(str(last_pt), {
            "format_version": FORMAT_VERSION,
            "epoch": epoch_done,
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "history": history,
            "best_val_rmse": best_val_rmse,
            "best_epoch": best_epoch,
            "train_seconds": train_seconds,
            "run_config": run_config,
        })

    def _save_best(epoch_done, val_rmse):
        save_checkpoint(str(best_pt), {
            "format_version": FORMAT_VERSION,
            "epoch": epoch_done,
            "val_rmse": val_rmse,
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "run_config": run_config,
            "backbone_config": backbone_cfg,
        })

    def _write_meta(status):
        epoch_seconds_mean = (
            sum(h["epoch_seconds"] for h in history) / len(history) if history else None
        )
        _atomic_json(meta_path, _run_meta_payload(
            run_config, status, len(history), best_epoch, best_val_rmse,
            n_params, train_seconds, epoch_seconds_mean, device,
        ))

    def _write_history():
        _atomic_csv(history_path, history, HISTORY_COLUMNS)

    # --- Vòng lặp epoch ---
    for epoch in range(start_epoch, epochs + 1):
        generator = getattr(train_loader, "generator", None)
        if generator is not None:
            generator.manual_seed(seed + epoch)

        t0 = time.time()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device,
            scaler=scaler, fp16=use_fp16, grad_clip=grad_clip,
            log_every=log_every, epoch=epoch,
        )
        val = evaluate(model, val_loader, device, loss_fn=loss_fn, fp16=None)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_seconds = time.time() - t0

        val_rmse = metrics.rmse(val["y_true"], val["y_pred_raw"])
        val_mae = metrics.mae(val["y_true"], val["y_pred_raw"])
        val_rmse_raw = metrics.rmse(val["y_true"], val["y_pred_raw"], clip=False)
        train_seconds += epoch_seconds
        is_best = val_rmse < best_val_rmse
        if is_best:
            best_val_rmse = val_rmse
            best_epoch = epoch

        history.append({
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "val_loss": val["loss"],
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_rmse_raw": val_rmse_raw,
            "lr": train_stats["lr"],
            "epoch_seconds": epoch_seconds,
            "is_best": bool(is_best),
        })
        _write_history()
        _write_meta("running")
        if is_best:
            _save_best(epoch, val_rmse)
        _save_last(epoch)

        if stop_after_epoch is not None and epoch == stop_after_epoch:
            return {
                "run_name": run_name, "status": "running",
                "epochs_done": len(history), "best_epoch": best_epoch,
                "best_val_rmse": best_val_rmse, "train_seconds": train_seconds,
                "epoch_seconds_mean": epoch_seconds,
                "n_params": n_params, "history": pd.DataFrame(history),
                "checkpoint_dir": str(ckpt_run_dir), "results_dir": str(res_run_dir),
            }

    # Hoàn tất
    _write_meta("completed")
    if not keep_last and last_pt.exists():
        last_pt.unlink()

    epoch_seconds_mean = sum(h["epoch_seconds"] for h in history) / len(history)
    return {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "train_seconds": train_seconds,
        "epoch_seconds_mean": epoch_seconds_mean,
        "n_params": n_params,
        "history": pd.DataFrame(history),
        "checkpoint_dir": str(ckpt_run_dir),
        "results_dir": str(res_run_dir),
    }


def compare_runs(run_names, results_dir) -> pd.DataFrame:
    """Bảng so sánh RMSE Val tốt nhất của nhiều lần chạy."""
    rows = []
    for name in run_names:
        run_dir = Path(results_dir) / "runs" / name
        history_path = run_dir / "history.csv"
        if not history_path.exists():
            raise FileNotFoundError(f"Thiếu history.csv của lần chạy {name!r}")
        df = pd.read_csv(history_path)
        best_idx = df["val_rmse"].idxmin()
        best = df.loc[best_idx]
        meta = {}
        meta_path = run_dir / "run_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows.append({
            "run_name": name,
            "best_epoch": int(best["epoch"]),
            "best_val_rmse": float(best["val_rmse"]),
            "best_val_mae": float(best["val_mae"]),
            "train_seconds": meta.get("train_seconds"),
            "epoch_seconds_mean": meta.get("epoch_seconds_mean"),
        })
    out = pd.DataFrame(rows)
    first = out["best_val_rmse"].iloc[0]
    out["rel_diff_pct_vs_first"] = 100.0 * (first - out["best_val_rmse"]) / first
    return out
