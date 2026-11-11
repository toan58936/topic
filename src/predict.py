"""Suy luận và xuất kết quả dự đoán Test.

Spec: docs/define&designt/04_engine_and_utils.md mục 2.7
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig

from src import metrics
from src.engine import evaluate
from src.model_builder import BACKBONES, build_model
from src.utils import get_device, load_checkpoint, write_atomic


def _resolve_device(device):
    if device is None:
        return get_device("auto")
    return torch.device(device) if not isinstance(device, torch.device) else device


def _autocast_ctx(use_fp16: bool):
    import contextlib
    if use_fp16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _resolve_fp16(fp16, device: torch.device):
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


def predict(
    texts, model, tokenizer, *, device=None, max_length=256, batch_size=32,
    fp16=None, clip=True,
):
    """Dự đoán điểm cho một chuỗi (-> float) hoặc danh sách chuỗi (-> ndarray)."""
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    if not isinstance(texts, (list, tuple)) or len(texts) == 0:
        raise ValueError("texts phải là chuỗi đơn hoặc danh sách chuỗi không rỗng")
    if not all(isinstance(t, str) for t in texts):
        raise ValueError("Mọi phần tử của texts phải là chuỗi")
    if max_length < 1:
        raise ValueError(f"max_length phải >= 1, nhận: {max_length}")
    if batch_size < 1:
        raise ValueError(f"batch_size phải >= 1, nhận: {batch_size}")

    device = _resolve_device(device)
    use_fp16 = _resolve_fp16(fp16, device)
    was_training = model.training
    model.eval()
    model.to(device)
    preds = []
    try:
        with torch.no_grad(), _autocast_ctx(use_fp16):
            for i in range(0, len(texts), batch_size):
                batch_texts = list(texts[i : i + batch_size])
                enc = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                out = model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                ).float().cpu().numpy()
                preds.append(out)
    finally:
        if was_training:
            model.train()

    arr = np.concatenate(preds) if preds else np.array([])
    if clip:
        arr = np.clip(arr, 1.0, 5.0)
    return float(arr[0]) if single else arr


def load_trained_model(checkpoint_path, device=None, *, backbone_config=None):
    """Dựng lại mô hình từ best_model.pt, trả về (model eval, info)."""
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    run_config = ckpt.get("run_config")
    if run_config is None:
        raise ValueError(f"Checkpoint {checkpoint_path} thiếu run_config")
    model_state = ckpt.get("model_state_dict")
    if model_state is None:
        raise ValueError(f"Checkpoint {checkpoint_path} thiếu model_state_dict")

    backbone = run_config.get("backbone")
    pooling = run_config.get("pooling", "mean")
    dropout = run_config.get("dropout", 0.1)

    if backbone_config is None and ckpt.get("backbone_config"):
        model_type = BACKBONES[backbone]["model_type"]
        cfg_dict = dict(ckpt["backbone_config"])
        cfg_dict.pop("model_type", None)
        backbone_config = AutoConfig.for_model(model_type, **cfg_dict)
    if backbone_config is None:
        warnings.warn(
            f"Checkpoint {checkpoint_path} không có backbone_config; "
            "tải AutoConfig.from_pretrained (cần mạng hoặc cache).",
            UserWarning,
            stacklevel=2,
        )
        backbone_config = AutoConfig.from_pretrained(BACKBONES[backbone]["hf_name"])

    model = build_model(
        backbone, pooling=pooling, dropout=dropout,
        pretrained=False, backbone_config=backbone_config,
    )
    model.load_state_dict(model_state)
    model.eval()
    device = _resolve_device(device)
    model.to(device)

    info = {
        "run_name": run_config.get("run_name"),
        "backbone": backbone,
        "epoch": ckpt.get("epoch"),
        "val_rmse": ckpt.get("val_rmse"),
        "run_config": run_config,
    }
    return model, info


def export_test_predictions(
    model, test_loader, model_name, out_dir, *, device=None, fp16=None,
    overwrite=False, run_info=None,
) -> dict:
    """Chạy mô hình trên test_loader đúng một lần, ghi <model>_test.csv + meta json."""
    out_dir = Path(out_dir)
    csv_path = out_dir / f"{model_name}_test.csv"
    meta_path = out_dir / f"{model_name}_test_meta.json"
    if csv_path.exists() and not overwrite:
        raise FileExistsError(f"Đã tồn tại {csv_path}; dùng overwrite=True để ghi đè")

    device = _resolve_device(device)
    model.to(device)
    t0 = time.time()
    result = evaluate(model, test_loader, device, fp16=fp16)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.time() - t0

    df = pd.DataFrame({
        "review_id": result["review_id"],
        "true_stars": result["y_true"].astype(int),
        "pred_raw": result["y_pred_raw"],
        "pred": np.clip(result["y_pred_raw"], 1.0, 5.0),
    })
    write_atomic(str(csv_path), lambda p: df.to_csv(p, index=False))

    meta = {
        "model": model_name,
        "run_name": (run_info or {}).get("run_name", model_name),
        "checkpoint_epoch": (run_info or {}).get("epoch"),
        "n": len(df),
        "rmse_raw": metrics.rmse(result["y_true"], result["y_pred_raw"], clip=False),
        "inference_seconds": inference_seconds,
        "fp16": _resolve_fp16(fp16, device) if fp16 is not None else device.type == "cuda",
        "device": str(device),
    }
    write_atomic(
        str(meta_path),
        lambda p: Path(p).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"),
    )
    return meta


def _read_pred_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("review_id", "true_stars", "pred_raw", "pred"):
        if col not in df.columns:
            raise ValueError(f"{path} thiếu cột {col!r}")
    return df


def merge_test_predictions(
    model_names, preds_dir, test_parquet, out_dir, *,
    baseline_preds=None, overwrite=False,
) -> pd.DataFrame:
    """Ghép dự đoán các mô hình (+ baseline) thành test_predictions.csv."""
    if not model_names:
        raise ValueError("model_names không được rỗng")
    preds_dir = Path(preds_dir)
    out_dir = Path(out_dir)

    frames = []
    for name in model_names:
        path = preds_dir / f"{name}_test.csv"
        if not path.exists():
            raise FileNotFoundError(f"Thiếu file dự đoán: {path}")
        frames.append(_read_pred_csv(path))

    ref_ids = frames[0]["review_id"].tolist()
    ref_true = frames[0]["true_stars"].tolist()
    for name, df in zip(model_names, frames):
        if df["review_id"].tolist() != ref_ids:
            raise ValueError(f"{name}_test.csv có review_id khác thứ tự/nội dung")
        if not np.array_equal(df["true_stars"].to_numpy(), np.asarray(ref_true)):
            raise ValueError(f"{name}_test.csv có true_stars không khớp")

    out = pd.DataFrame({"review_id": ref_ids, "true_stars": np.asarray(ref_true, dtype=np.int64)})

    if baseline_preds is not None:
        if isinstance(baseline_preds, (str, Path)):
            base = pd.read_csv(baseline_preds)
        else:
            base = baseline_preds
        for col in ("review_id", "pred_baseline"):
            if col not in base.columns:
                raise ValueError(f"baseline_preds thiếu cột {col!r}")
        base_ids = set(base["review_id"])
        out_ids = set(out["review_id"])
        if base_ids != out_ids:
            raise ValueError(
                f"baseline_preds có review_id không khớp: "
                f"thiếu {sorted(out_ids - base_ids)[:5]}, "
                f"thừa {sorted(base_ids - out_ids)[:5]}"
            )
        base = base.set_index("review_id").loc[out["review_id"]].reset_index()
        out["pred_baseline"] = base["pred_baseline"].to_numpy()
        out["abs_err_baseline"] = (out["true_stars"] - out["pred_baseline"]).abs()

    for name, df in zip(model_names, frames):
        out[f"pred_{name}"] = df["pred"].to_numpy()

    for name in model_names:
        out[f"abs_err_{name}"] = (out["true_stars"] - out[f"pred_{name}"]).abs()

    # Cột abs_err_baseline phải đứng sau các cột pred (theo spec: pred_* rồi abs_err_*)
    cols = ["review_id", "true_stars"]
    if "pred_baseline" in out.columns:
        cols.append("pred_baseline")
    cols += [f"pred_{n}" for n in model_names]
    if "abs_err_baseline" in out.columns:
        cols.append("abs_err_baseline")
    cols += [f"abs_err_{n}" for n in model_names]
    out = out[cols]

    out_csv = out_dir / "test_predictions.csv"
    out_text_csv = out_dir / "test_predictions_with_text.csv"
    for p in (out_csv, out_text_csv):
        if p.exists() and not overwrite:
            raise FileExistsError(f"Đã tồn tại {p}; dùng overwrite=True để ghi đè")

    test_df = pd.read_parquet(test_parquet, columns=["review_id", "text"])
    text_map = dict(zip(test_df["review_id"], test_df["text"]))
    missing = [r for r in out["review_id"] if r not in text_map]
    if missing:
        raise ValueError(f"test_parquet thiếu review_id: {missing[:5]}")

    with_text = out.copy()
    with_text.insert(1, "text", out["review_id"].map(text_map))

    write_atomic(str(out_csv), lambda p: out.to_csv(p, index=False))
    write_atomic(str(out_text_csv), lambda p: with_text.to_csv(p, index=False))
    return out


def collect_summary_inputs(model_runs, results_dir) -> dict:
    """Gom số liệu cho build_model_summary từ run_meta.json và meta dự đoán."""
    results_dir = Path(results_dir)
    param_counts, train_seconds, inference_seconds = {}, {}, {}
    rmse_raw_by_model, seconds_per_epoch = {}, {}
    split_id = None
    for model, run_name in model_runs.items():
        run_meta_path = results_dir / "runs" / run_name / "run_meta.json"
        if not run_meta_path.exists():
            raise FileNotFoundError(f"Thiếu run_meta.json: {run_meta_path}")
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        param_counts[model] = run_meta.get("n_params")
        train_seconds[model] = run_meta.get("train_seconds")
        seconds_per_epoch[model] = run_meta.get("epoch_seconds_mean")
        if split_id is None:
            split_id = run_meta.get("split_id")

        pred_meta_path = results_dir / "preds" / f"{model}_test_meta.json"
        if not pred_meta_path.exists():
            raise FileNotFoundError(f"Thiếu meta dự đoán: {pred_meta_path}")
        pred_meta = json.loads(pred_meta_path.read_text(encoding="utf-8"))
        inference_seconds[model] = pred_meta.get("inference_seconds")
        rmse_raw_by_model[model] = pred_meta.get("rmse_raw")
        if split_id is None:
            split_id = pred_meta.get("split_id")

    return {
        "param_counts": param_counts,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "rmse_raw_by_model": rmse_raw_by_model,
        "seconds_per_epoch": seconds_per_epoch,
        "split_id": split_id,
    }
