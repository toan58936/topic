"""Tests cho src/predict.py — spec docs/tests/04_engine_and_utils_tests.md nhóm E."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import src.train as train_mod
from conftest import FakeTokenizer, base_cfg, make_loader, tiny_model
from src.engine import evaluate
from src.predict import (
    collect_summary_inputs,
    export_test_predictions,
    load_trained_model,
    merge_test_predictions,
    predict,
)


def _cpu():
    return torch.device("cpu")


# ---------------- P1-P4: predict ----------------

def test_p1_predict_single_list_and_clip():
    model = tiny_model()
    tok = FakeTokenizer()

    single = predict("great food", model, tok)
    assert isinstance(single, float) and 1.0 <= single <= 5.0

    texts = ["great food", "terrible service", "okay-ish meal",
             "loved the pasta", "would not recommend"]
    arr = predict(texts, model, tok)
    assert isinstance(arr, np.ndarray) and arr.shape == (5,)
    assert ((arr >= 1.0) & (arr <= 5.0)).all()
    assert arr[0] == pytest.approx(single, abs=1e-5)

    # Ép bias head = 10 -> clip cho 5.0, clip=False cho ~10
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(10.0)
    model.eval()
    out_clip = predict("anything", model, tok, clip=True)
    out_raw = predict("anything", model, tok, clip=False)
    assert out_clip == pytest.approx(5.0)
    assert out_raw == pytest.approx(10.0, abs=0.5)

    with torch.no_grad():
        model.head.bias.fill_(-3.0)
    assert predict("anything", model, tok, clip=True) == pytest.approx(1.0)


def test_p2_predict_batch_size_invariance():
    model = tiny_model()
    tok = FakeTokenizer()
    texts = [
        "short",
        "a much longer review with many words to tokenize here",
        "mid length text for the model",
        "x",
        "another fairly long piece of review text with extra words",
        "tiny",
        "the quick brown fox jumps over the lazy dog again and again",
    ]
    r1 = predict(texts, model, tok, batch_size=1)
    r3 = predict(texts, model, tok, batch_size=3)
    r32 = predict(texts, model, tok, batch_size=32)
    assert np.allclose(r1, r3, atol=1e-5)
    assert np.allclose(r1, r32, atol=1e-5)


def test_p3_predict_invalid_inputs():
    model = tiny_model()
    tok = FakeTokenizer()
    with pytest.raises(ValueError):
        predict([], model, tok)
    with pytest.raises(ValueError):
        predict(["ok", 123], model, tok)
    with pytest.raises(ValueError):
        predict(None, model, tok)
    with pytest.raises(ValueError):
        predict("ok", model, tok, batch_size=0)
    with pytest.raises(ValueError):
        predict("ok", model, tok, max_length=0)


def test_p4_predict_restores_mode():
    tok = FakeTokenizer()
    model = tiny_model()
    model.train()
    predict("hello world", model, tok)
    assert model.training
    model.eval()
    predict("hello world", model, tok)
    assert not model.training
    assert all(p.grad is None for p in model.parameters())


# ---------------- P5: load_trained_model ----------------

def _trained_run(tmp_path, epochs=2):
    cfg = base_cfg(tmp_path, epochs=epochs)
    loaders = (
        make_loader(24, 4, seed=0, shuffle=True),
        make_loader(8, 4, seed=1),
        make_loader(4, 4, seed=2),
    )
    result = train_mod.run_training(cfg, model=tiny_model(), loaders=loaders)
    ckpt_path = Path(result["checkpoint_dir"]) / "best_model.pt"
    return ckpt_path, result, loaders[1]


def test_p5_load_trained_model_roundtrip(tmp_path, monkeypatch):
    ckpt_path, result, val_loader = _trained_run(tmp_path)

    model, info = load_trained_model(str(ckpt_path), backbone_config=None)
    assert isinstance(info["run_name"], str)
    assert info["backbone"] == "distilbert"
    assert "epoch" in info and "val_rmse" in info and "run_config" in info
    assert not model.training

    # evaluate trên Val cho cùng val_rmse trong checkpoint
    res = evaluate(model, val_loader, _cpu())
    rmse_again = float(np.sqrt(np.mean(
        (np.clip(res["y_pred_raw"], 1, 5) - res["y_true"]) ** 2)))
    assert rmse_again == pytest.approx(info["val_rmse"], abs=1e-5)

    # File không tồn tại
    with pytest.raises(FileNotFoundError):
        load_trained_model(str(tmp_path / "nope.pt"))

    # Checkpoint chỉ có model_state_dict -> ValueError
    bad_path = tmp_path / "bad.pt"
    torch.save({"model_state_dict": {}}, bad_path)
    with pytest.raises(ValueError):
        load_trained_model(str(bad_path))

    # Không truyền backbone_config và checkpoint có khoá backbone_config ->
    # AutoConfig.from_pretrained bị chặn (mô phỏng offline) vẫn phải thành công
    def _boom(*a, **kw):
        raise OSError("no network")
    monkeypatch.setattr("src.predict.AutoConfig.from_pretrained", _boom)
    model2, _ = load_trained_model(str(ckpt_path))
    assert model2 is not None


# ---------------- P6: export_test_predictions ----------------

def test_p6_export_test_predictions(tmp_path):
    model = tiny_model()
    tok = FakeTokenizer()
    test_loader = make_loader(6, 4, seed=3)
    out_dir = tmp_path / "preds"

    meta = export_test_predictions(model, test_loader, "bert", out_dir)
    df = pd.read_csv(out_dir / "bert_test.csv")
    assert list(df.columns) == ["review_id", "true_stars", "pred_raw", "pred"]
    assert df["review_id"].tolist() == [f"r{i}" for i in range(6)]
    assert np.allclose(df["pred"], np.clip(df["pred_raw"], 1, 5))
    assert df["true_stars"].dtype.kind in "iu"

    meta_json = json.loads((out_dir / "bert_test_meta.json").read_text())
    assert meta_json["n"] == 6
    expected_rmse_raw = float(np.sqrt(np.mean(
        (df["pred_raw"].to_numpy() - df["true_stars"].to_numpy()) ** 2)))
    assert meta_json["rmse_raw"] == pytest.approx(expected_rmse_raw)
    assert meta_json["inference_seconds"] > 0
    assert meta["model"] == "bert"

    # Gọi lần hai -> FileExistsError; overwrite=True -> ghi được
    with pytest.raises(FileExistsError):
        export_test_predictions(model, test_loader, "bert", out_dir)
    meta2 = export_test_predictions(model, test_loader, "bert", out_dir, overwrite=True)
    assert meta2["n"] == 6

    # Mô hình bias 10 -> pred_raw ~10, pred=5, rmse_raw lớn
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(10.0)
    model.eval()
    out_dir2 = tmp_path / "preds2"
    meta3 = export_test_predictions(model, test_loader, "bert", out_dir2)
    df3 = pd.read_csv(out_dir2 / "bert_test.csv")
    assert (df3["pred_raw"] > 9).all()
    assert (df3["pred"] == 5.0).all()
    assert meta3["rmse_raw"] > expected_rmse_raw


# ---------------- P7: merge_test_predictions ----------------

def _make_preds_dir(tmp_path, names=("bert", "roberta"), swap=False, bad_true=False):
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    ids = ["r0", "r1", "r2"]
    trues = np.array([4.0, 2.0, 5.0])
    data = {
        "bert":    {"pred_raw": [3.6, 2.2, 4.4], "pred": [3.6, 2.2, 4.4]},
        "roberta": {"pred_raw": [4.1, 1.8, 5.3], "pred": [4.1, 1.8, 5.0]},
    }
    for name in names:
        order = [2, 0, 1] if (swap and name == "roberta") else [0, 1, 2]
        df = pd.DataFrame({
            "review_id": [ids[i] for i in order],
            "true_stars": trues[order].astype(int) if not bad_true else trues[order],
            "pred_raw": [data[name]["pred_raw"][i] for i in order],
            "pred": [data[name]["pred"][i] for i in order],
        })
        if bad_true and name == "bert":
            df.loc[0, "true_stars"] = 99
        df.to_csv(preds_dir / f"{name}_test.csv", index=False)
    return preds_dir, ids


def _make_test_parquet(tmp_path, ids, drop=None):
    rows = [{"review_id": r, "text": f"text of {r}"} for r in ids if r != drop]
    path = tmp_path / "test.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_p7_merge_test_predictions(tmp_path):
    ids = ["r0", "r1", "r2"]
    preds_dir, _ = _make_preds_dir(tmp_path)
    test_parquet = _make_test_parquet(tmp_path, ids)
    out_dir = tmp_path / "merged"

    baseline = pd.DataFrame({
        "review_id": ids,
        "pred_baseline": [4.0, 2.5, 4.8],
    })
    out = merge_test_predictions(
        ["bert", "roberta"], preds_dir, test_parquet, out_dir,
        baseline_preds=baseline,
    )
    expected_cols = [
        "review_id", "true_stars", "pred_baseline", "pred_bert", "pred_roberta",
        "abs_err_baseline", "abs_err_bert", "abs_err_roberta",
    ]
    assert list(out.columns) == expected_cols
    assert out["true_stars"].dtype.kind in "iu"
    # abs_err: |4 - 3.6| = 0.4
    assert out.loc[0, "abs_err_bert"] == pytest.approx(0.4, abs=1e-6)

    with_text = pd.read_csv(out_dir / "test_predictions_with_text.csv")
    assert list(with_text.columns)[:3] == ["review_id", "text", "true_stars"]
    assert with_text.loc[0, "text"] == "text of r0"
    assert (out_dir / "test_predictions.csv").exists()

    # Không baseline -> không có cột baseline
    out_dir2 = tmp_path / "merged2"
    out2 = merge_test_predictions(["bert", "roberta"], preds_dir, test_parquet, out_dir2)
    assert "pred_baseline" not in out2.columns
    assert "abs_err_baseline" not in out2.columns

    # Lỗi: review_id đổi thứ tự
    tmp2 = tmp_path / "case2"
    tmp2.mkdir()
    preds_dir2, _ = _make_preds_dir(tmp2, swap=True)
    tp2 = _make_test_parquet(tmp2, ids)
    with pytest.raises(ValueError):
        merge_test_predictions(["bert", "roberta"], preds_dir2, tp2, tmp2 / "o")

    # Lỗi: true_stars khác nhau
    tmp3 = tmp_path / "case3"
    tmp3.mkdir()
    preds_dir3, _ = _make_preds_dir(tmp3, bad_true=True)
    tp3 = _make_test_parquet(tmp3, ids)
    with pytest.raises(ValueError):
        merge_test_predictions(["bert", "roberta"], preds_dir3, tp3, tmp3 / "o")

    # Lỗi: baseline thiếu review_id
    bad_base = baseline.iloc[:-1]
    with pytest.raises(ValueError):
        merge_test_predictions(["bert", "roberta"], preds_dir, test_parquet,
                               tmp_path / "o3", baseline_preds=bad_base)

    # Lỗi: test_parquet thiếu review_id
    tp_missing = _make_test_parquet(tmp_path, ids, drop="r1")
    with pytest.raises(ValueError):
        merge_test_predictions(["bert", "roberta"], preds_dir, tp_missing, tmp_path / "o4")

    # Lỗi: model_names rỗng
    with pytest.raises(ValueError):
        merge_test_predictions([], preds_dir, test_parquet, tmp_path / "o5")

    # Lỗi: gọi lần hai không overwrite -> FileExistsError
    with pytest.raises(FileExistsError):
        merge_test_predictions(["bert", "roberta"], preds_dir, test_parquet, out_dir,
                               baseline_preds=baseline)


# ---------------- P8: collect_summary_inputs ----------------

def test_p8_collect_summary_inputs(tmp_path):
    results_dir = tmp_path / "results"
    # run_meta cho run_name bert_mse và roberta
    for run_name, n_params, train_s, ep_s in (
        ("bert_mse", 1000, 60.0, 20.0),
        ("roberta", 2000, 90.0, 30.0),
    ):
        d = results_dir / "runs" / run_name
        d.mkdir(parents=True)
        (d / "run_meta.json").write_text(json.dumps({
            "run_name": run_name, "n_params": n_params,
            "train_seconds": train_s, "epoch_seconds_mean": ep_s,
        }))
    # meta dự đoán cho model bert và roberta
    for model, inf_s, rmse_raw in (("bert", 1.5, 0.9), ("roberta", 2.5, 0.8)):
        d = results_dir / "preds"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{model}_test_meta.json").write_text(json.dumps({
            "model": model, "inference_seconds": inf_s, "rmse_raw": rmse_raw,
        }))

    out = collect_summary_inputs({"bert": "bert_mse", "roberta": "roberta"}, results_dir)
    assert out["param_counts"] == {"bert": 1000, "roberta": 2000}
    assert out["train_seconds"] == {"bert": 60.0, "roberta": 90.0}
    assert out["inference_seconds"] == {"bert": 1.5, "roberta": 2.5}
    assert out["rmse_raw_by_model"] == {"bert": 0.9, "roberta": 0.8}
    assert out["seconds_per_epoch"] == {"bert": 20.0, "roberta": 30.0}

    # Thiếu meta -> FileNotFoundError
    (results_dir / "preds" / "bert_test_meta.json").unlink()
    with pytest.raises(FileNotFoundError):
        collect_summary_inputs({"bert": "bert_mse", "roberta": "roberta"}, results_dir)
