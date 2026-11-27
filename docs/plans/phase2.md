# Phase 2 Checklist: Model & Training Infrastructure

> Model builder, utils, engine, train, predict. Owner: C (Model), D (Training), Leader (review).

---

## 2.1 Spec & Code

### Model Builder (C)
- [x] Spec `docs/03_model.md` written
- [x] Spec reviewed by Leader
- [x] Code `src/model_builder.py` written:
  - [x] `BACKBONES` dict: bert, roberta, distilbert with hf_name, model_type, drop_pooler
  - [x] `build_model(backbone, pooling, dropout, pretrained, backbone_config, output_bias_init)`
  - [x] `TextRegressor` class: backbone, mean/cls pooling, dropout, Linear head
  - [x] `mean_pool(last_hidden_state, attention_mask)` — masked mean, float32, zero for all-pad
  - [x] `cls_pool(last_hidden_state)` — position 0
  - [x] `build_tokenizer(backbone)` — AutoTokenizer.from_pretrained
  - [x] `count_parameters(model, trainable_only=False)`
  - [x] No `token_type_ids` in forward
  - [x] BERT/RoBERTa: `add_pooling_layer=False`, DistilBERT: no pooler

### Training Infra (D)
- [x] Spec `docs/04_engine_and_utils.md` written
- [x] Spec reviewed by Leader
- [x] Code `src/utils.py`: `set_seed`, checkpoint save/load (weights_only=True)
- [x] Code `src/engine.py`:
  - [x] `train_epoch(model, loader, optimizer, scheduler, loss_fn, fp16, grad_clip)` → `train_one_epoch`
  - [x] `eval_epoch(model, loader, loss_fn, fp16)` → `evaluate`
  - [x] Huber (delta=1.0) & MSE loss support
  - [x] fp16 autocast, output .float() before loss
  - [x] Checkpoint per epoch: model, optimizer, scheduler, scaler, history, best_val_rmse, backbone_config, run_config (`last.pt`, `best_model.pt`)
  - [x] History CSV per run (`history.csv` + `run_meta.json`)
- [x] Code `src/train.py`: CLI config, `set_seed` before `build_model`, log run_meta
- [x] Code `src/predict.py`: load weights (build_model pretrained=False + state_dict), predict, clip, export test_predictions.csv, merge validation
- [x] Hàm `main()` cho `python -m src.predict` implemented

---

## 2.2 Tests (CPU, Synthetic Data + Tiny Random Config)

- [x] `tests/test_model_builder.py` written & PASS (14/14)
  - [x] All 3 backbones build with tiny random config
  - [x] Forward: batch [2, 16] → output [2]
  - [x] Mean pooling: padding doesn't affect result
  - [x] CLS pooling: position 0
  - [x] Head bias init = mean train label (when provided)
  - [x] Param count: backbone + head (no pooler)
  - [x] Tokenizer correct for each backbone

- [x] `tests/test_utils.py` written & PASS (4/4)
  - [x] `set_seed` reproducible
  - [x] Checkpoint save/load with weights_only=True

- [x] `tests/test_engine.py` written & PASS (13/13)
  - [x] Loss computation (Huber, MSE)
  - [x] LR schedule (linear warmup)
  - [x] Fit: history, best checkpoint, resume contiguous
  - [x] Test not iterated during training

- [x] `tests/test_train.py` written & PASS (5/5)
  - [x] Config parsing (2e-5 parsed correctly, reject unknown keys)
  - [x] `set_seed` called before `build_model`
  - [x] CLI works

- [x] `tests/test_predict.py` written & PASS (9/9)
  - [x] `predict(text)` returns float in [1.0, 5.0]
  - [x] Export & merge test_predictions.csv
  - [x] Merge validation: stops on split_id/review_id mismatch

---

## 2.3 Verification (Kaggle GPU) — **COMPLETED**

- [x] Kaggle: `git clone`, `pip install -r requirements.txt` (< 10 min)
- [x] Smoke test: 8 samples, 1 epoch, batch=16, fp16, BERT Huber
  - [x] No errors
  - [x] Loss valid (not NaN)
  - [x] Checkpoint saved to `/kaggle/working/checkpoints/`
- [x] Short GPU run: 100 steps (or 1 epoch), measure `seconds_per_epoch`, VRAM
  - [x] `seconds_per_epoch` recorded for Milestone 1 (~595s BERT, ~350s DistilBERT, ~680s RoBERTa)
  - [x] VRAM < 14GB (T4 16GB)
  - [x] No OOM

---

## 2.4 Deliverables

- [x] `model_builder.py`, `utils.py`, `engine.py`, `train.py`, `predict.py` + tests committed
- [x] All pytest CPU PASS (60/60 not slow)
- [x] Smoke test PASS (Kaggle)
- [x] Short GPU run PASS, `seconds_per_epoch` recorded
- [x] PRs merged to `develop`

---

## 2.5 AI Log

- [x] `docs/ai_log.md` updated with Phase 2 entries (C, D)

---

## 2.6 Phase Gate: Phase 2 → Phase 3

**PASS Criteria** (Leader approve):
- [x] All 5 test files PASS (CPU)
- [x] 3 backbones build with tiny config
- [x] Smoke test + short GPU run PASS
- [x] `seconds_per_epoch` recorded
- [x] Code merged to `develop`

**Sign-off**:
- Owner C: _______________ / Date: _______________
- Owner D: _______________ / Date: _______________
- Leader: _______________ / Date: _______________