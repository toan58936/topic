# Phase 2 Checklist: Model & Training Infrastructure

> Model builder, utils, engine, train, predict. Owner: C (Model), D (Training), Leader (review).

---

## 2.1 Spec & Code

### Model Builder (C)
- [ ] Spec `docs/03_model.md` written
- [ ] Spec reviewed by Leader
- [ ] Code `src/model_builder.py` written:
  - [ ] `BACKBONES` dict: bert, roberta, distilbert with hf_name, model_type, drop_pooler
  - [ ] `build_model(backbone, pooling, dropout, pretrained, backbone_config, output_bias_init)`
  - [ ] `TextRegressor` class: backbone, mean/cls pooling, dropout, Linear head
  - [ ] `mean_pool(last_hidden_state, attention_mask)` — masked mean, float32, zero for all-pad
  - [ ] `cls_pool(last_hidden_state)` — position 0
  - [ ] `build_tokenizer(backbone)` — AutoTokenizer.from_pretrained
  - [ ] `count_parameters(model, trainable_only=False)`
  - [ ] No `token_type_ids` in forward
  - [ ] BERT/RoBERTa: `add_pooling_layer=False`, DistilBERT: no pooler

### Training Infra (D)
- [ ] Spec `docs/04_engine_and_utils.md` written
- [ ] Spec reviewed by Leader
- [ ] Code `src/utils.py`: `set_seed`, checkpoint save/load (weights_only=True)
- [ ] Code `src/engine.py`:
  - [ ] `train_epoch(model, loader, optimizer, scheduler, loss_fn, fp16, grad_clip)`
  - [ ] `eval_epoch(model, loader, loss_fn, fp16)`
  - [ ] Huber (delta=1.0) & MSE loss support
  - [ ] fp16 autocast, output .float() before loss
  - [ ] Checkpoint per epoch: model, optimizer, scheduler, rng_state, epoch, best_val_rmse, backbone_config, run_config
  - [ ] History CSV per run
- [ ] Code `src/train.py`: CLI config, `set_seed` before `build_model`, log run_meta
- [ ] Code `src/predict.py`: load weights (build_model pretrained=False + state_dict), predict, clip, export test_predictions.csv, merge validation

---

## 2.2 Tests (CPU, Synthetic Data + Tiny Random Config)

- [ ] `tests/test_model_builder.py` written & PASS
  - [ ] All 3 backbones build with tiny random config
  - [ ] Forward: batch [2, 16] → output [2]
  - [ ] Mean pooling: padding doesn't affect result
  - [ ] CLS pooling: position 0
  - [ ] Head bias init = mean train label (when provided)
  - [ ] Param count: backbone + head (no pooler)
  - [ ] Tokenizer correct for each backbone

- [ ] `tests/test_utils.py` written & PASS
  - [ ] `set_seed` reproducible
  - [ ] Checkpoint save/load with weights_only=True

- [ ] `tests/test_engine.py` written & PASS
  - [ ] Loss computation (Huber, MSE)
  - [ ] LR schedule (linear warmup)
  - [ ] Fit: history, best checkpoint, resume contiguous
  - [ ] Test not iterated during training

- [ ] `tests/test_train.py` written & PASS
  - [ ] Config parsing (2e-5 parsed correctly, reject unknown keys)
  - [ ] `set_seed` called before `build_model`
  - [ ] CLI works

- [ ] `tests/test_predict.py` written & PASS
  - [ ] `predict(text)` returns float in [1.0, 5.0]
  - [ ] Export & merge test_predictions.csv
  - [ ] Merge validation: stops on split_id/review_id mismatch

---

## 2.3 Verification (Colab GPU)

- [ ] Colab: `git clone`, mount Drive, `pip install -r requirements.txt` (< 10 min)
- [ ] Smoke test: 8 samples, 1 epoch, batch=16, fp16, BERT Huber
  - [ ] No errors
  - [ ] Loss valid (not NaN)
  - [ ] Checkpoint saved to Drive
- [ ] Short GPU run: 100 steps (or 1 epoch), measure `seconds_per_epoch`, VRAM
  - [ ] `seconds_per_epoch` recorded for Milestone 1
  - [ ] VRAM < 14GB (T4 16GB)
  - [ ] No OOM

---

## 2.4 Deliverables

- [ ] `model_builder.py`, `utils.py`, `engine.py`, `train.py`, `predict.py` + tests committed
- [ ] All pytest CPU PASS
- [ ] Smoke test PASS (Colab)
- [ ] Short GPU run PASS, `seconds_per_epoch` recorded
- [ ] PRs merged to `develop`

---

## 2.5 AI Log

- [ ] `docs/ai_log.md` updated with Phase 2 entries (C, D)

---

## 2.6 Phase Gate: Phase 2 → Phase 3

**PASS Criteria** (Leader approve):
- [ ] All 5 test files PASS (CPU)
- [ ] 3 backbones build with tiny config
- [ ] Smoke test + short GPU run PASS
- [ ] `seconds_per_epoch` recorded
- [ ] Code merged to `develop`

**Sign-off**:
- Owner C: _______________ / Date: _______________
- Owner D: _______________ / Date: _______________
- Leader: _______________ / Date: _______________