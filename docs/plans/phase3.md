# Phase 3 Checklist: Smoke Test & Short GPU Run

> Verify pipeline works on Kaggle GPU before full training. Owner: D (Training), C (Model support).

---

## 3.1 Kaggle Environment Setup (D) — **COMPLETED**

- [x] Mở Kaggle Notebook (GPU T4/P100)
- [x] Gắn Dataset Yelp làm Input: `/kaggle/input/yelp-dataset/` (file `review.json`)
- [x] Clone repo:
  ```bash
  cd /kaggle/working
  git clone <repo-url> repo
  cd repo
  ```
- [x] Install dependencies:
  ```bash
  pip install -q -r requirements.txt
  ```
- [x] Verify imports:
  ```python
  import torch, transformers
  print(torch.__version__, transformers.__version__)
  print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
  ```
- [x] Copy data từ Input Dataset về working (faster I/O):
  ```bash
  cp /kaggle/input/yelp-topic7-splits/*.parquet /kaggle/working/data/
  cp /kaggle/input/yelp-topic7-splits/split_stats.json /kaggle/working/data/
  ```

---

## 3.2 Smoke Test (8 Samples, 1 Epoch) — **COMPLETED**

- [x] Tạo config smoke test: `configs/smoke_test.yaml`
  ```yaml
  backbone: "bert"
  pooling: "mean"
  loss: "huber"
  epochs: 1
  batch_size: 16
  lr: 2e-5
  max_steps: 8  # only 8 batches
  fp16: true
  gradient_checkpointing: false
  seed: 42
  ```
- [x] Chạy:
  ```bash
  python -m src.train --config configs/smoke_test.yaml
  ```
- [x] Verify:
  - [x] Không lỗi import/runtime
  - [x] Loss hợp lệ (không NaN/Inf)
  - [x] Mỗi batch ra output shape [B]
  - [x] Checkpoint ghi được (`/kaggle/working/checkpoints/bert/last.pt`)
  - [x] `history.csv` có 1 dòng: epoch, train_loss, val_rmse, val_mae, epoch_seconds

---

## 3.3 Short GPU Run (100 Steps hoặc 1 Epoch Đầy Đủ) — **COMPLETED**

- [x] Config: `configs/short_run.yaml` (same as smoke_test nhưng epochs=1, max_steps=100 hoặc bỏ max_steps)
- [x] Chạy:
  ```bash
  python -m src.train --config configs/short_run.yaml
  ```
- [x] Measure & Record:
  - [x] `seconds_per_epoch` (từ `history.csv` hoặc log): BERT ~595s, DistilBERT ~350s, RoBERTa ~680s
  - [x] Peak VRAM usage: < 14GB on T4 16GB
  - [x] Train loss trend (giảm sau epoch 1)
  - [x] Val RMSE/MAE hợp lý (không NaN)

---

## 3.4 Troubleshooting Common Issues

| Issue | Fix |
|-------|-----|
| OOM (Out of Memory) | `batch_size: 8`, `gradient_checkpointing: true` |
| Import error transformers | `pip install -U transformers accelerate` |
| Loss NaN | Check fp16: ép `.float()` trước loss, giảm lr |
| Checkpoint not saving | Check `/kaggle/working` path permissions, disk space |
| Dataset not found | Verify `/kaggle/input/yelp-topic7-splits/` mounted as Input |

---

## 3.5 Resume Verification — **COMPLETED**

- [x] Giả lập ngắt: stop notebook, reopen
- [x] Resume từ checkpoint:
  ```bash
  python -m src.train --config configs/short_run.yaml --resume auto
  ```
- [x] Verify:
  - [x] Tiếp tục từ epoch đúng
  - [x] Optimizer/scheduler state restored
  - [x] RNG state restored (reproducible)

---

## 3.6 Estimate Full Training Time — **COMPLETED**

- [x] Based on `seconds_per_epoch` × 3 epochs × 3 models
- [x] Add buffer: +30% for Kaggle overhead, checkpoint save, eval
- [x] Record estimate in `ai_log.md` for Leader planning

---

## 3.7 Deliverables — **ALL COMPLETED**

- [x] Smoke test PASS (Kaggle)
- [x] Short GPU run PASS, `seconds_per_epoch` recorded
- [x] VRAM usage recorded (< 14GB on T4)
- [x] Resume mechanism verified
- [x] Full training time estimate recorded
- [x] Any config adjustments (batch_size, gradient_checkpointing) documented

---

## 3.8 AI Log

- [x] `docs/ai_log.md` updated với Phase 3 entries (D)

---

## 3.9 Phase Gate: Phase 3 → Phase 4

**PASS Criteria** (Leader approve):
- [x] Smoke test PASS
- [x] Short GPU run PASS
- [x] `seconds_per_epoch` recorded for Milestone 1
- [x] Resume works
- [x] No blocking issues (OOM fixed, import errors fixed)

**Sign-off**:
- Owner D: _______________ / Date: _______________
- Leader: _______________ / Date: _______________