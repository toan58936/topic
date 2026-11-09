# Phase 3 Checklist: Smoke Test & Short GPU Run

> Verify pipeline works on GPU before full training. Owner: D (Training), C (Model support).

---

## 3.1 Colab Environment Setup (D)

- [ ] Mở Colab notebook: `notebooks/colab_runner.ipynb`
- [ ] Mount Google Drive:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
- [ ] Clone repo:
  ```bash
  cd /content/drive/MyDrive/project
  git clone <repo-url> repo
  cd repo
  ```
- [ ] Install dependencies:
  ```bash
  pip install -q -r requirements.txt
  ```
- [ ] Verify imports:
  ```python
  import torch, transformers
  print(torch.__version__, transformers.__version__)
  print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
  ```
- [ ] Copy data từ Drive về local Colab (faster I/O):
  ```bash
  cp /content/drive/MyDrive/project/data/*.parquet /content/
  ```

---

## 3.2 Smoke Test (8 Samples, 1 Epoch)

- [ ] Tạo config smoke test: `configs/smoke_test.yaml`
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
- [ ] Chạy:
  ```bash
  python -m src.train --config configs/smoke_test.yaml
  ```
- [ ] Verify:
  - [ ] Không lỗi import/runtime
  - [ ] Loss hợp lệ (không NaN/Inf)
  - [ ] Mỗi batch ra output shape [B]
  - [ ] Checkpoint ghi được (`/content/drive/MyDrive/project/checkpoints/bert/epoch_1.pt`)
  - [ ] `history.csv` có 1 dòng: epoch, train_loss, val_rmse, val_mae, epoch_seconds

---

## 3.3 Short GPU Run (100 Steps hoặc 1 Epoch Đầy Đủ)

- [ ] Config: `configs/short_run.yaml` (same as smoke_test nhưng epochs=1, max_steps=100 hoặc bỏ max_steps)
- [ ] Chạy:
  ```bash
  python -m src.train --config configs/short_run.yaml
  ```
- [ ] Measure & Record:
  - [ ] `seconds_per_epoch` (từ `history.csv` hoặc log)
  - [ ] Peak VRAM usage (`nvidia-smi` hoặc `torch.cuda.max_memory_allocated()`)
  - [ ] Train loss trend (giảm sau epoch 1)
  - [ ] Val RMSE/MAE hợp lý (không NaN)

---

## 3.4 Troubleshooting Common Issues

| Issue | Fix |
|-------|-----|
| OOM (Out of Memory) | `batch_size: 8`, `gradient_checkpointing: true` |
| Drive mount timeout | Re-mount, check quota |
| Import error transformers | `pip install -U transformers accelerate` |
| Loss NaN | Check fp16: ép `.float()` trước loss, giảm lr |
| Checkpoint not saving | Check Drive path permissions, disk space |

---

## 3.5 Resume Verification

- [ ] Giả lập ngắt: stop notebook, reopen
- [ ] Resume từ checkpoint:
  ```bash
  python -m src.train --config configs/short_run.yaml --resume auto
  ```
- [ ] Verify:
  - [ ] Tiếp tục từ epoch đúng
  - [ ] Optimizer/scheduler state restored
  - [ ] RNG state restored (reproducible)

---

## 3.6 Estimate Full Training Time

- [ ] Based on `seconds_per_epoch` × 3 epochs × 3 models
- [ ] Add buffer: +30% for Colab overhead, checkpoint save, eval
- [ ] Record estimate in `ai_log.md` for Leader planning

---

## 3.7 Deliverables

- [ ] Smoke test PASS (Colab)
- [ ] Short GPU run PASS, `seconds_per_epoch` recorded
- [ ] VRAM usage recorded (< 14GB on T4)
- [ ] Resume mechanism verified
- [ ] Full training time estimate recorded
- [ ] Any config adjustments (batch_size, gradient_checkpointing) documented

---

## 3.8 AI Log

- [ ] `docs/ai_log.md` updated với Phase 3 entries (D)

---

## 3.9 Phase Gate: Phase 3 → Phase 4

**PASS Criteria** (Leader approve):
- [ ] Smoke test PASS
- [ ] Short GPU run PASS
- [ ] `seconds_per_epoch` recorded for Milestone 1
- [ ] Resume works
- [ ] No blocking issues (OOM fixed, import errors fixed)

**Sign-off**:
- Owner D: _______________ / Date: _______________
- Leader: _______________ / Date: _______________