# Phase 4 Checklist: Full BERT Training (Huber + MSE) — Loss Rule

> Full BERT training with two loss functions, apply loss rule. Owner: D (Training), Leader (decision).

---

## 4.1 Training Runs — **COMPLETED**

### Run 1: BERT + Huber (δ=1.0)
- [x] Config: `configs/train_bert_huber.yaml`
  - backbone: bert, pooling: mean, loss: huber, epochs: 3, batch: 16, lr: 2e-5, fp16: true
- [x] Run on Kaggle: `python -m src.train --config configs/train_bert_huber.yaml`
- [x] Verify:
  - [x] 3 epochs completed
  - [x] Checkpoint per epoch saved to `/kaggle/working/checkpoints/bert/` (last.pt, best_model.pt)
  - [x] `best_model.pt` = epoch with lowest Val RMSE (clipped)
  - [x] `history.csv` has: epoch, train_loss, val_rmse, val_mae, epoch_seconds
  - [x] `run_meta.json` saved (backbone, pooling, loss, seed, lr, batch, epochs, torch/transformers version, GPU, commit_hash, split_id)
  - [x] Sync kết quả về Drive (`drive/project/checkpoints/bert/`)

### Run 2: BERT + MSE
- [x] Config: `configs/train_bert_mse.yaml` (same as above, loss: mse)
- [x] Run on Kaggle
- [x] Verify same artifacts as Run 1
- [x] Sync kết quả về Drive (`drive/project/checkpoints/bert/`)

---

## 4.2 Loss Rule Application — **COMPLETED**

**Rule**: Chuyển sang MSE cho cả 3 backbone **nếu và chỉ nếu**:
- `RMSE_MSE <= 0.99 * RMSE_Huber` (≥1% improvement relative) **VÀ**
- `MAE_MSE <= 1.01 * MAE_Huber` (không tệ hơn >1% relative)

**Evaluation**: Trên **Val set**, cuối Phase 4, **không đổi sau khi thấy Test**.

- [x] Extract Val RMSE/MAE from both `history.csv` (best epoch)
- [x] Compute ratios:
  - BERT Huber: RMSE=0.5608, MAE=0.3411
  - BERT MSE: RMSE=0.5628, MAE=0.3417
  - RMSE ratio: 1.0036 > 0.99 (MSE worse)
  - MAE ratio: 1.0018 < 1.01 (MSE comparable)
- [x] Decision: **Keep Huber** (default) — MSE không đáp ứng điều kiện RMSE improvement ≥1%
- [x] Record in `docs/ai_log.md`:
  - RMSE_Huber=0.5608, MAE_Huber=0.3411, RMSE_MSE=0.5628, MAE_MSE=0.3417
  - Ratios computed: RMSE_ratio=1.0036, MAE_ratio=1.0018
  - Decision: Keep Huber + reasoning
- [x] Leader approves decision

---

## 4.3 Config Update for Phase 5 — **COMPLETED**

- [x] Update default loss in training configs for Phase 5:
  - `configs/train_distilbert.yaml` → loss: huber
  - `configs/train_roberta.yaml` → loss: huber
- [x] Commit config changes

---

## 4.4 Deliverables — **ALL COMPLETED**

- [x] BERT Huber checkpoints on `/kaggle/working/checkpoints/bert/` + Drive
- [x] BERT MSE checkpoints on `/kaggle/working/checkpoints/bert/` + Drive
- [x] Loss rule decision recorded in `ai_log.md` + Leader approved
- [x] Phase 5 configs updated with chosen loss (Huber)
- [x] Milestone 1 data ready: `split_stats.json`, Baseline Val, threshold, `seconds_per_epoch`, Huber vs MSE Val, loss decided

---

## 4.5 AI Log

- [x] `docs/ai_log.md` updated with Phase 4 entries (D, Leader)

---

## 4.6 Phase Gate: Phase 4 → Phase 5

**PASS Criteria** (Leader approve):
- [x] Both BERT runs completed 3 epochs
- [x] Checkpoints + history + run_meta on Drive
- [x] Loss rule applied correctly, decision documented
- [x] Milestone 1 deliverables complete

**Sign-off**:
- Owner D: _______________ / Date: _______________
- Leader: _______________ / Date: _______________