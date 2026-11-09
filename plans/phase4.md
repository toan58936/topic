# Phase 4 Checklist: Full BERT Training (Huber + MSE) — Loss Rule

> Full BERT training with two loss functions, apply loss rule. Owner: D (Training), Leader (decision).

---

## 4.1 Training Runs

### Run 1: BERT + Huber (δ=1.0)
- [ ] Config: `configs/train_bert_huber.yaml`
  - backbone: bert, pooling: mean, loss: huber, epochs: 3, batch: 16, lr: 2e-5, fp16: true
- [ ] Run on Colab: `python -m src.train --config configs/train_bert_huber.yaml`
- [ ] Verify:
  - [ ] 3 epochs completed
  - [ ] Checkpoint per epoch saved to Drive (`drive/project/checkpoints/bert/epoch_{1,2,3}.pt`)
  - [ ] `best_model.pt` = epoch with lowest Val RMSE (clipped)
  - [ ] `history.csv` has: epoch, train_loss, val_rmse, val_mae, epoch_seconds
  - [ ] `run_meta.json` saved (backbone, pooling, loss, seed, lr, batch, epochs, torch/transformers version, GPU, commit_hash, split_id)

### Run 2: BERT + MSE
- [ ] Config: `configs/train_bert_mse.yaml` (same as above, loss: mse)
- [ ] Run on Colab
- [ ] Verify same artifacts as Run 1

---

## 4.2 Loss Rule Application (Define Mục 9, CHECKLIST #9)

**Rule**: Chuyển sang MSE cho cả 3 backbone **nếu và chỉ nếu**:
- `RMSE_MSE <= 0.99 * RMSE_Huber` (≥1% improvement relative) **VÀ**
- `MAE_MSE <= 1.01 * MAE_Huber` (không tệ hơn >1% relative)

**Evaluation**: Trên **Val set**, cuối Phase 4, **không đổi sau khi thấy Test**.

- [ ] Extract Val RMSE/MAE from both `history.csv` (last epoch or best epoch)
- [ ] Compute ratios
- [ ] Decision:
  - [ ] **Keep Huber** (default) — nếu MSE không đáp ứng cả 2 điều kiện
  - [ ] **Switch to MSE** — nếu MSE đáp ứng cả 2 điều kiện
- [ ] Record in `docs/ai_log.md`:
  - RMSE_Huber, MAE_Huber, RMSE_MSE, MAE_MSE
  - Ratios computed
  - Decision + reasoning
- [ ] Leader approves decision

---

## 4.3 Config Update for Phase 5

- [ ] Update default loss in training configs for Phase 5:
  - `configs/train_distilbert.yaml` → loss: [chosen loss]
  - `configs/train_roberta.yaml` → loss: [chosen loss]
- [ ] Commit config changes

---

## 4.4 Deliverables

- [ ] BERT Huber checkpoints on Drive
- [ ] BERT MSE checkpoints on Drive
- [ ] Loss rule decision recorded in `ai_log.md` + Leader approved
- [ ] Phase 5 configs updated with chosen loss
- [ ] Milestone 1 data ready: `split_stats.json`, Baseline Val, threshold, `seconds_per_epoch`, Huber vs MSE Val, loss decided

---

## 4.5 AI Log

- [ ] `docs/ai_log.md` updated with Phase 4 entries (D, Leader)

---

## 4.6 Phase Gate: Phase 4 → Phase 5

**PASS Criteria** (Leader approve):
- [ ] Both BERT runs completed 3 epochs
- [ ] Checkpoints + history + run_meta on Drive
- [ ] Loss rule applied correctly, decision documented
- [ ] Milestone 1 deliverables complete

**Sign-off**:
- Owner D: _______________ / Date: _______________
- Leader: _______________ / Date: _______________