# Baseline Neural Network Surrogate - Training Summary

**Date:** April 29, 2026  
**Model:** PricingSurrogate (Baseline)  
**Run ID:** nn-baseline-20260429_001905  
**Status:** ✅ FINISHED

---

## Model Architecture

```
Input (4)
  ├─ Linear(4 → 128) + SiLU
  ├─ Linear(128 → 128) + SiLU
  ├─ Linear(128 → 128) + SiLU
  ├─ Linear(128 → 128) + SiLU
  └─ Linear(128 → 3)

Output Constraints:
  • Price:  sigmoid → [0, 1]
  • Delta:  sigmoid → [0, 1]
  • Gamma:  ReLU    → [0, ∞)

Parameters: 50,563
File size: 201.6 KB
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW (lr=0.001, weight_decay=1e-4) |
| **Scheduler** | CosineAnnealingLR (T_max=200) |
| **Loss Function** | Weighted relative MSE |
| **Loss Weights** | λ_price=1.0, λ_delta=0.5, λ_gamma=0.1 |
| **Batch Size** | 1024 |
| **Epochs** | 200 |
| **Training Time** | 3.4 minutes |

---

## Dataset Splits

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | 70,000 | Parameter training |
| Val | 15,000 | Early stopping & hyperparameter tuning |
| Test | 15,000 | Final evaluation (ablation baseline) |

---

## Performance Metrics

### Overall Loss
- **Train (final):** 0.004079
- **Val (final):** 0.004508
- **Test (final):** 0.004592

### Per-Output Metrics (Test Set)

#### Call Price
```
Target range: [0.000000, 0.657054]
Pred range:   [0.000000, 0.657202]
MAE:   0.001208  ✅ Excellent
RMSE:  0.001923  ✅ Excellent
Loss:  0.000345
```

#### Delta
```
Target range: [0.000000, 1.000000]
Pred range:   [0.000000, 0.999541]
MAE:   0.008026  ✅ Good
RMSE:  0.012065  ✅ Good
Loss:  0.001163
```

#### Gamma
```
Target range: [0.000000, 45.766624]
Pred range:   [0.000000, 7.499783]
MAE:   0.363090  ⚠️ Underfitting
RMSE:  1.577677  ⚠️ Underfitting
Loss:  0.036661
```

---

## Key Observations

### Strengths
- ✅ **Call Price:** Excellent predictions with RMSE < 0.002
- ✅ **Delta:** Good predictions across full [0,1] range
- ✅ **Training Stability:** Smooth convergence with no instabilities
- ✅ **Overfitting Prevention:** Minimal gap between train/val losses

### Areas for Improvement (Row 1 of Ablation Table)
- ⚠️ **Gamma Predictions:** Capped at ~7.5 instead of reaching ~46
  - Suggests need for deeper network or specialized architecture
  - Loss weight (0.1) may need adjustment
  - Possible: Add skip connections, batch norm, or separate gamma head

---

## File Locations

| File | Location | Size |
|------|----------|------|
| Model | `models/nn/baseline.pt` | 201.6 KB |
| History | `models/nn/baseline_history.pt` | - |
| MLflow Artifacts | `experiments/<run_id>/` | - |
| Training Log | Terminal output above | - |

---

## MLflow Tracking

### Parameters Logged (17)
- model_name, model_type, hidden_dim, n_layers, activation, dropout
- n_params, epochs, batch_size, learning_rate, weight_decay
- optimizer, scheduler, lambda_price/delta/gamma, device

### Metrics Logged (17)
- final_train/val/test_loss, best_val_loss, best_epoch
- test_loss_price/delta/gamma
- val_loss_price/delta/gamma
- train_loss_price/delta/gamma

### Artifacts Saved
- `pytorch_model/` - Full PyTorch model directory
- Training history

---

## Next Steps

1. **Analyze Gamma Underfitting**
   - Check if loss weight needs tuning
   - Consider architectural changes (e.g., separate heads)

2. **Improvement Experiments** (Rows 2-N of ablation table)
   - Add batch normalization
   - Increase hidden_dim or n_layers
   - Try residual connections
   - Adjust output activation for gamma

3. **Validation Protocol**
   - Compare all models on same test set
   - Track metrics per moneyness bucket
   - Analyze error distribution

4. **Paper Preparation**
   - Table 1: Ablation results (this baseline = Row 1)
   - Figure: Training curves
   - Figure: Error analysis by moneyness

---

## Commands to Reproduce

```bash
# View results in MLflow UI
mlflow ui

# Retrain baseline
python -m src.train_nn

# Evaluate baseline on test set
python -c "from src.train_nn import PricingSurrogate; ..."
```

---

## Status: ✅ BASELINE COMPLETE - Ready for Ablation Studies
