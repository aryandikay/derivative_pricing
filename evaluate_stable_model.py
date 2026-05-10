"""Evaluate the final stable model trained with stable loss function."""

import numpy as np
import torch
from pathlib import Path
from src.evaluate_nn import PricingSurrogate
from src.data import black_scholes_call

# Load test data
test_data = np.load('data/processed/test.npz')
X_test_scaled = test_data['X_test_scaled']
y_test = test_data['y_test']
X_test_original = test_data['X_test_original']

# Extract prices
true_price = y_test[:, 0]

# Load model
device = torch.device('cpu')
model = PricingSurrogate(n_inputs=4, n_outputs=3, hidden_dim=128, n_layers=4, activation='silu')
model_path = Path('models/nn/final_stable_model.pt')

if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}")

checkpoint = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint)
model.eval()
model.to(device)

print(f"Loaded model from: {model_path}")
print()

# === ATM Check (Assertion) ===
# ATM: S/K = 1.0, T = 1.0, sigma = 0.2, r = 0.05
atm_moneyness = np.array([[1.0, 1.0, 0.2, 0.05]])
atm_scaled = atm_moneyness / np.array([[1.0, 1.0, 1.0, 1.0]])  # No scaling needed for standardized
# Actually need to scale with the scaler used during training
import joblib
scaler = joblib.load('models/input_scaler.pkl')
atm_scaled = scaler.transform(atm_moneyness)

atm_tensor = torch.FloatTensor(atm_scaled).to(device)
with torch.no_grad():
    atm_output = model(atm_tensor)
    atm_nn_price = atm_output[0, 0].item()

atm_bs_price = black_scholes_call(1.0, 1.0, 0.05, 0.2)

# Check if within ±5%
atm_error_pct = abs(atm_nn_price - atm_bs_price) / atm_bs_price * 100

print(f"ATM Test:")
print(f"BS: {atm_bs_price:.6f}")
print(f"NN: {atm_nn_price:.6f}")
print(f"Error: {atm_error_pct:.2f}%")
print()

if atm_error_pct > 5.0:
    raise AssertionError(f"Loaded wrong model checkpoint. ATM error {atm_error_pct:.2f}% exceeds 5% threshold")

# === Get predictions ===
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    outputs = model(X_test_tensor)
    pred_price = outputs[:, 0].cpu().numpy()

# === Metrics ===
print(f"Distribution Check:")
print(f"Mean true: {np.mean(true_price):.6f}")
print(f"Mean pred: {np.mean(pred_price):.6f}")
print()

# MAPE on filtered data (price > 1e-3)
mask = true_price > 1e-3
filtered_true = true_price[mask]
filtered_pred = pred_price[mask]
rel_error = np.abs(filtered_pred - filtered_true) / filtered_true
mape = np.mean(rel_error) * 100

print(f"MAPE: {mape:.2f}%")
