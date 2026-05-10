import torch
import numpy as np
import joblib
from src.nn_model import PricingSurrogate
from src.data import black_scholes_call

device  = torch.device('cpu')
scaler  = joblib.load('outputs/router_v1/scaler.pkl')  # use router scaler

model = PricingSurrogate(hidden_dim=128, n_layers=4, activation='silu')
model.load_state_dict(
    torch.load('models/nn/best_model.pt', map_location=device))
model.eval()

# Test multiple inputs across the pricing surface
test_cases = [
    (1.00, 0.50, 0.20, 0.05, "ATM  6m 20vol"),
    (0.90, 0.25, 0.20, 0.05, "OTM  3m 20vol"),
    (1.10, 1.00, 0.20, 0.05, "ITM  1yr 20vol"),
    (1.00, 1.00, 0.30, 0.05, "ATM  1yr 30vol"),
    (1.00, 2.00, 0.20, 0.05, "ATM  2yr 20vol"),
]

print(f"{'Case':<20} {'BS True':>10} {'NN Pred':>10} "
      f"{'Error%':>8} {'Status':>8}")
print("-" * 60)

all_pass = True
for m, T, sig, r, label in test_cases:
    bs = black_scholes_call(m, T, r, sig)
    raw = np.array([[m, T, sig, r]])
    scaled = torch.FloatTensor(scaler.transform(raw))
    
    with torch.no_grad():
        out = model(scaled)
    pred  = out[0, 0].item()
    error = abs(pred - bs) / (bs + 1e-8) * 100
    
    status = 'PASS' if error < 1.0 else 'FAIL'
    if error >= 1.0:
        all_pass = False
    
    print(f"{label:<20} {bs:>10.6f} {pred:>10.6f} "
          f"{error:>7.2f}% {status:>8}")

print(f"\nAll cases pass: {all_pass}")

if not all_pass:
    # Check if the pattern reveals which feature is wrongly scaled
    print("\nScaler parameters:")
    feature_names = ['moneyness', 'T', 'sigma', 'r']
    for i, name in enumerate(feature_names):
        print(f"  {name:<12}: mean={scaler.mean_[i]:.4f}, "
              f"std={scaler.scale_[i]:.4f}")
    
    print("\nExpected approximate scaler parameters:")
    print("  moneyness : mean≈1.00, std≈0.17")
    print("  T         : mean≈0.50, std≈0.55  (years)")
    print("  sigma     : mean≈0.40, std≈0.22")
    print("  r         : mean≈0.05, std≈0.03")
    print("\nIf your actual values differ significantly, ")
    print("the scaler was fitted on wrong data or wrong units.")