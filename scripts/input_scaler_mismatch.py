import joblib
import numpy as np

scaler_health = joblib.load('models/input_scaler.pkl')
scaler_router = joblib.load('outputs/router_v1/scaler.pkl')

print("Health check scaler (models/input_scaler.pkl):")
print(f"  Feature means: {scaler_health.mean_}")
print(f"  Feature stds:  {scaler_health.scale_}")

print("\nRouter scaler (outputs/router_v1/scaler.pkl):")
print(f"  Feature means: {scaler_router.mean_}")
print(f"  Feature stds:  {scaler_router.scale_}")

means_match = np.allclose(scaler_health.mean_, scaler_router.mean_)
stds_match  = np.allclose(scaler_health.scale_, scaler_router.scale_)

print(f"\nMeans match: {means_match}")
print(f"Stds match:  {stds_match}")

# Show what each scaler does to the ATM test input
test_input = np.array([[1.0, 0.5, 0.20, 0.05]])

scaled_health = scaler_health.transform(test_input)
scaled_router = scaler_router.transform(test_input)

print(f"\nATM input raw:            {test_input[0]}")
print(f"Scaled by health scaler:  {scaled_health[0]}")
print(f"Scaled by router scaler:  {scaled_router[0]}")
print(f"Max difference:           {np.abs(scaled_health - scaled_router).max():.6f}")