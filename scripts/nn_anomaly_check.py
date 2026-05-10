import os
import sys
import numpy as np
import torch
import joblib

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.nn_model import PricingSurrogate
from src.router import UncertaintyRouter
from src.data import black_scholes_call


def line():
    print('-' * 90)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cpu')

    # Load standalone NN + global scaler
    scaler = joblib.load('models/input_scaler.pkl')
    model = PricingSurrogate(hidden_dim=128, n_layers=4, activation='silu').to(device)
    model.load_state_dict(torch.load('models/nn/best_model.pt', map_location=device))
    model.eval()

    # Load router bundle
    router = UncertaintyRouter.from_saved('outputs/router_v1/', device=device)
    router_scaler = router.scaler
    router_model = getattr(router, 'nn', None)
    if router_model is None:
        router_model = router.nn_model
    router_model.eval()

    x_raw = np.array([1.0, 1.0, 0.2, 0.05], dtype=np.float64)

    print('NN DIAGNOSTIC CHECKS')
    line()

    # Prompt 1: Compare raw NN outputs (router vs standalone)
    scaled_router = router_scaler.transform([x_raw])
    tensor_router = torch.FloatTensor(scaled_router).to(device)
    with torch.no_grad():
        out_router = router_model(tensor_router)

    scaled_test = scaler.transform([x_raw])
    tensor_test = torch.FloatTensor(scaled_test).to(device)
    with torch.no_grad():
        out_test = model(tensor_test)

    print('Prompt 1 - SAME input output comparison')
    print('Router NN output     :', out_router.cpu().numpy())
    print('Standalone NN output :', out_test.cpu().numpy())
    p1_diff = np.max(np.abs(out_router.cpu().numpy() - out_test.cpu().numpy()))
    print('Max abs diff         :', float(p1_diff))
    print('Model inconsistency? :', 'YES' if p1_diff > 1e-7 else 'NO')
    line()

    # Prompt 2: feature order corruption
    print('Prompt 2 - Feature order / scaling check')
    print('Input raw       :', x_raw)
    print('Scaled (router) :', scaled_router)
    print('Scaled (test)   :', scaled_test)
    scale_diff = np.max(np.abs(scaled_router - scaled_test))
    print('Scaled diff max :', float(scale_diff))
    print('Likely order mismatch?:', 'YES' if scale_diff > 1e-8 else 'NO')
    line()

    # Prompt 3: double scaling
    print('Prompt 3 - Double scaling check')
    first = scaler.transform([x_raw])
    second = scaler.transform(first)
    print('Before scaling  :', x_raw)
    print('After 1st scale :', first)
    print('After 2nd scale :', second)
    print('2nd-scale drift magnitude:', float(np.max(np.abs(second - first))))
    line()

    # Prompt 4: dtype mismatch
    print('Prompt 4 - dtype check')
    print('Model dtype:', next(model.parameters()).dtype)
    print('Input dtype:', tensor_test.dtype)
    tensor_test = tensor_test.float()
    model = model.float()
    print('After fix -> Model dtype:', next(model.parameters()).dtype)
    print('After fix -> Input dtype:', tensor_test.dtype)
    line()

    # Prompt 5: Verify router actually uses NN output
    print('Prompt 5 - Router internals consistency check')
    # Compute router NN raw output directly
    with torch.no_grad():
        nn_out_inside = router_model(tensor_router)
    nn_price_inside = float(nn_out_inside[0, 0].item())

    # Router public API output
    r_price, r_delta, r_gamma, r_unc, r_route, r_meta = router.price(1.0, 1.0, 0.2, 0.05)
    print('NN raw output inside router path (direct NN):', nn_price_inside)
    print('Router returned price                     :', float(r_price))
    print('Router route                              :', r_route)
    print('Router uncertainty                        :', float(r_unc))
    print('Router overriding NN output?              :', 'YES' if abs(float(r_price) - nn_price_inside) > 1e-8 else 'NO')
    line()

    # Prompt 6: Direct Black-Scholes comparison
    print('Prompt 6 - Direct BS comparison')
    bs = black_scholes_call(1.0, 1.0, 0.05, 0.2)
    print('BS  :', float(bs))
    print('NN  :', float(out_test[0, 0].item()))
    print('Router price:', float(r_price))
    bs_nn_rel = abs(float(out_test[0, 0].item()) - float(bs)) / (abs(float(bs)) + 1e-12) * 100
    print('NN rel error (%):', float(bs_nn_rel))
    line()

    # Prompt 7: model corruption in memory
    print('Prompt 7 - Parameter NaN scan')
    any_nan = False
    nan_report = []
    for name, param in model.named_parameters():
        has_nan = bool(torch.isnan(param).any().item())
        any_nan = any_nan or has_nan
        nan_report.append((name, has_nan))
        print(f'{name:40s} NaN={has_nan}')
    print('Any NaN in model params?:', any_nan)
    line()

    # Additional integrity checks
    print('Extra checks - state dict and scaler identity')
    standalone_sd = torch.load('models/nn/best_model.pt', map_location='cpu')
    router_sd = torch.load('outputs/router_v1/nn_model.pt', map_location='cpu')
    key = next(iter(standalone_sd.keys()))
    wdiff = torch.max(torch.abs(standalone_sd[key] - router_sd[key])).item()
    print('Weight max diff (best_model vs router nn_model):', wdiff)

    same_scaler = True
    try:
        router_scaler_file = joblib.load('outputs/router_v1/scaler.pkl')
        model_scaler_file = joblib.load('models/input_scaler.pkl')
        same_scaler = np.allclose(router_scaler_file.mean_, model_scaler_file.mean_) and np.allclose(router_scaler_file.scale_, model_scaler_file.scale_)
    except Exception:
        same_scaler = False
    print('Router scaler equals model scaler?:', same_scaler)
    line()

    anomalies = []
    if p1_diff > 1e-7:
        anomalies.append('Router NN and standalone NN give different outputs on same input.')
    if scale_diff > 1e-8:
        anomalies.append('Router scaler and standalone scaler transform input differently.')
    if abs(float(r_price) - nn_price_inside) > 1e-8:
        anomalies.append('Router price output does not use NN price even when route=nn.')
    if bs_nn_rel > 1.0:
        anomalies.append('Standalone NN has >1% error on ATM BS sanity point.')
    if any_nan:
        anomalies.append('NaNs detected in NN parameters.')

    print('ANOMALY SUMMARY')
    if anomalies:
        for i, a in enumerate(anomalies, 1):
            print(f'{i}. {a}')
    else:
        print('No anomalies detected in requested checks.')


if __name__ == '__main__':
    main()
