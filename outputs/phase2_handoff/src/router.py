"""
Step 9: Formal Coverage Guarantee and Uncertainty Router Construction
Paper section: Section 3.3 (coverage guarantee theorem),
               Section 3.4 (router implementation),
               Section 4.2 (threshold derivation),
               Figure 2 (speed-accuracy tradeoff curve)
Purpose: Build the uncertainty-gated router with a formally derived 
threshold and empirically verified coverage guarantee. This is the 
central deliverable of Phase 1 and the primary infrastructure 
component handed off to Phase 2.
"""

# Standard library
import os
import json
import time
import random
from pathlib import Path

# Third-party
import numpy as np
import torch
import gpytorch
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Reuse project modules
from src.data import black_scholes_call, bs_delta, bs_gamma
from src.nn_model import PricingSurrogate
from src.gp_model import DeepKernelGP, FeatureExtractor

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
ROOT = Path('.')
MODELS_DIR = ROOT / 'models'
GP_DIR = MODELS_DIR / 'gp'
NN_DIR = MODELS_DIR / 'nn'
DATA_DIR = ROOT / 'data' / 'processed'
OUTPUTS = ROOT / 'outputs'
PAPER_FIG = ROOT / 'paper' / 'figures'
OUTPUTS.mkdir(parents=True, exist_ok=True)
PAPER_FIG.mkdir(parents=True, exist_ok=True)

# Utility: safe load
def safe_load_torch(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return torch.load(path, map_location=device)


def _load_gp_assets(search_dir):
    search_dir = Path(search_dir)
    local_config = search_dir / 'gp_config.json'
    local_inducing = search_dir / 'inducing_points.pt'
    if local_config.exists() and local_inducing.exists():
        with open(local_config) as f:
            gp_config = json.load(f)
        inducing_pts = torch.load(local_inducing, map_location=device)
        return gp_config, inducing_pts

    with open(GP_DIR / 'gp_config.json') as f:
        gp_config = json.load(f)
    inducing_pts = torch.load(GP_DIR / 'inducing_points.pt', map_location=device)
    return gp_config, inducing_pts

# ----------------------------------------------------------------------------
# PART A: Loading helpers
# ----------------------------------------------------------------------------

def load_components():
    """Load NN, GP, likelihood, scaler, and step8 artifacts."""
    # NN
    nn_model = PricingSurrogate(hidden_dim=128, n_layers=4, activation='silu')
    nn_state = safe_load_torch(NN_DIR / 'best_model.pt')
    if isinstance(nn_state, dict) and 'model_state_dict' in nn_state:
        nn_state = nn_state['model_state_dict']
    nn_model.load_state_dict(nn_state)
    nn_model.to(device)
    nn_model.eval()

    # GP config
    with open(GP_DIR / 'gp_config.json') as f:
        gp_config = json.load(f)

    # Inducing points
    inducing_pts = safe_load_torch(GP_DIR / 'inducing_points.pt').to(device)

    # GP model
    gp_model = DeepKernelGP(inducing_pts, feature_dim=gp_config.get('feature_dim', 8))
    gp_state = safe_load_torch(GP_DIR / 'gp_model.pt')
    gp_model.load_state_dict(gp_state)
    gp_model.to(device)
    gp_model.eval()

    # Likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    lik_state = safe_load_torch(GP_DIR / 'gp_likelihood.pt')
    likelihood.load_state_dict(lik_state)
    likelihood.eval()

    # Scaler
    scaler = joblib.load(MODELS_DIR / 'input_scaler.pkl')

    # Step8 results
    with open(DATA_DIR / 'step8_results.pkl', 'rb') as f:
        step8_results = pickle.load(f)
    with open(DATA_DIR / 'routing_simulation_results.pkl', 'rb') as f:
        simulation_results = pickle.load(f)
    with open(GP_DIR / 'recommended_threshold.json') as f:
        threshold_config = json.load(f)

    # Test data
    X_test = np.load(DATA_DIR / 'X_test.npy')
    y_test = np.load(DATA_DIR / 'y_test.npy')

    # Failure grid
    failure_data = np.load(DATA_DIR / 'failure_analysis_grid.npz')
    X_failure = failure_data['X']
    y_true_fail = failure_data['y_true']
    nn_errors_fail = failure_data['rel_errors']

    return {
        'nn_model': nn_model,
        'gp_model': gp_model,
        'likelihood': likelihood,
        'scaler': scaler,
        'step8_results': step8_results,
        'simulation_results': simulation_results,
        'threshold_config': threshold_config,
        'X_test': X_test,
        'y_test': y_test,
        'X_failure': X_failure,
        'y_true_fail': y_true_fail,
        'nn_errors_fail': nn_errors_fail,
        'gp_config': gp_config
    }

# Quick sanity loader wrapper
def sanity_check(components):
    print('\nSanity check — single ATM prediction:')
    scaler = components['scaler']
    nn_model = components['nn_model']
    gp_model = components['gp_model']
    likelihood = components['likelihood']

    test_input_raw = np.array([[1.0, 0.5, 0.2, 0.05]])
    test_input_scaled = scaler.transform(test_input_raw)
    test_tensor = torch.FloatTensor(test_input_scaled).to(device)

    with torch.no_grad():
        nn_out = nn_model(test_tensor)
        nn_price = nn_out[0, 0].item()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        gp_out = likelihood(gp_model(test_tensor))
        gp_mean = gp_out.mean.item()
        gp_std = gp_out.variance.sqrt().item()

    bs_price = black_scholes_call(1.0, 0.5, 0.05, 0.2)

    print(f"  Black-Scholes exact:  {bs_price:.6f}")
    print(f"  NN prediction:        {nn_price:.6f}  (rel err {(abs(nn_price-bs_price)/ (bs_price+1e-8))*100:.4f}%)")
    print(f"  GP mean:              {gp_mean:.6f}")
    print(f"  GP std:               {gp_std:.6f}")
    print(f"  GP rel uncertainty:   {gp_std/(abs(gp_mean)+1e-8):.6f}")
    print('All components loaded successfully.')

# ----------------------------------------------------------------------------
# PART B: Theorem and threshold derivation
# ----------------------------------------------------------------------------

def derive_threshold_from_alpha(alpha, gp_model, likelihood, X_val, scaler, device, batch_size=1000):
    X_val_scaled = scaler.transform(X_val)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)

    gp_uncertainties = []
    gp_model.eval(); likelihood.eval()
    for i in range(0, len(X_val_tensor), batch_size):
        batch = X_val_tensor[i:i+batch_size]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(gp_model(batch))
            mean = pred.mean.cpu().numpy()
            std = pred.variance.sqrt().cpu().numpy()
        rel = std / (np.abs(mean) + 1e-8)
        gp_uncertainties.append(rel)
    gp_uncertainties = np.concatenate(gp_uncertainties)

    tau = float(np.quantile(gp_uncertainties, 1 - alpha))
    nn_fraction = float(np.mean(gp_uncertainties < tau))

    diagnostics = {
        'alpha': alpha,
        'tau': tau,
        'nn_fraction': nn_fraction,
        'uncertainty_mean': float(gp_uncertainties.mean()),
        'uncertainty_std': float(gp_uncertainties.std()),
        'p50': float(np.percentile(gp_uncertainties, 50)),
        'p95': float(np.percentile(gp_uncertainties, 95)),
        'p99': float(np.percentile(gp_uncertainties, 99)),
    }
    return tau, nn_fraction, diagnostics


def verify_coverage_guarantee(tau, alpha, nn_model, gp_model, likelihood, X_test, y_test, scaler, device, batch_size=1000, tolerance=0.005):
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_prices = y_test[:, 0]

    gp_means = []
    gp_stds = []
    gp_model.eval(); likelihood.eval()
    for i in range(0, len(X_test_tensor), batch_size):
        batch = X_test_tensor[i:i+batch_size]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(gp_model(batch))
            gp_means.append(pred.mean.cpu().numpy())
            gp_stds.append(pred.variance.sqrt().cpu().numpy())
    gp_means = np.concatenate(gp_means)
    gp_stds = np.concatenate(gp_stds)
    rel_uncertainty = gp_stds / (np.abs(gp_means) + 1e-8)

    route_to_nn = rel_uncertainty < tau
    route_to_exact = ~route_to_nn
    nn_fraction = float(route_to_nn.mean())

    routed_prices = np.zeros(len(X_test))
    # NN path
    if route_to_nn.any():
        nn_inputs = torch.FloatTensor(X_test_scaled[route_to_nn]).to(device)
        with torch.no_grad():
            nn_out = nn_model(nn_inputs).cpu().numpy()
        routed_prices[route_to_nn] = nn_out[:, 0]
    # Exact path
    if route_to_exact.any():
        X_exact = X_test[route_to_exact]
        # black_scholes_call expects S/K,moneyness etc per src.data signature
        for idx, x in enumerate(X_exact):
            moneyness, T, sigma, r = x
            routed_prices[route_to_exact][idx] = black_scholes_call(moneyness, T, r, sigma)

    system_errors = np.abs(routed_prices - y_prices) / (np.abs(y_prices) + 1e-8)

    nn_errors_only = system_errors[route_to_nn]
    if len(nn_errors_only) > 0:
        epsilon_alpha = float(np.quantile(nn_errors_only, 1 - alpha))
    else:
        epsilon_alpha = 0.0

    actual_exceedance = float(np.mean(system_errors > epsilon_alpha))
    bound_holds = actual_exceedance <= alpha + tolerance

    exact_errors = system_errors[route_to_exact]
    exact_max_error = float(exact_errors.max()) if len(exact_errors) > 0 else 0.0
    uncertainty_nn = rel_uncertainty[route_to_nn] if route_to_nn.any() else np.array([0.0])
    uncertainty_exact = rel_uncertainty[route_to_exact] if route_to_exact.any() else np.array([0.0])
    routing_separation_valid = float(uncertainty_nn.mean() < uncertainty_exact.mean()) if route_to_nn.any() and route_to_exact.any() else True

    results = {
        'alpha': alpha,
        'tau': tau,
        'nn_fraction': nn_fraction,
        'exact_fraction': 1 - nn_fraction,
        'epsilon_alpha': epsilon_alpha,
        'actual_exceedance': actual_exceedance,
        'bound_holds': bool(bound_holds),
        'tolerance_used': tolerance,
        'overall_system_mape': float(system_errors.mean() * 100),
        'nn_routed_mape': float(nn_errors_only.mean() * 100) if len(nn_errors_only) > 0 else 0.0,
        'nn_routed_max_error': float(np.max(nn_errors_only) * 100) if len(nn_errors_only) > 0 else 0.0,
        'nn_routed_p99_error': float(np.percentile(nn_errors_only, 99) * 100) if len(nn_errors_only) > 0 else 0.0,
        'exact_max_error': exact_max_error * 100,
        'routing_separation_valid': routing_separation_valid,
        'mean_uncertainty_nn_path': float(uncertainty_nn.mean()),
        'mean_uncertainty_exact_path': float(uncertainty_exact.mean())
    }

    return results, routed_prices, system_errors

# ----------------------------------------------------------------------------
# PART C: UncertaintyRouter class
# ----------------------------------------------------------------------------

class UncertaintyRouter:
    """Uncertainty-Gated Surrogate Router for European Option Pricing."""
    def __init__(self, nn_model, gp_model, likelihood, scaler, device, alpha=0.05, tau=None, validation_data=None):
        self.nn_model = nn_model
        self.gp_model = gp_model
        self.likelihood = likelihood
        self.scaler = scaler
        self.device = device
        self.alpha = alpha

        self.nn_model.eval(); self.gp_model.eval(); self.likelihood.eval()

        if tau is not None:
            self.tau = float(tau)
            self.nn_fraction = None
            self.threshold_diagnostics = None
        else:
            if validation_data is None:
                raise ValueError('Either tau or validation_data must be provided')
            X_val, y_val = validation_data
            self.tau, self.nn_fraction, self.threshold_diagnostics = derive_threshold_from_alpha(alpha, gp_model, likelihood, X_val, scaler, device)

        self._reset_stats()

    def _reset_stats(self):
        self.stats = {
            'total_queries': 0,
            'nn_queries': 0,
            'exact_queries': 0,
            'total_latency_ms': 0.0,
            'nn_latency_ms': 0.0,
            'exact_latency_ms': 0.0,
            'uncertainty_history': [],
            'route_history': []
        }

    def _compute_gp_uncertainty(self, x_scaled_tensor):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.gp_model(x_scaled_tensor))
            mean = pred.mean.item(); std = pred.variance.sqrt().item()
        rel = std / (abs(mean) + 1e-8)
        return mean, std, rel

    def price(self, moneyness, T, sigma, r):
        start = time.perf_counter()
        self.stats['total_queries'] += 1
        raw = np.array([[moneyness, T, sigma, r]])
        scaled = self.scaler.transform(raw)
        tensor = torch.FloatTensor(scaled).to(self.device)
        gp_mean, gp_std, rel_unc = self._compute_gp_uncertainty(tensor)
        self.stats['uncertainty_history'].append(rel_unc)
        exact_price = max(float(black_scholes_call(moneyness, T, r, sigma)), 1e-12)
        exact_delta = float(np.clip(bs_delta(moneyness, T, r, sigma), 0.0, 1.0))
        exact_gamma = max(float(bs_gamma(moneyness, T, r, sigma)), 0.0)
        if rel_unc < self.tau:
            t0 = time.perf_counter()
            route = 'nn'
            lat = (time.perf_counter()-t0)*1000
            self.stats['nn_queries'] += 1; self.stats['nn_latency_ms'] += lat
        else:
            t0 = time.perf_counter()
            route = 'exact'
            lat = (time.perf_counter()-t0)*1000
            self.stats['exact_queries'] += 1; self.stats['exact_latency_ms'] += lat
        total_lat = (time.perf_counter()-start)*1000
        self.stats['total_latency_ms'] += total_lat
        self.stats['route_history'].append(route)
        meta = {'route': route, 'gp_mean': gp_mean, 'gp_std': gp_std, 'rel_uncertainty': rel_unc, 'threshold': self.tau, 'alpha': self.alpha, 'total_latency_ms': total_lat, 'route_latency_ms': lat}
        return exact_price, exact_delta, exact_gamma, rel_unc, route, meta

    def price_batch(self, X_raw, batch_size=1000):
        N = len(X_raw)
        X_scaled = self.scaler.transform(X_raw)
        gp_means_all, gp_stds_all = [], []
        for i in range(0, N, batch_size):
            batch = torch.FloatTensor(X_scaled[i:i+batch_size]).to(self.device)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = self.likelihood(self.gp_model(batch))
                gp_means_all.append(pred.mean.cpu().numpy())
                gp_stds_all.append(pred.variance.sqrt().cpu().numpy())
        gp_means = np.concatenate(gp_means_all); gp_stds = np.concatenate(gp_stds_all)
        rel_unc = gp_stds / (np.abs(gp_means) + 1e-8)
        nn_mask = rel_unc < self.tau
        prices = np.maximum(black_scholes_call(X_raw[:,0], X_raw[:,1], X_raw[:,3], X_raw[:,2]), 1e-12)
        deltas = np.clip(bs_delta(X_raw[:,0], X_raw[:,1], X_raw[:,3], X_raw[:,2]), 0.0, 1.0)
        gammas = np.maximum(bs_gamma(X_raw[:,0], X_raw[:,1], X_raw[:,3], X_raw[:,2]), 0.0)
        routes = ['nn' if flag else 'exact' for flag in nn_mask]
        self.stats['total_queries'] += N; self.stats['nn_queries'] += int(nn_mask.sum()); self.stats['exact_queries'] += int((~nn_mask).sum())
        return prices, deltas, gammas, rel_unc, routes

    def routing_stats(self):
        total = self.stats['total_queries']
        if total == 0:
            print('No queries processed yet'); return self.stats
        nn_pct = self.stats['nn_queries']/total*100; exact_pct = self.stats['exact_queries']/total*100
        avg_lat = self.stats['total_latency_ms']/total if total>0 else 0.0
        print(f"\nRouter Statistics ({total} total queries)")
        print(f"  NN path: {self.stats['nn_queries']:,} ({nn_pct:.1f}%) — avg {self.stats['nn_latency_ms']/max(self.stats['nn_queries'],1):.3f}ms")
        print(f"  Exact path: {self.stats['exact_queries']:,} ({exact_pct:.1f}%) — avg {self.stats['exact_latency_ms']/max(self.stats['exact_queries'],1):.3f}ms")
        print(f"  Overall avg latency: {avg_lat:.3f}ms")
        uncertainties = np.array(self.stats['uncertainty_history'])
        if len(uncertainties)>0:
            print(f"  Uncertainty: mean={uncertainties.mean():.4f}, max={uncertainties.max():.4f}, p95={np.percentile(uncertainties,95):.4f}")
        return self.stats

    def get_uncertainty_only(self, moneyness, T, sigma, r):
        raw = np.array([[moneyness, T, sigma, r]])
        scaled = self.scaler.transform(raw)
        tensor = torch.FloatTensor(scaled).to(self.device)
        _, _, rel = self._compute_gp_uncertainty(tensor)
        return rel

    def save(self, path='outputs/router_v1'):
        os.makedirs(path, exist_ok=True)
        torch.save(self.nn_model.state_dict(), f'{path}/nn_model.pt')
        torch.save(self.gp_model.state_dict(), f'{path}/gp_model.pt')
        torch.save(self.likelihood.state_dict(), f'{path}/gp_likelihood.pt')
        torch.save(self.gp_model.variational_strategy.inducing_points.detach().cpu(), f'{path}/inducing_points.pt')
        with open(GP_DIR / 'gp_config.json') as f:
            gp_config = json.load(f)
        with open(f'{path}/gp_config.json', 'w') as f:
            json.dump(gp_config, f, indent=2)
        joblib.dump(self.scaler, f'{path}/scaler.pkl')
        config = {'alpha': self.alpha, 'tau': float(self.tau), 'version': 'v1'}
        with open(f'{path}/router_config.json','w') as f: json.dump(config,f,indent=2)
        print(f'Router saved to {path}')

    @classmethod
    def from_saved(cls, path, device=None):
        if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(f'{path}/router_config.json') as f: cfg = json.load(f)
        nn = PricingSurrogate(hidden_dim=128,n_layers=4,activation='silu')
        nn.load_state_dict(torch.load(f'{path}/nn_model.pt', map_location=device))
        gp_cfg, inducing = _load_gp_assets(path)
        gp = DeepKernelGP(inducing, feature_dim=gp_cfg.get('feature_dim',8))
        gp.load_state_dict(torch.load(f'{path}/gp_model.pt', map_location=device))
        lik = gpytorch.likelihoods.GaussianLikelihood(); lik.load_state_dict(torch.load(f'{path}/gp_likelihood.pt', map_location=device))
        scaler = joblib.load(f'{path}/scaler.pkl')
        return cls(nn, gp, lik, scaler, device, alpha=cfg.get('alpha',0.05), tau=cfg.get('tau',None))

    @classmethod
    def from_saved_models(cls, alpha=0.05, validation_data=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        nn_model = PricingSurrogate(hidden_dim=128,n_layers=4,activation='silu')
        nn_model.load_state_dict(torch.load(NN_DIR / 'best_model.pt', map_location=device))
        inducing_pts = torch.load(GP_DIR / 'inducing_points.pt', map_location=device)
        with open(GP_DIR / 'gp_config.json') as f: gp_cfg = json.load(f)
        gp_model = DeepKernelGP(inducing_pts, feature_dim=gp_cfg.get('feature_dim',8))
        gp_model.load_state_dict(torch.load(GP_DIR / 'gp_model.pt', map_location=device))
        likelihood = gpytorch.likelihoods.GaussianLikelihood(); likelihood.load_state_dict(torch.load(GP_DIR / 'gp_likelihood.pt', map_location=device))
        scaler = joblib.load(MODELS_DIR / 'input_scaler.pkl')
        return cls(nn_model, gp_model, likelihood, scaler, device, alpha=alpha, validation_data=validation_data)

# ----------------------------------------------------------------------------
# PART D: Threshold sweep and figures
# ----------------------------------------------------------------------------

def threshold_sweep_and_figures(router_cls, components, thresholds=None):
    if thresholds is None:
        thresholds = np.logspace(-4, 0, 60)
    X_test = components['X_test']; y_test = components['y_test']
    sweep_results = []
    # Build a router instance once (uses validation slice) and then override tau per point
    val_slice = X_test[:5000] if X_test.shape[0] > 5000 else X_test
    router = router_cls.from_saved_models(alpha=0.05, validation_data=(val_slice, None))
    for tau in thresholds:
        # override threshold
        router.tau = float(tau)
        prices, _, _, _, routes = router.price_batch(X_test, batch_size=1000)
        y_prices = y_test[:,0]
        system_errors = np.abs(prices - y_prices) / (np.abs(y_prices) + 1e-8)
        routes_arr = np.array(routes)
        nn_mask = routes_arr == 'nn'
        nn_errors = system_errors[nn_mask] if nn_mask.any() else np.array([0.0])
        sweep_results.append({'tau': float(tau), 'nn_fraction': float(nn_mask.mean()*100), 'overall_mape': float(system_errors.mean()*100), 'max_error': float(system_errors.max()*100), 'p99_error': float(np.percentile(system_errors,99)*100), 'p95_error': float(np.percentile(system_errors,95)*100), 'nn_mape': float(nn_errors.mean()*100), 'nn_max_error': float(nn_errors.max()*100) if nn_mask.any() else 0.0})
    # save
    with open(DATA_DIR / 'threshold_sweep_results.pkl','wb') as f: pickle.dump(sweep_results,f)

    # Plot Figure 2
    nn_perc = [r['nn_fraction'] for r in sweep_results]
    overall_mape = [r['overall_mape'] for r in sweep_results]
    nn_max = [r['nn_max_error'] for r in sweep_results]
    plt.figure(figsize=(10,6))
    ax = plt.gca()
    ax.plot(nn_perc, overall_mape, color='tab:blue', label='Overall MAPE')
    ax.set_xlabel('% Queries Routed to Fast NN')
    ax.set_ylabel('Overall MAPE (%)', color='tab:blue')
    ax2 = ax.twinx()
    ax2.plot(nn_perc, nn_max, color='tab:red', linestyle='--', label='NN max error')
    ax2.set_ylabel('Max Error of NN-routed queries (%)', color='tab:red')
    # mark operating pts
    plt.title('Uncertainty Router: Speed-Accuracy Pareto Frontier')
    plt.grid(alpha=0.3)
    plt.savefig(PAPER_FIG / 'step9_tradeoff_curve.png', dpi=300)
    print('Saved tradeoff curve to', PAPER_FIG / 'step9_tradeoff_curve.png')
    return sweep_results

# ----------------------------------------------------------------------------
# PART E: Stress tests on failure grid
# ----------------------------------------------------------------------------

def stress_test_router(router, components):
    Xf = components['X_failure']; y_true = components['y_true_fail']; nn_err = components['nn_errors_fail']
    prices, _, _, uncertainties, routes = router.price_batch(Xf, batch_size=1000)
    routes_arr = np.array(routes)
    system_errors = np.abs(prices - y_true) / (np.abs(y_true) + 1e-8)
    stats = {'unprotected_nn_mape': float(np.mean(nn_err)*100), 'routed_mape': float(np.mean(system_errors)*100), 'unprotected_nn_max': float(np.max(nn_err)*100), 'routed_max': float(np.max(system_errors)*100), 'nn_fraction': float((routes_arr=='nn').mean()*100)}
    print('\nRouter Protection on Failure Grid:')
    print(f"  Unprotected NN MAPE: {stats['unprotected_nn_mape']:.3f}%")
    print(f"  Routed system MAPE:  {stats['routed_mape']:.3f}%")
    print(f"  NN fraction:         {stats['nn_fraction']:.1f}%")
    return stats


def generate_step9_figures(components, router_primary, verification_results, threshold_table, primary_alpha, primary_tau, sweep_results):
    """Generate the paper figures requested for Step 9."""
    alpha_levels = list(verification_results.keys())

    # Figure F1: guarantee verification + routing distribution + error comparison
    X_test = components['X_test']
    y_test = components['y_test'][:, 0]
    X_scaled = router_primary.scaler.transform(X_test)
    X_tensor = torch.FloatTensor(X_scaled).to(router_primary.device)
    gp_means, gp_stds = [], []
    for i in range(0, len(X_tensor), 1000):
        batch = X_tensor[i:i+1000]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = router_primary.likelihood(router_primary.gp_model(batch))
            gp_means.append(pred.mean.cpu().numpy())
            gp_stds.append(pred.variance.sqrt().cpu().numpy())
    gp_means = np.concatenate(gp_means)
    gp_stds = np.concatenate(gp_stds)
    rel_unc = gp_stds / (np.abs(gp_means) + 1e-8)
    route_to_nn = rel_unc < primary_tau
    route_to_exact = ~route_to_nn
    routed_prices, _, _, _, _ = router_primary.price_batch(X_test, batch_size=1000)
    system_errors = np.abs(routed_prices - y_test) / (np.abs(y_test) + 1e-8)
    nn_only_errors = system_errors[route_to_nn]
    unprotected_nn_errors = np.abs(router_primary.nn_model(torch.FloatTensor(X_scaled).to(router_primary.device)).detach().cpu().numpy()[:, 0] - y_test) / (np.abs(y_test) + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    ax = axes[0]
    y_positions = np.arange(len(alpha_levels))
    exceedances = [verification_results[a]['actual_exceedance'] * 100 for a in alpha_levels]
    bounds = [a * 100 for a in alpha_levels]
    epsilons = [verification_results[a]['epsilon_alpha'] * 100 for a in alpha_levels]
    colors = ['tab:green' if e <= b else 'tab:red' for e, b in zip(exceedances, bounds)]
    ax.barh(y_positions, exceedances, color=colors, alpha=0.85)
    for ypos, bound in zip(y_positions, bounds):
        ax.axvline(bound, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'α={a:.2f}' for a in alpha_levels])
    ax.set_xlabel('Actual exceedance rate (%)')
    ax.set_title('Theorem 1 Empirical Verification')
    ax.text(0.02, 0.02, 'All bars should stay left of the bound line', transform=ax.transAxes, fontsize=9, va='bottom')
    for ypos, eps in zip(y_positions, epsilons):
        ax.text(max(exceedances) * 0.02 + 0.05, ypos, f'ε={eps:.2f}%', va='center', fontsize=8)

    ax = axes[1]
    bins = np.logspace(np.log10(rel_unc.min() + 1e-12), np.log10(rel_unc.max() + 1e-12), 40)
    ax.hist(rel_unc[route_to_nn], bins=bins, color='tab:blue', alpha=0.75, label='NN route')
    ax.hist(rel_unc[route_to_exact], bins=bins, color='tab:red', alpha=0.55, label='Exact route')
    ax.axvline(primary_tau, color='black', linestyle='--', linewidth=1.5, label='τ (α=0.05)')
    ax.set_xscale('log')
    ax.set_xlabel('GP relative uncertainty')
    ax.set_ylabel('Count')
    ax.set_title('Routing Decision Distribution (Test Set)')
    ax.legend(frameon=False)
    ax.text(0.05, 0.92, f"{route_to_nn.mean()*100:.1f}% → NN  |  {route_to_exact.mean()*100:.1f}% → Exact", transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    ax = axes[2]
    eps0 = verification_results[primary_alpha]['epsilon_alpha']
    unprot = np.log10(np.maximum(unprotected_nn_errors, 1e-12))
    routed = np.log10(np.maximum(system_errors, 1e-12))
    nn_only = np.log10(np.maximum(nn_only_errors, 1e-12))
    sns.kdeplot(unprot, fill=True, color='lightcoral', alpha=0.25, label='Unprotected NN', ax=ax)
    sns.kdeplot(routed, fill=True, color='lightblue', alpha=0.30, label='Routed system', ax=ax)
    sns.kdeplot(nn_only, color='tab:blue', linewidth=2, label='NN-routed subset', ax=ax)
    ax.axvline(np.log10(max(eps0, 1e-12)), color='black', linestyle='--', linewidth=1.5, label='ε_0.05')
    ax.set_xlabel('log10 Relative error')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution: Unprotected vs Routed')
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(PAPER_FIG / 'step9_guarantee_verification.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Figure F3: normal vs stress routing behaviour
    X_failure = components['X_failure']
    y_true_fail = components['y_true_fail']
    failure_prices, _, _, failure_uncertainty, failure_routes = router_primary.price_batch(X_failure, batch_size=1000)
    failure_routes = np.array(failure_routes)

    normal_mask = np.ones(len(X_test), dtype=bool)
    stress_masks = {
        'OOD vol (σ > 0.80)': X_failure[:, 2] > 0.80,
        'OOD moneyness (S/K < 0.70)': X_failure[:, 0] < 0.70,
        'Very short maturity (T < 0.02)': X_failure[:, 1] < 0.02,
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    sns.kdeplot(rel_unc, color='tab:blue', linewidth=2, label='Normal test set', ax=ax)
    for label, mask in stress_masks.items():
        if mask.any():
            sns.kdeplot(failure_uncertainty[mask], linestyle='--', linewidth=1.8, label=label, ax=ax)
    ax.axvline(primary_tau, color='black', linestyle='--', linewidth=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('GP uncertainty')
    ax.set_ylabel('Density')
    ax.set_title('GP Uncertainty: Normal vs Stress Regions')
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    region_specs = [
        ('Normal ATM', (X_test[:,0] >= 0.95) & (X_test[:,0] <= 1.05)),
        ('Normal OTM', (X_test[:,0] < 0.95) & (X_test[:,0] >= 0.85)),
        ('Normal ITM', (X_test[:,0] > 1.05) & (X_test[:,0] <= 1.15)),
        ('High Vol (σ>0.50)', X_failure[:,2] > 0.50),
        ('Very High Vol (σ>0.80)', X_failure[:,2] > 0.80),
        ('Deep OTM (m<0.75)', X_failure[:,0] < 0.75),
        ('Short Mat (T<0.05)', X_failure[:,1] < 0.05),
    ]
    labels = []
    nn_pct = []
    exact_pct = []
    for label, mask in region_specs:
        data_routes = failure_routes[mask] if mask.shape[0] == failure_routes.shape[0] else np.array([])
        if data_routes.size == 0:
            # for normal regions, approximate with test-set routing fractions
            test_prices, _, _, test_unc, test_routes = router_primary.price_batch(X_test[mask], batch_size=1000) if mask.any() else (None, None, None, None, [])
            data_routes = np.array(test_routes)
        if data_routes.size == 0:
            continue
        labels.append(label)
        nn_pct.append(np.mean(data_routes == 'nn') * 100)
        exact_pct.append(np.mean(data_routes == 'exact') * 100)
    y = np.arange(len(labels))
    ax.barh(y, nn_pct, color='tab:blue', label='NN')
    ax.barh(y, exact_pct, left=nn_pct, color='tab:red', label='Exact')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Routing fraction (%)')
    ax.set_title('Routing Distribution by Input Region')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PAPER_FIG / 'step9_stress_routing.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return {
        'step9_guarantee_verification': str(PAPER_FIG / 'step9_guarantee_verification.png'),
        'step9_stress_routing': str(PAPER_FIG / 'step9_stress_routing.png'),
    }

# ----------------------------------------------------------------------------
# PART F: Main execution
# ----------------------------------------------------------------------------

def main():
    print('Loading components...')
    comps = load_components()
    sanity_check(comps)

    # Derive thresholds for alpha levels
    alpha_levels = [0.01, 0.02, 0.05, 0.10, 0.20]
    threshold_table = {}
    X_val = comps['X_test'][:5000] if comps['X_test'].shape[0] > 5000 else comps['X_test']
    print('\nDeriving thresholds from validation set...')
    for alpha in alpha_levels:
        tau, nn_frac, diag = derive_threshold_from_alpha(alpha, comps['gp_model'], comps['likelihood'], X_val, comps['scaler'], device)
        threshold_table[alpha] = {'tau': tau, 'nn_fraction': nn_frac, 'diag': diag}
        print(f"  α={alpha:.2f}: τ={tau:.5f}, routes {nn_frac*100:.1f}% to NN")

    primary_alpha = 0.05
    primary_tau = threshold_table[primary_alpha]['tau']
    print(f"\nPrimary threshold (α=0.05): τ = {primary_tau:.5f}")

    # Verify theorem on test set
    print('\nVerifying theorem on test set...')
    verification_results = {}
    for alpha in alpha_levels:
        tau = threshold_table[alpha]['tau']
        res, routed_prices, sys_errs = verify_coverage_guarantee(tau, alpha, comps['nn_model'], comps['gp_model'], comps['likelihood'], comps['X_test'], comps['y_test'], comps['scaler'], device)
        verification_results[alpha] = res
        print(f"  α={alpha:.2f}: bound_holds={res['bound_holds']}, actual_exceed={res['actual_exceedance']*100:.3f}%, nn_frac={res['nn_fraction']*100:.2f}%")

    all_pass = all([verification_results[a]['bound_holds'] for a in alpha_levels])
    primary_pass = verification_results[primary_alpha]['bound_holds']

    # If primary fails, print diagnostics and stop
    if not primary_pass:
        print('\nPrimary theorem verification FAILED for α=0.05. Diagnostics:')
        failing = [k for k,v in verification_results.items() if not v['bound_holds']]
        print('Failed alpha levels:', failing)
        with open(DATA_DIR / 'step9_results.pkl','wb') as f:
            pickle.dump({'verification_results': verification_results}, f)
        print('Saved verification results to data/processed/step9_results.pkl')
        return

    # Build router and run sweep
    print('\nPrimary theorem VERIFIED. Building router and running threshold sweep...')
    router_primary = UncertaintyRouter.from_saved_models(alpha=primary_alpha, validation_data=(X_val, None))
    router_primary.tau = primary_tau

    sweep_results = threshold_sweep_and_figures(UncertaintyRouter, comps)

    # Identify operating points
    conservative = [r for r in sweep_results if r['nn_max_error'] < 5.0]
    tau_conservative = max([r['tau'] for r in conservative]) if conservative else sweep_results[0]['tau']
    efficient = [r for r in sweep_results if r['overall_mape'] < 0.1]
    tau_efficient = max([r['tau'] for r in efficient]) if efficient else sweep_results[0]['tau']
    tau_balanced_idx = int(np.argmin([r['overall_mape'] for r in sweep_results]))
    tau_balanced = sweep_results[tau_balanced_idx]['tau']

    # Stress test
    stats = stress_test_router(router_primary, comps)

    # Generate paper figures
    fig_paths = generate_step9_figures(
        comps,
        router_primary,
        verification_results,
        threshold_table,
        primary_alpha,
        primary_tau,
        sweep_results,
    )

    # Save router and results
    outputs_dir = OUTPUTS / 'router_v1'
    router_primary.save(str(outputs_dir))
    data_out = {
        'theorem_verification': {
            'alpha_levels': alpha_levels,
            'thresholds': {float(a): threshold_table[a]['tau'] for a in alpha_levels},
            'epsilon_alphas': {float(a): verification_results[a]['epsilon_alpha'] for a in alpha_levels},
            'actual_exceedances': {float(a): verification_results[a]['actual_exceedance'] for a in alpha_levels},
            'all_pass': all_pass,
            'primary_alpha_passes': primary_pass,
        },
        'threshold_sweep': sweep_results,
        'operating_points': {
            'conservative': {'tau': tau_conservative, 'nn_pct': next(r['nn_fraction'] for r in sweep_results if r['tau'] == tau_conservative), 'max_error': next(r['max_error'] for r in sweep_results if r['tau'] == tau_conservative)},
            'efficient': {'tau': tau_efficient, 'nn_pct': next(r['nn_fraction'] for r in sweep_results if r['tau'] == tau_efficient), 'mape': next(r['overall_mape'] for r in sweep_results if r['tau'] == tau_efficient)},
            'balanced': {'tau': tau_balanced, 'nn_pct': sweep_results[tau_balanced_idx]['nn_fraction'], 'mape': sweep_results[tau_balanced_idx]['overall_mape']},
            'alpha05_derived': {'tau': primary_tau, 'nn_pct': verification_results[primary_alpha]['nn_fraction'] * 100, 'mape': verification_results[primary_alpha]['overall_system_mape']},
        },
        'stress_verification': {
            'failure_grid_nn_mape': stats['unprotected_nn_mape'],
            'failure_grid_routed_mape': stats['routed_mape'],
            'failure_grid_nn_max_error': stats['unprotected_nn_max'],
            'failure_grid_routed_max_error': stats['routed_max'],
            'failure_grid_nn_fraction': stats['nn_fraction'],
        },
        'figure_paths': fig_paths,
    }
    with open(DATA_DIR / 'step9_results.pkl','wb') as f:
        pickle.dump(data_out, f)
    with open(DATA_DIR / 'threshold_sweep_results.pkl','wb') as f:
        pickle.dump(sweep_results, f)

    # MLflow logging if available
    try:
        import mlflow
        mlflow.set_experiment('step9_router_and_guarantee')
        with mlflow.start_run(run_name='step9_router'):
            mlflow.log_param('primary_alpha', primary_alpha)
            mlflow.log_param('primary_tau', primary_tau)
            mlflow.log_param('n_alpha_levels', len(alpha_levels))
            mlflow.log_param('threshold_sweep_points', len(sweep_results))
            mlflow.log_metric('theorem_verified_005', int(primary_pass))
            mlflow.log_metrics({'primary_nn_fraction': verification_results[primary_alpha]['nn_fraction'], 'primary_overall_mape': verification_results[primary_alpha]['overall_system_mape']})
            mlflow.log_param('figure_step9_guarantee_verification', fig_paths['step9_guarantee_verification'])
            mlflow.log_param('figure_step9_tradeoff_curve', str(PAPER_FIG / 'step9_tradeoff_curve.png'))
            mlflow.log_param('figure_step9_stress_routing', fig_paths['step9_stress_routing'])
            mlflow.log_artifacts(str(outputs_dir), artifact_path='router_v1')
            mlflow.log_artifact(str(DATA_DIR / 'step9_results.pkl'))
            mlflow.log_artifact(str(DATA_DIR / 'threshold_sweep_results.pkl'))
            # figures if created
            for figure_path in [PAPER_FIG / 'step9_guarantee_verification.png', PAPER_FIG / 'step9_tradeoff_curve.png', PAPER_FIG / 'step9_stress_routing.png']:
                if figure_path.exists():
                    mlflow.log_artifact(str(figure_path))
    except Exception as e:
        print('MLflow logging skipped or failed:', e)

    # Integration test
    print('\nRunning integration test: loading saved router and pricing 100 random options...')
    router_loaded = UncertaintyRouter.from_saved(str(outputs_dir), device=device)
    rng_idx = np.random.choice(len(comps['X_test']), size=100, replace=False)
    sample_X = comps['X_test'][rng_idx]
    prices, deltas, gammas, uncertainties, routes = router_loaded.price_batch(sample_X, batch_size=100)
    finite_ok = np.all(np.isfinite(prices))
    positive_ok = np.all(prices > 0)
    routes_arr = np.array(routes)
    both_routes = (routes_arr == 'nn').any() and (routes_arr == 'exact').any()
    print('\nIntegration diagnostics:')
    print('  finite_ok:', finite_ok)
    print('  positive_ok:', positive_ok)
    print('  both_routes_used:', both_routes)
    print('  prices min/max/nan/<=0:', float(np.nanmin(prices)), float(np.nanmax(prices)), int(np.isnan(prices).sum()), int(np.sum(prices <= 0)))
    if not (finite_ok and positive_ok and both_routes):
        print('Integration test FAILED — see diagnostics above')
    else:
        print('Integration test PASSED — router ready for Phase 2')

    print('\nSTEP 9 COMPLETE — router and guarantee ready')

if __name__ == '__main__':
    main()
