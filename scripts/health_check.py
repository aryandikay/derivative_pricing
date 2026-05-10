#!/usr/bin/env python3
"""
Comprehensive Phase 1 model health check (READ-ONLY evaluation).
Saves outputs:
 - paper/figures/model_health_check.png
 - paper/health_check_report.txt
 - data/processed/health_check_results.pkl

Run from workspace root.
"""
import os
import sys
import time
import json
import pickle
try:
    import joblib
except Exception:
    joblib = None
import datetime
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import gpytorch

from scipy.stats import norm, spearmanr, pearsonr

# LOWESS
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except Exception:
    lowess = None

ROOT = os.path.dirname(os.path.dirname(__file__))
# ensure src import
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import black_scholes_call
from src.nn_model import PricingSurrogate
from src.gp_model import DeepKernelGP
from src.router import UncertaintyRouter

# Utility
def safe_load(path, loader, name):
    if not os.path.exists(path):
        print(f"MISSING: {path} (required: {name})")
        return None
    try:
        return loader(path)
    except Exception as e:
        print(f"ERROR loading {path}: {e}")
        return None

# thresholds
thresholds = {
    'nn_overall_mape':    {'pass': 0.5,  'warn': 1.0,  'unit': '%'},
    'nn_atm_mape':        {'pass': 0.2,  'warn': 0.5,  'unit': '%'},
    'nn_deep_otm_mape':   {'pass': 1.5,  'warn': 3.0,  'unit': '%'},
    'nn_max_error':       {'pass': 5.0,  'warn': 10.0, 'unit': '%'},
    'nn_delta_mae':       {'pass': 0.005,'warn': 0.01, 'unit': 'abs'},
    'nn_latency_ms':      {'pass': 1.0,  'warn': 3.0,  'unit': 'ms'},
    'gp_95ci_coverage':   {'pass': 95.0, 'warn': 93.0, 'unit': '%', 'direction': 'above'},
    'gp_ece':             {'pass': 0.02, 'warn': 0.05, 'unit': 'float'},
    'spearman_corr':      {'pass': 0.6,  'warn': 0.4,  'unit': 'float', 'direction': 'above'},
    'router_nn_fraction': {'pass': 90.0, 'warn': 80.0, 'unit': '%', 'direction': 'above'},
    'system_mape':        {'pass': 0.5,  'warn': 1.0,  'unit': '%'},
    'theorem_holds':      {'pass': True, 'warn': None, 'unit': 'bool'},
    'gfc_mape_reduction': {'pass': 60.0, 'warn': 30.0, 'unit': '%', 'direction': 'above'},
}

def status_symbol(value, threshold_dict, key):
    t = threshold_dict[key]
    direction = t.get('direction', 'below')
    if t['unit'] == 'bool':
        if value == t['pass']:
            return '✓ PASS', 'green'
        else:
            return '✗ FAIL', 'red'
    if direction == 'above':
        if value >= t['pass']: return '✓ PASS', 'green'
        elif value >= t['warn']: return '⚠ WARN', 'yellow'
        else: return '✗ FAIL', 'red'
    else:
        if value <= t['pass']: return '✓ PASS', 'green'
        elif value <= t['warn']: return '⚠ WARN', 'yellow'
        else: return '✗ FAIL', 'red'

# Set seeds
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

results = {'timestamp': datetime.datetime.utcnow().isoformat()}

# Paths
paths = {
    'nn_state': 'models/nn/best_model.pt',
    'nn_traced': 'models/nn/best_model_traced.pt',
    'gp_config': 'models/gp/gp_config.json',
    'gp_inducing': 'models/gp/inducing_points.pt',
    'gp_state': 'models/gp/gp_model.pt',
    'gp_lik': 'models/gp/gp_likelihood.pt',
    'scaler': 'models/input_scaler.pkl',
    'router_dir': 'outputs/router_v1/',
    'X_test': 'data/processed/X_test.npy',
    'y_test': 'data/processed/y_test.npy',
    'failure_grid': 'data/processed/failure_analysis_grid.npz',
}

# Load scaler (try joblib, fall back to pickle)
def load_scaler_fallback(p):
    if joblib is not None:
        return joblib.load(p)
    else:
        with open(p,'rb') as f:
            return pickle.load(f)

scaler = safe_load(paths['scaler'], load_scaler_fallback, 'input_scaler')

# Load NN
nn_model = None
if os.path.exists(paths['nn_state']):
    try:
        nn_model = PricingSurrogate(hidden_dim=128, n_layers=4, activation='silu')
        sd = torch.load(paths['nn_state'], map_location=device)
        nn_model.load_state_dict(sd)
        nn_model.eval().to(device)
        print('NN loaded')
    except Exception as e:
        print('Failed loading NN:', e)
else:
    print('Missing NN state:', paths['nn_state'])

# Load TorchScript
nn_traced = None
if os.path.exists(paths['nn_traced']):
    try:
        nn_traced = torch.jit.load(paths['nn_traced'], map_location=device)
        nn_traced.eval()
        print('TorchScript loaded')
    except Exception as e:
        print('Failed loading TorchScript:', e)
else:
    print('Missing traced model:', paths['nn_traced'])

# Load GP
gp_model = None
likelihood = None
gp_config = None
if os.path.exists(paths['gp_config']):
    try:
        with open(paths['gp_config']) as f:
            gp_config = json.load(f)
    except Exception as e:
        print('Failed loading gp_config:', e)
else:
    print('Missing gp_config:', paths['gp_config'])

if os.path.exists(paths['gp_inducing']) and gp_config is not None:
    try:
        inducing = torch.load(paths['gp_inducing'], map_location=device)
        gp_model = DeepKernelGP(inducing, feature_dim=gp_config['feature_dim'])
        gp_sd = torch.load(paths['gp_state'], map_location=device) if os.path.exists(paths['gp_state']) else None
        if gp_sd is not None:
            gp_model.load_state_dict(gp_sd)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        if os.path.exists(paths['gp_lik']):
            likelihood.load_state_dict(torch.load(paths['gp_lik'], map_location=device))
        gp_model.eval().to(device)
        likelihood.eval()
        print('GP loaded')
    except Exception as e:
        print('Failed loading GP:', e)
else:
    print('Missing GP files or config')

# Load router
router = None
if os.path.isdir(paths['router_dir']):
    try:
        router = UncertaintyRouter.from_saved(paths['router_dir'])
        print('Router loaded')
    except Exception as e:
        print('Failed loading router:', e)
else:
    print('Missing router dir:', paths['router_dir'])

# Load test data
X_test = safe_load(paths['X_test'], lambda p: np.load(p), 'X_test')
y_test = safe_load(paths['y_test'], lambda p: np.load(p), 'y_test')

# Load cached steps for reference if present
cached = {}
for key in ['step7_results.pkl','step8_results.pkl','step9_results.pkl','step10_results.pkl','gp_calibration.pkl']:
    p = os.path.join('data/processed', key)
    if os.path.exists(p):
        try:
            with open(p, 'rb') as f:
                cached[key] = pickle.load(f)
        except Exception:
            cached[key] = None

# SANITY CHECKS
print('\n=== SECTION 1: SANITY CHECKS ===')
if nn_model is None or gp_model is None or likelihood is None or router is None or scaler is None:
    print('One or more core components are missing. Will continue but some sections will be skipped.')

sanity_ok = True
try:
    test_raw = np.array([[1.0, 0.5, 0.20, 0.05]])
    if scaler is not None:
        test_scaled = scaler.transform(test_raw)
    else:
        test_scaled = test_raw
    test_tensor = torch.FloatTensor(test_scaled).to(device)
    bs_exact = black_scholes_call(1.0, 0.5, 0.05, 0.20)
    nn_price = None
    if nn_model is not None:
        with torch.no_grad():
            nn_out = nn_model(test_tensor)
            nn_price = float(nn_out[0,0].item())
    gp_mean = gp_std = None
    if gp_model is not None and likelihood is not None:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            gp_pred = likelihood(gp_model(test_tensor))
            gp_mean = float(gp_pred.mean.item())
            gp_std  = float(gp_pred.variance.sqrt().item())
    r_price = r_unc = r_route = None
    if router is not None:
        r_price, r_delta, r_gamma, r_unc, r_route, r_meta = router.price(1.0, 0.5, 0.20, 0.05)
    print('SANITY: BS exact:', bs_exact)
    print('  NN price:', nn_price)
    print('  GP mean/std:', gp_mean, gp_std)
    print('  Router price/route:', r_price, r_route)
    nn_error = abs(nn_price - bs_exact) / (bs_exact + 1e-12) * 100 if nn_price is not None else np.inf
    if nn_price is None or nn_error >= 1.0:
        print(f'FAIL: NN error on ATM = {nn_error:.3f}% >= 1%')
        sanity_ok = False
    if gp_mean is None or abs(gp_mean - bs_exact) / (bs_exact + 1e-12) >= 0.05:
        print('FAIL: GP prediction very far from BS')
        sanity_ok = False
    if r_price is None or r_price <= 0:
        print('FAIL: Router returned non-positive price')
        sanity_ok = False
    if sanity_ok:
        print('SANITY CHECK PASSED — All models operational')
    else:
        print('SANITY CHECK FAILED — stopping as requested')
        results['sanity_ok'] = False
        # Save partial results and exit
        os.makedirs('data/processed', exist_ok=True)
        with open('data/processed/health_check_results.pkl','wb') as f:
            pickle.dump(results, f)
        sys.exit(1)
except Exception as e:
    print('Sanity check failed with exception:', e)
    results['sanity_ok'] = False
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/health_check_results.pkl','wb') as f:
        pickle.dump(results, f)
    sys.exit(1)

results['sanity_ok'] = True

# SECTION 2: NN DEEP EVAL
print('\n=== SECTION 2: NN SURROGATE EVALUATION ===')
nn_metrics = {}
if X_test is None or y_test is None or nn_model is None:
    print('Skipping NN evaluation due to missing data or model')
else:
    X_test_scaled = scaler.transform(X_test) if scaler is not None else X_test
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_prices = y_test[:,0]
    y_deltas = y_test[:,1]
    y_gammas = y_test[:,2]
    nn_model.eval()
    with torch.no_grad():
        nn_preds = nn_model(X_test_tensor).cpu().numpy()
    nn_price_pred = nn_preds[:,0]
    nn_delta_pred = nn_preds[:,1]
    nn_gamma_pred = nn_preds[:,2]
    price_errors = np.abs(nn_price_pred - y_prices) / (y_prices + 1e-8)
    overall_mape = price_errors.mean() * 100
    overall_rmse = np.sqrt(np.mean((nn_price_pred - y_prices)**2))
    overall_max = price_errors.max() * 100
    overall_p99 = np.percentile(price_errors,99) * 100
    overall_p95 = np.percentile(price_errors,95) * 100
    relative_bias = np.mean((nn_price_pred - y_prices)/(y_prices+1e-8))*100
    nn_metrics.update({'overall_mape': overall_mape, 'overall_rmse': overall_rmse,
                       'overall_max': overall_max, 'overall_p99': overall_p99,
                       'overall_p95': overall_p95, 'relative_bias': relative_bias})
    # buckets by moneyness
    moneyness = X_test[:,0]
    buckets = {
        'deep_otm': (0.70,0.85), 'otm':(0.85,0.95), 'atm':(0.95,1.05),
        'itm':(1.05,1.15), 'deep_itm':(1.15,1.30)
    }
    bucket_stats = {}
    for k,(a,b) in buckets.items():
        mask = (moneyness >= a) & (moneyness < b)
        vals = price_errors[mask]
        bucket_stats[k] = {'mape': vals.mean()*100 if len(vals)>0 else np.nan,
                           'max': vals.max()*100 if len(vals)>0 else np.nan,
                           'n': int(mask.sum())}
    nn_metrics['by_moneyness'] = bucket_stats
    # time buckets
    T = X_test[:,1]
    t_buckets = {'very_short':(None,0.05),'short':(0.05,0.25),'medium':(0.25,1.0),'long':(1.0,None)}
    tb = {}
    for k,(a,b) in t_buckets.items():
        if a is None:
            mask = T < b
        elif b is None:
            mask = T >= a
        else:
            mask = (T >= a) & (T < b)
        vals = price_errors[mask]
        tb[k] = {'mape': vals.mean()*100 if len(vals)>0 else np.nan,'n':int(mask.sum())}
    nn_metrics['by_time'] = tb
    # vol buckets
    sigma = X_test[:,2]
    v_buckets = {'low':(None,0.20),'mid':(0.20,0.50),'high':(0.50,None)}
    vb = {}
    for k,(a,b) in v_buckets.items():
        if a is None:
            mask = sigma < b
        elif b is None:
            mask = sigma >= a
        else:
            mask = (sigma >= a) & (sigma < b)
        vals = price_errors[mask]
        vb[k] = {'mape': vals.mean()*100 if len(vals)>0 else np.nan,'n':int(mask.sum())}
    nn_metrics['by_vol'] = vb
    # greeks
    delta_mae = np.mean(np.abs(nn_delta_pred - y_deltas))
    delta_mape = np.mean(np.abs(nn_delta_pred - y_deltas)/(y_deltas+0.01))*100
    gamma_mae = np.mean(np.abs(nn_gamma_pred - y_gammas))
    nn_metrics.update({'delta_mae': delta_mae,'delta_mape': delta_mape,'gamma_mae': gamma_mae})
    # latency
    print('Warming up NN...')
    for _ in range(100): _ = nn_model(test_tensor)
    times = []
    for _ in range(200):
        start = time.perf_counter()
        with torch.no_grad(): _ = nn_model(test_tensor)
        times.append((time.perf_counter() - start)*1000)
    latency_mean = np.mean(times)
    latency_p99 = np.percentile(times,99)
    start = time.perf_counter()
    with torch.no_grad(): _ = nn_model(X_test_tensor)
    batch_time = (time.perf_counter() - start)*1000
    throughput = len(X_test)/(batch_time/1000)
    nn_metrics.update({'latency_mean_ms': latency_mean,'latency_p99_ms': latency_p99,'throughput': throughput})
    # pass/fail decisions
    nn_metrics['pass_overall_mape'] = status_symbol(overall_mape, thresholds, 'nn_overall_mape')
    atm_mape = bucket_stats['atm']['mape']
    nn_metrics['pass_atm_mape'] = status_symbol(atm_mape, thresholds, 'nn_atm_mape')
    nn_metrics['pass_deep_otm_mape'] = status_symbol(bucket_stats['deep_otm']['mape'], thresholds, 'nn_deep_otm_mape')
    nn_metrics['pass_delta_mae'] = status_symbol(delta_mae, thresholds, 'nn_delta_mae')
    nn_metrics['pass_latency'] = status_symbol(latency_mean, thresholds, 'nn_latency_ms')
    nn_metrics['pass_max_error'] = status_symbol(overall_max, thresholds, 'nn_max_error')

results['nn'] = nn_metrics

# SECTION 3: GP CALIBRATION
print('\n=== SECTION 3: GP CALIBRATION ===')
gp_metrics = {}
if gp_model is None or likelihood is None or X_test is None:
    print('Skipping GP calibration due to missing components')
else:
    X_test_scaled = scaler.transform(X_test) if scaler is not None else X_test
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    gp_means_all = []
    gp_stds_all = []
    batch = 1000
    for i in range(0, len(X_test_tensor), batch):
        b = X_test_tensor[i:i+batch]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(gp_model(b))
            gp_means_all.append(pred.mean.cpu().numpy())
            gp_stds_all.append(pred.variance.sqrt().cpu().numpy())
        if i % 10000 == 0:
            print(f'GP progress: {i}/{len(X_test_tensor)}')
    gp_means = np.concatenate(gp_means_all)
    gp_stds = np.concatenate(gp_stds_all)
    gp_rel_unc = gp_stds/(gp_means+1e-8)
    gp_metrics['unc_mean'] = float(np.mean(gp_rel_unc))
    gp_metrics['unc_median'] = float(np.median(gp_rel_unc))
    gp_metrics['unc_p95'] = float(np.percentile(gp_rel_unc,95))
    gp_metrics['unc_max'] = float(np.max(gp_rel_unc))
    # coverage
    coverages = {}
    for conf in [0.50,0.68,0.80,0.90,0.95,0.99]:
        z = norm.ppf((1+conf)/2)
        lower = gp_means - z*gp_stds
        upper = gp_means + z*gp_stds
        coverage = np.mean((y_prices >= lower) & (y_prices <= upper))
        gap = coverage - conf
        status = 'OVERCONFIDENT — DANGEROUS' if gap < -0.02 else ('PASS' if gap >= 0 else 'SLIGHTLY LOW — MONITOR')
        coverages[conf] = {'coverage': float(coverage), 'gap': float(gap), 'status': status}
    gp_metrics['coverages'] = coverages
    # ECE
    bins = np.linspace(0,1,11)
    ece = 0.0
    for i in range(10):
        conf_low, conf_high = bins[i], bins[i+1]
        conf_mid = (conf_low + conf_high)/2
        z = norm.ppf((1+conf_mid)/2)
        lower = gp_means - z*gp_stds
        upper = gp_means + z*gp_stds
        in_interval = (y_prices >= lower) & (y_prices <= upper)
        actual_cov = in_interval.mean()
        ece += abs(actual_cov - conf_mid)
    ece /= 10
    gp_metrics['ece'] = float(ece)
    # regional checks
    moneyness = X_test[:,0]
    regions = {
        'otm': moneyness < 0.95,
        'atm': (moneyness >= 0.95) & (moneyness < 1.05),
        'high_vol': X_test[:,2] > 0.5,
        'short_T': X_test[:,1] < 0.25
    }
    regions_cov = {}
    for name,mask in regions.items():
        if mask.sum()==0:
            regions_cov[name] = None
            continue
        mm = gp_means[mask]
        ss = gp_stds[mask]
        lower = mm - 1.96*ss
        upper = mm + 1.96*ss
        cov = np.mean((y_prices[mask]>=lower)&(y_prices[mask]<=upper))
        regions_cov[name] = float(cov)
    gp_metrics['regions_cov'] = regions_cov
    # pct above tau
    tau = getattr(router,'tau', None)
    if tau is not None:
        pct_above_tau = np.mean(gp_rel_unc >= tau)*100
    else:
        pct_above_tau = None
    gp_metrics['pct_above_tau'] = float(pct_above_tau) if pct_above_tau is not None else None
    gp_metrics['pass_95ci'] = status_symbol(coverages[0.95]['coverage']*100, thresholds, 'gp_95ci_coverage')
    gp_metrics['pass_ece'] = status_symbol(ece, thresholds, 'gp_ece')

results['gp'] = gp_metrics

# SECTION 4: UNCERTAINTY-ERROR ALIGNMENT
print('\n=== SECTION 4: UNCERTAINTY-ERROR ALIGNMENT ===')
align = {}
if os.path.exists(paths['failure_grid']) and gp_model is not None:
    f = np.load(paths['failure_grid'])
    X_fail = f['X']
    y_fail = f['y_true']
    nn_fail = f['rel_errors']
    X_fail_scaled = scaler.transform(X_fail) if scaler is not None else X_fail
    gp_unc_fail = []
    batch = 1000
    for i in range(0, len(X_fail_scaled), batch):
        b = torch.FloatTensor(X_fail_scaled[i:i+batch]).to(device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(gp_model(b))
            mean = pred.mean.cpu().numpy()
            std = pred.variance.sqrt().cpu().numpy()
        gp_unc_fail.append(std/(mean+1e-8))
        if i % 10000 == 0:
            print(f'Failure grid GP progress: {i}/{len(X_fail_scaled)}')
    gp_unc_fail = np.concatenate(gp_unc_fail)
    spearman_corr, spearman_pval = spearmanr(gp_unc_fail, nn_fail)
    pearson_corr, pearson_pval = pearsonr(gp_unc_fail, nn_fail)
    align['spearman'] = float(spearman_corr)
    align['pearson'] = float(pearson_corr)
    # decile analysis
    decile_edges = np.percentile(gp_unc_fail, np.arange(0,110,10))
    deciles = []
    for d in range(10):
        mask = (gp_unc_fail >= decile_edges[d]) & (gp_unc_fail < decile_edges[d+1])
        if mask.sum()==0:
            deciles.append({'mean_unc': None,'mean_err': None,'severe_pct': None})
            continue
        mean_unc = float(gp_unc_fail[mask].mean())
        mean_err = float(nn_fail[mask].mean()*100)
        severe_pct = float(np.mean(nn_fail[mask] > 0.05) * 100)
        deciles.append({'mean_unc': mean_unc,'mean_err_pct': mean_err,'severe_pct': severe_pct})
    align['deciles'] = deciles
    cached_spearman = None
    if 'step8_results.pkl' in cached and cached['step8_results.pkl'] is not None:
        cached_spearman = cached['step8_results.pkl'].get('alignment',{}).get('spearman_corr')
    align['cached_spearman'] = cached_spearman
    align['drift'] = abs(spearman_corr - cached_spearman) if cached_spearman is not None else None
else:
    print('Skipping alignment: missing failure grid or GP')

results['alignment'] = align

# SECTION 5: ROUTER LIVE VALIDATION
print('\n=== SECTION 5: ROUTER LIVE VALIDATION ===')
router_res = {}
if router is None or X_test is None:
    print('Skipping router section')
else:
    router._reset_stats()
    r_prices, r_deltas, r_gammas, r_uncs, r_routes = router.price_batch(X_test, batch_size=1000)
    r_routes_arr = np.array(r_routes)
    r_errors = np.abs(r_prices - y_prices)/(y_prices+1e-8)
    nn_mask = r_routes_arr == 'nn'
    exact_mask = r_routes_arr == 'exact'
    nn_fraction = float(nn_mask.mean()*100)
    exact_fraction = float(exact_mask.mean()*100)
    system_mape = float(r_errors.mean()*100)
    system_max = float(r_errors.max()*100)
    system_p99 = float(np.percentile(r_errors,99)*100)
    nn_errors_only = r_errors[nn_mask]
    nn_mape_routed = float(nn_errors_only.mean()*100) if nn_mask.any() else 0.0
    nn_max_routed = float(nn_errors_only.max()*100) if nn_mask.any() else 0.0
    exact_errors = r_errors[exact_mask]
    exact_max = float(exact_errors.max()*100) if exact_mask.any() else 0.0
    # theorem
    alpha = 0.05
    epsilon_alpha = np.quantile(nn_errors_only, 1-alpha) if nn_mask.any() else 0
    actual_exceedance = float(np.mean(r_errors > epsilon_alpha)*100)
    theorem_bound = alpha*100
    theorem_passes = actual_exceedance <= theorem_bound + 0.5
    unc_nn = float(np.mean(r_uncs[nn_mask]) if nn_mask.any() else np.nan)
    unc_exact = float(np.mean(r_uncs[exact_mask]) if exact_mask.any() else np.nan)
    separation_ok = unc_nn < unc_exact
    router_res.update({'nn_fraction_pct': nn_fraction,'exact_fraction_pct': exact_fraction,'system_mape_pct': system_mape,
                       'system_max_pct': system_max,'system_p99_pct': system_p99,'nn_mape_routed_pct': nn_mape_routed,
                       'exact_max_pct': exact_max,'epsilon_alpha_pct': epsilon_alpha*100,'actual_exceedance_pct': actual_exceedance,
                       'theorem_passes': bool(theorem_passes),'unc_nn': unc_nn,'unc_exact': unc_exact,'separation_ok': bool(separation_ok)})
    router_res['pass_router_nn_fraction'] = status_symbol(nn_fraction, thresholds, 'router_nn_fraction')
    router_res['pass_theorem'] = status_symbol(theorem_passes, thresholds, 'theorem_holds')

results['router'] = router_res

# SECTION 6: STRESS TEST SPOT CHECK
print('\n=== SECTION 6: STRESS TEST SPOT CHECK ===')
stress = {}
scenarios = ['gfc_2008','covid_2020','zirp','vol_spike','normal']
for scenario_name in scenarios:
    p = f'data/stress_scenarios/{scenario_name}.npz'
    if not os.path.exists(p):
        print('Missing scenario:', scenario_name)
        continue
    data = np.load(p)
    X_stress = data['X'][:2000]
    y_stress = data['price'][:2000]
    X_stress_scaled = scaler.transform(X_stress) if scaler is not None else X_stress
    X_stress_tensor = torch.FloatTensor(X_stress_scaled).to(device)
    with torch.no_grad():
        nn_stress_pred = nn_model(X_stress_tensor)[:,0].cpu().numpy()
    nn_stress_errors = np.abs(nn_stress_pred - y_stress)/(y_stress+1e-8)
    r_stress_prices, _, _, r_stress_uncs, r_stress_routes = router.price_batch(X_stress, batch_size=500)
    r_stress_routes_arr = np.array(r_stress_routes)
    r_stress_errors = np.abs(r_stress_prices - y_stress)/(y_stress+1e-8)
    nn_mape_stress = float(nn_stress_errors.mean()*100)
    router_mape_stress = float(r_stress_errors.mean()*100)
    nn_pct_stress = float(np.mean(r_stress_routes_arr == 'nn')*100)
    mean_unc_stress = float(np.mean(r_stress_uncs))
    protection_active = router_mape_stress < nn_mape_stress/2
    stress[scenario_name] = {'nn_mape_pct': nn_mape_stress,'router_mape_pct': router_mape_stress,'nn_routed_pct': nn_pct_stress,'mean_gp_unc': mean_unc_stress,'protection_active': protection_active}

results['stress'] = stress

# SECTION 7: PLOT DASHBOARD
print('\n=== SECTION 7: PLOTTING DASHBOARD ===')
os.makedirs('paper/figures', exist_ok=True)
fig = plt.figure(figsize=(20,16))
gs = fig.add_gridspec(3,3)
# Panel (1,1) Pred vs True
ax = fig.add_subplot(gs[0,0])
if 'nn' in results and X_test is not None:
    true = y_test[:,0]
    pred = nn_price_pred
    rel_err = price_errors
    sc = ax.scatter(true+1e-12, pred+1e-12, c=np.clip(rel_err*100/1.0,0,1), cmap='RdYlGn_r', s=6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.plot([true.min(),true.max()],[true.min(),true.max()],color='k',linewidth=0.8)
    ax.set_title('NN: Predicted vs True Price')
    ax.text(0.05,0.95,f"MAPE: {overall_mape:.3f}%",transform=ax.transAxes,verticalalignment='top')
# Panel (1,2) error by moneyness boxplot
ax = fig.add_subplot(gs[0,1])
try:
    groups = []
    labels = []
    for k in ['deep_otm','otm','atm','itm','deep_itm']:
        mask = (moneyness >= buckets[k][0]) & (moneyness < buckets[k][1])
        groups.append((price_errors[mask]*100))
        labels.append(k)
    sns.boxplot(data=groups, ax=ax)
    ax.set_xticklabels(labels)
    ax.set_yscale('log')
    ax.axhline(0.5, color='gray', linestyle='--')
    ax.set_title('NN Error by Moneyness')
except Exception:
    ax.text(0.5,0.5,'Insufficient data for boxplot',ha='center')
# Panel (1,3) GP calibration
ax = fig.add_subplot(gs[0,2])
if 'gp' in results and 'coverages' in gp_metrics:
    confs = []
    covs = []
    for conf,info in gp_metrics['coverages'].items():
        confs.append(conf)
        covs.append(info['coverage'])
    ax.plot([0,1],[0,1],'k--')
    ax.plot(confs, covs, marker='o')
    ax.set_xlabel('Stated confidence')
    ax.set_ylabel('Empirical coverage')
    ax.set_title('GP Calibration')
    ax.text(0.05,0.95,f"ECE: {gp_metrics.get('ece',np.nan):.4f}",transform=ax.transAxes,verticalalignment='top')
# Panel (2,1) Uncertainty vs Error
ax = fig.add_subplot(gs[1,0])
try:
    sample_idx = np.random.choice(len(gp_rel_unc), size=min(10000,len(gp_rel_unc)), replace=False)
    x = gp_rel_unc[sample_idx]
    y = price_errors[sample_idx]
    ax.scatter(x+1e-12,y+1e-12,c='gray',s=6)
    if lowess is not None:
        lw = lowess(y, x, frac=0.3)
        ax.plot(lw[:,0], lw[:,1], color='r')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title('Uncertainty vs Error Alignment')
    ax.text(0.05,0.95,f"Spearman: {align.get('spearman',np.nan):.3f}",transform=ax.transAxes,verticalalignment='top')
    if tau is not None:
        ax.axvline(tau, color='k', linestyle='--')
except Exception:
    ax.text(0.5,0.5,'Insufficient data for plot',ha='center')
# Panel (2,2) Router distribution
ax = fig.add_subplot(gs[1,1])
try:
    unc = gp_rel_unc
    mask_nn = unc < router.tau
    ax.hist(unc[mask_nn], bins=50, color='blue', alpha=0.6, label='NN')
    ax.hist(unc[~mask_nn], bins=50, color='red', alpha=0.6, label='Exact')
    ax.axvline(router.tau, color='k', linestyle='--')
    ax.legend()
    ax.set_title('Router: Uncertainty Distribution')
    ax.text(0.05,0.95,f"NN%: {router_res.get('nn_fraction_pct',np.nan):.1f}%",transform=ax.transAxes,verticalalignment='top')
except Exception:
    ax.text(0.5,0.5,'Insufficient data for plot',ha='center')
# Panel (2,3) System error comparison
ax = fig.add_subplot(gs[1,2])
try:
    bars = [results['nn']['overall_mape'], np.nan, results['router']['system_mape_pct']]
    labels = ['NN','GP','Router']
    colors = ['red','orange','green']
    ax.bar(labels, bars, color=colors)
    ax.axhline(0.5, color='gray', linestyle='--')
    ax.set_title('System MAPE Comparison')
except Exception:
    ax.text(0.5,0.5,'Insufficient data for plot',ha='center')
# Panel (3,1) Stress test comparison
ax = fig.add_subplot(gs[2,0])
try:
    groups = []
    labels = []
    nn_vals = []
    router_vals = []
    for name in scenarios:
        if name in stress:
            labels.append(name)
            nn_vals.append(stress[name]['nn_mape_pct'])
            router_vals.append(stress[name]['router_mape_pct'])
    x = np.arange(len(labels))
    ax.bar(x-0.2, nn_vals, width=0.4, color='red', label='NN')
    ax.bar(x+0.2, router_vals, width=0.4, color='green', label='Router')
    ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title('Stress Test: NN vs Router MAPE')
    ax.legend()
except Exception:
    ax.text(0.5,0.5,'Insufficient data for plot',ha='center')
# Panel (3,2) NN routing % by scenario
ax = fig.add_subplot(gs[2,1])
try:
    labels = []
    vals = []
    for name in scenarios:
        if name in stress:
            labels.append(name)
            vals.append(stress[name]['nn_routed_pct'])
    ax.bar(labels, vals, color=['green' if v>90 else 'gray' for v in vals])
    ax.axhline(90, color='gray', linestyle='--')
    ax.set_title('Queries to NN by Scenario')
except Exception:
    ax.text(0.5,0.5,'Insufficient data for plot',ha='center')
# Panel (3,3) Theorem 1 verification visual
ax = fig.add_subplot(gs[2,2])
try:
    alphas = [0.01,0.02,0.05,0.10,0.20]
    actuals = []
    bounds = []
    colors = []
    for a in alphas:
        eps = np.quantile(nn_errors_only, 1-a) if nn_mask.any() else 0
        actual = float(np.mean(r_errors > eps)*100)
        bound = a*100
        actuals.append(actual)
        bounds.append(bound)
        colors.append('green' if actual < bound else 'red')
    x = np.arange(len(alphas))
    ax.bar(x, actuals, color=colors)
    ax.plot(x, bounds, linestyle='--', color='red')
    ax.set_xticks(x); ax.set_xticklabels([str(a) for a in alphas])
    ax.set_title('Theorem 1: Bound vs Actual Exceedance')
except Exception:
    ax.text(0.5,0.5,'Insufficient data for plot',ha='center')

plt.tight_layout()
fig_path = 'paper/figures/model_health_check.png'
plt.savefig(fig_path, dpi=300)
print('Saved figure to', fig_path)

# SECTION 8: FINAL REPORT
print('\n=== SECTION 8: FINAL REPORT ===')
report_lines = []
report_lines.append('PHASE 1 MODEL HEALTH CHECK — COMPLETE REPORT')
report_lines.append('')
# NN section
report_lines.append('AREA 1: NEURAL NETWORK SURROGATE')
nn = results.get('nn',{})
report_lines.append(f"  overall_mape: {nn.get('overall_mape'):.4f}%  target: {thresholds['nn_overall_mape']['pass']}%  {nn.get('pass_overall_mape')}")
report_lines.append(f"  atm_mape: {nn.get('by_moneyness',{}).get('atm',{}).get('mape'):.4f}%  target: {thresholds['nn_atm_mape']['pass']}%  {nn.get('pass_atm_mape')}")
report_lines.append(f"  deep_otm_mape: {nn.get('by_moneyness',{}).get('deep_otm',{}).get('mape'):.4f}%  target: {thresholds['nn_deep_otm_mape']['pass']}%  {nn.get('pass_deep_otm_mape')}")
report_lines.append(f"  delta_mae: {nn.get('delta_mae'):.6f}  target: {thresholds['nn_delta_mae']['pass']}  {nn.get('pass_delta_mae')}")
report_lines.append(f"  latency_mean_ms: {nn.get('latency_mean_ms'):.4f}  target: {thresholds['nn_latency_ms']['pass']}  {nn.get('pass_latency')}")
report_lines.append('')
# GP section
report_lines.append('AREA 2: GAUSSIAN PROCESS CALIBRATION')
gp = results.get('gp',{})
if gp:
    cov95 = gp['coverages'][0.95]['coverage']*100
    report_lines.append(f"  95% CI coverage: {cov95:.2f}%  target: {thresholds['gp_95ci_coverage']['pass']}%  {gp.get('pass_95ci')}")
    report_lines.append(f"  ECE: {gp.get('ece'):.4f}  target: {thresholds['gp_ece']['pass']}  {gp.get('pass_ece')}")
    report_lines.append(f"  unc_mean: {gp.get('unc_mean'):.6f}")
else:
    report_lines.append('  GP metrics unavailable')
report_lines.append('')
# Alignment
report_lines.append('AREA 3: UNCERTAINTY-ERROR ALIGNMENT')
al = results.get('alignment',{})
if al and 'spearman' in al:
    report_lines.append(f"  Spearman: {al['spearman']:.4f}  target: {thresholds['spearman_corr']['pass']}  {status_symbol(al['spearman'], thresholds, 'spearman_corr')}")
else:
    report_lines.append('  Alignment metrics unavailable')
report_lines.append('')
# Router
report_lines.append('AREA 4: ROUTER PERFORMANCE')
rr = results.get('router',{})
if rr:
    report_lines.append(f"  NN fraction: {rr.get('nn_fraction_pct'):.2f}%  target: {thresholds['router_nn_fraction']['pass']}%  {rr.get('pass_router_nn_fraction')}")
    report_lines.append(f"  System MAPE: {rr.get('system_mape_pct'):.4f}%  target: {thresholds['system_mape']['pass']}%  {status_symbol(rr.get('system_mape_pct'), thresholds, 'system_mape')}")
    report_lines.append(f"  Theorem passes: {rr.get('theorem_passes')}  {rr.get('pass_theorem')}")
else:
    report_lines.append('  Router metrics unavailable')
report_lines.append('')
# Stress
report_lines.append('AREA 5: STRESS TEST BEHAVIOUR')
st = results.get('stress',{})
for name,v in st.items():
    report_lines.append(f"  {name}: NN MAPE={v['nn_mape_pct']:.2f}%, Router MAPE={v['router_mape_pct']:.2f}%, NN routed%={v['nn_routed_pct']:.1f}%")
report_lines.append('')
# Verdict counts
n_pass = n_warn = n_fail = 0
# Collect statuses from various checks
status_checks = []
# NN statuses
for key in ['pass_overall_mape','pass_atm_mape','pass_deep_otm_mape','pass_delta_mae','pass_latency','pass_max_error']:
    val = nn.get(key)
    if val is None: continue
    sym = val[0]
    if 'PASS' in sym: n_pass += 1
    elif 'WARN' in sym: n_warn += 1
    else: n_fail += 1
# GP statuses
for key in ['pass_95ci','pass_ece']:
    val = gp.get(key)
    if val is None: continue
    sym = val[0]
    if 'PASS' in sym: n_pass += 1
    elif 'WARN' in sym: n_warn += 1
    else: n_fail += 1
# Alignment
if 'spearman' in al:
    sym = status_symbol(al['spearman'], thresholds, 'spearman_corr')[0]
    if 'PASS' in sym: n_pass+=1
    elif 'WARN' in sym: n_warn+=1
    else: n_fail+=1
# Router
if rr:
    for key in ['pass_router_nn_fraction','pass_theorem']:
        val = rr.get(key)
        if val is None: continue
        sym = val[0]
        if 'PASS' in sym: n_pass+=1
        elif 'WARN' in sym: n_warn+=1
        else: n_fail+=1

results['overall'] = {'n_pass': n_pass,'n_warn': n_warn,'n_fail': n_fail}
# Final verdict
if n_fail == 0 and n_warn <= 2:
    verdict = 'READY FOR PHASE 2'
    verdict_detail = 'All critical metrics pass. Minor warnings noted above.'
elif n_fail == 0:
    verdict = 'CONDITIONALLY READY — REVIEW WARNINGS'
    verdict_detail = f"{n_warn} warnings require review before Phase 2."
elif n_fail <= 2:
    verdict = 'NOT READY — FIX FAILURES BEFORE PROCEEDING'
    verdict_detail = f"{n_fail} critical failures must be addressed."
else:
    verdict = 'SYSTEM FAILURE — SIGNIFICANT ISSUES FOUND'
    verdict_detail = f"{n_fail} failures indicate fundamental problems."

report_lines.append('FINAL VERDICT')
report_lines.append(f"  Total metrics checked: {n_pass+n_warn+n_fail}")
report_lines.append(f"  PASS: {n_pass}")
report_lines.append(f"  WARN: {n_warn}")
report_lines.append(f"  FAIL: {n_fail}")
report_lines.append(f"  VERDICT: {verdict}")
report_lines.append(f"  {verdict_detail}")

report_text = '\n'.join(report_lines)
with open('paper/health_check_report.txt','w') as f:
    f.write(report_text)
print('Saved report to paper/health_check_report.txt')

# Save machine-readable results
with open('data/processed/health_check_results.pkl','wb') as f:
    pickle.dump(results, f)
print('Saved results to data/processed/health_check_results.pkl')

print('\nHealth check complete.')
