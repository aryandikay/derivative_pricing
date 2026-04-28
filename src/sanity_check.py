"""Sanity check visualizations for generated dataset."""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import black_scholes_call, bs_delta_call


def visualize_sanity_checks():
    """Create sanity check plots for dataset."""
    
    # Load the generated dataset
    dataset_path = Path("data/raw/dataset_100k.npz")
    data = np.load(dataset_path)
    
    # Extract features and targets
    moneyness = data['features_moneyness']
    T = data['features_T']
    r = data['features_r']
    sigma = data['features_sigma']
    call_prices = data['targets_call_price']
    deltas = data['targets_delta']
    
    print("\n" + "="*70)
    print("DATASET SANITY CHECK VISUALIZATIONS")
    print("="*70)
    
    # ====================================================================
    # SANITY CHECK 1: Price vs Moneyness for Fixed T and Sigma
    # ====================================================================
    print("\n[1] Price vs Moneyness (S-Curve Check)")
    print("-" * 70)
    
    # Find samples with T close to 1.0 year and sigma = 0.2
    target_T = 1.0
    target_sigma = 0.2
    
    # Find closest match
    T_tolerance = 0.05
    sigma_tolerance = 0.01
    
    mask = (np.abs(T - target_T) < T_tolerance) & (np.abs(sigma - target_sigma) < sigma_tolerance)
    
    if mask.sum() > 0:
        idx = mask
        subset_moneyness = moneyness[idx]
        subset_prices = call_prices[idx]
        
        # Sort by moneyness for visualization
        sort_idx = np.argsort(subset_moneyness)
        M = subset_moneyness[sort_idx]
        P = subset_prices[sort_idx]
        
        print(f"\n  Found {mask.sum()} samples near T≈{target_T}, σ≈{target_sigma}")
        print(f"\n  Moneyness Range (sorted):")
        print(f"    Min: {M.min():.4f}")
        print(f"    Max: {M.max():.4f}")
        print(f"    Mean: {M.mean():.4f}")
        
        print(f"\n  Call Price Range:")
        print(f"    Min: {P.min():.6f}")
        print(f"    Max: {P.max():.6f}")
        print(f"    Mean: {P.mean():.6f}")
        
        # Check S-curve shape: prices should increase monotonically with moneyness
        price_diffs = np.diff(P)
        is_monotonic = np.all(price_diffs >= -1e-10)  # Allow tiny numerical errors
        
        print(f"\n  S-Curve Shape Check:")
        print(f"    Monotonic increase: {'✓ PASS' if is_monotonic else '✗ FAIL'}")
        print(f"    Negative diffs: {(price_diffs < 0).sum()}")
        
        # Visual representation
        print(f"\n  ASCII Price Curve (T≈1.0yr, σ≈20%):")
        print(f"  {'Moneyness':>10} | {'Price':>10} | Chart")
        print(f"  {'-'*10}-+-{'-'*10}-+{'-'*30}")
        
        # Sample every 10th point for display
        step = max(1, len(M) // 20)
        for i in range(0, len(M), step):
            bar_len = int((P[i] / P.max()) * 30)
            print(f"  {M[i]:10.4f} | {P[i]:10.6f} | {'█' * bar_len}")
    
    # ====================================================================
    # SANITY CHECK 2: Delta vs Moneyness
    # ====================================================================
    print("\n[2] Delta vs Moneyness (0 to 1 Check)")
    print("-" * 70)
    
    mask = (np.abs(T - target_T) < T_tolerance) & (np.abs(sigma - target_sigma) < sigma_tolerance)
    
    if mask.sum() > 0:
        idx = mask
        subset_moneyness = moneyness[idx]
        subset_deltas = deltas[idx]
        
        # Sort by moneyness
        sort_idx = np.argsort(subset_moneyness)
        M = subset_moneyness[sort_idx]
        D = subset_deltas[sort_idx]
        
        print(f"\n  Delta Range:")
        print(f"    Min: {D.min():.6f}")
        print(f"    Max: {D.max():.6f}")
        print(f"    Mean: {D.mean():.6f}")
        
        # Check delta is in [0, 1]
        valid_range = np.all((D >= 0) & (D <= 1))
        
        # Check monotonic increase
        delta_diffs = np.diff(D)
        is_monotonic = np.all(delta_diffs >= -1e-10)
        
        print(f"\n  Delta Bounds Check:")
        print(f"    All in [0,1]: {'✓ PASS' if valid_range else '✗ FAIL'}")
        print(f"    Monotonic: {'✓ PASS' if is_monotonic else '✗ FAIL'}")
        
        # Visual
        print(f"\n  ASCII Delta Curve (T≈1.0yr, σ≈20%):")
        print(f"  {'Moneyness':>10} | {'Delta':>10} | Chart (scaled to [0,1])")
        print(f"  {'-'*10}-+-{'-'*10}-+{'-'*30}")
        
        step = max(1, len(M) // 20)
        for i in range(0, len(M), step):
            bar_len = int(D[i] * 30)
            print(f"  {M[i]:10.4f} | {D[i]:10.6f} | {'█' * bar_len}")
    
    # ====================================================================
    # SANITY CHECK 3: Overall Dataset Statistics
    # ====================================================================
    print("\n[3] Overall Dataset Statistics")
    print("-" * 70)
    
    print(f"\n  Features (Input Variables):")
    print(f"    Moneyness - Mean: {moneyness.mean():.4f}, Std: {moneyness.std():.4f}")
    print(f"    T (years) - Mean: {T.mean():.4f}, Std: {T.std():.4f}")
    print(f"    r (%) - Mean: {r.mean()*100:.4f}, Std: {r.std()*100:.4f}")
    print(f"    σ (%) - Mean: {sigma.mean()*100:.4f}, Std: {sigma.std()*100:.4f}")
    
    print(f"\n  Targets (Output Variables):")
    print(f"    Call price - Mean: {call_prices.mean():.6f}, Std: {call_prices.std():.6f}")
    print(f"    Delta - Mean: {deltas.mean():.6f}, Std: {deltas.std():.6f}")
    
    gamma = data['targets_gamma']
    vega = data['targets_vega']
    theta = data['targets_theta']
    
    print(f"    Gamma - Mean: {gamma.mean():.6f}, Std: {gamma.std():.6f}")
    print(f"    Vega - Mean: {vega.mean():.6f}, Std: {vega.std():.6f}")
    print(f"    Theta - Mean: {theta.mean():.6f}, Std: {theta.std():.6f}")
    
    # ====================================================================
    # SANITY CHECK 4: Moneyness-related insights
    # ====================================================================
    print("\n[4] Moneyness Distribution (Market-Realistic)")
    print("-" * 70)
    
    bins = np.linspace(0.7, 1.3, 13)
    counts, _ = np.histogram(moneyness, bins=bins)
    
    print(f"\n  Moneyness Histogram:")
    print(f"  {' Range      | Count  | Density (%) | Chart'}")
    print(f"  {'-'*50}")
    
    for i in range(len(counts)):
        range_start = bins[i]
        range_end = bins[i+1]
        count = counts[i]
        density = 100 * count / len(moneyness)
        bar_len = int(density / 2)
        print(f"  [{range_start:.2f}-{range_end:.2f}] | {count:6d} | {density:10.2f} | {'█' * bar_len}")
    
    print("\n" + "="*70)
    print("✓ ALL SANITY CHECKS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    visualize_sanity_checks()
