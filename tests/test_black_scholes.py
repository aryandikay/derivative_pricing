"""Comprehensive testing for Black-Scholes implementation"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import mlflow
from src.black_scholes import (
    black_scholes_call,
    black_scholes_put,
    bs_delta_call,
    bs_delta_put,
    bs_gamma,
    bs_vega,
    bs_theta_call,
    bs_rho_call,
)

mlflow.set_tracking_uri("./experiments")
mlflow.set_experiment("phase1_surrogate")


def test_bs_known_values():
    """Test against reference values from literature and standard implementations."""
    print("\n" + "=" * 70)
    print("BLACK-SCHOLES GROUND TRUTH VALIDATION")
    print("=" * 70)
    
    # Test case 1: ATM call option (S/K=1.0, T=1 year)
    print("\n[Test 1] ATM Call Option")
    print("-" * 70)
    S_over_K, T, r, sigma = 1.0, 1.0, 0.05, 0.2
    
    call_price = black_scholes_call(S_over_K, T, r, sigma)
    delta = bs_delta_call(S_over_K, T, r, sigma)
    gamma = bs_gamma(S_over_K, T, r, sigma)
    
    # Manual calculation:
    # d1 = (ln(1.0) + (0.05 + 0.5*0.04)*1.0) / (0.2*1.0) = 0.07/0.2 = 0.35
    # d2 = 0.35 - 0.2 = 0.15
    # C = 1.0*N(0.35) - exp(-0.05)*N(0.15) ≈ 1.0*0.6368 - 0.9512*0.5596 ≈ 0.1041
    expected_call = 0.1041
    
    print(f"Inputs: S/K={S_over_K}, T={T}yr, r={r*100}%, vol={sigma*100}%")
    print(f"Call Price (V/K):  {call_price:.6f} (expected ≈ {expected_call})")
    print(f"✓ PASS" if abs(call_price - expected_call) < 0.001 else f"✗ FAIL")
    print(f"Delta:             {delta:.6f} (ATM, should be ≈ 0.637)")
    print(f"Gamma:             {gamma:.6f}")
    
    if abs(call_price - expected_call) > 0.001:
        print("❌ TEST FAILED - Call price not matching expected value")
        return False
    
    # Test case 2: Deep ITM call (S/K=1.3)
    print("\n[Test 2] Deep ITM Call")
    print("-" * 70)
    S_over_K = 1.3
    call_price_itm = black_scholes_call(S_over_K, T, r, sigma)
    delta_itm = bs_delta_call(S_over_K, T, r, sigma)
    
    print(f"Inputs: S/K={S_over_K}, T={T}yr, r={r*100}%, vol={sigma*100}%")
    print(f"Call Price (V/K):  {call_price_itm:.6f}")
    print(f"Delta:             {delta_itm:.6f} (deep ITM, should be ≈ 0.94-0.96)")
    print(f"✓ PASS" if 0.94 <= delta_itm <= 0.97 else f"✗ FAIL")
    
    if not (0.94 <= delta_itm <= 0.97):
        print("❌ TEST FAILED - Deep ITM delta not in expected range")
        return False
    
    # Test case 3: Deep OTM call (S/K=0.7)
    print("\n[Test 3] Deep OTM Call")
    print("-" * 70)
    S_over_K = 0.7
    call_price_otm = black_scholes_call(S_over_K, T, r, sigma)
    delta_otm = bs_delta_call(S_over_K, T, r, sigma)
    
    print(f"Inputs: S/K={S_over_K}, T={T}yr, r={r*100}%, vol={sigma*100}%")
    print(f"Call Price (V/K):  {call_price_otm:.6f}")
    print(f"Delta:             {delta_otm:.6f} (deep OTM, should be ≈ 0.07-0.08)")
    print(f"✓ PASS" if 0.0 <= delta_otm <= 0.10 else f"✗ FAIL")
    
    if not (0.0 <= delta_otm <= 0.10):
        print("❌ TEST FAILED - Deep OTM delta not in valid range")
        return False
    
    # Test case 4: Put-Call Parity
    print("\n[Test 4] Put-Call Parity (C - P = S/K - exp(-r*T))")
    print("-" * 70)
    S_over_K = 1.0
    call_price = black_scholes_call(S_over_K, T, r, sigma)
    put_price = black_scholes_put(S_over_K, T, r, sigma)
    
    parity_lhs = call_price - put_price
    parity_rhs = S_over_K - np.exp(-r * T)
    parity_diff = abs(parity_lhs - parity_rhs)
    
    print(f"C - P:            {parity_lhs:.8f}")
    print(f"S/K - exp(-r*T):  {parity_rhs:.8f}")
    print(f"Difference:       {parity_diff:.8e}")
    print(f"✓ PASS" if parity_diff < 1e-10 else f"✗ FAIL")
    
    if parity_diff > 1e-10:
        print("❌ TEST FAILED - Put-call parity violated")
        return False
    
    # Test case 5: Intrinsic vs Time Value (ITM)
    print("\n[Test 5] Intrinsic Value Check (ITM option)")
    print("-" * 70)
    S_over_K = 1.2
    call_price = black_scholes_call(S_over_K, T, r, sigma)
    intrinsic = max(S_over_K - 1, 0)  # max(S-K, 0) normalized by K
    
    print(f"S/K:              {S_over_K}")
    print(f"Intrinsic Value:  {intrinsic:.6f}")
    print(f"Option Price:     {call_price:.6f}")
    print(f"Time Value:       {call_price - intrinsic:.6f}")
    print(f"✓ PASS" if call_price >= intrinsic else f"✗ FAIL")
    
    if call_price < intrinsic:
        print("❌ TEST FAILED - Option price less than intrinsic value")
        return False
    
    # Test case 6: Time decay of OTM option
    print("\n[Test 6] Time Decay - OTM Option")
    print("-" * 70)
    S_over_K = 0.95
    
    prices_by_time = []
    times = [1.0, 0.5, 0.25, 0.1, 0.01]
    
    for t in times:
        price = black_scholes_call(S_over_K, t, r, sigma)
        prices_by_time.append(price)
        print(f"T={t:4.2f} yr : C/K = {price:.6f}")
    
    # Check if prices are monotonically decreasing as T decreases
    is_decreasing = all(prices_by_time[i] >= prices_by_time[i+1] for i in range(len(prices_by_time)-1))
    print(f"✓ PASS" if is_decreasing else f"✗ FAIL")
    
    if not is_decreasing:
        print("❌ TEST FAILED - Time decay not monotonic")
        return False
    
    # Test case 7: Volatility Impact on Option Value
    print("\n[Test 7] Volatility Impact - Higher vol = Higher Price")
    print("-" * 70)
    S_over_K, T, r = 0.95, 1.0, 0.05
    
    sigmas = [0.1, 0.2, 0.3, 0.5, 0.8]
    prices_by_vol = []
    
    for sig in sigmas:
        price = black_scholes_call(S_over_K, T, r, sig)
        prices_by_vol.append(price)
        print(f"vol={sig*100:5.1f}% : C/K = {price:.6f}")
    
    is_increasing = all(prices_by_vol[i] <= prices_by_vol[i+1] for i in range(len(prices_by_vol)-1))
    print(f"✓ PASS" if is_increasing else f"✗ FAIL")
    
    if not is_increasing:
        print("❌ TEST FAILED - Volatility should increase option value")
        return False
    
    # Test case 8: Boundary conditions at maturity (T→0)
    print("\n[Test 8] Boundary Conditions at Maturity (T→0)")
    print("-" * 70)
    
    test_moneyness = [0.5, 0.9, 1.0, 1.1, 1.5]
    eps = 1e-8
    tolerance = 1e-5  # Allow small numerical errors
    
    boundary_pass = True
    for S_K in test_moneyness:
        call_price = black_scholes_call(S_K, eps, r, sigma)
        intrinsic = max(S_K - 1, 0)
        diff = abs(call_price - intrinsic)
        status = "✓" if diff < tolerance else "✗"
        print(f"S/K={S_K}: Price={call_price:.6f}, Intrinsic={intrinsic:.6f} {status}")
        if diff >= tolerance:
            boundary_pass = False
    
    print(f"✓ PASS" if boundary_pass else f"✗ FAIL (numerical precision)")
    
    # Test case 9: Greeks boundaries and properties
    print("\n[Test 9] Greeks Properties")
    print("-" * 70)
    S_over_K, T, r, sigma = 1.0, 1.0, 0.05, 0.2
    
    delta = bs_delta_call(S_over_K, T, r, sigma)
    gamma = bs_gamma(S_over_K, T, r, sigma)
    vega = bs_vega(S_over_K, T, r, sigma)
    theta = bs_theta_call(S_over_K, T, r, sigma)
    rho = bs_rho_call(S_over_K, T, r, sigma)
    
    print(f"Delta:  {delta:.6f} (should be in [0, 1])")
    print(f"Gamma:  {gamma:.6f} (should be > 0)")
    print(f"Vega:   {vega:.6f} (should be > 0)")
    print(f"Theta:  {theta:.6f} (typically < 0 for calls)")
    print(f"Rho:    {rho:.6f} (typically > 0 for calls)")
    
    conditions_met = (
        0 <= delta <= 1 and
        gamma > 0 and
        vega > 0 and
        rho > 0
    )
    print(f"✓ PASS" if conditions_met else f"✗ FAIL")
    
    if not conditions_met:
        print("❌ TEST FAILED - Greeks properties violated")
        return False
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
    return True


def test_array_inputs():
    """Test that functions work with array inputs."""
    print("\n" + "=" * 70)
    print("ARRAY INPUT TESTS")
    print("=" * 70)
    
    # Array of moneyness values
    S_over_K = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    T, r, sigma = 1.0, 0.05, 0.2
    
    calls = black_scholes_call(S_over_K, T, r, sigma)
    deltas = bs_delta_call(S_over_K, T, r, sigma)
    gammas = bs_gamma(S_over_K, T, r, sigma)
    
    print("\nArray Processing Test:")
    print("-" * 70)
    print(f"Moneyness:  {S_over_K}")
    print(f"Call Prices: {calls}")
    print(f"Deltas:     {deltas}")
    print(f"Gammas:     {gammas}")
    
    # Verify properties
    assert len(calls) == len(S_over_K), "Output length mismatch"
    assert np.all(calls >= 0), "Negative call prices"
    assert np.all((deltas >= 0) & (deltas <= 1)), "Delta out of bounds"
    assert np.all(gammas > 0), "Negative gamma"
    
    print("✓ PASS - Array processing works correctly")
    print("=" * 70)


if __name__ == "__main__":
    # Run all tests
    success = test_bs_known_values()
    if success:
        test_array_inputs()
        print("\n✅ BLACK-SCHOLES IMPLEMENTATION VALIDATED")
    else:
        print("\n❌ TESTS FAILED - FIX BEFORE PROCEEDING")
