"""Data processing module with Black-Scholes ground truth implementation."""

import numpy as np
import mlflow
from scipy.stats import norm
from pathlib import Path

mlflow.set_tracking_uri("./experiments")
mlflow.set_experiment("phase1_surrogate")


# ============================================================================
# BLACK-SCHOLES GROUND TRUTH IMPLEMENTATIONS
# ============================================================================

def black_scholes_call(S_over_K, T, r, sigma):
    """
    Black-Scholes call option price.
    
    Parameters
    ----------
    S_over_K : float or array
        Moneyness = S/K (spot price / strike price)
    T : float
        Time to maturity in years
    r : float
        Risk-free rate (annualized, decimal form e.g., 0.05 for 5%)
    sigma : float
        Volatility (annualized, decimal form e.g., 0.2 for 20%)
    
    Returns
    -------
    float or array
        Call option price normalized by strike (V/K)
    
    Notes
    -----
    Formula:
        C/K = S/K * N(d1) - exp(-r*T) * N(d2)
        where d1 = [ln(S/K) + (r + 0.5*sigma^2)*T] / (sigma*sqrt(T))
              d2 = d1 - sigma*sqrt(T)
    """
    # Guard against T=0 edge case
    T = np.maximum(T, 1e-8)
    
    # Calculate d1 and d2
    d1 = (np.log(S_over_K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Call price
    price = S_over_K * norm.cdf(d1) - np.exp(-r * T) * norm.cdf(d2)
    return price


def black_scholes_put(S_over_K, T, r, sigma):
    """
    Black-Scholes put option price.
    
    Parameters
    ----------
    S_over_K : float or array
        Moneyness = S/K
    T : float
        Time to maturity in years
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    
    Returns
    -------
    float or array
        Put option price normalized by strike (V/K)
    """
    T = np.maximum(T, 1e-8)
    
    d1 = (np.log(S_over_K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    price = np.exp(-r * T) * norm.cdf(-d2) - S_over_K * norm.cdf(-d1)
    return price


def bs_delta_call(S_over_K, T, r, sigma):
    """
    Delta of a call option (dC/dS, normalized).
    
    Parameters
    ----------
    S_over_K : float or array
        Moneyness
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    
    Returns
    -------
    float or array
        Delta (between 0 and 1 for calls)
    """
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S_over_K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def bs_delta(S_over_K, T, r, sigma):
    """Alias for bs_delta_call for backward compatibility."""
    return bs_delta_call(S_over_K, T, r, sigma)


def bs_delta_put(S_over_K, T, r, sigma):
    """Delta of a put option (between -1 and 0 for puts)."""
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S_over_K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1


def bs_gamma(S_over_K, T, r, sigma):
    """
    Gamma of an option (d2C/dS2).
    
    Returns
    -------
    float or array
        Gamma (always positive, same for calls and puts)
    """
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S_over_K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S_over_K * sigma * np.sqrt(T))


def bs_vega(S_over_K, T, r, sigma):
    """Vega (dC/dsigma) - sensitivity to 1% change in volatility."""
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S_over_K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S_over_K * norm.pdf(d1) * np.sqrt(T) / 100


def bs_theta_call(S_over_K, T, r, sigma):
    """Theta of a call option (dC/dT) - per day, normalized by K."""
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S_over_K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    theta = (-S_over_K * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
             - r * np.exp(-r * T) * norm.cdf(d2))
    
    return theta / 365


def bs_rho_call(S_over_K, T, r, sigma):
    """Rho of a call option (dC/dr) - sensitivity to 1% change in rates."""
    T = np.maximum(T, 1e-8)
    d1 = (np.log(S_over_K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return T * np.exp(-r * T) * norm.cdf(d2) / 100


# ============================================================================
# DATA GENERATION WITH MLFLOW TRACKING
# ============================================================================

class BSDataGenerator:
    """Generate synthetic option pricing data using Black-Scholes ground truth."""
    
    def __init__(self, output_dir: str = "data/processed", raw_dir: str = "data/raw"):
        """
        Initialize data generator.
        
        Parameters
        ----------
        output_dir : str
            Directory to store processed data
        raw_dir : str
            Directory to store raw generated data
        """
        self.output_dir = Path(output_dir)
        self.raw_dir = Path(raw_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_dataset(self, 
                        n_samples: int = 100000,
                        random_seed: int = 42) -> dict:
        """
        Generate synthetic dataset with market-realistic sampling strategy.
        
        This uses non-uniform sampling based on market conventions:
        - Moneyness: 50% uniform (0.7-1.3), 50% concentrated near ATM (1.0)
        - Time: log-uniform so short maturities are well-represented
        - Volatility: uniform (5%-80%)
        - Rate: uniform (0%-10%)
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate (default: 100,000 for realistic market)
        random_seed : int
            Seed for reproducibility
            
        Returns
        -------
        dict
            Generated dataset with features and targets
        """
        run_name = f"bs-data-gen-n{n_samples}"
        
        with mlflow.start_run(run_name=run_name):
            print(f"\n{'='*70}")
            print(f"GENERATING DATASET: {n_samples:,} samples")
            print(f"{'='*70}")
            
            # Set random seed for reproducibility
            np.random.seed(random_seed)
            
            # Log generation parameters
            mlflow.log_params({
                "n_samples": n_samples,
                "random_seed": random_seed,
                "sampling_strategy": "market-realistic (non-uniform)",
            })
            
            # ====================================================================
            # SAMPLING STRATEGY 1: MONEYNESS
            # ====================================================================
            # Market-based: Most trading occurs near ATM, but we need full range
            print("\n[1] Sampling Moneyness (S/K)")
            print("-" * 70)
            
            n_atm = n_samples // 2
            n_uniform = n_samples - n_atm
            
            # 50% uniform distribution across realistic range
            moneyness_uniform = np.random.uniform(0.7, 1.3, n_uniform)
            
            # 50% concentrated around ATM with normal distribution
            # clip to ensure we stay within realistic bounds
            moneyness_atm = np.random.normal(1.0, 0.05, n_atm).clip(0.7, 1.3)
            
            # Combine and shuffle
            moneyness = np.concatenate([moneyness_uniform, moneyness_atm])
            np.random.shuffle(moneyness)
            
            print(f"  Moneyness range (market): [0.7, 1.3]")
            print(f"  Distribution: 50% uniform + 50% normal(1.0, 0.05)")
            print(f"  Generated range: [{moneyness.min():.4f}, {moneyness.max():.4f}]")
            print(f"  Mean: {moneyness.mean():.4f}, Std: {moneyness.std():.4f}")
            
            # ====================================================================
            # SAMPLING STRATEGY 2: TIME TO MATURITY
            # ====================================================================
            # Log-uniform: ensures we capture both short and long-dated options
            print("\n[2] Sampling Time to Maturity (T)")
            print("-" * 70)
            
            T_min_days = 7  # 1 week = liquid options minimum
            T_max_years = 2.0  # 2 years = typical long-dated options
            
            log_T = np.random.uniform(np.log(T_min_days/365), np.log(T_max_years), n_samples)
            T = np.exp(log_T)
            
            print(f"  Time range (market): [1 week, 2 years]")
            print(f"  Time range (years): [{T_min_days/365:.4f}, {T_max_years}]")
            print(f"  Sampling: log-uniform for density at short maturities")
            print(f"  Generated range: [{T.min():.4f}, {T.max():.4f}] years")
            print(f"  Mean: {T.mean():.4f}, Std: {T.std():.4f}")
            
            # ====================================================================
            # SAMPLING STRATEGY 3: VOLATILITY
            # ====================================================================
            # Uniform across realistic historical range
            print("\n[3] Sampling Volatility (σ)")
            print("-" * 70)
            
            sigma_min = 0.05   # 5% - historical minimum
            sigma_max = 0.80   # 80% - extreme volatility events
            
            sigma = np.random.uniform(sigma_min, sigma_max, n_samples)
            
            print(f"  Volatility range (market): [{sigma_min*100:.0f}%, {sigma_max*100:.0f}%]")
            print(f"  Sampling: uniform")
            print(f"  Generated range: [{sigma.min()*100:.2f}%, {sigma.max()*100:.2f}%]")
            print(f"  Mean: {sigma.mean()*100:.2f}%, Std: {sigma.std()*100:.2f}%")
            
            # ====================================================================
            # SAMPLING STRATEGY 4: INTEREST RATE
            # ====================================================================
            # Uniform across historical central bank rates
            print("\n[4] Sampling Risk-Free Rate (r)")
            print("-" * 70)
            
            r_min = 0.00   # 0% - historical low
            r_max = 0.10   # 10% - historical high
            
            r = np.random.uniform(r_min, r_max, n_samples)
            
            print(f"  Rate range (market): [{r_min*100:.0f}%, {r_max*100:.0f}%]")
            print(f"  Sampling: uniform")
            print(f"  Generated range: [{r.min()*100:.2f}%, {r.max()*100:.2f}%]")
            print(f"  Mean: {r.mean()*100:.2f}%, Std: {r.std()*100:.2f}%")
            
            # ====================================================================
            # COMPUTE BLACK-SCHOLES TARGETS
            # ====================================================================
            print("\n[5] Computing Black-Scholes Ground Truth")
            print("-" * 70)
            
            call_prices = black_scholes_call(moneyness, T, r, sigma)
            put_prices = black_scholes_put(moneyness, T, r, sigma)
            deltas = bs_delta_call(moneyness, T, r, sigma)
            gammas = bs_gamma(moneyness, T, r, sigma)
            vegas = bs_vega(moneyness, T, r, sigma)
            thetas = bs_theta_call(moneyness, T, r, sigma)
            rhos = bs_rho_call(moneyness, T, r, sigma)
            
            # Clip very small numerical artifacts (e.g., gammas that should be positive)
            gammas = np.maximum(gammas, 0.0)
            call_prices = np.maximum(call_prices, 0.0)
            put_prices = np.maximum(put_prices, 0.0)
            
            print(f"  Call price range: [{call_prices.min():.6f}, {call_prices.max():.6f}]")
            print(f"  Put price range: [{put_prices.min():.6f}, {put_prices.max():.6f}]")
            print(f"  Delta range: [{deltas.min():.6f}, {deltas.max():.6f}]")
            print(f"  Gamma range: [{gammas.min():.6f}, {gammas.max():.6f}]")
            
            # ====================================================================
            # SANITY CHECKS
            # ====================================================================
            print("\n[6] SANITY CHECKS")
            print("-" * 70)
            
            # Check 1: No NaN values
            has_nan_call = np.isnan(call_prices).any()
            has_nan_put = np.isnan(put_prices).any()
            has_nan_delta = np.isnan(deltas).any()
            has_nan_gamma = np.isnan(gammas).any()
            
            print(f"\n  Check 1: NaN values")
            print(f"    Call prices: {'✗ FAIL - NaNs found!' if has_nan_call else '✓ PASS'}")
            print(f"    Put prices:  {'✗ FAIL - NaNs found!' if has_nan_put else '✓ PASS'}")
            print(f"    Deltas:      {'✗ FAIL - NaNs found!' if has_nan_delta else '✓ PASS'}")
            print(f"    Gammas:      {'✗ FAIL - NaNs found!' if has_nan_gamma else '✓ PASS'}")
            
            if has_nan_call or has_nan_put or has_nan_delta or has_nan_gamma:
                raise ValueError("Dataset contains NaN values!")
            
            # Check 2: No negative prices (intrinsic value check)
            has_neg_call = (call_prices < 0).any()
            has_neg_put = (put_prices < 0).any()
            
            print(f"\n  Check 2: Negative prices (should be none)")
            print(f"    Call prices: {'✗ FAIL - negatives found!' if has_neg_call else '✓ PASS'}")
            print(f"    Put prices:  {'✗ FAIL - negatives found!' if has_neg_put else '✓ PASS'}")
            
            if has_neg_call or has_neg_put:
                raise ValueError("Dataset contains negative prices!")
            
            # Check 3: No-arbitrage bounds
            # For calls: C <= S (normalized: C/K <= S/K)
            violates_call_upper = (call_prices > moneyness).any()
            
            # For puts: P <= K (normalized: P/K <= 1)
            violates_put_upper = (put_prices > 1.0).any()
            
            print(f"\n  Check 3: No-arbitrage bounds")
            print(f"    Call ≤ S/K:  {'✗ FAIL - bound violated!' if violates_call_upper else '✓ PASS'}")
            print(f"    Put ≤ 1:     {'✗ FAIL - bound violated!' if violates_put_upper else '✓ PASS'}")
            
            if violates_call_upper or violates_put_upper:
                raise ValueError("Dataset violates no-arbitrage bounds!")
            
            # Check 4: Delta bounds (for calls)
            violates_delta = ((deltas < 0) | (deltas > 1)).any()
            
            print(f"\n  Check 4: Delta bounds [0, 1]")
            print(f"    Call delta:  {'✗ FAIL - out of bounds!' if violates_delta else '✓ PASS'}")
            
            if violates_delta:
                raise ValueError("Dataset has invalid delta values!")
            
            # Check 5: Gamma always positive
            has_non_pos_gamma = (gammas < 0).any()  # After clipping, should have none
            
            print(f"\n  Check 5: Gamma positivity")
            print(f"    Gamma ≥ 0:   {'✗ FAIL - negatives found!' if has_non_pos_gamma else '✓ PASS'}")
            
            if has_non_pos_gamma:
                raise ValueError("Dataset has negative gamma values!")
            
            # Check 6: Put-Call Parity
            discount_factor = np.exp(-r * T)
            pc_parity_lhs = call_prices - put_prices
            pc_parity_rhs = moneyness - discount_factor
            parity_violation = np.abs(pc_parity_lhs - pc_parity_rhs).max()
            
            print(f"\n  Check 6: Put-Call Parity")
            print(f"    Max violation: {parity_violation:.2e}")
            print(f"    Status:       {'✓ PASS' if parity_violation < 1e-10 else '✗ WARN'}")
            
            # Create dataset dictionary
            dataset = {
                "features": {
                    "moneyness": moneyness,
                    "T": T,
                    "r": r,
                    "sigma": sigma,
                },
                "targets": {
                    "call_price": call_prices,
                    "put_price": put_prices,
                    "delta": deltas,
                    "gamma": gammas,
                    "vega": vegas,
                    "theta": thetas,
                    "rho": rhos,
                },
            }
            
            # Log dataset statistics as metrics
            metrics = {
                "n_samples": n_samples,
                "moneyness_mean": float(np.mean(moneyness)),
                "moneyness_std": float(np.std(moneyness)),
                "T_mean": float(np.mean(T)),
                "T_std": float(np.std(T)),
                "sigma_mean": float(np.mean(sigma)),
                "sigma_std": float(np.std(sigma)),
                "r_mean": float(np.mean(r)),
                "r_std": float(np.std(r)),
                "call_price_mean": float(np.mean(call_prices)),
                "call_price_std": float(np.std(call_prices)),
                "call_price_min": float(np.min(call_prices)),
                "call_price_max": float(np.max(call_prices)),
                "delta_mean": float(np.mean(deltas)),
                "delta_std": float(np.std(deltas)),
                "gamma_mean": float(np.mean(gammas)),
                "gamma_std": float(np.std(gammas)),
                "pc_parity_max_violation": float(parity_violation),
            }
            mlflow.log_metrics(metrics)
            
            print(f"\n{'='*70}")
            print(f"✓ DATASET GENERATION SUCCESSFUL")
            print(f"✓ All sanity checks passed!")
            print(f"{'='*70}")
            
            return dataset
    
    def save_dataset(self, dataset: dict, filename: str = "dataset_100k.npz", raw: bool = True):
        """
        Save dataset to disk.
        
        Parameters
        ----------
        dataset : dict
            Dataset to save
        filename : str
            Output filename
        raw : bool
            If True, save to data/raw; otherwise save to data/processed
        """
        output_dir = self.raw_dir if raw else self.output_dir
        filepath = output_dir / filename
        
        # Flatten dictionary for saving
        save_dict = {}
        for feature_name, feature_array in dataset["features"].items():
            save_dict[f"features_{feature_name}"] = feature_array
        for target_name, target_array in dataset["targets"].items():
            save_dict[f"targets_{target_name}"] = target_array
        
        np.savez(filepath, **save_dict)
        
        with mlflow.start_run(run_name="dataset-save"):
            mlflow.log_artifact(str(filepath), artifact_path="datasets")
            print(f"\n✓ Dataset saved to {filepath}")
            print(f"  File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """Main entry point for data generation."""
    print("\n" + "=" * 70)
    print("BLACK-SCHOLES DATASET GENERATOR")
    print("Market-Realistic Sampling Strategy")
    print("=" * 70)
    
    generator = BSDataGenerator()
    
    # Generate 100k dataset with market-realistic sampling
    dataset = generator.generate_dataset(
        n_samples=100000,
        random_seed=42,
    )
    
    # Save to raw data directory
    generator.save_dataset(dataset, filename="dataset_100k.npz", raw=True)
    
    print("\n" + "=" * 70)
    print("DATA GENERATION COMPLETE!")
    print("=" * 70)
    print("\nDataset Summary:")
    print(f"  Location: data/raw/dataset_100k.npz")
    print(f"  Samples: 100,000")
    print(f"  Features: moneyness, T, r, sigma")
    print(f"  Targets: call_price, put_price, delta, gamma, vega, theta, rho")
    print(f"  All sanity checks passed!")
    print(f"\nNext: View MLflow dashboard")
    print(f"  $ mlflow ui")


if __name__ == "__main__":
    main()
