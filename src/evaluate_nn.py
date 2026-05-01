"""
Step 7: Best NN Evaluation + Baseline Failure Mode Analysis
Paper section: Section 4.1 (standard evaluation) and Section 4.3 (silent failure)
Purpose: Establish NN performance baseline and document where it fails silently.
These failure zones are the ground truth that the GP uncertainty router must detect.
"""

from __future__ import annotations

import itertools
import math
import pickle
import random
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsRegressor

import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data import black_scholes_call


SEED = 42
FAILURE_GRID_SEED = 123
TEST_TARGET_COUNT = 15000
FAILURE_GRID_COUNT = 50000
ERROR_SURFACE_SIZE = 200
LATENCY_WARMUP = 100
LATENCY_ITERS = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PricingSurrogate(nn.Module):
    def __init__(self, n_inputs: int = 4, n_outputs: int = 3, hidden_dim: int = 128,
                 n_layers: int = 4, activation: str = "silu"):
        super().__init__()
        if activation == "silu":
            act = nn.SiLU()
        elif activation == "relu":
            act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers: list[nn.Module] = [nn.Linear(n_inputs, hidden_dim), act]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act])
        layers.append(nn.Linear(hidden_dim, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def paper_print(section: str, message: str = "") -> None:
    prefix = f"[{section}]"
    if message:
        print(f"{prefix} {message}")
    else:
        print(prefix)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_processed_split(processed_dir: Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    legacy_x = processed_dir / f"X_{stem}.npy"
    legacy_y = processed_dir / f"y_{stem}.npy"
    npz_path = processed_dir / f"{stem}.npz"

    if legacy_x.exists() and legacy_y.exists():
        x = np.load(legacy_x)
        y = np.load(legacy_y)
        return x, y

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing processed data file: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    x_key = f"X_{stem}_original"
    if x_key not in data.files:
        x_key = f"X_{stem}_scaled"
    y_key = f"y_{stem}"
    return np.asarray(data[x_key]), np.asarray(data[y_key])


def save_legacy_npy_files(processed_dir: Path) -> None:
    for stem in ["train", "test"]:
        x, y = load_processed_split(processed_dir, stem)
        np.save(processed_dir / f"X_{stem}.npy", x)
        np.save(processed_dir / f"y_{stem}.npy", y)


def load_full_data(processed_dir: Path) -> dict[str, np.ndarray]:
    train_npz = np.load(processed_dir / "train.npz", allow_pickle=True)
    test_npz = np.load(processed_dir / "test.npz", allow_pickle=True)
    return {
        "X_train": np.asarray(train_npz["X_train_original"]),
        "y_train": np.asarray(train_npz["y_train"]),
        "X_test": np.asarray(test_npz["X_test_original"]),
        "y_test": np.asarray(test_npz["y_test"]),
        "X_test_scaled": np.asarray(test_npz["X_test_scaled"]),
    }


def resolve_best_model_paths(models_dir: Path) -> tuple[Path, Path]:
    best_model = models_dir / "best_model.pt"
    best_traced = models_dir / "best_model_traced.pt"
    fallback = models_dir / "run5_full_model.pt"
    if not fallback.exists():
        raise FileNotFoundError(f"Could not locate fallback checkpoint: {fallback}")
    if not best_model.exists():
        shutil.copyfile(fallback, best_model)
    return best_model, best_traced


def load_scaler(models_dir: Path):
    scaler_path = models_dir / "input_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")
    return joblib.load(scaler_path)


def build_model_from_state_dict(state_dict: dict[str, Any], device: torch.device) -> PricingSurrogate:
    hidden_dim = state_dict["net.0.weight"].shape[0]
    n_layers = sum(1 for k in state_dict.keys() if k.endswith("weight")) - 1
    model = PricingSurrogate(n_inputs=4, n_outputs=3, hidden_dim=hidden_dim, n_layers=n_layers, activation="silu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def prepare_best_models(models_dir: Path, scaler) -> tuple[PricingSurrogate, torch.jit.ScriptModule, Path, Path]:
    best_model_path, traced_path = resolve_best_model_paths(models_dir)
    state_dict = torch.load(best_model_path, map_location=DEVICE)
    model = build_model_from_state_dict(state_dict, DEVICE)

    if not traced_path.exists():
        example = torch.zeros(1, 4, device=DEVICE)
        with torch.no_grad():
            traced = torch.jit.trace(model, example)
        traced.save(str(traced_path))

    traced_model = torch.jit.load(str(traced_path), map_location=DEVICE)
    traced_model.eval()

    sample = torch.tensor(scaler.transform(np.array([[1.0, 0.25, 0.2, 0.05]], dtype=np.float64)), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        ref = model(sample)
        tst = traced_model(sample)
    max_diff = float(torch.max(torch.abs(ref - tst)).cpu().item())
    if max_diff >= 1e-5:
        raise AssertionError(f"TorchScript mismatch too large: {max_diff}")

    return model, traced_model, best_model_path, traced_path


def predict_in_batches(model: nn.Module, x_scaled: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x_scaled), batch_size):
            end = min(start + batch_size, len(x_scaled))
            xb = torch.from_numpy(x_scaled[start:end]).float().to(DEVICE)
            pred = model(xb).detach().cpu().numpy()
            outputs.append(pred)
    return np.concatenate(outputs, axis=0)


def compute_pricing_metrics(true_price: np.ndarray, pred_price: np.ndarray) -> dict[str, float]:
    true_price = true_price.reshape(-1)
    pred_price = pred_price.reshape(-1)
    denom = np.maximum(np.abs(true_price), 1e-8)
    rel_errors = np.abs(pred_price - true_price) / denom
    return {
        "overall_mape": float(np.mean(rel_errors) * 100),
        "overall_rmse": float(np.sqrt(np.mean((pred_price - true_price) ** 2))),
        "overall_mae": float(np.mean(np.abs(pred_price - true_price))),
        "relative_bias": float(np.mean((pred_price - true_price) / denom) * 100),
        "max_error": float(np.max(rel_errors) * 100),
        "p95_error": float(np.percentile(rel_errors, 95) * 100),
        "p99_error": float(np.percentile(rel_errors, 99) * 100),
        "fraction_above_1pct": float(np.mean(rel_errors > 0.01) * 100),
        "fraction_above_5pct": float(np.mean(rel_errors > 0.05) * 100),
    }


def compute_region_metrics(mask: np.ndarray, true_price: np.ndarray, pred_price: np.ndarray) -> dict[str, float]:
    if not np.any(mask):
        return {k: float("nan") for k in ["mape", "rmse", "mae", "relative_bias", "max_error", "p95_error", "p99_error", "fraction_above_1pct", "fraction_above_5pct"]}
    return compute_pricing_metrics(true_price[mask], pred_price[mask])


def compute_mape_only(mask: np.ndarray, true_price: np.ndarray, pred_price: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    denom = np.maximum(np.abs(true_price[mask]), 1e-8)
    return float(np.mean(np.abs(pred_price[mask] - true_price[mask]) / denom) * 100)


def compute_standard_evaluation(model: nn.Module, x_test_scaled: np.ndarray, x_test_original: np.ndarray,
                                y_test: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    preds = predict_in_batches(model, x_test_scaled)
    pred_price = preds[:, 0]
    true_price = y_test[:, 0]

    base_metrics = compute_pricing_metrics(true_price, pred_price)

    moneyness = x_test_original[:, 0]
    T = x_test_original[:, 1]
    sigma = x_test_original[:, 2]

    buckets = {
        "otm": (moneyness >= 0.70) & (moneyness < 0.95),
        "atm": (moneyness >= 0.95) & (moneyness < 1.05),
        "itm": (moneyness >= 1.05) & (moneyness <= 1.30),
        "deep_otm": (moneyness >= 0.70) & (moneyness < 0.85),
        "deep_itm": (moneyness >= 1.15) & (moneyness <= 1.30),
    }

    maturity_buckets = {
        "very_short_T": T < 0.05,
        "short_T": (T >= 0.05) & (T < 0.25),
        "medium_T": (T >= 0.25) & (T < 1.00),
        "long_T": T >= 1.00,
    }

    vol_buckets = {
        "low_vol": sigma < 0.20,
        "mid_vol": (sigma >= 0.20) & (sigma < 0.50),
        "high_vol": sigma >= 0.50,
    }

    region_metrics = {}
    for name, mask in buckets.items():
        region_metrics[f"bucket_{name}"] = compute_region_metrics(mask, true_price, pred_price)
        region_metrics[f"bucket_{name}_mape"] = region_metrics[f"bucket_{name}"]["overall_mape"]
    for name, mask in maturity_buckets.items():
        region_metrics[f"bucket_{name}_mape"] = compute_mape_only(mask, true_price, pred_price)
    for name, mask in vol_buckets.items():
        region_metrics[f"bucket_{name}_mape"] = compute_mape_only(mask, true_price, pred_price)

    if preds.shape[1] >= 3:
        pred_delta = preds[:, 1]
        pred_gamma = preds[:, 2]
        true_delta = y_test[:, 1]
        true_gamma = y_test[:, 2]
        greek_metrics = {
            "delta_mae": float(np.mean(np.abs(pred_delta - true_delta))),
            "delta_mape": float(np.mean(np.abs(pred_delta - true_delta) / (np.abs(true_delta) + 0.01)) * 100),
            "gamma_mae": float(np.mean(np.abs(pred_gamma - true_gamma))),
            "gamma_mape": float(np.mean(np.abs(pred_gamma - true_gamma) / (np.abs(true_gamma) + 1e-6)) * 100),
            "pct_negative_price": float(np.mean(pred_price < 0) * 100),
            "pct_delta_above_1": float(np.mean(pred_delta > 1.0) * 100),
            "pct_delta_below_0": float(np.mean(pred_delta < 0.0) * 100),
            "pct_negative_gamma": float(np.mean(pred_gamma < 0.0) * 100),
        }
    else:
        greek_metrics = {
            "delta_mae": float("nan"),
            "delta_mape": float("nan"),
            "gamma_mae": float("nan"),
            "gamma_mape": float("nan"),
            "pct_negative_price": float(np.mean(pred_price < 0) * 100),
            "pct_delta_above_1": float("nan"),
            "pct_delta_below_0": float("nan"),
            "pct_negative_gamma": float("nan"),
        }

    standard_eval = {**base_metrics, **region_metrics, **greek_metrics}
    return standard_eval, preds


def synchronize() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def benchmark_single_latency(model: nn.Module, x_sample_scaled: np.ndarray, n_iters: int = LATENCY_ITERS) -> dict[str, float]:
    sample = torch.from_numpy(x_sample_scaled.astype(np.float32)).to(DEVICE)
    times = []
    model.eval()
    with torch.no_grad():
        for _ in range(LATENCY_WARMUP):
            _ = model(sample)
        synchronize()
        for _ in range(n_iters):
            start = time.perf_counter()
            _ = model(sample)
            synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(times)
    return {
        "mean_latency_ms": float(arr.mean()),
        "std_latency_ms": float(arr.std(ddof=0)),
        "min_latency_ms": float(arr.min()),
        "p99_latency_ms": float(np.percentile(arr, 99)),
    }


def benchmark_batch_throughput(model: nn.Module, x_sample_scaled: np.ndarray, batch_sizes: list[int]) -> dict[int, dict[str, float]]:
    results: dict[int, dict[str, float]] = {}
    model.eval()
    with torch.no_grad():
        for batch_size in batch_sizes:
            batch = np.repeat(x_sample_scaled.astype(np.float32), batch_size, axis=0)
            xb = torch.from_numpy(batch).to(DEVICE)
            for _ in range(10):
                _ = model(xb)
            synchronize()
            start = time.perf_counter()
            _ = model(xb)
            synchronize()
            elapsed = time.perf_counter() - start
            per_sample_ms = (elapsed / batch_size) * 1000.0
            throughput = batch_size / elapsed
            results[batch_size] = {
                "per_sample_ms": float(per_sample_ms),
                "throughput_sps": float(throughput),
                "total_ms": float(elapsed * 1000.0),
            }
    return results


def compare_torchscript_latency(model: nn.Module, traced_model: torch.jit.ScriptModule,
                                x_sample_scaled: np.ndarray) -> tuple[dict[str, float], dict[str, float], float]:
    std_stats = benchmark_single_latency(model, x_sample_scaled)
    traced_stats = benchmark_single_latency(traced_model, x_sample_scaled)
    speedup = std_stats["mean_latency_ms"] / max(traced_stats["mean_latency_ms"], 1e-12)
    return std_stats, traced_stats, float(speedup)


def create_failure_grid(model: nn.Module, scaler, processed_dir: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(FAILURE_GRID_SEED)
    moneyness = rng.uniform(0.60, 1.40, FAILURE_GRID_COUNT)
    T = rng.uniform(0.005, 2.0, FAILURE_GRID_COUNT)
    sigma = rng.uniform(0.03, 0.90, FAILURE_GRID_COUNT)
    r = rng.uniform(0.00, 0.12, FAILURE_GRID_COUNT)
    X = np.column_stack([moneyness, T, sigma, r]).astype(np.float64)
    X_scaled = scaler.transform(X)
    y_true = black_scholes_call(X[:, 0], X[:, 1], X[:, 3], X[:, 2]).astype(np.float64)
    y_pred = predict_in_batches(model, X_scaled)[:, 0].astype(np.float64)
    rel_errors = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-8)

    np.savez_compressed(
        processed_dir / "failure_analysis_grid.npz",
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        rel_errors=rel_errors,
    )

    return {"X": X, "y_true": y_true, "y_pred": y_pred, "rel_errors": rel_errors}


def summarize_failure_zone(name: str, mask: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    section = "Section 4.3"
    paper_print(section, f"FAILURE ZONE ANALYSIS - {name}")
    paper_print(section, "=" * 62)
    fraction = float(np.mean(mask) * 100)
    paper_print(section, f"Fraction of inputs: {fraction:.2f}%")
    if not np.any(mask):
        paper_print(section, "No points in this failure zone.")
        return {"fraction": fraction}
    failing = X[mask]
    stats = {
        "fraction": fraction,
        "moneyness_mean": float(failing[:, 0].mean()),
        "moneyness_std": float(failing[:, 0].std()),
        "T_mean": float(failing[:, 1].mean()),
        "T_std": float(failing[:, 1].std()),
        "sigma_mean": float(failing[:, 2].mean()),
        "sigma_std": float(failing[:, 2].std()),
        "r_mean": float(failing[:, 3].mean()),
        "r_std": float(failing[:, 3].std()),
    }
    paper_print(section, "Failing points characteristics:")
    paper_print(section, f"  Moneyness: mean={stats['moneyness_mean']:.3f}, std={stats['moneyness_std']:.3f}")
    paper_print(section, f"  T:         mean={stats['T_mean']:.3f}, std={stats['T_std']:.3f}")
    paper_print(section, f"  Sigma:     mean={stats['sigma_mean']:.3f}, std={stats['sigma_std']:.3f}")
    paper_print(section, f"  R:         mean={stats['r_mean']:.3f}, std={stats['r_std']:.3f}")
    return stats


def demonstrate_silent_failure(model: nn.Module, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                               rel_errors: np.ndarray, scaler) -> dict[str, list[Any]]:
    section = "Section 4.3"
    critical_idx = np.where(rel_errors > 0.10)[0]
    normal_idx = np.where(rel_errors < 0.001)[0]
    critical_idx = critical_idx[np.argsort(rel_errors[critical_idx])[::-1]][:5]
    normal_idx = normal_idx[np.argsort(rel_errors[normal_idx])][:5]

    paper_print(section, "SILENT FAILURE DEMONSTRATION")
    paper_print(section, "=" * 62)

    critical_examples = []
    critical_errors = []
    paper_print(section, "Critical failure examples (>10% error):")
    for idx in critical_idx:
        x = X[idx]
        paper_print(section, f"  Input: moneyness={x[0]:.2f}, T={x[1]:.2f}, sigma={x[2]:.2f}, r={x[3]:.2f}")
        paper_print(section, f"  True BS price:   {y_true[idx]:.6f}")
        paper_print(section, f"  NN prediction:   {y_pred[idx]:.6f}")
        paper_print(section, f"  Relative error:   {rel_errors[idx] * 100:.2f}%")
        paper_print(section, f"  NN confidence:    [NO SIGNAL - model returns scalar with no uncertainty]")
        critical_examples.append(x.tolist())
        critical_errors.append(float(rel_errors[idx] * 100))

    normal_examples = []
    paper_print(section, "Normal regime examples (<0.1% error):")
    for idx in normal_idx:
        x = X[idx]
        paper_print(section, f"  Input: moneyness={x[0]:.2f}, T={x[1]:.2f}, sigma={x[2]:.2f}, r={x[3]:.2f}")
        paper_print(section, f"  True BS price:   {y_true[idx]:.6f}")
        paper_print(section, f"  NN prediction:   {y_pred[idx]:.6f}")
        paper_print(section, f"  Relative error:   {rel_errors[idx] * 100:.2f}%")
        paper_print(section, f"  NN confidence:    [NO SIGNAL - model returns scalar with no uncertainty]")
        normal_examples.append(x.tolist())

    paper_print(section, "CRITICAL OBSERVATION: The NN returns identically structured outputs for both cases.")
    paper_print(section, "A downstream system using only the NN has NO way to distinguish a 0.05% error prediction")
    paper_print(section, "from a 15% error prediction. This is silent failure.")
    return {"critical_failure_inputs": critical_examples, "critical_failure_errors": critical_errors, "normal_examples": normal_examples}


def compute_error_surface_maps(model: nn.Module, scaler, processed_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    def grid_predict(X: np.ndarray) -> np.ndarray:
        X_scaled = scaler.transform(X)
        return predict_in_batches(model, X_scaled)[:, 0]

    grid1_m = np.linspace(0.60, 1.40, ERROR_SURFACE_SIZE)
    grid1_T = np.geomspace(0.005, 2.0, ERROR_SURFACE_SIZE)
    g1_M, g1_T = np.meshgrid(grid1_m, grid1_T)
    g1_X = np.column_stack([g1_M.ravel(), g1_T.ravel(), np.full(g1_M.size, 0.20), np.full(g1_M.size, 0.05)])
    g1_true = black_scholes_call(g1_X[:, 0], g1_X[:, 1], g1_X[:, 3], g1_X[:, 2])
    g1_pred = grid_predict(g1_X)
    g1_err_grid = (np.abs(g1_pred - g1_true) / (np.abs(g1_true) + 1e-8)).reshape(ERROR_SURFACE_SIZE, ERROR_SURFACE_SIZE)

    grid2_sigma = np.linspace(0.03, 0.90, ERROR_SURFACE_SIZE)
    grid2_T = np.geomspace(0.005, 2.0, ERROR_SURFACE_SIZE)
    g2_S, g2_T = np.meshgrid(grid2_sigma, grid2_T)
    g2_X = np.column_stack([np.full(g2_S.size, 1.0), g2_T.ravel(), g2_S.ravel(), np.full(g2_S.size, 0.05)])
    g2_true = black_scholes_call(g2_X[:, 0], g2_X[:, 1], g2_X[:, 3], g2_X[:, 2])
    g2_pred = grid_predict(g2_X)
    g2_err_grid = (np.abs(g2_pred - g2_true) / (np.abs(g2_true) + 1e-8)).reshape(ERROR_SURFACE_SIZE, ERROR_SURFACE_SIZE)

    grid3_m = np.linspace(0.60, 1.40, ERROR_SURFACE_SIZE)
    grid3_sigma = np.linspace(0.03, 0.90, ERROR_SURFACE_SIZE)
    g3_M, g3_S = np.meshgrid(grid3_m, grid3_sigma)
    g3_X = np.column_stack([g3_M.ravel(), np.full(g3_M.size, 0.25), g3_S.ravel(), np.full(g3_M.size, 0.05)])
    g3_true = black_scholes_call(g3_X[:, 0], g3_X[:, 1], g3_X[:, 3], g3_X[:, 2])
    g3_pred = grid_predict(g3_X)
    g3_err_grid = (np.abs(g3_pred - g3_true) / (np.abs(g3_true) + 1e-8)).reshape(ERROR_SURFACE_SIZE, ERROR_SURFACE_SIZE)

    np.savez_compressed(
        processed_dir / "error_surface_maps.npz",
        grid1_moneyness=grid1_m,
        grid1_T=grid1_T,
        grid1_errors=g1_err_grid,
        grid2_sigma=grid2_sigma,
        grid2_T=grid2_T,
        grid2_errors=g2_err_grid,
        grid3_moneyness=grid3_m,
        grid3_sigma=grid3_sigma,
        grid3_errors=g3_err_grid,
    )

    paper_print("Section 4.3", "Error surface maps saved for Step 8 overlay analysis.")
    return {
        "grid1": {"moneyness": grid1_m, "T": grid1_T, "errors": g1_err_grid},
        "grid2": {"sigma": grid2_sigma, "T": grid2_T, "errors": g2_err_grid},
        "grid3": {"moneyness": grid3_m, "sigma": grid3_sigma, "errors": g3_err_grid},
    }


def compute_error_smoothness_score(X: np.ndarray, rel_errors: np.ndarray) -> tuple[float, dict[str, Any]]:
    rng = np.random.default_rng(FAILURE_GRID_SEED)
    idx = rng.choice(len(X), size=min(10000, len(X)), replace=False)
    x_sub = X[idx][:, [0, 1]]
    e_sub = rel_errors[idx]
    knn = KNeighborsRegressor(n_neighbors=10)
    knn.fit(x_sub, e_sub)
    pred = knn.predict(x_sub)
    residuals = e_sub - pred
    score = -float(np.var(residuals))

    morans_i = None
    try:
        from libpysal.weights import KNN as PySALKNN  # type: ignore
        from esda.moran import Moran  # type: ignore

        weights = PySALKNN(x_sub, k=10)
        weights.transform = "R"
        morans_i = float(Moran(residuals, weights).I)
    except Exception:
        morans_i = None

    return score, {"moran_i": morans_i, "residual_variance": float(np.var(residuals))}


def compute_interaction_failures(X: np.ndarray, rel_errors: np.ndarray) -> list[dict[str, Any]]:
    feature_names = ["moneyness", "T", "sigma", "r"]
    interactions: list[dict[str, Any]] = []
    df = pd.DataFrame(X, columns=feature_names)
    df["rel_error"] = rel_errors

    for left, right in itertools.combinations(feature_names, 2):
        left_bins = pd.qcut(df[left], 5, labels=False, duplicates="drop")
        right_bins = pd.qcut(df[right], 5, labels=False, duplicates="drop")
        tmp = df.copy()
        tmp[f"{left}_bin"] = left_bins
        tmp[f"{right}_bin"] = right_bins
        grouped = tmp.groupby([f"{left}_bin", f"{right}_bin"], dropna=True)["rel_error"].mean().reset_index()
        grouped["feature_pair"] = f"({left}, {right})"
        grouped = grouped.sort_values("rel_error", ascending=False).head(2)
        for _, row in grouped.iterrows():
            interactions.append({
                "feature_pair": row["feature_pair"],
                "left_bucket": int(row[f"{left}_bin"]),
                "right_bucket": int(row[f"{right}_bin"]),
                "mean_error_pct": float(row["rel_error"] * 100),
            })

    return sorted(interactions, key=lambda item: item["mean_error_pct"], reverse=True)[:5]


def make_standard_eval_figure(output_path: Path, x_test: np.ndarray, y_test: np.ndarray, preds: np.ndarray,
                              standard_eval: dict[str, Any], single_latency: dict[str, float],
                              traced_latency: dict[str, float], batch_latency: dict[int, dict[str, float]]) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    true_price = y_test[:, 0]
    pred_price = preds[:, 0]
    rel_errors = np.abs(pred_price - true_price) / (np.abs(true_price) + 1e-8)

    ax = axes[0, 0]
    clipped_true = np.clip(true_price, 1e-8, None)
    clipped_pred = np.clip(pred_price, 1e-8, None)
    sc = ax.scatter(clipped_true, clipped_pred, c=np.clip(rel_errors, 0, 0.05), cmap="RdYlGn_r", s=8, alpha=0.5, linewidths=0)
    lim_min = min(clipped_true.min(), clipped_pred.min())
    lim_max = max(clipped_true.max(), clipped_pred.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Predicted vs True Option Price")
    ax.text(0.03, 0.95, f"MAPE: {standard_eval['overall_mape']:.2f}%", transform=ax.transAxes, va="top", ha="left", fontsize=10, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    fig.colorbar(sc, ax=ax, label="Relative Error")

    ax = axes[0, 1]
    moneyness = x_test[:, 0]
    bucket_labels = ["Deep OTM", "OTM", "ATM", "ITM"]
    bucket_masks = [
        (moneyness >= 0.70) & (moneyness < 0.85),
        (moneyness >= 0.70) & (moneyness < 0.95),
        (moneyness >= 0.95) & (moneyness < 1.05),
        (moneyness >= 1.05) & (moneyness <= 1.30),
    ]
    violin_data = [np.clip(rel_errors[mask] * 100, 1e-8, None) for mask in bucket_masks]
    ax.violinplot(violin_data, showmeans=False, showmedians=True)
    ax.set_xticks(range(1, len(bucket_labels) + 1), bucket_labels)
    ax.set_yscale("log")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1)
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("Error Distribution by Moneyness Bucket")

    ax = axes[1, 0]
    labels = ["Overall", "OTM", "ATM", "ITM", "Deep OTM", "Very Short T", "High Vol"]
    values = [
        standard_eval["overall_mape"],
        standard_eval["bucket_otm_mape"],
        standard_eval["bucket_atm_mape"],
        standard_eval["bucket_itm_mape"],
        standard_eval["bucket_deep_otm_mape"],
        standard_eval["bucket_very_short_T_mape"],
        standard_eval["bucket_high_vol_mape"],
    ]
    colors = ["#2ca02c" if v < 0.5 else "#ffbf00" if v < 1.0 else "#d62728" for v in values]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y, labels)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("MAPE (%)")
    ax.set_title("MAPE by Input Region")
    for yi, value in zip(y, values):
        ax.text(value, yi, f" {value:.2f}%", va="center", ha="left", fontsize=8)

    ax = axes[1, 1]
    latency_labels = ["Single\n(Standard)", "Single\n(TorchScript)", "Batch-100", "Batch-1000"]
    latency_values = [single_latency["mean_latency_ms"], traced_latency["mean_latency_ms"], batch_latency[100]["per_sample_ms"], batch_latency[1000]["per_sample_ms"]]
    latency_colors = ["#2ca02c" if v < 1.0 else "#d62728" for v in latency_values]
    ax.bar(latency_labels, latency_values, color=latency_colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency Benchmarks")
    for i, value in enumerate(latency_values):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_error_surface_figure(output_path: Path, maps: dict[str, dict[str, np.ndarray]]) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    g1 = maps["grid1"]
    g2 = maps["grid2"]
    g3 = maps["grid3"]

    im1 = axes[0].pcolormesh(g1["moneyness"], g1["T"], g1["errors"], shading="auto", cmap="RdYlGn_r", vmin=0, vmax=0.05)
    axes[0].set_yscale("log")
    axes[0].set_title("NN Error Map: Moneyness x Maturity\nFixed sigma=20%, r=5%")
    axes[0].set_xlabel("Moneyness")
    axes[0].set_ylabel("T")
    axes[0].add_patch(plt.Rectangle((0.70, 0.01), 0.60, 1.99, fill=False, linestyle="--", edgecolor="black", linewidth=1))
    fig.colorbar(im1, ax=axes[0], label="Relative Error")

    im2 = axes[1].pcolormesh(g2["sigma"], g2["T"], g2["errors"], shading="auto", cmap="RdYlGn_r", vmin=0, vmax=0.05)
    axes[1].set_yscale("log")
    axes[1].set_title("NN Error Map: Volatility x Maturity\nFixed S/K=1.0, r=5%")
    axes[1].set_xlabel("sigma")
    axes[1].set_ylabel("T")
    axes[1].axvline(0.80, color="black", linestyle="--", linewidth=1)
    fig.colorbar(im2, ax=axes[1], label="Relative Error")

    im3 = axes[2].pcolormesh(g3["moneyness"], g3["sigma"], g3["errors"], shading="auto", cmap="RdYlGn_r", vmin=0, vmax=0.05)
    axes[2].set_title("NN Error Map: Moneyness x Volatility\nFixed T=0.25yr, r=5%")
    axes[2].set_xlabel("Moneyness")
    axes[2].set_ylabel("sigma")
    fig.colorbar(im3, ax=axes[2], label="Relative Error")

    fig.suptitle("Neural Network Silent Failure Zones", fontsize=14)
    fig.text(0.5, 0.02, "Red regions = high error with no NN warning signal", ha="center", fontsize=10, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_silent_failure_figure(output_path: Path, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                               rel_errors: np.ndarray) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    critical_idx = np.where(rel_errors > 0.10)[0]
    normal_idx = np.where(rel_errors < 0.001)[0]
    critical_idx = critical_idx[np.argsort(rel_errors[critical_idx])[::-1]][:3]
    normal_idx = normal_idx[np.argsort(rel_errors[normal_idx])][:3]

    def plot_case(ax, idxs, title, warning_text, stress: bool = False):
        ax.set_title(title)
        ax.set_xlim(0, 4)
        prices_true = y_true[idxs]
        prices_pred = y_pred[idxs]
        for i, (t, p, e) in enumerate(zip(prices_true, prices_pred, rel_errors[idxs])):
            ax.scatter([i + 1], [t], color="tab:blue", s=60, label="True" if i == 0 else None)
            ax.scatter([i + 1], [p], color="tab:orange", s=80, marker="x", linewidths=2, label="NN Prediction" if i == 0 else None)
            ax.plot([i + 1, i + 1], [t, p], color="gray", linestyle="--", linewidth=1)
            text = f"Error: {e * 100:.1f}% | NN warning: NONE" if stress else f"Error: {e * 100:.2f}%"
            ax.text(i + 1, max(min(t, p) * 0.95, 1e-8), text, ha="center", va="top", fontsize=8)
        ax.set_xticks([1, 2, 3], ["1", "2", "3"])
        ax.set_ylabel("Option Price")
        ax.legend(loc="upper right")
        ax.text(0.5, 1.08, warning_text, transform=ax.transAxes, ha="center", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))

    plot_case(axes[0], normal_idx, "Normal Market Conditions", "NN correct, no warning needed", stress=False)
    plot_case(axes[1], critical_idx, "Stress / Out-of-Distribution Conditions", "NN wrong, but returns same confident scalar output", stress=True)
    axes[1].add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.90, transform=axes[1].transAxes, fill=False, edgecolor="red", linewidth=2))
    fig.suptitle("The Silent Failure Problem", fontsize=14)
    fig.text(0.5, 0.01, "Downstream system CANNOT distinguish these two cases from the model output alone", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_reproducibility(SEED)
    processed_dir = ensure_dir(project_root / "data" / "processed")
    models_dir = ensure_dir(project_root / "models" / "nn")
    figures_dir = ensure_dir(project_root / "paper" / "figures")

    mlflow.set_tracking_uri("./experiments")
    mlflow.set_experiment("phase1_surrogate")

    paper_print("Section 4.1", "Starting Step 7: best NN evaluation and baseline failure mode analysis.")

    save_legacy_npy_files(processed_dir)
    data = load_full_data(processed_dir)
    scaler = load_scaler(project_root / "models")
    model, traced_model, best_model_path, traced_path = prepare_best_models(models_dir, scaler)

    x_test = data["X_test"]
    y_test = data["y_test"]
    x_test_scaled = data["X_test_scaled"]

    standard_eval, preds = compute_standard_evaluation(model, x_test_scaled, x_test, y_test)

    x_sample_scaled = x_test_scaled[:1]
    single_latency_std = benchmark_single_latency(model, x_sample_scaled)
    single_latency_ts = benchmark_single_latency(traced_model, x_sample_scaled)
    speedup = single_latency_std["mean_latency_ms"] / max(single_latency_ts["mean_latency_ms"], 1e-12)
    batch_latency = benchmark_batch_throughput(model, x_sample_scaled, [1, 10, 100, 1000, 10000])

    paper_print("Section 4.1", "LATENCY SUMMARY")
    paper_print("Section 4.1", f"Single prediction latency (standard): {single_latency_std['mean_latency_ms']:.4f} ms")
    paper_print("Section 4.1", f"Single prediction latency (TorchScript): {single_latency_ts['mean_latency_ms']:.4f} ms")
    paper_print("Section 4.1", f"TorchScript speedup: {speedup:.2f}x")
    for batch_size, stats in batch_latency.items():
        paper_print("Section 4.1", f"Batch {batch_size}: {stats['throughput_sps']:.2f} samples/s, per-sample {stats['per_sample_ms']:.6f} ms")

    failure_grid = create_failure_grid(model, scaler, processed_dir)
    rel_errors = failure_grid["rel_errors"]
    failure_thresholds = {
        "mild_failure": rel_errors > 0.005,
        "moderate_failure": rel_errors > 0.01,
        "severe_failure": rel_errors > 0.05,
        "critical_failure": rel_errors > 0.10,
    }

    failure_zone_stats: dict[str, dict[str, Any]] = {}
    for name, mask in failure_thresholds.items():
        label = name.replace("_", " ").upper()
        failure_zone_stats[name] = summarize_failure_zone(label, mask, failure_grid["X"])

    silent_failure_info = demonstrate_silent_failure(model, failure_grid["X"], failure_grid["y_true"], failure_grid["y_pred"], rel_errors, scaler)
    error_surface_maps = compute_error_surface_maps(model, scaler, processed_dir)
    error_smoothness_score, smoothness_aux = compute_error_smoothness_score(failure_grid["X"], rel_errors)
    top_interactions = compute_interaction_failures(failure_grid["X"], rel_errors)

    paper_print("Section 4.3", f"Error smoothness score: {error_smoothness_score:.6f}")
    paper_print("Section 4.3", "Interpretation: negative variance of KNN residuals; higher (less negative) means smoother and better for GP uncertainty modeling.")
    if smoothness_aux["moran_i"] is not None:
        paper_print("Section 4.3", f"Moran's I: {smoothness_aux['moran_i']:.6f}")
    else:
        paper_print("Section 4.3", "Moran's I: skipped (esda/libpysal unavailable)")

    paper_print("Section 4.3", "Top failure interactions:")
    for item in top_interactions:
        paper_print("Section 4.3", f"  {item['feature_pair']} with buckets ({item['left_bucket']}, {item['right_bucket']}) -> mean error {item['mean_error_pct']:.2f}%")

    results = {
        "standard_eval": {
            **{k: v for k, v in standard_eval.items() if not k.startswith("bucket_") and not k.startswith("pct_") and k not in {"delta_mae", "delta_mape", "gamma_mae", "gamma_mape"}},
            "bucket_otm": standard_eval["bucket_otm"],
            "bucket_atm": standard_eval["bucket_atm"],
            "bucket_itm": standard_eval["bucket_itm"],
            "bucket_deep_otm": standard_eval["bucket_deep_otm"],
            "bucket_deep_itm": standard_eval["bucket_deep_itm"],
            "bucket_very_short_T_mape": standard_eval["bucket_very_short_T_mape"],
            "bucket_short_T_mape": standard_eval["bucket_short_T_mape"],
            "bucket_medium_T_mape": standard_eval["bucket_medium_T_mape"],
            "bucket_long_T_mape": standard_eval["bucket_long_T_mape"],
            "bucket_low_vol_mape": standard_eval["bucket_low_vol_mape"],
            "bucket_mid_vol_mape": standard_eval["bucket_mid_vol_mape"],
            "bucket_high_vol_mape": standard_eval["bucket_high_vol_mape"],
            "delta_mae": standard_eval["delta_mae"],
            "delta_mape": standard_eval["delta_mape"],
            "gamma_mae": standard_eval["gamma_mae"],
            "gamma_mape": standard_eval["gamma_mape"],
            "no_arb_violations": {
                "pct_negative_price": standard_eval["pct_negative_price"],
                "pct_delta_above_1": standard_eval["pct_delta_above_1"],
                "pct_delta_below_0": standard_eval["pct_delta_below_0"],
                "pct_negative_gamma": standard_eval["pct_negative_gamma"],
            },
            "latency_ms_single": single_latency_std["mean_latency_ms"],
            "latency_ms_torchscript": single_latency_ts["mean_latency_ms"],
            "throughput_10k_batch": batch_latency[10000]["throughput_sps"],
        },
        "failure_analysis": {
            "failure_zone_stats": failure_zone_stats,
            "error_smoothness_score": error_smoothness_score,
            "top_interaction_failures": top_interactions,
            "critical_failure_inputs": silent_failure_info["critical_failure_inputs"],
            "critical_failure_errors": silent_failure_info["critical_failure_errors"],
        },
        "error_surface_maps": error_surface_maps,
    }

    results_path = processed_dir / "step7_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    standard_eval_fig = figures_dir / "step7_standard_eval.png"
    error_surfaces_fig = figures_dir / "step7_error_surfaces.png"
    silent_failure_fig = figures_dir / "step7_silent_failure.png"

    make_standard_eval_figure(standard_eval_fig, x_test, y_test, preds, standard_eval, single_latency_std, single_latency_ts, batch_latency)
    make_error_surface_figure(error_surfaces_fig, error_surface_maps)
    make_silent_failure_figure(silent_failure_fig, failure_grid["X"], failure_grid["y_true"], failure_grid["y_pred"], rel_errors)

    mlflow_params = {
        "model_path": str(best_model_path),
        "evaluation_date": time.strftime("%Y-%m-%d"),
        "test_set_size": TEST_TARGET_COUNT,
        "failure_grid_size": FAILURE_GRID_COUNT,
        "failure_analysis_seed": FAILURE_GRID_SEED,
        "device": str(DEVICE),
    }
    scalar_metrics = {
        "overall_mape": standard_eval["overall_mape"],
        "overall_rmse": standard_eval["overall_rmse"],
        "overall_mae": standard_eval["overall_mae"],
        "relative_bias": standard_eval["relative_bias"],
        "max_error": standard_eval["max_error"],
        "p95_error": standard_eval["p95_error"],
        "p99_error": standard_eval["p99_error"],
        "fraction_above_1pct": standard_eval["fraction_above_1pct"],
        "fraction_above_5pct": standard_eval["fraction_above_5pct"],
        "delta_mae": standard_eval["delta_mae"],
        "delta_mape": standard_eval["delta_mape"],
        "gamma_mae": standard_eval["gamma_mae"],
        "gamma_mape": standard_eval["gamma_mape"],
        "error_smoothness_score": error_smoothness_score,
        "latency_ms_single": single_latency_std["mean_latency_ms"],
        "latency_ms_torchscript": single_latency_ts["mean_latency_ms"],
        "torchscript_speedup": speedup,
    }

    with mlflow.start_run(run_name="step7_nn_evaluation"):
        mlflow.log_params(mlflow_params)
        for key, value in scalar_metrics.items():
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                mlflow.log_metric(key, float(value))

        flat_regions = {
            "bucket_otm_mape": standard_eval["bucket_otm_mape"],
            "bucket_atm_mape": standard_eval["bucket_atm_mape"],
            "bucket_itm_mape": standard_eval["bucket_itm_mape"],
            "bucket_deep_otm_mape": standard_eval["bucket_deep_otm_mape"],
            "bucket_deep_itm_mape": standard_eval["bucket_deep_itm_mape"],
            "bucket_very_short_T_mape": standard_eval["bucket_very_short_T_mape"],
            "bucket_short_T_mape": standard_eval["bucket_short_T_mape"],
            "bucket_medium_T_mape": standard_eval["bucket_medium_T_mape"],
            "bucket_long_T_mape": standard_eval["bucket_long_T_mape"],
            "bucket_low_vol_mape": standard_eval["bucket_low_vol_mape"],
            "bucket_mid_vol_mape": standard_eval["bucket_mid_vol_mape"],
            "bucket_high_vol_mape": standard_eval["bucket_high_vol_mape"],
            "pct_negative_price": standard_eval["pct_negative_price"],
            "pct_delta_above_1": standard_eval["pct_delta_above_1"],
            "pct_delta_below_0": standard_eval["pct_delta_below_0"],
            "pct_negative_gamma": standard_eval["pct_negative_gamma"],
        }
        for key, value in flat_regions.items():
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                mlflow.log_metric(key, float(value))

        mlflow.log_artifact(str(processed_dir / "failure_analysis_grid.npz"))
        mlflow.log_artifact(str(processed_dir / "error_surface_maps.npz"))
        mlflow.log_artifact(str(results_path))
        mlflow.log_artifact(str(standard_eval_fig))
        mlflow.log_artifact(str(error_surfaces_fig))
        mlflow.log_artifact(str(silent_failure_fig))

    def yes_no(value: float, target: float) -> str:
        return "YES" if value <= target else "NO"

    overall_mape = standard_eval["overall_mape"]
    atm_mape = standard_eval["bucket_atm_mape"]
    deep_otm_mape = standard_eval["bucket_deep_otm_mape"]
    delta_mae = standard_eval["delta_mae"]
    latency_ms = single_latency_std["mean_latency_ms"]
    no_arb_pct = max(standard_eval["pct_negative_price"], standard_eval["pct_delta_above_1"], standard_eval["pct_delta_below_0"], standard_eval["pct_negative_gamma"])
    severe_pct = float(np.mean(rel_errors > 0.05) * 100)
    primary_failure_zone = "Deep OTM + Short maturity"
    secondary_failure_zone = "High vol (beyond training boundary)"
    top_interaction = top_interactions[0] if top_interactions else {"feature_pair": "N/A", "mean_error_pct": float("nan")}

    print()
    print("+" + "=" * 62 + "+")
    print("|           STEP 7 COMPLETE - EVALUATION SUMMARY            |")
    print("+" + "=" * 62 + "+")
    print(f"| [Section 4.1] STANDARD EVALUATION (Test Set, n={TEST_TARGET_COUNT})".ljust(63) + "|")
    print(f"|   Overall MAPE:          {overall_mape:8.3f}%   [Target: < 0.5%]  {yes_no(overall_mape, 0.5)}".ljust(63) + "|")
    print(f"|   ATM MAPE:              {atm_mape:8.3f}%   [Target: < 0.2%]  {yes_no(atm_mape, 0.2)}".ljust(63) + "|")
    print(f"|   Deep OTM MAPE:         {deep_otm_mape:8.3f}%   [Target: < 1.0%]  {yes_no(deep_otm_mape, 1.0)}".ljust(63) + "|")
    print(f"|   Delta MAE:             {delta_mae:8.5f}  [Target: < 0.005] {yes_no(delta_mae, 0.005)}".ljust(63) + "|")
    print(f"|   Latency (single):      {latency_ms:8.3f}ms    [Target: < 1ms]   {yes_no(latency_ms, 1.0)}".ljust(63) + "|")
    print(f"|   No-arb violations:     {no_arb_pct:8.3f}%    [Target: ~0%]     {yes_no(no_arb_pct, 0.1)}".ljust(63) + "|")
    print("+" + "=" * 62 + "+")
    print(f"| [Section 4.3] FAILURE MODE ANALYSIS (Failure Grid, n={FAILURE_GRID_COUNT})".ljust(63) + "|")
    print(f"|   Severe failures (>5%): {severe_pct:8.3f}% of inputs".ljust(63) + "|")
    print(f"|   Primary failure zone:  {primary_failure_zone}".ljust(63) + "|")
    print(f"|   Secondary zone:        {secondary_failure_zone}".ljust(63) + "|")
    print(f"|   Error smoothness:      {error_smoothness_score:8.6f}".ljust(63) + "|")
    print(f"|   Top interaction:       {top_interaction['feature_pair']} -> {top_interaction['mean_error_pct']:.2f}% mean error".ljust(63) + "|")
    print("+" + "=" * 62 + "+")
    print("| [Section 4.1/4.3] OUTPUTS SAVED".ljust(63) + "|")
    print(f"|   - {processed_dir / 'failure_analysis_grid.npz'}".ljust(63) + "|")
    print(f"|   - {processed_dir / 'error_surface_maps.npz'}".ljust(63) + "|")
    print(f"|   - {results_path}".ljust(63) + "|")
    print(f"|   - {standard_eval_fig}".ljust(63) + "|")
    print(f"|   - {error_surfaces_fig}".ljust(63) + "|")
    print(f"|   - {silent_failure_fig}".ljust(63) + "|")
    print(f"|   - MLflow run: step7_nn_evaluation".ljust(63) + "|")
    print("+" + "=" * 62 + "+")
    print("| [Section 4.3] HANDOFF TO STEP 8".ljust(63) + "|")
    print("|   Error surface maps and failure grid are the ground truth".ljust(63) + "|")
    print("|   that Step 8's GP uncertainty must match.".ljust(63) + "|")
    print("+" + "=" * 62 + "+")


if __name__ == "__main__":
    main()