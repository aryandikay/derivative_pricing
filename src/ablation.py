"""
STEP 6: ABLATION STUDY FOR SURROGATE PRICING MODEL

This script runs a systematic ablation study across 7 experiment configurations
to determine which model architecture and loss formulation produces the smoothest,
most well-behaved error surface for GP-gated routing in subsequent steps.

The key innovation is the "routing suitability" metric set, which measures not
just prediction accuracy (MAPE) but also error smoothness—a critical requirement
for downstream uncertainty-guided routing.

PAPER SECTION: Methods → Model Selection & Routing Foundation
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
import mlflow
import mlflow.pytorch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import time
import warnings

warnings.filterwarnings('ignore')

# Setup
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import black_scholes_call, bs_delta_call, bs_gamma

# MLflow setup
mlflow.set_tracking_uri("./experiments")
mlflow.set_experiment("phase1_surrogate")

# Random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*70}")
print(f"DEVICE: {DEVICE}")
print(f"{'='*70}\n")


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class PricingSurrogate(nn.Module):
    """
    Flexible surrogate model for option pricing.
    
    Can output:
    - Single target: just price
    - Multi-task: [price, delta, gamma]
    
    Can use different activations (SiLU, ReLU) for ablation.
    """
    
    def __init__(self, n_inputs=4, n_outputs=1, hidden_dim=128, n_layers=4, 
                 activation='silu'):
        """
        Parameters
        ----------
        n_inputs : int
            Input dimensionality (4: moneyness, T, sigma, r)
        n_outputs : int
            Output dimensionality (1 for price only, 3 for [price, delta, gamma])
        hidden_dim : int
            Hidden layer dimension (128 standard)
        n_layers : int
            Number of hidden layers (4 standard)
        activation : str
            'silu' or 'relu'
        """
        super().__init__()
        
        self.n_outputs = n_outputs
        self.activation_name = activation
        
        # Activation function
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(n_inputs, hidden_dim))
        layers.append(self.activation)
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, n_outputs))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, compute_gradients=False):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 4)
        compute_gradients : bool
            If True, enable gradients for autograd computation (needed for PINN)
            
        Returns
        -------
        torch.Tensor
            Output of shape (batch_size, n_outputs)
        """
        if compute_gradients:
            x.requires_grad_(True)
        
        return self.net(x)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def mse_loss(pred, target):
    """Standard MSE loss."""
    return torch.mean((pred - target) ** 2)


def relative_mse_loss(pred, target, epsilon=1e-8):
    """
    Relative MSE loss: mean of ((pred - true) / (true + epsilon))^2
    
    Better for multi-scale targets (price ranges 0-1, delta 0-1, etc)
    and reduces scale-dependent bias.
    """
    relative_error = (pred - target) / (torch.abs(target) + epsilon)
    return torch.mean(relative_error ** 2)


def compute_pde_residual(model, X_scaled, X_original, scaler, lambda_pde=0.01):
    """
    Compute Black-Scholes PDE residual for PINN regularization.
    
    The Black-Scholes PDE is:
        -dV/dT + 0.5 * sigma^2 * moneyness^2 * d2V/dm2 + r * moneyness * dV/dm - r * V = 0
    
    Parameters
    ----------
    model : nn.Module
        Surrogate model (takes normalized inputs)
    X_scaled : torch.Tensor
        Normalized inputs (batch_size, 4)
    X_original : torch.Tensor
        Original scale inputs (batch_size, 4) for PDE coefficient reconstruction
    scaler : StandardScaler
        Fitted scaler for inverse transform
    lambda_pde : float
        Weight for PDE loss
        
    Returns
    -------
    torch.Tensor
        PDE residual loss
    """
    
    # Enable gradients for autograd
    X_scaled.requires_grad_(True)
    
    # Forward pass through model to get price prediction
    V = model(X_scaled)[:, 0:1]  # Extract price output (first column)
    
    # Extract original-scale parameters for PDE coefficients
    # X_original columns: [moneyness, T, sigma, r]
    moneyness = X_original[:, 0]
    T = X_original[:, 1]
    sigma = X_original[:, 2]
    r = X_original[:, 3]
    
    # Compute dV/dT (derivative w.r.t. time feature, index 1)
    dV = torch.autograd.grad(
        outputs=V.sum(),
        inputs=X_scaled,
        create_graph=True,
        retain_graph=True
    )[0]
    
    dV_dT = dV[:, 1:2]  # Time is at index 1
    
    # Compute dV/dm (derivative w.r.t. moneyness feature, index 0)
    dV_dm = dV[:, 0:1]  # Moneyness is at index 0
    
    # Compute d2V/dm2 (second derivative w.r.t. moneyness)
    d2V = torch.autograd.grad(
        outputs=dV_dm.sum(),
        inputs=X_scaled,
        create_graph=True,
        retain_graph=True
    )[0]
    
    d2V_dm2 = d2V[:, 0:1]  # Second derivative w.r.t. moneyness
    
    # Reconstruct PDE: -dV/dT + 0.5 * sigma^2 * moneyness^2 * d2V/dm2 + r * moneyness * dV/dm - r * V
    pde_residual = (
        -dV_dT +
        0.5 * (sigma.unsqueeze(1) ** 2) * (moneyness.unsqueeze(1) ** 2) * d2V_dm2 +
        r.unsqueeze(1) * moneyness.unsqueeze(1) * dV_dm -
        r.unsqueeze(1) * V
    )
    
    # PDE loss is mean squared residual
    pde_loss = torch.mean(pde_residual ** 2)
    
    return pde_loss, torch.abs(pde_residual).mean().item()  # Return loss and magnitude


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model(config, train_loader, val_loader, X_val_original, scaler):
    """
    Train a surrogate model with specified configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dict with model hyperparameters
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    X_val_original : np.ndarray
        Original-scale validation inputs (for PINN)
    scaler : StandardScaler
        Fitted scaler
        
    Returns
    -------
    dict
        Training history and trained model
    """
    
    # Extract config
    n_outputs = config['n_outputs']
    use_pinn = config['use_pinn']
    use_greeks = config['use_greeks']
    loss_type = config['loss_type']
    lambda_price = config.get('lambda_price', 1.0)
    lambda_delta = config.get('lambda_delta', 0.5)
    lambda_gamma = config.get('lambda_gamma', 0.1)
    lambda_pde = config.get('lambda_pde', 0.01)
    activation = config['activation']
    hidden_dim = config['hidden_dim']
    n_layers = config['n_layers']
    epochs = config['epochs']
    lr = config['lr']
    weight_decay = config['weight_decay']
    
    # Build model
    model = PricingSurrogate(
        n_inputs=4,
        n_outputs=n_outputs,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation=activation
    ).to(DEVICE)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate schedule: cosine annealing
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_price_mape': [],
        'pde_residual_mag': [] if use_pinn else None,
    }
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(X_batch, compute_gradients=False)
            
            # Compute data loss
            if n_outputs == 1:  # Price only
                if loss_type == 'mse':
                    loss = mse_loss(pred, y_batch[:, 0:1])
                else:  # relative_mse
                    loss = relative_mse_loss(pred, y_batch[:, 0:1])
            else:  # Multi-task
                pred_price = pred[:, 0:1]
                pred_delta = pred[:, 1:2]
                pred_gamma = pred[:, 2:3]
                
                true_price = y_batch[:, 0:1]
                true_delta = y_batch[:, 1:2]
                true_gamma = y_batch[:, 2:3]
                
                if loss_type == 'mse':
                    loss_price = mse_loss(pred_price, true_price)
                    loss_delta = mse_loss(pred_delta, true_delta)
                    loss_gamma = mse_loss(pred_gamma, true_gamma)
                else:  # relative_mse
                    loss_price = relative_mse_loss(pred_price, true_price)
                    loss_delta = relative_mse_loss(pred_delta, true_delta)
                    loss_gamma = relative_mse_loss(pred_gamma, true_gamma)
                
                loss = lambda_price * loss_price + lambda_delta * loss_delta + lambda_gamma * loss_gamma
            
            # Add PINN regularization if enabled
            if use_pinn:
                # Sample collocation points for PDE constraint
                # Use batch inputs as collocation points (could also sample randomly)
                X_batch_np = X_batch.detach().cpu().numpy()
                
                # Inverse transform to get original scale
                X_original_batch = scaler.inverse_transform(X_batch_np)
                X_original_batch_tensor = torch.from_numpy(X_original_batch).float().to(DEVICE)
                
                pde_loss, _ = compute_pde_residual(
                    model, X_batch, X_original_batch_tensor, scaler, lambda_pde
                )
                
                loss = loss + lambda_pde * pde_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            n_batches += 1
        
        train_loss_avg = train_loss_sum / n_batches
        history['train_loss'].append(train_loss_avg)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            price_mape_sum = 0.0
            n_val_batches = 0
            
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                pred = model(X_batch)
                
                # Compute validation loss (same as training)
                if n_outputs == 1:
                    if loss_type == 'mse':
                        val_loss = mse_loss(pred, y_batch[:, 0:1])
                    else:
                        val_loss = relative_mse_loss(pred, y_batch[:, 0:1])
                    pred_price = pred
                else:
                    pred_price = pred[:, 0:1]
                    true_price = y_batch[:, 0:1]
                    
                    if loss_type == 'mse':
                        val_loss = mse_loss(pred_price, true_price) if n_outputs == 1 else \
                            mse_loss(pred[:, 0:1], true_price) + \
                            mse_loss(pred[:, 1:2], y_batch[:, 1:2]) + \
                            mse_loss(pred[:, 2:3], y_batch[:, 2:3])
                    else:
                        val_loss = relative_mse_loss(pred_price, true_price) if n_outputs == 1 else \
                            relative_mse_loss(pred[:, 0:1], true_price) + \
                            relative_mse_loss(pred[:, 1:2], y_batch[:, 1:2]) + \
                            relative_mse_loss(pred[:, 2:3], y_batch[:, 2:3])
                
                # Compute MAPE on price
                price_mape = torch.mean(
                    torch.abs(pred_price - y_batch[:, 0:1]) / (torch.abs(y_batch[:, 0:1]) + 1e-8)
                ) * 100
                
                val_loss_sum += val_loss.item()
                price_mape_sum += price_mape.item()
                n_val_batches += 1
            
            val_loss_avg = val_loss_sum / n_val_batches
            val_mape_avg = price_mape_sum / n_val_batches
            
            history['val_loss'].append(val_loss_avg)
            history['val_price_mape'].append(val_mape_avg)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping (optional, set patience high to let model train fully)
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss_avg:.6f} | "
                  f"Val Loss: {val_loss_avg:.6f} | "
                  f"Val MAPE: {val_mape_avg:.4f}%")
        
        # Check for NaN
        if np.isnan(train_loss_avg) or np.isnan(val_loss_avg):
            print(f"❌ NaN detected at epoch {epoch+1}. Training diverged.")
            return None
    
    return {
        'model': model,
        'history': history,
        'config': config,
    }


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_routing_suitability_metrics(model, X_test, y_test, X_test_original):
    """
    Compute metrics specifically designed for GP routing suitability.
    
    Parameters
    ----------
    model : nn.Module
        Trained model
    X_test : torch.Tensor or np.ndarray
        Scaled test inputs
    y_test : np.ndarray
        True test targets
    X_test_original : np.ndarray
        Original-scale test inputs (for moneyness bucketing)
        
    Returns
    -------
    dict
        Routing suitability metrics
    """
    
    if isinstance(X_test, np.ndarray):
        X_test = torch.from_numpy(X_test).float()
    
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(DEVICE)
        pred = model(X_test).cpu().numpy()
    
    # Extract price predictions and errors
    pred_price = pred[:, 0] if pred.ndim > 1 else pred
    true_price = y_test[:, 0] if y_test.ndim > 1 else y_test
    
    relative_errors = np.abs(pred_price - true_price) / (np.abs(true_price) + 1e-8)
    
    # Error smoothness: KNN-based local error variance
    # Intuition: smooth error surface means nearby inputs have similar errors
    # which is good for routing (GP can learn the error pattern)
    X_features = X_test_original[:, [0, 1]]  # moneyness, T
    
    knn = KNeighborsRegressor(n_neighbors=10, n_jobs=-1)
    knn.fit(X_features, relative_errors)
    knn_pred_errors = knn.predict(X_features)
    
    error_smoothness_score = np.var(relative_errors - knn_pred_errors)
    # Lower is better: lower variance means smoother error surface
    
    # Worst-case error
    worst_case_error = np.max(relative_errors) * 100
    
    # 99th percentile error
    p99_error = np.percentile(relative_errors, 99) * 100
    
    # Fraction with error > 1%
    fraction_above_1pct = np.mean(relative_errors > 0.01) * 100
    
    return {
        'error_smoothness_score': error_smoothness_score,
        'worst_case_error': worst_case_error,
        'p99_error': p99_error,
        'fraction_errors_above_1pct': fraction_above_1pct,
    }


def evaluate_model(model, X_test, y_test, X_test_original, config):
    """
    Comprehensive model evaluation.
    
    Parameters
    ----------
    model : nn.Module
        Trained model
    X_test : np.ndarray
        Scaled test inputs
    y_test : np.ndarray
        True test targets
    X_test_original : np.ndarray
        Original-scale test inputs
    config : dict
        Model configuration
        
    Returns
    -------
    dict
        Evaluation metrics
    """
    
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).float().to(DEVICE)
        pred = model(X_test_tensor).cpu().numpy()
    
    # Extract components
    pred_price = pred[:, 0] if pred.ndim > 1 else pred
    true_price = y_test[:, 0] if y_test.ndim > 1 else y_test
    
    # Price MAPE
    price_mape_overall = np.mean(np.abs(pred_price - true_price) / (np.abs(true_price) + 1e-8)) * 100
    
    # MAPE by moneyness bucket
    moneyness = X_test_original[:, 0]
    
    otm_mask = moneyness < 0.95
    atm_mask = (moneyness >= 0.95) & (moneyness <= 1.05)
    itm_mask = moneyness > 1.05
    
    mape_otm = np.mean(np.abs(pred_price[otm_mask] - true_price[otm_mask]) / 
                       (np.abs(true_price[otm_mask]) + 1e-8)) * 100 if otm_mask.sum() > 0 else np.nan
    
    mape_atm = np.mean(np.abs(pred_price[atm_mask] - true_price[atm_mask]) / 
                       (np.abs(true_price[atm_mask]) + 1e-8)) * 100 if atm_mask.sum() > 0 else np.nan
    
    mape_itm = np.mean(np.abs(pred_price[itm_mask] - true_price[itm_mask]) / 
                       (np.abs(true_price[itm_mask]) + 1e-8)) * 100 if itm_mask.sum() > 0 else np.nan
    
    metrics = {
        'test_mape_overall': price_mape_overall,
        'test_mape_otm': mape_otm,
        'test_mape_atm': mape_atm,
        'test_mape_itm': mape_itm,
    }
    
    # Delta and gamma errors (if applicable)
    if pred.shape[1] >= 2:
        pred_delta = pred[:, 1]
        true_delta = y_test[:, 1]
        metrics['test_delta_mae'] = mean_absolute_error(true_delta, pred_delta)
    
    if pred.shape[1] >= 3:
        pred_gamma = pred[:, 2]
        true_gamma = y_test[:, 2]
        metrics['test_gamma_mae'] = mean_absolute_error(true_gamma, pred_gamma)
    
    # Inference latency
    X_dummy = torch.randn(1, 4).to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        # Warmup
        for _ in range(100):
            _ = model(X_dummy)
        
        # Timed runs
        start = time.time()
        for _ in range(10000):
            _ = model(X_dummy)
        end = time.time()
    
    latency_ms = ((end - start) / 10000) * 1000
    metrics['latency_ms'] = latency_ms
    
    # Routing suitability metrics
    routing_metrics = compute_routing_suitability_metrics(model, X_test, y_test, X_test_original)
    metrics.update(routing_metrics)
    
    return metrics


# ============================================================================
# MAIN ABLATION STUDY
# ============================================================================

def run_ablation_study():
    """Run all 7 ablation experiments."""
    
    print("\n" + "="*70)
    print("STARTING ABLATION STUDY")
    print("="*70)
    
    # Load preprocessed data
    print("\n[LOADING DATA]")
    train_data = np.load('data/processed/train.npz', allow_pickle=True)
    val_data = np.load('data/processed/val.npz', allow_pickle=True)
    test_data = np.load('data/processed/test.npz', allow_pickle=True)
    
    X_train = train_data['X_train_scaled']
    y_train = train_data['y_train']
    X_val = val_data['X_val_scaled']
    y_val = val_data['y_val']
    X_test = test_data['X_test_scaled']
    y_test = test_data['y_test']
    X_test_original = test_data['X_test_original']
    
    scaler = joblib.load('models/input_scaler.pkl')
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float()
    )
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float()
    )
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    # Define 7 ablation configurations
    ablation_configs = [
        {
            'run_name': 'run1_baseline',
            'n_outputs': 1,
            'loss_type': 'mse',
            'use_greeks': False,
            'use_pinn': False,
            'lambda_price': 1.0,
            'lambda_delta': 0.0,
            'lambda_gamma': 0.0,
            'lambda_pde': 0.0,
            'activation': 'silu',
            'hidden_dim': 128,
            'n_layers': 4,
            'epochs': 200,
            'batch_size': 1024,
            'lr': 1e-3,
            'weight_decay': 1e-4,
        },
        {
            'run_name': 'run2_relative_mse',
            'n_outputs': 1,
            'loss_type': 'relative_mse',
            'use_greeks': False,
            'use_pinn': False,
            'lambda_price': 1.0,
            'lambda_delta': 0.0,
            'lambda_gamma': 0.0,
            'lambda_pde': 0.0,
            'activation': 'silu',
            'hidden_dim': 128,
            'n_layers': 4,
            'epochs': 200,
            'batch_size': 1024,
            'lr': 1e-3,
            'weight_decay': 1e-4,
        },
        {
            'run_name': 'run3_multi_task',
            'n_outputs': 3,
            'loss_type': 'relative_mse',
            'use_greeks': True,
            'use_pinn': False,
            'lambda_price': 1.0,
            'lambda_delta': 0.5,
            'lambda_gamma': 0.1,
            'lambda_pde': 0.0,
            'activation': 'silu',
            'hidden_dim': 128,
            'n_layers': 4,
            'epochs': 200,
            'batch_size': 1024,
            'lr': 1e-3,
            'weight_decay': 1e-4,
        },
        {
            'run_name': 'run4_pinn_only',
            'n_outputs': 1,
            'loss_type': 'relative_mse',
            'use_greeks': False,
            'use_pinn': True,
            'lambda_price': 1.0,
            'lambda_delta': 0.0,
            'lambda_gamma': 0.0,
            'lambda_pde': 0.01,
            'activation': 'silu',
            'hidden_dim': 128,
            'n_layers': 4,
            'epochs': 200,
            'batch_size': 1024,
            'lr': 1e-3,
            'weight_decay': 1e-4,
        },
        {
            'run_name': 'run5_full_model',
            'n_outputs': 3,
            'loss_type': 'relative_mse',
            'use_greeks': True,
            'use_pinn': True,
            'lambda_price': 1.0,
            'lambda_delta': 0.5,
            'lambda_gamma': 0.1,
            'lambda_pde': 0.01,
            'activation': 'silu',
            'hidden_dim': 128,
            'n_layers': 4,
            'epochs': 200,
            'batch_size': 1024,
            'lr': 1e-3,
            'weight_decay': 1e-4,
        },
        {
            'run_name': 'run6_relu_activation',
            'n_outputs': 3,
            'loss_type': 'relative_mse',
            'use_greeks': True,
            'use_pinn': True,
            'lambda_price': 1.0,
            'lambda_delta': 0.5,
            'lambda_gamma': 0.1,
            'lambda_pde': 0.01,
            'activation': 'relu',
            'hidden_dim': 128,
            'n_layers': 4,
            'epochs': 200,
            'batch_size': 1024,
            'lr': 1e-3,
            'weight_decay': 1e-4,
        },
    ]
    
    # Add Run 7 sub-runs (lambda_pde sweep)
    for lambda_pde_val in [0.001, 0.01, 0.1]:
        ablation_configs.append({
            'run_name': f'run7_lambda_pde_{lambda_pde_val}',
            'n_outputs': 3,
            'loss_type': 'relative_mse',
            'use_greeks': True,
            'use_pinn': True,
            'lambda_price': 1.0,
            'lambda_delta': 0.5,
            'lambda_gamma': 0.1,
            'lambda_pde': lambda_pde_val,
            'activation': 'silu',
            'hidden_dim': 128,
            'n_layers': 4,
            'epochs': 200,
            'batch_size': 1024,
            'lr': 1e-3,
            'weight_decay': 1e-4,
        })
    
    # Run all ablations
    results = []
    
    for i, config in enumerate(ablation_configs):
        print(f"\n{'='*70}")
        print(f"RUN {i+1}: {config['run_name'].upper()}")
        print(f"{'='*70}")
        print(f"Outputs: {config['n_outputs']}")
        print(f"Loss: {config['loss_type']}")
        print(f"Use Greeks: {config['use_greeks']}")
        print(f"Use PINN: {config['use_pinn']}")
        if config['use_pinn']:
            print(f"Lambda PDE: {config['lambda_pde']}")
        print(f"Activation: {config['activation']}")
        
        try:
            with mlflow.start_run(run_name=config['run_name']) as run:
                # Log parameters
                mlflow.log_params({
                    'run_name': config['run_name'],
                    'n_outputs': config['n_outputs'],
                    'loss_type': config['loss_type'],
                    'use_greeks': config['use_greeks'],
                    'use_pinn': config['use_pinn'],
                    'lambda_price': config['lambda_price'],
                    'lambda_delta': config['lambda_delta'],
                    'lambda_gamma': config['lambda_gamma'],
                    'lambda_pde': config['lambda_pde'],
                    'activation': config['activation'],
                    'hidden_dim': config['hidden_dim'],
                    'n_layers': config['n_layers'],
                    'epochs': config['epochs'],
                    'batch_size': config['batch_size'],
                    'lr': config['lr'],
                })
                
                # Train
                print("\nTraining...")
                result = train_model(config, train_loader, val_loader, X_val, scaler)
                
                if result is None:
                    print("❌ Training failed or diverged!")
                    mlflow.log_param("status", "FAILED")
                    continue
                
                model = result['model']
                history = result['history']
                
                # Log training curves
                for epoch, (train_loss, val_loss, val_mape) in enumerate(
                    zip(history['train_loss'], history['val_loss'], history['val_price_mape'])
                ):
                    mlflow.log_metrics({
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_price_mape': val_mape,
                    }, step=epoch)
                
                # Evaluate
                print("Evaluating...")
                metrics = evaluate_model(model, X_test, y_test, X_test_original, config)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Save model
                model_path = Path('models/nn') / f"{config['run_name']}.pt"
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(str(model_path), artifact_path="models")
                
                # Prepare results dict
                result_dict = {'run_name': config['run_name']}
                result_dict.update(config)
                result_dict.update(metrics)
                
                results.append(result_dict)
                
                print(f"\n✓ Run {i+1} completed!")
                print(f"  MAPE (overall): {metrics['test_mape_overall']:.4f}%")
                print(f"  Error Smoothness: {metrics['error_smoothness_score']:.6f}")
                print(f"  Worst-case Error: {metrics['worst_case_error']:.4f}%")
                
        except Exception as e:
            print(f"❌ Run {i+1} failed with error: {str(e)}")
            mlflow.log_param("status", f"FAILED: {str(e)}")
            continue
    
    return results, train_loader, val_loader, X_test, y_test, X_test_original, scaler


if __name__ == "__main__":
    print("STARTING STEP 6: ABLATION STUDY")
    
    results, train_loader, val_loader, X_test, y_test, X_test_original, scaler = run_ablation_study()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*70)
    print(results_df.to_string())
    
    print("\n✓ Ablation study complete! Check MLflow for detailed results.")
