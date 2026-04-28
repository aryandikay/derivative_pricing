"""Neural Network Surrogate with PINN Regularization (Physics-Informed)."""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import mlflow
import mlflow.pytorch
import joblib
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.train_nn import PricingSurrogate, combined_loss, relative_mse

mlflow.set_tracking_uri("./experiments")
mlflow.set_experiment("phase1_surrogate")


# ============================================================================
# PDE RESIDUAL COMPUTATION
# ============================================================================

def bs_pde_residual(model, collocation_points, scaler, device='cpu'):
    """
    Compute Black-Scholes PDE residual at collocation points.
    
    PDE: -dV/dT + 0.5*sigma^2*S/K^2*d2V/dS2 + r*S/K*dV/dS - r*V = 0
    
    All derivatives computed via autograd on normalized inputs.
    
    Parameters
    ----------
    model : nn.Module
        Neural network model
    collocation_points : torch.Tensor
        Collocation points in scaled space, shape (N, 4)
        Columns: [moneyness, T, sigma, r]
    scaler : StandardScaler
        Fitted scaler for inverse transform
    device : str
        Device to use ('cpu' or 'cuda')
    
    Returns
    -------
    torch.Tensor
        Mean squared PDE residual
    """
    # Enable gradient computation
    x = collocation_points.clone().requires_grad_(True)
    x = x.to(device)
    
    # Transform to original scale for PDE coefficients
    x_np = x.detach().cpu().numpy()
    x_orig_np = scaler.inverse_transform(x_np)
    x_orig = torch.from_numpy(x_orig_np).to(device).float()
    
    # Extract variables from original scale
    m = x_orig[:, 0]      # moneyness S/K
    T = x_orig[:, 1]      # time to maturity
    sigma = x_orig[:, 2]  # volatility
    r = x_orig[:, 1]      # risk-free rate (using T index by mistake - FIX THIS)
    
    # Actually get r correctly
    r = x_orig[:, 3]      # risk-free rate
    
    # Evaluate model to get price
    V = model(x)[:, 0]    # Only use price output for PDE
    
    # First derivatives (using automatic differentiation)
    dV_dx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
    
    dV_dT = dV_dx[:, 1]   # dV/dT (time derivative)
    dV_dm = dV_dx[:, 0]   # dV/dm (spatial derivative w.r.t. moneyness)
    
    # Second derivative (d2V/dm2)
    d2V_dm2 = torch.autograd.grad(dV_dm.sum(), x, create_graph=True)[0][:, 0]
    
    # Black-Scholes PDE residual
    # -dV/dT + 0.5*sigma^2*(S/K)^2*d2V/dS2 + r*(S/K)*dV/dS - r*V = 0
    residual = (-dV_dT 
                + 0.5 * sigma**2 * m**2 * d2V_dm2 
                + r * m * dV_dm 
                - r * V)
    
    # Mean squared residual
    pde_loss = torch.mean(residual**2)
    
    return pde_loss


# ============================================================================
# TRAINING WITH PINN REGULARIZATION
# ============================================================================

class PINNTrainer:
    """Train neural network with Physics-Informed Neural Network regularization."""
    
    def __init__(self, model_dir: str = "models/nn", 
                 processed_dir: str = "data/processed"):
        """
        Initialize PINN trainer.
        
        Parameters
        ----------
        model_dir : str
            Directory to save models
        processed_dir : str
            Directory with preprocessed splits
        """
        self.model_dir = Path(model_dir)
        self.processed_dir = Path(processed_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n✓ Using device: {self.device}")
    
    def load_data(self) -> tuple:
        """Load preprocessed splits and scaler."""
        print("\n" + "="*70)
        print("LOADING PREPROCESSED DATA")
        print("="*70)
        
        # Load training data
        train_data = np.load(self.processed_dir / "train.npz")
        X_train = train_data['X_train_scaled'].astype(np.float32)
        y_train = train_data['y_train'].astype(np.float32)
        
        # Load validation data
        val_data = np.load(self.processed_dir / "val.npz")
        X_val = val_data['X_val_scaled'].astype(np.float32)
        y_val = val_data['y_val'].astype(np.float32)
        
        # Load test data
        test_data = np.load(self.processed_dir / "test.npz")
        X_test = test_data['X_test_scaled'].astype(np.float32)
        y_test = test_data['y_test'].astype(np.float32)
        
        # Load scaler
        scaler = joblib.load(Path("models/input_scaler.pkl"))
        
        print(f"\nTrain: X {X_train.shape}, y {y_train.shape}")
        print(f"Val:   X {X_val.shape}, y {y_val.shape}")
        print(f"Test:  X {X_test.shape}, y {y_test.shape}")
        print(f"Scaler: {type(scaler).__name__}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, scaler
    
    def train(self, 
              model_name: str = "pinn-lambda0.01",
              lambda_pde: float = 0.01,
              hidden_dim: int = 128,
              n_layers: int = 4,
              activation: str = 'silu',
              dropout: float = 0.0,
              epochs: int = 200,
              batch_size: int = 1024,
              collocation_batch_size: int = 256,
              lr: float = 1e-3,
              weight_decay: float = 1e-4,
              lambdas: tuple = (1.0, 0.5, 0.1)):
        """
        Train PINN-regularized model with MLflow tracking.
        
        Parameters
        ----------
        model_name : str
            Name of model (for saving and logging)
        lambda_pde : float
            Weight for PDE regularization loss
        hidden_dim : int
            Hidden dimension size
        n_layers : int
            Number of hidden layers
        activation : str
            Activation function
        dropout : float
            Dropout rate
        epochs : int
            Number of training epochs
        batch_size : int
            Data batch size
        collocation_batch_size : int
            Number of collocation points per batch
        lr : float
            Learning rate
        weight_decay : float
            L2 regularization weight
        lambdas : tuple
            Loss weights for (price, delta, gamma)
        """
        
        print("\n" + "="*70)
        print(f"TRAINING: {model_name.upper()}")
        print("="*70)
        
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = self.load_data()
        
        # Create model
        print(f"\n[1] Creating Model")
        print("-" * 70)
        
        model = PricingSurrogate(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            dropout=dropout
        )
        model = model.to(self.device)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model architecture:")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num layers: {n_layers}")
        print(f"  Activation: {activation}")
        print(f"  Total parameters: {n_params:,}")
        
        # Create data loaders
        print(f"\n[2] Creating Data Loaders")
        print("-" * 70)
        
        X_train_tensor = torch.from_numpy(X_train).to(self.device)
        y_train_tensor = torch.from_numpy(y_train).to(self.device)
        X_val_tensor = torch.from_numpy(X_val).to(self.device)
        y_val_tensor = torch.from_numpy(y_val).to(self.device)
        X_test_tensor = torch.from_numpy(X_test).to(self.device)
        y_test_tensor = torch.from_numpy(y_test).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, pin_memory=False)
        
        n_batches = len(train_loader)
        print(f"Train batches: {n_batches} (batch_size={batch_size})")
        print(f"Collocation batch size: {collocation_batch_size}")
        
        # Setup optimizer and scheduler
        print(f"\n[3] Setting up Optimizer & Scheduler")
        print("-" * 70)
        
        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        
        print(f"Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
        print(f"Scheduler: CosineAnnealingLR (T_max={epochs})")
        
        # Training loop with MLflow
        print(f"\n[4] Starting MLflow Run")
        print("-" * 70)
        
        run_name = f"nn-{model_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters
            mlflow.log_params({
                'model_name': model_name,
                'model_type': 'neural_network_pinn',
                'hidden_dim': hidden_dim,
                'n_layers': n_layers,
                'activation': activation,
                'dropout': dropout,
                'n_params': n_params,
                'epochs': epochs,
                'batch_size': batch_size,
                'collocation_batch_size': collocation_batch_size,
                'learning_rate': lr,
                'weight_decay': weight_decay,
                'optimizer': 'AdamW',
                'scheduler': 'CosineAnnealingLR',
                'lambda_price': lambdas[0],
                'lambda_delta': lambdas[1],
                'lambda_gamma': lambdas[2],
                'lambda_pde': lambda_pde,
                'device': str(self.device),
            })
            
            print(f"\nRun name: {run_name}")
            print(f"Lambda_PDE: {lambda_pde}")
            
            print(f"\n[5] Training Loop with PINN Regularization")
            print("-" * 70)
            print(f"{'Epoch':>6} | {'Data Loss':>12} | {'PDE Loss':>12} | "
                  f"{'Total Loss':>12} | {'Val Loss':>12} | {'Time':>6}")
            print("-" * 70)
            
            best_val_loss = float('inf')
            best_epoch = 0
            train_losses = []
            pde_losses = []
            val_losses = []
            
            import time
            start_time = time.time()
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                data_loss_epoch = 0.0
                pde_loss_epoch = 0.0
                total_loss_epoch = 0.0
                
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    
                    # Data loss (labeled points)
                    pred = model(X_batch)
                    data_loss = combined_loss(pred, y_batch, lambdas=lambdas)
                    
                    # PDE loss (unlabeled collocation points)
                    collocation = torch.FloatTensor(collocation_batch_size, 4).uniform_(-2, 2)
                    pde_loss = bs_pde_residual(model, collocation, scaler, device=self.device)
                    
                    # Total loss
                    total_loss = data_loss + lambda_pde * pde_loss
                    
                    # Backward pass
                    total_loss.backward()
                    optimizer.step()
                    
                    data_loss_epoch += data_loss.item()
                    pde_loss_epoch += pde_loss.item()
                    total_loss_epoch += total_loss.item()
                
                data_loss_epoch /= n_batches
                pde_loss_epoch /= n_batches
                total_loss_epoch /= n_batches
                train_losses.append(data_loss_epoch)
                pde_losses.append(pde_loss_epoch)
                
                # Validation phase
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val_tensor)
                    val_loss = combined_loss(val_pred, y_val_tensor, lambdas=lambdas)
                
                val_loss_item = val_loss.item()
                val_losses.append(val_loss_item)
                
                # Learning rate scheduling
                current_lr = optimizer.param_groups[0]['lr']
                scheduler.step()
                
                # Track best model
                if val_loss_item < best_val_loss:
                    best_val_loss = val_loss_item
                    best_epoch = epoch
                
                # Log metrics every epoch
                mlflow.log_metric('data_loss', data_loss_epoch, step=epoch)
                mlflow.log_metric('pde_loss', pde_loss_epoch, step=epoch)
                mlflow.log_metric('total_train_loss', total_loss_epoch, step=epoch)
                mlflow.log_metric('val_loss', val_loss_item, step=epoch)
                mlflow.log_metric('learning_rate', current_lr, step=epoch)
                
                # Print progress
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    elapsed = time.time() - start_time
                    print(f"{epoch+1:6d} | {data_loss_epoch:12.6f} | "
                          f"{pde_loss_epoch:12.6f} | {total_loss_epoch:12.6f} | "
                          f"{val_loss_item:12.6f} | {elapsed/60:6.1f}m")
            
            # Evaluation on test set
            print(f"\n[6] Evaluating on Test Set")
            print("-" * 70)
            
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train_tensor)
                train_eval_loss = combined_loss(train_pred, y_train_tensor, lambdas=lambdas)
                
                val_pred = model(X_val_tensor)
                val_eval_loss = combined_loss(val_pred, y_val_tensor, lambdas=lambdas)
                
                test_pred = model(X_test_tensor)
                test_eval_loss = combined_loss(test_pred, y_test_tensor, lambdas=lambdas)
                
                # Per-output losses
                l_test_price = relative_mse(test_pred[:, 0], y_test_tensor[:, 0])
                l_test_delta = relative_mse(test_pred[:, 1], y_test_tensor[:, 1])
                l_test_gamma = relative_mse(test_pred[:, 2], y_test_tensor[:, 2])
            
            print(f"\nFinal Losses:")
            print(f"  Train: {train_eval_loss.item():.6f}")
            print(f"  Val:   {val_eval_loss.item():.6f}")
            print(f"  Test:  {test_eval_loss.item():.6f}")
            
            print(f"\nPer-Output Losses (Test):")
            print(f"  Price: {l_test_price.item():.6f}")
            print(f"  Delta: {l_test_delta.item():.6f}")
            print(f"  Gamma: {l_test_gamma.item():.6f}")
            
            # Log final metrics
            mlflow.log_metrics({
                'final_train_loss': train_eval_loss.item(),
                'final_val_loss': val_eval_loss.item(),
                'final_test_loss': test_eval_loss.item(),
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'test_loss_price': l_test_price.item(),
                'test_loss_delta': l_test_delta.item(),
                'test_loss_gamma': l_test_gamma.item(),
            })
            
            # Save model
            print(f"\n[7] Saving Model & Artifacts")
            print("-" * 70)
            
            model_path = self.model_dir / f"{model_name}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"✓ Saved model: {model_path}")
            
            # Log model as artifact
            mlflow.pytorch.log_model(model, artifact_path="pytorch_model")
            print(f"✓ Logged PyTorch model to MLflow")
            
            # Save training history
            history = {
                'data_losses': train_losses,
                'pde_losses': pde_losses,
                'val_losses': val_losses,
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss,
                'lambda_pde': lambda_pde,
            }
            history_path = self.model_dir / f"{model_name}_history.pt"
            torch.save(history, history_path)
            mlflow.log_artifact(str(history_path), artifact_path="models")
            print(f"✓ Saved training history: {history_path}")
            
            print(f"\n" + "="*70)
            print(f"✓ TRAINING COMPLETE")
            print("="*70)
            print(f"\nModel saved to: {model_path}")
            print(f"MLflow run: {run_name}")
            print(f"Total training time: {(time.time() - start_time)/60:.1f} minutes")
            
            return model, history


def main():
    """Main entry point for PINN lambda sweep experiments."""
    print("\n" + "="*70)
    print("PINN REGULARIZATION: LAMBDA SWEEP EXPERIMENTS")
    print("="*70)
    
    trainer = PINNTrainer(model_dir="models/nn", 
                          processed_dir="data/processed")
    
    # Lambda sweep experiments
    lambda_values = [0.001, 0.01, 0.1]
    models = {}
    histories = {}
    
    for lambda_pde in lambda_values:
        print(f"\n\n{'='*70}")
        print(f"EXPERIMENT: Lambda_PDE = {lambda_pde}")
        print(f"{'='*70}")
        
        model_name = f"pinn-lambda{lambda_pde}"
        
        model, history = trainer.train(
            model_name=model_name,
            lambda_pde=lambda_pde,
            hidden_dim=128,
            n_layers=4,
            activation='silu',
            dropout=0.0,
            epochs=200,
            batch_size=1024,
            collocation_batch_size=256,
            lr=1e-3,
            weight_decay=1e-4,
            lambdas=(1.0, 0.5, 0.1)
        )
        
        models[lambda_pde] = model
        histories[lambda_pde] = history
    
    print("\n" + "="*70)
    print("PINN LAMBDA SWEEP COMPLETE")
    print("="*70)
    print("\nResults Summary:")
    for lambda_pde in lambda_values:
        best_loss = histories[lambda_pde]['best_val_loss']
        print(f"  Lambda={lambda_pde:5.3f}: Best val loss = {best_loss:.6f}")
    
    print("\nNext steps:")
    print("  1. View results: mlflow ui")
    print("  2. Compare with baseline: baseline test loss = 0.004592")
    print("  3. Analyze PDE constraint effectiveness")


if __name__ == "__main__":
    main()
