"""Neural Network Surrogate Model Training with MLflow Tracking."""

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

mlflow.set_tracking_uri("./experiments")
mlflow.set_experiment("phase1_surrogate")


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class PricingSurrogate(nn.Module):
    """
    Baseline neural network surrogate for option pricing.
    
    Architecture:
    - Input layer: 4 features (moneyness, T, sigma, r)
    - Hidden layers: n_layers × hidden_dim units with SiLU activation
    - Output layer: 3 targets (price, delta, gamma) with bounds enforcement
    
    Output bounds:
    - Price: [0, 1] via sigmoid
    - Delta: [0, 1] via sigmoid
    - Gamma: [0, inf) via ReLU
    """
    
    def __init__(self, hidden_dim: int = 128, n_layers: int = 4, 
                 activation: str = 'silu', dropout: float = 0.0):
        """
        Initialize PricingSurrogate model.
        
        Parameters
        ----------
        hidden_dim : int
            Dimension of hidden layers
        n_layers : int
            Number of hidden layers
        activation : str
            Activation function ('silu' or 'relu')
        dropout : float
            Dropout rate (0.0 = no dropout)
        """
        super().__init__()
        
        # Choose activation function
        if activation == 'silu':
            act = nn.SiLU()
        elif activation == 'relu':
            act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build hidden layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(4, hidden_dim))
        layers.append(act)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer (unbounded)
        layers.append(nn.Linear(hidden_dim, 3))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with output bounds.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch_size, 4)
        
        Returns
        -------
        torch.Tensor
            Output predictions with bounds: [0,1], [0,1], [0,inf)
        """
        out = self.net(x)
        
        # Apply output constraints
        # Price (sigmoid to [0, 1])
        out[:, 0] = torch.sigmoid(out[:, 0])
        
        # Delta (sigmoid to [0, 1])
        out[:, 1] = torch.sigmoid(out[:, 1])
        
        # Gamma (ReLU to [0, inf))
        out[:, 2] = torch.relu(out[:, 2])
        
        return out


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def relative_mse(pred: torch.Tensor, target: torch.Tensor, 
                 epsilon: float = 1e-2) -> torch.Tensor:
    """
    Relative Mean Squared Error with robust epsilon.
    
    Useful for outputs with different scales (price, delta, gamma).
    Normalized by target magnitude to make loss scale-independent.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted values
    target : torch.Tensor
        Target values
    epsilon : float
        Small constant to avoid division by zero (default 1e-2 for stability)
    
    Returns
    -------
    torch.Tensor
        Relative MSE loss (scalar)
    """
    return torch.mean(((pred - target) / (torch.abs(target) + epsilon))**2)


def combined_loss(pred: torch.Tensor, target: torch.Tensor, 
                  lambdas: tuple = (1.0, 0.5, 0.1)) -> torch.Tensor:
    """
    Combined loss: weighted sum of relative MSE for each output.
    
    Parameters
    ----------
    pred : torch.Tensor
        Predicted values, shape (batch_size, 3)
    target : torch.Tensor
        Target values, shape (batch_size, 3)
    lambdas : tuple
        Weights for (price, delta, gamma) losses
    
    Returns
    -------
    torch.Tensor
        Combined loss (scalar)
    """
    l_price = relative_mse(pred[:, 0], target[:, 0])
    l_delta = relative_mse(pred[:, 1], target[:, 1])
    l_gamma = relative_mse(pred[:, 2], target[:, 2])
    
    return lambdas[0] * l_price + lambdas[1] * l_delta + lambdas[2] * l_gamma


# ============================================================================
# TRAINING
# ============================================================================

class NNTrainer:
    """Train neural network surrogate model with MLflow tracking."""
    
    def __init__(self, model_dir: str = "models/nn", 
                 processed_dir: str = "data/processed"):
        """
        Initialize trainer.
        
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
        """
        Load preprocessed train/val/test splits.
        
        Returns
        -------
        tuple
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
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
        
        print(f"\nTrain: X {X_train.shape}, y {y_train.shape}")
        print(f"Val:   X {X_val.shape}, y {y_val.shape}")
        print(f"Test:  X {X_test.shape}, y {y_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train(self, 
              model_name: str = "baseline",
              hidden_dim: int = 128,
              n_layers: int = 4,
              activation: str = 'silu',
              dropout: float = 0.0,
              epochs: int = 200,
              batch_size: int = 1024,
              lr: float = 1e-3,
              weight_decay: float = 1e-4,
              lambdas: tuple = (1.0, 0.5, 0.1)):
        """
        Train the baseline NN model with MLflow tracking.
        
        Parameters
        ----------
        model_name : str
            Name of model (for saving and logging)
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
            Batch size
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
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()
        
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
        print(f"  Dropout: {dropout}")
        print(f"  Total parameters: {n_params:,}")
        print(model)
        
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
                'model_type': 'neural_network',
                'hidden_dim': hidden_dim,
                'n_layers': n_layers,
                'activation': activation,
                'dropout': dropout,
                'n_params': n_params,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': lr,
                'weight_decay': weight_decay,
                'optimizer': 'AdamW',
                'scheduler': 'CosineAnnealingLR',
                'lambda_price': lambdas[0],
                'lambda_delta': lambdas[1],
                'lambda_gamma': lambdas[2],
                'device': str(self.device),
            })
            
            print(f"\nRun name: {run_name}")
            print(f"\n[5] Training Loop")
            print("-" * 70)
            print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | "
                  f"{'LR':>10} | {'Time':>6}")
            print("-" * 70)
            
            best_val_loss = float('inf')
            best_epoch = 0
            train_losses = []
            val_losses = []
            
            import time
            start_time = time.time()
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss_epoch = 0.0
                
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    pred = model(X_batch)
                    loss = combined_loss(pred, y_batch, lambdas=lambdas)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    train_loss_epoch += loss.item()
                
                train_loss_epoch /= n_batches
                train_losses.append(train_loss_epoch)
                
                # Validation phase
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val_tensor)
                    val_loss = combined_loss(val_pred, y_val_tensor, 
                                            lambdas=lambdas)
                
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
                mlflow.log_metric('train_loss', train_loss_epoch, step=epoch)
                mlflow.log_metric('val_loss', val_loss_item, step=epoch)
                mlflow.log_metric('learning_rate', current_lr, step=epoch)
                
                # Print progress
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    elapsed = time.time() - start_time
                    print(f"{epoch+1:6d} | {train_loss_epoch:12.6f} | "
                          f"{val_loss_item:12.6f} | {current_lr:10.2e} | "
                          f"{elapsed/60:6.1f}m")
            
            # Evaluation on test set
            print(f"\n[6] Evaluating on Test Set")
            print("-" * 70)
            
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train_tensor)
                train_eval_loss = combined_loss(train_pred, y_train_tensor, 
                                               lambdas=lambdas)
                
                val_pred = model(X_val_tensor)
                val_eval_loss = combined_loss(val_pred, y_val_tensor, 
                                             lambdas=lambdas)
                
                test_pred = model(X_test_tensor)
                test_eval_loss = combined_loss(test_pred, y_test_tensor, 
                                              lambdas=lambdas)
                
                # Per-output losses
                l_train_price = relative_mse(train_pred[:, 0], y_train_tensor[:, 0])
                l_train_delta = relative_mse(train_pred[:, 1], y_train_tensor[:, 1])
                l_train_gamma = relative_mse(train_pred[:, 2], y_train_tensor[:, 2])
                
                l_val_price = relative_mse(val_pred[:, 0], y_val_tensor[:, 0])
                l_val_delta = relative_mse(val_pred[:, 1], y_val_tensor[:, 1])
                l_val_gamma = relative_mse(val_pred[:, 2], y_val_tensor[:, 2])
                
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
                'val_loss_price': l_val_price.item(),
                'val_loss_delta': l_val_delta.item(),
                'val_loss_gamma': l_val_gamma.item(),
                'train_loss_price': l_train_price.item(),
                'train_loss_delta': l_train_delta.item(),
                'train_loss_gamma': l_train_gamma.item(),
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
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss,
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
    """Main entry point for baseline NN training."""
    print("\n" + "="*70)
    print("BASELINE NEURAL NETWORK TRAINING")
    print("="*70)
    
    trainer = NNTrainer(model_dir="models/nn", 
                        processed_dir="data/processed")
    
    # Train baseline model
    model, history = trainer.train(
        model_name="baseline",
        hidden_dim=128,
        n_layers=4,
        activation='silu',
        dropout=0.0,
        epochs=200,
        batch_size=1024,
        lr=1e-3,
        weight_decay=1e-4,
        lambdas=(1.0, 0.5, 0.1)
    )
    
    print("\n" + "="*70)
    print("BASELINE TRAINING PIPELINE COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. View results: mlflow ui")
    print("  2. Train improved models: python -m src.train_nn_improvements")


if __name__ == "__main__":
    main()
