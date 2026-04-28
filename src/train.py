"""Model training module with MLflow tracking."""

import mlflow
import numpy as np
from pathlib import Path
from src.data import BSDataGenerator, black_scholes_call

mlflow.set_tracking_uri("./experiments")
mlflow.set_experiment("phase1_surrogate")


class ModelTrainer:
    """Base class for surrogate model training with MLflow tracking."""
    
    def __init__(self, model_type: str = "nn"):
        """
        Initialize trainer.
        
        Parameters
        ----------
        model_type : str
            Type of model (nn, gp, mlp, etc.)
        """
        self.model_type = model_type
        self.data_dir = Path("data/processed")
        self.model_dir = Path(f"models/{model_type}")
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def load_training_data(self, dataset_file: str = "bs_dataset.npz"):
        """
        Load Black-Scholes training data.
        
        Parameters
        ----------
        dataset_file : str
            Path to saved dataset
            
        Returns
        -------
        tuple
            (X, y) training data
        """
        filepath = self.data_dir / dataset_file
        
        if not filepath.exists():
            print(f"Generating training data: {dataset_file}")
            generator = BSDataGenerator()
            dataset = generator.generate_dataset(n_samples=10000)
            generator.save_dataset(dataset, dataset_file)
        
        # Load the data
        data = np.load(filepath)
        
        # Stack features
        X = np.column_stack([
            data['features_S_over_K'],
            data['features_T'],
            data['features_r'],
            data['features_sigma'],
        ])
        
        # Target: call price
        y = data['targets_call_price']
        
        return X, y
    
    def train(self, train_params: dict = None):
        """
        Train surrogate model with MLflow tracking.
        
        Parameters
        ----------
        train_params : dict
            Training hyperparameters
        """
        if train_params is None:
            train_params = {
                "model_type": self.model_type,
                "test_split": 0.2,
            }
        
        run_name = f"{self.model_type}-surrogate-training"
        
        with mlflow.start_run(run_name=run_name):
            # Log training parameters
            mlflow.log_params(train_params)
            
            # Load data
            print(f"Loading training data...")
            X, y = self.load_training_data()
            
            # Split data
            test_split = train_params.get("test_split", 0.2)
            n_train = int(len(X) * (1 - test_split))
            
            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]
            
            # Log data metrics
            mlflow.log_metrics({
                "n_train_samples": len(X_train),
                "n_test_samples": len(X_test),
                "input_dim": X_train.shape[1],
                "y_train_mean": float(np.mean(y_train)),
                "y_train_std": float(np.std(y_train)),
                "y_test_mean": float(np.mean(y_test)),
                "y_test_std": float(np.std(y_test)),
            })
            
            print(f"✓ Training set: {len(X_train)} samples")
            print(f"✓ Test set: {len(X_test)} samples")
            print(f"✓ Run logged to MLflow: {run_name}")
            
            # TODO: Implement actual model training here
            # For now, just a placeholder that stores the data
            return X_train, X_test, y_train, y_test


def train_surrogate_model(model_type: str = "nn"):
    """
    Train a surrogate model to approximate Black-Scholes pricing.
    
    Parameters
    ----------
    model_type : str
        Type of surrogate model
    """
    trainer = ModelTrainer(model_type=model_type)
    trainer.train({
        "model_type": model_type,
        "test_split": 0.2,
    })


if __name__ == "__main__":
    print("=" * 70)
    print("SURROGATE MODEL TRAINING")
    print("=" * 70)
    
    # Example: Train neural network surrogate
    train_surrogate_model(model_type="nn")
    
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)
