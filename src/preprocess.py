"""Dataset preprocessing and train/val/test splitting with MLflow tracking."""

import sys
from pathlib import Path
import numpy as np
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

mlflow.set_tracking_uri("./experiments")
mlflow.set_experiment("phase1_surrogate")


class DataPreprocessor:
    """Preprocess and split Black-Scholes dataset for model training."""
    
    def __init__(self, 
                 raw_dir: str = "data/raw",
                 processed_dir: str = "data/processed",
                 models_dir: str = "models"):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        raw_dir : str
            Directory with raw dataset
        processed_dir : str
            Directory to save processed splits
        models_dir : str
            Directory to save scaler and models
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)
        
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = None
        
    def load_raw_dataset(self, filename: str = "dataset_100k.npz") -> dict:
        """
        Load raw dataset from data/raw/.
        
        Parameters
        ----------
        filename : str
            Name of raw dataset file
            
        Returns
        -------
        dict
            Loaded dataset
        """
        filepath = self.raw_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        print(f"\nLoading dataset: {filepath}")
        data = np.load(filepath)
        
        print(f"✓ Loaded {len(data.files)} arrays")
        for key in sorted(data.files):
            arr = data[key]
            print(f"  {key:30} shape={arr.shape}")
        
        return data
    
    def preprocess_and_split(self,
                            dataset_filename: str = "dataset_100k.npz",
                            target_vars: list = None,
                            val_split: float = 0.15,
                            test_split: float = 0.15,
                            random_seed: int = 42):
        """
        Preprocess dataset and create train/val/test splits.
        
        Strategy:
        - Stratified by moneyness buckets to ensure representative splits
        - Train: 70%, Validation: 15%, Test: 15%
        - Normalize inputs ONLY (not targets)
        - Save scaler for future predictions
        
        Parameters
        ----------
        dataset_filename : str
            Name of raw dataset file
        target_vars : list
            Target variables to use (default: ['call_price', 'delta', 'gamma'])
        val_split : float
            Validation set fraction
        test_split : float
            Test set fraction
        random_seed : int
            Random seed for reproducibility
            
        Returns
        -------
        dict
            Preprocessed data with train/val/test splits
        """
        if target_vars is None:
            target_vars = ['call_price', 'delta', 'gamma']
        
        run_name = f"preprocess-{dataset_filename.replace('.npz', '')}"
        
        with mlflow.start_run(run_name=run_name):
            print("\n" + "="*70)
            print("DATASET PREPROCESSING AND SPLITTING")
            print("="*70)
            
            # ====================================================================
            # STEP 1: LOAD RAW DATASET
            # ====================================================================
            print("\n[1] Loading Raw Dataset")
            print("-" * 70)
            
            data = self.load_raw_dataset(dataset_filename)
            
            n_samples = len(data['features_moneyness'])
            print(f"\n✓ Total samples: {n_samples:,}")
            
            # ====================================================================
            # STEP 2: EXTRACT FEATURES AND TARGETS
            # ====================================================================
            print("\n[2] Extracting Features and Targets")
            print("-" * 70)
            
            # Features: moneyness, T, sigma, r
            X = np.column_stack([
                data['features_moneyness'],
                data['features_T'],
                data['features_sigma'],
                data['features_r'],
            ])
            
            feature_names = ['moneyness', 'T', 'sigma', 'r']
            print(f"\nFeatures ({len(feature_names)}):")
            for i, name in enumerate(feature_names):
                print(f"  {i}: {name:15} range=[{X[:, i].min():.6f}, {X[:, i].max():.6f}]")
            
            # Targets
            y = np.column_stack([data[f'targets_{var}'] for var in target_vars])
            
            print(f"\nTargets ({len(target_vars)}):")
            for i, name in enumerate(target_vars):
                print(f"  {i}: {name:15} range=[{y[:, i].min():.6f}, {y[:, i].max():.6f}]")
            
            # ====================================================================
            # STEP 3: STRATIFIED SPLIT BY MONEYNESS
            # ====================================================================
            print("\n[3] Stratified Split by Moneyness Bucket")
            print("-" * 70)
            
            # Create moneyness buckets for stratification
            moneyness = data['features_moneyness']
            moneyness_bins = [0.7, 0.85, 0.95, 1.05, 1.15, 1.3]
            moneyness_bucket = np.digitize(moneyness, bins=moneyness_bins)
            
            print(f"\nMoneyness buckets: {moneyness_bins}")
            unique, counts = np.unique(moneyness_bucket, return_counts=True)
            print(f"\nBucket distribution:")
            for b, c in zip(unique, counts):
                print(f"  Bucket {b}: {c:6,} samples ({100*c/len(moneyness):5.1f}%)")
            
            # First split: 70% train, 30% temp (for val+test)
            print(f"\nSplit 1: Train/Temp (70/30)")
            X_train, X_temp, y_train, y_temp, bucket_train, bucket_temp = train_test_split(
                X, y, moneyness_bucket,
                test_size=0.30,
                random_state=random_seed,
                stratify=moneyness_bucket
            )
            
            print(f"  Train: {len(X_train):6,} samples ({100*len(X_train)/len(X):5.1f}%)")
            print(f"  Temp:  {len(X_temp):6,} samples ({100*len(X_temp)/len(X):5.1f}%)")
            
            # Second split: 50% validation, 50% test from temp
            print(f"\nSplit 2: Validation/Test (50/50 of temp)")
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=0.50,
                random_state=random_seed,
                stratify=bucket_temp
            )
            
            print(f"  Val:  {len(X_val):6,} samples ({100*len(X_val)/len(X):5.1f}%)")
            print(f"  Test: {len(X_test):6,} samples ({100*len(X_test)/len(X):5.1f}%)")
            
            # ====================================================================
            # STEP 4: FIT SCALER ON TRAINING DATA ONLY
            # ====================================================================
            print("\n[4] Fitting StandardScaler on Training Data")
            print("-" * 70)
            
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            print(f"\nScaler fitted on {len(X_train):,} training samples")
            print(f"\nFeature scaling (mean, std after fit):")
            for i, name in enumerate(feature_names):
                mean = self.scaler.mean_[i]
                std = self.scaler.scale_[i]
                print(f"  {name:15} mean={mean:10.6f}, std={std:10.6f}")
            
            # Apply scaler to validation and test
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            print(f"\n✓ Scaler applied to all splits")
            
            # ====================================================================
            # STEP 5: SAVE SPLITS AND SCALER
            # ====================================================================
            print("\n[5] Saving Splits and Scaler")
            print("-" * 70)
            
            # Save training split
            train_path = self.processed_dir / "train.npz"
            np.savez(train_path,
                    X_train_scaled=X_train_scaled,
                    y_train=y_train,
                    X_train_original=X_train,
                    feature_names=feature_names,
                    target_names=np.array(target_vars))
            print(f"✓ Saved: {train_path}")
            
            # Save validation split
            val_path = self.processed_dir / "val.npz"
            np.savez(val_path,
                    X_val_scaled=X_val_scaled,
                    y_val=y_val,
                    X_val_original=X_val,
                    feature_names=feature_names,
                    target_names=np.array(target_vars))
            print(f"✓ Saved: {val_path}")
            
            # Save test split
            test_path = self.processed_dir / "test.npz"
            np.savez(test_path,
                    X_test_scaled=X_test_scaled,
                    y_test=y_test,
                    X_test_original=X_test,
                    feature_names=feature_names,
                    target_names=np.array(target_vars))
            print(f"✓ Saved: {test_path}")
            
            # Save scaler
            scaler_path = self.models_dir / "input_scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            print(f"✓ Saved: {scaler_path}")
            
            # ====================================================================
            # STEP 6: LOG TO MLFLOW
            # ====================================================================
            print("\n[6] Logging to MLflow")
            print("-" * 70)
            
            # Log parameters
            mlflow.log_params({
                "dataset": dataset_filename,
                "n_samples": n_samples,
                "n_features": len(feature_names),
                "n_targets": len(target_vars),
                "target_vars": ",".join(target_vars),
                "train_ratio": 0.70,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "stratify_by": "moneyness_buckets",
                "random_seed": random_seed,
                "scaler": "StandardScaler",
                "normalize_inputs": True,
                "normalize_targets": False,
            })
            
            # Log metrics
            mlflow.log_metrics({
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "n_features": len(feature_names),
                "n_targets": len(target_vars),
                "X_train_mean": float(np.mean(X_train_scaled)),
                "X_train_std": float(np.std(X_train_scaled)),
                "X_val_mean": float(np.mean(X_val_scaled)),
                "X_val_std": float(np.std(X_val_scaled)),
                "X_test_mean": float(np.mean(X_test_scaled)),
                "X_test_std": float(np.std(X_test_scaled)),
                "y_train_mean": float(np.mean(y_train)),
                "y_train_std": float(np.std(y_train)),
            })
            
            # Log artifacts
            mlflow.log_artifact(str(train_path), artifact_path="splits")
            mlflow.log_artifact(str(val_path), artifact_path="splits")
            mlflow.log_artifact(str(test_path), artifact_path="splits")
            mlflow.log_artifact(str(scaler_path), artifact_path="scalers")
            
            print(f"\n✓ Logged to MLflow")
            print(f"  Artifacts saved to experiments/")
            
            # ====================================================================
            # STEP 7: SUMMARY
            # ====================================================================
            print("\n" + "="*70)
            print("✓ PREPROCESSING COMPLETE")
            print("="*70)
            
            print(f"\nOutput Summary:")
            print(f"  Train set: {len(X_train):6,} samples (70%)")
            print(f"  Val set:   {len(X_val):6,} samples (15%)")
            print(f"  Test set:  {len(X_test):6,} samples (15%)")
            
            print(f"\nInput Features (scaled):")
            for i, name in enumerate(feature_names):
                print(f"  {i}: {name:15} shape=({len(X_train_scaled)},)")
            
            print(f"\nOutput Targets (NOT scaled):")
            for i, name in enumerate(target_vars):
                print(f"  {i}: {name:15} shape=({len(y_train)},)")
            
            print(f"\nSaved Locations:")
            print(f"  Splits:  {self.processed_dir}/")
            print(f"  Scaler:  {scaler_path}")
            
            return {
                'X_train_scaled': X_train_scaled,
                'X_val_scaled': X_val_scaled,
                'X_test_scaled': X_test_scaled,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'X_train_original': X_train,
                'X_val_original': X_val,
                'X_test_original': X_test,
                'scaler': self.scaler,
                'feature_names': feature_names,
                'target_names': target_vars,
            }
    
    def load_preprocessed_dataset(self, split: str = 'train') -> dict:
        """
        Load a preprocessed split.
        
        Parameters
        ----------
        split : str
            'train', 'val', or 'test'
            
        Returns
        -------
        dict
            Split data
        """
        if split.lower() not in ['train', 'val', 'test']:
            raise ValueError("split must be 'train', 'val', or 'test'")
        
        filepath = self.processed_dir / f"{split.lower()}.npz"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Split not found: {filepath}. Run preprocess first.")
        
        data = np.load(filepath, allow_pickle=True)
        
        return {
            f'X_{split}_scaled': data[f'X_{split}_scaled'],
            f'y_{split}': data[f'y_{split}'],
            f'X_{split}_original': data[f'X_{split}_original'],
            'feature_names': data['feature_names'].tolist(),
            'target_names': data['target_names'].tolist(),
        }
    
    def load_scaler(self) -> StandardScaler:
        """Load the fitted scaler."""
        scaler_path = self.models_dir / "input_scaler.pkl"
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}. Run preprocess first.")
        
        return joblib.load(scaler_path)


def main():
    """Main entry point for preprocessing."""
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE")
    print("="*70)
    
    preprocessor = DataPreprocessor()
    
    # Run preprocessing
    splits = preprocessor.preprocess_and_split(
        dataset_filename="dataset_100k.npz",
        target_vars=['call_price', 'delta', 'gamma'],
        random_seed=42
    )
    
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE COMPLETE")
    print("="*70)
    print("\nDirectory Structure:")
    print(f"  data/processed/")
    print(f"    ├── train.npz    (70,000 samples)")
    print(f"    ├── val.npz      (15,000 samples)")
    print(f"    └── test.npz     (15,000 samples)")
    print(f"\n  models/")
    print(f"    └── input_scaler.pkl")
    print("\nNext: Train surrogate models")
    print("  python -m src.train")


if __name__ == "__main__":
    main()
