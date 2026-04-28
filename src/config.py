"""Configuration and project initialization."""

import mlflow
from pathlib import Path

mlflow.set_tracking_uri("./experiments")
mlflow.set_experiment("phase1_surrogate")


def initialize_project():
    """Initialize project structure and MLflow tracking."""
    print("=" * 70)
    print("INITIALIZING PHASE 1 SURROGATE PRICING PROJECT")
    print("=" * 70)
    
    # Create necessary directories
    dirs = [
        "data/raw",
        "data/processed",
        "models/nn",
        "models/gp",
        "outputs",
        "experiments",
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory: {dir_path}")
    
    # Setup MLflow
    print(f"\n✓ MLflow tracking URI: ./experiments")
    print(f"✓ Experiment name: phase1_surrogate")
    
    # Create .gitignore if needed
    gitignore_path = Path(".gitignore")
    gitignore_entries = [
        "mlruns/",
        ".mlflow/",
        "*.egg-info/",
        "__pycache__/",
        "*.pyc",
        ".venv/",
        "*.npz",
    ]
    
    if gitignore_path.exists():
        with open(gitignore_path, "r") as f:
            current = f.read()
        
        with open(gitignore_path, "a") as f:
            for entry in gitignore_entries:
                if entry not in current:
                    f.write(f"{entry}\n")
                    print(f"✓ Added to .gitignore: {entry}")
    
    print("\n" + "=" * 70)
    print("PROJECT INITIALIZATION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Generate data: python -m src.data")
    print("  2. View MLflow: mlflow ui")
    print("  3. Train model: python -m src.train")


if __name__ == "__main__":
    initialize_project()
