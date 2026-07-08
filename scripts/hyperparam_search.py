#!/usr/bin/env python3
"""
Hyperparameter Grid Search for YOLO Training.

Searches over:
- Learning rate
- Batch size
- Optimizer
- Data split ratios
- Model variants
"""
import argparse
import itertools
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OUTPUTS_DIR


def generate_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations from parameter grid."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]


def train_with_params(
    data_yaml: str,
    model: str,
    params: Dict[str, Any],
    project_dir: Path,
    run_name: str,
    epochs: int = 50,
    device: str = "0",
) -> Dict[str, Any]:
    """Train a model with specific hyperparameters."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Ultralytics not installed. Run: pip install ultralytics")

    print(f"\n{'='*60}")
    print(f"Training: {run_name}")
    print(f"Params: {params}")
    print(f"{'='*60}")

    # Load model
    yolo = YOLO(model)

    # Build training config
    train_config = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": params.get("imgsz", 640),
        "batch": params.get("batch", 16),
        "lr0": params.get("lr", 0.01),
        "optimizer": params.get("optimizer", "SGD"),
        "patience": 20,  # Early stopping
        "project": str(project_dir),
        "name": run_name,
        "device": device,
        "exist_ok": True,
        "verbose": False,  # Less output during search
        # Disable heavy augmentations for faster search
        "augment": params.get("augment", False),
    }

    # Add warmup params based on optimizer
    if params.get("optimizer") == "AdamW":
        train_config["warmup_epochs"] = 3
        train_config["warmup_momentum"] = 0.8
    else:
        train_config["warmup_epochs"] = 5

    try:
        # Train
        results = yolo.train(**train_config)

        # Get metrics
        metrics = {
            "mAP50": float(results.box.map50) if hasattr(results, 'box') else 0.0,
            "mAP50-95": float(results.box.map) if hasattr(results, 'box') else 0.0,
            "precision": float(results.box.mp) if hasattr(results, 'box') else 0.0,
            "recall": float(results.box.mr) if hasattr(results, 'box') else 0.0,
        }

        # Validate on test set
        val_results = yolo.val(split="test")
        metrics["test_mAP50"] = float(val_results.box.map50)
        metrics["test_mAP50-95"] = float(val_results.box.map)

        return {
            "status": "completed",
            "params": params,
            "metrics": metrics,
            "run_dir": str(project_dir / run_name),
        }

    except Exception as e:
        return {
            "status": "failed",
            "params": params,
            "error": str(e),
        }


def run_grid_search(
    data_yaml: str,
    model: str,
    param_grid: Dict[str, List[Any]],
    output_dir: Path,
    epochs: int = 50,
    device: str = "0",
    max_trials: int = None,
) -> List[Dict[str, Any]]:
    """Run hyperparameter grid search."""

    combinations = generate_grid(param_grid)
    total = len(combinations)

    if max_trials and max_trials < total:
        import random
        random.shuffle(combinations)
        combinations = combinations[:max_trials]
        print(f"Randomly sampling {max_trials} of {total} combinations")

    print(f"\n{'#'*60}")
    print(f"# Hyperparameter Grid Search")
    print(f"# Model: {model}")
    print(f"# Total combinations: {len(combinations)}")
    print(f"{'#'*60}")
    print(f"\nParameter grid:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, params in enumerate(combinations):
        run_name = f"run_{i:03d}_" + "_".join(f"{k}={v}" for k, v in params.items())
        run_name = run_name[:100]  # Limit length

        print(f"\n[{i+1}/{len(combinations)}] {run_name}")

        result = train_with_params(
            data_yaml=data_yaml,
            model=model,
            params=params,
            project_dir=output_dir,
            run_name=run_name,
            epochs=epochs,
            device=device,
        )

        results.append(result)

        # Save intermediate results
        results_path = output_dir / "grid_search_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    # Analyze results
    completed = [r for r in results if r["status"] == "completed"]
    if completed:
        # Sort by test mAP50-95
        completed.sort(key=lambda x: x["metrics"].get("test_mAP50-95", 0), reverse=True)

        print(f"\n{'='*60}")
        print("Grid Search Results (Top 5)")
        print(f"{'='*60}")

        for i, r in enumerate(completed[:5]):
            print(f"\n#{i+1}: {r['metrics'].get('test_mAP50-95', 0):.4f} mAP@50-95")
            print(f"   Params: {r['params']}")
            print(f"   mAP@50: {r['metrics'].get('test_mAP50', 0):.4f}")

        # Save best config
        best = completed[0]
        best_path = output_dir / "best_config.json"
        with open(best_path, "w") as f:
            json.dump(best, f, indent=2)
        print(f"\nBest config saved to: {best_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Grid Search")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo26n.pt",
                        help="Base model")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Epochs per trial")
    parser.add_argument("--device", type=str, default="0",
                        help="Device")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Maximum number of trials (for random subset)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick search with fewer parameters")
    args = parser.parse_args()

    # Define parameter grid
    if args.quick:
        param_grid = {
            "lr": [0.01, 0.001],
            "batch": [16, 32],
            "optimizer": ["SGD", "AdamW"],
        }
    else:
        param_grid = {
            "lr": [0.1, 0.01, 0.001],
            "batch": [8, 16, 32],
            "optimizer": ["SGD", "AdamW"],
            "imgsz": [640],  # Keep fixed for now
        }

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = OUTPUTS_DIR / "hyperparam_search" / datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run search
    results = run_grid_search(
        data_yaml=args.data,
        model=args.model,
        param_grid=param_grid,
        output_dir=output_dir,
        epochs=args.epochs,
        device=args.device,
        max_trials=args.max_trials,
    )

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
