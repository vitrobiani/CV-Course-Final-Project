#!/usr/bin/env python3
"""
YOLO Training Pipeline with validation phases.

Supports YOLO26, YOLO11, and other Ultralytics models.
Implements the training guidelines:
- Phase A: Overfit single batch (sanity check)
- Phase B: 10% data run (scaling test)
- Phase C: Full training without augmentations (baseline)
- Phase D: Full training with augmentations (comparison)
"""
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OUTPUTS_DIR


def get_training_config(phase: str, model: str, data_yaml: str, **kwargs):
    """Get training configuration for each phase."""

    base_config = {
        "model": model,
        "data": data_yaml,
        "device": kwargs.get("device", "0"),
        "workers": kwargs.get("workers", 4),  # Reduced from 8 to avoid RAM bottleneck
        "cache": kwargs.get("cache", False),  # Cache images in RAM if True
        "exist_ok": True,
        "verbose": True,
    }

    if phase == "overfit":
        # Phase A: Overfit single batch
        return {
            **base_config,
            "epochs": kwargs.get("epochs", 50),
            "batch": kwargs.get("batch", 16),
            "imgsz": kwargs.get("imgsz", 640),
            "patience": 0,  # No early stopping
            "augment": False,
            "mosaic": 0.0,
            "mixup": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "fraction": 0.01,  # ~1% of data (small batch)
            "name": f"{Path(model).stem}_overfit",
        }

    elif phase == "quick":
        # Phase B: 10% data run
        return {
            **base_config,
            "epochs": kwargs.get("epochs", 20),
            "batch": kwargs.get("batch", 16),
            "imgsz": kwargs.get("imgsz", 640),
            "patience": 10,
            "augment": False,
            "fraction": 0.1,  # 10% of data
            "name": f"{Path(model).stem}_quick",
        }

    elif phase == "baseline":
        # Phase C: Full training without augmentations
        return {
            **base_config,
            "epochs": kwargs.get("epochs", 100),
            "batch": kwargs.get("batch", 16),
            "imgsz": kwargs.get("imgsz", 640),
            "patience": kwargs.get("patience", 30),
            "augment": False,
            "mosaic": 0.0,
            "mixup": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "name": f"{Path(model).stem}_baseline",
        }

    elif phase == "augmented":
        # Phase D: Full training with augmentations
        return {
            **base_config,
            "epochs": kwargs.get("epochs", 100),
            "batch": kwargs.get("batch", 16),
            "imgsz": kwargs.get("imgsz", 640),
            "patience": kwargs.get("patience", 30),
            "augment": True,
            # Safe augmentations for person/vehicle
            "mosaic": 1.0,
            "mixup": 0.1,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "flipud": 0.0,  # Don't flip vertically (cars/people don't go upside down)
            "fliplr": 0.5,  # Horizontal flip is safe
            "degrees": 0.0,  # No rotation (keep objects upright)
            "translate": 0.1,
            "scale": 0.5,
            "name": f"{Path(model).stem}_augmented",
        }

    elif phase == "full":
        # Full training with custom settings
        return {
            **base_config,
            "epochs": kwargs.get("epochs", 100),
            "batch": kwargs.get("batch", 16),
            "imgsz": kwargs.get("imgsz", 640),
            "patience": kwargs.get("patience", 30),
            "augment": kwargs.get("augment", True),
            "name": kwargs.get("name", f"{Path(model).stem}_full"),
        }

    else:
        raise ValueError(f"Unknown phase: {phase}")


def train_model(config: dict, project_dir: Path):
    """Train a YOLO model with the given configuration."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Ultralytics not installed. Run: pip install ultralytics")

    print(f"\n{'='*60}")
    print(f"Training: {config.get('name', 'model')}")
    print(f"{'='*60}")
    print(f"Model: {config['model']}")
    print(f"Data: {config['data']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch: {config['batch']}")
    print(f"Augment: {config.get('augment', False)}")
    print()

    # Load model
    model = YOLO(config.pop("model"))

    # Set project directory
    config["project"] = str(project_dir)

    # Train
    results = model.train(**config)

    return results, model


def run_validation_pipeline(
    data_yaml: str,
    model: str = "yolo26n.pt",
    project_dir: Path = None,
    device: str = "0",
    skip_phases: list = None,
    workers: int = 4,
):
    """
    Run the full validation pipeline.

    Phase A: Overfit single batch
    Phase B: 10% quick run
    Phase C: Baseline (no augmentation)
    Phase D: With augmentation
    """
    if project_dir is None:
        project_dir = OUTPUTS_DIR / "training" / datetime.now().strftime("%Y%m%d_%H%M%S")

    project_dir.mkdir(parents=True, exist_ok=True)
    skip_phases = skip_phases or []

    results_log = {
        "model": model,
        "data": data_yaml,
        "timestamp": datetime.now().isoformat(),
        "phases": {},
    }

    phases = [
        ("overfit", "Phase A: Overfit single batch"),
        ("quick", "Phase B: 10% quick run"),
        ("baseline", "Phase C: Baseline (no augmentation)"),
        ("augmented", "Phase D: With augmentation"),
    ]

    for phase_name, description in phases:
        if phase_name in skip_phases:
            print(f"\nSkipping {description}")
            continue

        print(f"\n{'#'*60}")
        print(f"# {description}")
        print(f"{'#'*60}")

        config = get_training_config(
            phase=phase_name,
            model=model,
            data_yaml=data_yaml,
            device=device,
            workers=workers,
        )

        try:
            results, trained_model = train_model(config, project_dir)

            # Log results
            results_log["phases"][phase_name] = {
                "status": "completed",
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
            }

            # Validate on test set if available
            if phase_name in ["baseline", "augmented"]:
                print(f"\nValidating {phase_name} model on test set...")
                val_results = trained_model.val(split="test")
                results_log["phases"][phase_name]["test_metrics"] = {
                    "mAP50": float(val_results.box.map50),
                    "mAP50-95": float(val_results.box.map),
                }

        except Exception as e:
            print(f"Error in {phase_name}: {e}")
            results_log["phases"][phase_name] = {
                "status": "failed",
                "error": str(e),
            }

    # Save results log
    log_path = project_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(results_log, f, indent=2)

    print(f"\n{'='*60}")
    print("Training pipeline complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {project_dir}")
    print(f"Log file: {log_path}")

    return results_log


def main():
    parser = argparse.ArgumentParser(description="YOLO Training Pipeline")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo26n.pt",
                        help="Model to train (default: yolo26n.pt)")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "overfit", "quick", "baseline", "augmented", "full"],
                        help="Training phase to run")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (overrides phase default)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")
    parser.add_argument("--device", type=str, default="0",
                        help="Device (0 for GPU, cpu for CPU)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of data loader workers (reduce if RAM limited)")
    parser.add_argument("--cache", action="store_true",
                        help="Cache images in RAM for faster training")
    parser.add_argument("--project", type=str, default=None,
                        help="Project directory for outputs")
    parser.add_argument("--name", type=str, default=None,
                        help="Run name")
    parser.add_argument("--skip", type=str, nargs="+", default=[],
                        help="Phases to skip")
    args = parser.parse_args()

    # Set project directory
    if args.project:
        project_dir = Path(args.project)
    else:
        project_dir = OUTPUTS_DIR / "training" / datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.phase == "all":
        # Run full validation pipeline
        run_validation_pipeline(
            data_yaml=args.data,
            model=args.model,
            project_dir=project_dir,
            device=args.device,
            skip_phases=args.skip,
            workers=args.workers,
        )
    else:
        # Run single phase - only pass epochs if explicitly provided
        kwargs = {
            "device": args.device,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "workers": args.workers,
        }
        if args.cache:
            kwargs["cache"] = "ram"
        if args.epochs:
            kwargs["epochs"] = args.epochs
        if args.name:
            kwargs["name"] = args.name

        config = get_training_config(
            phase=args.phase,
            model=args.model,
            data_yaml=args.data,
            **kwargs,
        )

        train_model(config, project_dir)


if __name__ == "__main__":
    main()
