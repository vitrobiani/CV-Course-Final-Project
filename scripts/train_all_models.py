#!/usr/bin/env python3
"""
Train all models for comparison.

Runs:
1. YOLO26n (nano - for RPi)
2. YOLO26s (small - for phone)
3. RetinaNet (Focal Loss - for imbalanced data)
4. MobileNet-SSDLite (edge alternative)

Optional:
5. Faster R-CNN (for comparison, but too slow for edge)
"""
import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OUTPUTS_DIR


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Train all models for comparison")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data.yaml")
    parser.add_argument("--epochs-yolo", type=int, default=100,
                        help="Epochs for YOLO models")
    parser.add_argument("--epochs-torch", type=int, default=50,
                        help="Epochs for Torchvision models")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="0",
                        help="Device")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--skip", type=str, nargs="+", default=[],
                        help="Models to skip")
    parser.add_argument("--only", type=str, nargs="+", default=[],
                        help="Only train these models")
    args = parser.parse_args()

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = OUTPUTS_DIR / "model_comparison" / datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).parent

    # Define all models to train
    models = [
        {
            "name": "yolo26n",
            "type": "yolo",
            "description": "YOLO26 Nano (edge - RPi)",
            "cmd": [
                sys.executable, str(scripts_dir / "train_yolo.py"),
                "--data", args.data,
                "--model", "yolo26n.pt",
                "--phase", "baseline",
                "--batch", str(args.batch),
                "--device", args.device,
                "--project", str(output_dir / "yolo"),
            ],
        },
        {
            "name": "yolo26n_aug",
            "type": "yolo",
            "description": "YOLO26 Nano with Augmentation",
            "cmd": [
                sys.executable, str(scripts_dir / "train_yolo.py"),
                "--data", args.data,
                "--model", "yolo26n.pt",
                "--phase", "augmented",
                "--batch", str(args.batch),
                "--device", args.device,
                "--project", str(output_dir / "yolo"),
            ],
        },
        {
            "name": "yolo26s",
            "type": "yolo",
            "description": "YOLO26 Small (phone)",
            "cmd": [
                sys.executable, str(scripts_dir / "train_yolo.py"),
                "--data", args.data,
                "--model", "yolo26s.pt",
                "--phase", "augmented",
                "--batch", str(args.batch),
                "--device", args.device,
                "--project", str(output_dir / "yolo"),
            ],
        },
        {
            "name": "retinanet",
            "type": "torchvision",
            "description": "RetinaNet (Focal Loss - imbalanced data)",
            "cmd": [
                sys.executable, str(scripts_dir / "train_torchvision.py"),
                "--data", args.data,
                "--model", "retinanet",
                "--epochs", str(args.epochs_torch),
                "--batch", str(min(args.batch, 8)),  # RetinaNet needs more memory
                "--device", "cuda" if args.device != "cpu" else "cpu",
                "--output", str(output_dir / "retinanet"),
            ],
        },
        {
            "name": "mobilenetv4",
            "type": "torchvision",
            "description": "MobileNetV4 + RetinaNet head (assignment advanced work)",
            "cmd": [
                sys.executable, str(scripts_dir / "train_torchvision.py"),
                "--data", args.data,
                "--model", "mobilenetv4",
                "--epochs", str(args.epochs_torch),
                "--batch", str(args.batch),
                "--device", "cuda" if args.device != "cpu" else "cpu",
                "--output", str(output_dir / "mobilenetv4"),
            ],
        },
        {
            "name": "mobilenet_ssd",
            "type": "torchvision",
            "description": "MobileNetV3-SSDLite (edge baseline)",
            "cmd": [
                sys.executable, str(scripts_dir / "train_torchvision.py"),
                "--data", args.data,
                "--model", "mobilenet_ssd",
                "--epochs", str(args.epochs_torch),
                "--batch", str(args.batch),
                "--device", "cuda" if args.device != "cpu" else "cpu",
                "--output", str(output_dir / "mobilenet_ssd"),
            ],
        },
        {
            "name": "fasterrcnn",
            "type": "torchvision",
            "description": "Faster R-CNN (comparison only - too slow for edge)",
            "cmd": [
                sys.executable, str(scripts_dir / "train_torchvision.py"),
                "--data", args.data,
                "--model", "fasterrcnn",
                "--epochs", str(args.epochs_torch),
                "--batch", str(min(args.batch, 4)),  # Faster R-CNN needs more memory
                "--device", "cuda" if args.device != "cpu" else "cpu",
                "--output", str(output_dir / "fasterrcnn"),
            ],
        },
    ]

    # Filter models
    if args.only:
        models = [m for m in models if m["name"] in args.only]
    if args.skip:
        models = [m for m in models if m["name"] not in args.skip]

    print(f"\n{'#'*60}")
    print(f"# Training {len(models)} models")
    print(f"# Output: {output_dir}")
    print(f"{'#'*60}")

    for m in models:
        print(f"  - {m['name']}: {m['description']}")

    results = {}

    for model in models:
        success = run_command(model["cmd"], model["description"])
        results[model["name"]] = {
            "success": success,
            "description": model["description"],
            "type": model["type"],
        }

        # Save intermediate results
        results_path = output_dir / "training_status.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")

    for name, result in results.items():
        status = "✓ Completed" if result["success"] else "✗ Failed"
        print(f"  {name}: {status}")

    print(f"\nResults saved to: {output_dir}")
    print("\nNext step: Run model comparison:")
    print(f"  python scripts/compare_models.py --data {args.data} ...")


if __name__ == "__main__":
    main()
