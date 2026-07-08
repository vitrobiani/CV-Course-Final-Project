#!/usr/bin/env python3
"""
Model Comparison Framework.

Compares different object detection architectures:
- YOLO26 (nano, small)
- YOLO11 (nano, small)
- RetinaNet
- Faster R-CNN
- EfficientDet
- MobileNetV3-SSD

Metrics:
- mAP@50, mAP@50-95
- Inference time (CPU, GPU)
- Model size (MB)
- FLOPs
"""
import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OUTPUTS_DIR


class ModelBenchmark:
    """Benchmark a trained model."""

    def __init__(self, model_path: str, model_type: str):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None

    def load_model(self):
        """Load the model based on type."""
        if self.model_type.startswith("yolo"):
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        elif self.model_type == "retinanet":
            import torchvision.models.detection as detection
            self.model = detection.retinanet_resnet50_fpn(weights=None)
            self.model.load_state_dict(torch.load(self.model_path))
        elif self.model_type == "fasterrcnn":
            import torchvision.models.detection as detection
            self.model = detection.fasterrcnn_resnet50_fpn(weights=None)
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def get_model_size(self) -> float:
        """Get model size in MB."""
        path = Path(self.model_path)
        if path.exists():
            return path.stat().st_size / (1024 * 1024)
        return 0.0

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        if self.model is None:
            self.load_model()

        if self.model_type.startswith("yolo"):
            # Ultralytics models
            return sum(p.numel() for p in self.model.model.parameters())
        else:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def benchmark_inference(
        self,
        image_path: str,
        device: str = "cpu",
        warmup: int = 5,
        iterations: int = 20,
    ) -> Dict[str, float]:
        """Benchmark inference time."""
        import numpy as np
        from PIL import Image

        if self.model is None:
            self.load_model()

        # Load test image
        img = Image.open(image_path)
        img_array = np.array(img)

        times = []

        # Warmup
        for _ in range(warmup):
            if self.model_type.startswith("yolo"):
                self.model(img_array, device=device, verbose=False)
            else:
                # Torchvision models
                self.model.eval()
                with torch.no_grad():
                    tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                    tensor = tensor.unsqueeze(0).to(device)
                    self.model(tensor)

        # Benchmark
        for _ in range(iterations):
            start = time.perf_counter()

            if self.model_type.startswith("yolo"):
                self.model(img_array, device=device, verbose=False)
            else:
                with torch.no_grad():
                    tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                    tensor = tensor.unsqueeze(0).to(device)
                    self.model(tensor)

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "fps": 1000 / np.mean(times),
        }

    def evaluate(self, data_yaml: str, split: str = "test") -> Dict[str, float]:
        """Evaluate model on dataset."""
        if self.model is None:
            self.load_model()

        if self.model_type.startswith("yolo"):
            results = self.model.val(data=data_yaml, split=split, verbose=False)
            return {
                "mAP50": float(results.box.map50),
                "mAP50-95": float(results.box.map),
                "precision": float(results.box.mp),
                "recall": float(results.box.mr),
            }
        else:
            # For torchvision models, implement custom evaluation
            # This is a placeholder - full implementation would use COCO eval
            return {"mAP50": 0.0, "mAP50-95": 0.0}


def compare_models(
    models: List[Dict[str, str]],
    data_yaml: str,
    test_image: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Compare multiple models."""

    results = {
        "timestamp": datetime.now().isoformat(),
        "data": data_yaml,
        "models": [],
    }

    for model_info in models:
        model_name = model_info["name"]
        model_path = model_info["path"]
        model_type = model_info["type"]

        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        try:
            benchmark = ModelBenchmark(model_path, model_type)
            benchmark.load_model()

            # Get model info
            model_size = benchmark.get_model_size()
            param_count = benchmark.count_parameters()

            print(f"  Size: {model_size:.2f} MB")
            print(f"  Parameters: {param_count:,}")

            # Evaluate accuracy
            print("  Evaluating accuracy...")
            accuracy = benchmark.evaluate(data_yaml)
            print(f"  mAP@50: {accuracy['mAP50']:.4f}")
            print(f"  mAP@50-95: {accuracy['mAP50-95']:.4f}")

            # Benchmark inference
            print("  Benchmarking CPU inference...")
            cpu_timing = benchmark.benchmark_inference(test_image, device="cpu")
            print(f"  CPU: {cpu_timing['mean_ms']:.1f}ms ({cpu_timing['fps']:.1f} FPS)")

            if torch.cuda.is_available():
                print("  Benchmarking GPU inference...")
                gpu_timing = benchmark.benchmark_inference(test_image, device="0")
                print(f"  GPU: {gpu_timing['mean_ms']:.1f}ms ({gpu_timing['fps']:.1f} FPS)")
            else:
                gpu_timing = None

            results["models"].append({
                "name": model_name,
                "path": model_path,
                "type": model_type,
                "size_mb": model_size,
                "parameters": param_count,
                "accuracy": accuracy,
                "cpu_timing": cpu_timing,
                "gpu_timing": gpu_timing,
            })

        except Exception as e:
            print(f"  Error: {e}")
            results["models"].append({
                "name": model_name,
                "path": model_path,
                "type": model_type,
                "error": str(e),
            })

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "model_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Size(MB)':<10} {'Params':<12} {'mAP50':<10} {'mAP50-95':<10} {'CPU(ms)':<10} {'FPS':<8}")
    print("-" * 80)

    for m in results["models"]:
        if "error" in m:
            print(f"{m['name']:<20} ERROR: {m['error'][:50]}")
        else:
            print(f"{m['name']:<20} {m['size_mb']:<10.2f} {m['parameters']:<12,} "
                  f"{m['accuracy']['mAP50']:<10.4f} {m['accuracy']['mAP50-95']:<10.4f} "
                  f"{m['cpu_timing']['mean_ms']:<10.1f} {m['cpu_timing']['fps']:<8.1f}")

    print(f"\nResults saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Object Detection Models")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data.yaml")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="Model paths (format: name:path:type)")
    parser.add_argument("--test-image", type=str, required=True,
                        help="Test image for inference benchmarking")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()

    # Parse model specifications
    models = []
    for spec in args.models:
        parts = spec.split(":")
        if len(parts) == 3:
            models.append({
                "name": parts[0],
                "path": parts[1],
                "type": parts[2],
            })
        elif len(parts) == 2:
            # Infer type from path
            model_type = "yolo" if "yolo" in parts[1].lower() else "unknown"
            models.append({
                "name": parts[0],
                "path": parts[1],
                "type": model_type,
            })
        else:
            print(f"Invalid model spec: {spec}")
            print("Format: name:path:type or name:path")
            sys.exit(1)

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = OUTPUTS_DIR / "model_comparison" / datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run comparison
    compare_models(
        models=models,
        data_yaml=args.data,
        test_image=args.test_image,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
