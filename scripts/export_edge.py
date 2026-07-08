#!/usr/bin/env python3
"""
Export trained models to edge deployment formats.

Supports:
- ONNX (recommended - portable, well-supported)
- TFLite (requires tensorflow)
- NCNN (optimized for ARM/mobile)

For RPi Zero 2 W deployment.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OUTPUTS_DIR


def get_model_size(path: Path) -> float:
    """Get model size in MB."""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    elif path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / (1024 * 1024)
    return 0.0


def export_model(
    model_path: str,
    format: str = "onnx",
    imgsz: int = 320,
    half: bool = False,
    int8: bool = False,
    data_yaml: str = None,
    output_dir: str = None,
    simplify: bool = True,
):
    """Export a YOLO model to edge format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Ultralytics not installed. Run: pip install ultralytics")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\n{'='*60}")
    print(f"Exporting: {model_path.name}")
    print(f"Format: {format}")
    print(f"Image size: {imgsz}")
    print(f"{'='*60}\n")

    # Load model
    model = YOLO(str(model_path))

    # Export arguments
    export_args = {
        "format": format,
        "imgsz": imgsz,
        "simplify": simplify,
    }

    # Half precision (FP16)
    if half:
        export_args["half"] = True
        print("Using FP16 (half precision)")

    # INT8 quantization
    if int8:
        export_args["int8"] = True
        if data_yaml:
            export_args["data"] = data_yaml
            print(f"Using INT8 quantization with calibration data: {data_yaml}")
        else:
            print("Warning: INT8 without calibration data may reduce accuracy")

    # Export
    try:
        exported_path = model.export(**export_args)
        exported_path = Path(exported_path)
    except Exception as e:
        print(f"Export failed: {e}")

        # Suggest alternatives
        if "tensorflow" in str(e).lower():
            print("\nTFLite export requires TensorFlow. Try:")
            print("  pip install tensorflow")
            print("\nOr use ONNX format instead (no extra dependencies):")
            print(f"  python scripts/export_edge.py --model {model_path} --format onnx")
        return None

    # Move to output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if exported_path.is_file():
            new_path = output_dir / exported_path.name
            exported_path.rename(new_path)
            exported_path = new_path
        elif exported_path.is_dir():
            import shutil
            new_path = output_dir / exported_path.name
            if new_path.exists():
                shutil.rmtree(new_path)
            shutil.move(str(exported_path), str(new_path))
            exported_path = new_path

    # Report results
    original_size = get_model_size(model_path)
    exported_size = get_model_size(exported_path)

    print(f"\n{'='*60}")
    print("Export Complete!")
    print(f"{'='*60}")
    print(f"Original model: {model_path}")
    print(f"Original size:  {original_size:.2f} MB")
    print(f"Exported to:    {exported_path}")
    print(f"Exported size:  {exported_size:.2f} MB")
    print(f"Size reduction: {(1 - exported_size/original_size)*100:.1f}%")

    # Format-specific notes
    if format == "onnx":
        print(f"\nTo run on RPi:")
        print(f"  pip install onnxruntime")
        print(f"  # Then use ONNX Runtime for inference")
    elif format == "tflite":
        print(f"\nTo run on RPi:")
        print(f"  pip install tflite-runtime")
        print(f"  # Or: pip install tensorflow")
    elif format == "ncnn":
        print(f"\nTo run on RPi:")
        print(f"  # NCNN is optimized for ARM - best performance")
        print(f"  # See: https://github.com/Tencent/ncnn")

    return exported_path


def validate_exported_model(
    original_path: str,
    exported_path: str,
    test_image: str = None,
):
    """Validate exported model produces similar results."""
    try:
        from ultralytics import YOLO
        import numpy as np
    except ImportError:
        print("Skipping validation - dependencies not available")
        return

    print(f"\n{'='*60}")
    print("Validating exported model...")
    print(f"{'='*60}")

    # Load models
    original = YOLO(original_path)
    exported = YOLO(exported_path)

    # Find a test image if not provided
    if test_image is None:
        # Try to find an image from the dataset
        for pattern in ["outputs/datasets/*/images/val/*", "outputs/datasets/*/images/test/*"]:
            images = list(Path(".").glob(pattern))
            if images:
                test_image = str(images[0])
                break

    if test_image is None:
        print("No test image found, skipping validation")
        return

    print(f"Test image: {test_image}")

    # Run inference
    results_orig = original(test_image, verbose=False)
    results_exp = exported(test_image, verbose=False)

    # Compare results
    boxes_orig = results_orig[0].boxes
    boxes_exp = results_exp[0].boxes

    print(f"Original model detections: {len(boxes_orig)}")
    print(f"Exported model detections: {len(boxes_exp)}")

    if len(boxes_orig) == len(boxes_exp):
        print("Detection count matches!")
    else:
        print("Warning: Detection counts differ (may be due to quantization)")


def export_all_formats(
    model_path: str,
    imgsz: int = 320,
    output_dir: str = None,
    data_yaml: str = None,
):
    """Export model to all supported formats."""

    formats = [
        ("onnx", {"simplify": True}),
        ("ncnn", {}),
    ]

    # Only try TFLite if TensorFlow is available
    try:
        import tensorflow
        formats.append(("tflite", {}))
    except ImportError:
        print("TensorFlow not available, skipping TFLite export")

    results = {}
    for fmt, extra_args in formats:
        print(f"\n{'#'*60}")
        print(f"# Exporting to {fmt.upper()}")
        print(f"{'#'*60}")

        try:
            path = export_model(
                model_path=model_path,
                format=fmt,
                imgsz=imgsz,
                output_dir=output_dir,
                data_yaml=data_yaml,
                **extra_args,
            )
            results[fmt] = {"status": "success", "path": str(path)}
        except Exception as e:
            print(f"Failed to export {fmt}: {e}")
            results[fmt] = {"status": "failed", "error": str(e)}

    # Summary
    print(f"\n{'='*60}")
    print("EXPORT SUMMARY")
    print(f"{'='*60}")
    for fmt, result in results.items():
        if result["status"] == "success":
            size = get_model_size(Path(result["path"]))
            print(f"  {fmt.upper()}: {result['path']} ({size:.2f} MB)")
        else:
            print(f"  {fmt.upper()}: FAILED - {result['error'][:50]}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Export YOLO models for edge deployment")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (.pt file)")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "tflite", "ncnn", "all"],
                        help="Export format (default: onnx)")
    parser.add_argument("--imgsz", type=int, default=320,
                        help="Image size for export (default: 320)")
    parser.add_argument("--half", action="store_true",
                        help="Use FP16 half precision")
    parser.add_argument("--int8", action="store_true",
                        help="Use INT8 quantization (best for edge)")
    parser.add_argument("--data", type=str, default=None,
                        help="data.yaml for INT8 calibration")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--validate", action="store_true",
                        help="Validate exported model")
    parser.add_argument("--test-image", type=str, default=None,
                        help="Test image for validation")
    args = parser.parse_args()

    if args.format == "all":
        export_all_formats(
            model_path=args.model,
            imgsz=args.imgsz,
            output_dir=args.output,
            data_yaml=args.data,
        )
    else:
        exported_path = export_model(
            model_path=args.model,
            format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            int8=args.int8,
            data_yaml=args.data,
            output_dir=args.output,
        )

        if args.validate and exported_path:
            validate_exported_model(
                original_path=args.model,
                exported_path=str(exported_path),
                test_image=args.test_image,
            )


if __name__ == "__main__":
    main()
