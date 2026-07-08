#!/usr/bin/env python3
"""
Quick demo script - runs object detection on an image and saves the result.

Usage:
    python scripts/demo.py                          # Use defaults
    python scripts/demo.py --image path/to/img.jpg  # Custom image
    python scripts/demo.py --model retinanet        # Different model

Available models: yolo26n, yolo26s, retinanet
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_yolo(model_path: Path, image_path: Path, output_path: Path, conf: float = 0.25):
    """Run YOLO inference."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = model(str(image_path), conf=conf, verbose=False)[0]

    # Save with bounding boxes drawn
    result_img = results.plot()

    from PIL import Image
    img = Image.fromarray(result_img[..., ::-1])  # BGR to RGB
    img.save(output_path)

    return len(results.boxes)


def run_retinanet(model_path: Path, image_path: Path, output_path: Path, conf: float = 0.25):
    """Run RetinaNet inference."""
    import torch
    from PIL import Image, ImageDraw
    from torchvision.transforms import functional as F
    from torchvision.models.detection import retinanet_resnet50_fpn_v2

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = retinanet_resnet50_fpn_v2(num_classes=3)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)[0]

    # Draw boxes
    draw = ImageDraw.Draw(image)
    class_names = {1: "person", 2: "vehicle"}
    colors = {1: "#3b82f6", 2: "#f97316"}

    count = 0
    for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
        if score > conf and label.item() in class_names:
            x1, y1, x2, y2 = box.cpu().numpy()
            color = colors[label.item()]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            text = f"{class_names[label.item()]} {score:.0%}"
            draw.text((x1, y1 - 15), text, fill=color)
            count += 1

    image.save(output_path)
    return count


def main():
    parser = argparse.ArgumentParser(description="Run object detection demo")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image (default: sample from dataset)")
    parser.add_argument("--model", type=str, default="yolo26n",
                        choices=["yolo26n", "yolo26s", "retinanet"],
                        help="Model to use (default: yolo26n)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: outputs/demo_result.jpg)")
    args = parser.parse_args()

    # Find default image if not specified
    if args.image is None:
        sample_images = list((PROJECT_ROOT / "outputs/datasets/sam_12k/test/images").glob("*.jpg"))
        if sample_images:
            args.image = str(sample_images[0])
        else:
            print("Error: No sample images found. Please specify --image")
            sys.exit(1)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Set output path
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = Path(args.output) if args.output else output_dir / "demo_result.jpg"

    # Find model
    models_dir = PROJECT_ROOT / "best_models"
    model_map = {
        "yolo26n": models_dir / "yolo26n_augmented.pt",
        "yolo26s": models_dir / "yolo26s_augmented.pt",
        "retinanet": models_dir / "retinanet_best.pt",
    }

    model_path = model_map[args.model]
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    print(f"Image: {image_path}")
    print(f"Model: {args.model} ({model_path.name})")
    print(f"Confidence: {args.conf}")
    print()

    # Run inference
    if args.model in ["yolo26n", "yolo26s"]:
        detections = run_yolo(model_path, image_path, output_path, args.conf)
    else:
        detections = run_retinanet(model_path, image_path, output_path, args.conf)

    print(f"Detections: {detections}")
    print(f"Result saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
