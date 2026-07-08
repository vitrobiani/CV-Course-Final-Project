#!/usr/bin/env python3
"""
Prepare YOLO dataset from SAM labels.

Creates train/val/test splits and generates data.yaml for training.
"""
import argparse
import random
import shutil
import sys
from pathlib import Path
from collections import defaultdict
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SAM_LABELS_DIR, FLICKR_IMAGES_DIR, OUTPUTS_DIR


def count_classes(label_path: Path) -> dict:
    """Count class occurrences in a label file."""
    counts = defaultdict(int)
    if not label_path.exists():
        return counts

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                counts[class_id] += 1
    return counts


def get_image_path(stem: str, images_dir: Path) -> Path:
    """Find image file for a label stem."""
    for ext in [".jpg", ".jpeg", ".png"]:
        img_path = images_dir / f"{stem}{ext}"
        if img_path.exists():
            return img_path
    return None


def prepare_dataset(
    labels_dir: Path,
    images_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    min_detections: int = 1,
    seed: int = 42,
    class_names: list = None,
):
    """
    Prepare YOLO dataset with train/val/test splits.

    Args:
        labels_dir: Directory containing label files
        images_dir: Directory containing images
        output_dir: Output directory for the dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        min_detections: Minimum detections per image to include
        seed: Random seed for reproducibility
        class_names: List of class names
    """
    random.seed(seed)

    if class_names is None:
        class_names = ["person", "vehicle"]

    print(f"Preparing dataset from {labels_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    # Find all valid label files (with corresponding images)
    label_files = list(labels_dir.glob("*.txt"))
    valid_samples = []
    class_counts = defaultdict(int)

    print(f"\nScanning {len(label_files)} label files...")

    for label_path in label_files:
        stem = label_path.stem
        img_path = get_image_path(stem, images_dir)

        if img_path is None:
            continue

        counts = count_classes(label_path)
        total_dets = sum(counts.values())

        if total_dets >= min_detections:
            valid_samples.append({
                "stem": stem,
                "label_path": label_path,
                "image_path": img_path,
                "counts": counts,
            })
            for cls_id, count in counts.items():
                class_counts[cls_id] += count

    print(f"Found {len(valid_samples)} valid samples")
    print(f"Class distribution:")
    for cls_id, count in sorted(class_counts.items()):
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        print(f"  {cls_name}: {count} instances")

    # Shuffle and split
    random.shuffle(valid_samples)

    n_total = len(valid_samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_samples = valid_samples[:n_train]
    val_samples = valid_samples[n_train:n_train + n_val]
    test_samples = valid_samples[n_train + n_val:]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    print(f"  Test: {len(test_samples)}")

    # Create directory structure
    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    for split_name, samples in splits.items():
        img_dir = output_dir / split_name / "images"
        lbl_dir = output_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        split_class_counts = defaultdict(int)

        for sample in samples:
            # Copy image
            dst_img = img_dir / sample["image_path"].name
            shutil.copy(sample["image_path"], dst_img)

            # Copy label
            dst_lbl = lbl_dir / sample["label_path"].name
            shutil.copy(sample["label_path"], dst_lbl)

            for cls_id, count in sample["counts"].items():
                split_class_counts[cls_id] += count

        print(f"\n{split_name.upper()} split class distribution:")
        for cls_id, count in sorted(split_class_counts.items()):
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            print(f"  {cls_name}: {count} instances")

    # Create data.yaml
    data_yaml = {
        "path": str(output_dir.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\nCreated data.yaml at {yaml_path}")

    # Create a summary file
    summary = {
        "total_images": len(valid_samples),
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "test_images": len(test_samples),
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "class_counts": {class_names[k]: v for k, v in class_counts.items()},
        "seed": seed,
    }

    summary_path = output_dir / "dataset_summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

    print(f"Created summary at {summary_path}")

    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset")
    parser.add_argument("--labels-dir", type=str, default=None,
                        help="Labels directory (default: SAM labels)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Images directory (default: Flickr)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Training split ratio (default: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Test split ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--name", type=str, default="sam_dataset",
                        help="Dataset name")
    args = parser.parse_args()

    # Set defaults
    labels_dir = Path(args.labels_dir) if args.labels_dir else SAM_LABELS_DIR / "labels"
    images_dir = Path(args.images_dir) if args.images_dir else FLICKR_IMAGES_DIR
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / "datasets" / args.name

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Warning: Ratios sum to {total_ratio}, normalizing...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio

    yaml_path = prepare_dataset(
        labels_dir=labels_dir,
        images_dir=images_dir,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"\n{'='*60}")
    print("Dataset preparation complete!")
    print(f"{'='*60}")
    print(f"\nTo train with Ultralytics YOLO:")
    print(f"  yolo detect train data={yaml_path} model=yolo26n.pt epochs=100")
    print(f"\nOr in Python:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('yolo26n.pt')")
    print(f"  model.train(data='{yaml_path}', epochs=100)")


if __name__ == "__main__":
    main()
