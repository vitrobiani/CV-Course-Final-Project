#!/usr/bin/env python3
"""
Run Test-Time Augmentation (TTA) validation.

TTA tests label consistency by applying transformations (flips, etc.)
and checking if the model produces consistent predictions.

Usage:
    python scripts/run_tta_validation.py --labeler sam|florence [--sample-size N]
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SAM_LABELS_DIR, FLORENCE_LABELS_DIR, FLICKR_IMAGES_DIR, COMPARISONS_DIR
from src.utils.dataset_utils import BoundingBox
from src.validation.tta_validator import TTAValidator
from src.labeling.sam_labeler import create_sam_labeler
from src.labeling.florence_labeler import create_florence_labeler


def load_labels_from_dir(labels_dir: Path) -> dict:
    """Load all labels from a directory."""
    labels = {}
    for label_file in labels_dir.glob("*.txt"):
        img_name = label_file.stem
        boxes = []
        with open(label_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        box = BoundingBox.from_yolo(line.strip())
                        box.class_name = "person" if box.class_id == 0 else "vehicle"
                        boxes.append(box)
                    except Exception:
                        pass
        labels[img_name] = boxes
    return labels


def main():
    parser = argparse.ArgumentParser(description="Run TTA validation")
    parser.add_argument("--labeler", type=str, required=True, choices=["sam", "florence"],
                        help="Which labeler to use")
    parser.add_argument("--sample-size", type=int, default=100,
                        help="Number of images to validate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Setup
    if args.labeler == "sam":
        labels_dir = SAM_LABELS_DIR / "labels"
        labeler = create_sam_labeler(device=args.device)
    else:
        labels_dir = FLORENCE_LABELS_DIR / "labels"
        labeler = create_florence_labeler(device=args.device)

    # Load existing labels
    if not labels_dir.exists():
        logger.error(f"Labels not found: {labels_dir}")
        sys.exit(1)

    labels = load_labels_from_dir(labels_dir)
    logger.info(f"Loaded {len(labels)} labels")

    # Load model
    logger.info(f"Loading {args.labeler} model...")
    labeler.load_model()

    # Run TTA validation
    logger.info("Running TTA validation...")
    validator = TTAValidator(labeler)
    results = validator.validate_dataset(
        labels=labels,
        images_dir=FLICKR_IMAGES_DIR,
        sample_size=args.sample_size,
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"TTA Validation Results ({args.labeler.upper()})")
    print("=" * 60)
    print(f"Images validated: {results['images_validated']}")
    print(f"Average consistency: {results['average_consistency']:.4f}")
    print(f"Total stable boxes: {results['total_stable_boxes']}")
    print(f"Total unstable boxes: {results['total_unstable_boxes']}")
    print(f"Stability rate: {results['stability_rate']:.4f}")
    print(f"\nLow consistency images ({len(results['low_consistency_images'])}):")
    for img in results['low_consistency_images'][:10]:
        print(f"  - {img}")
    if len(results['low_consistency_images']) > 10:
        print(f"  ... and {len(results['low_consistency_images']) - 10} more")

    # Save results
    import json
    output_file = COMPARISONS_DIR / f"tta_validation_{args.labeler}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
