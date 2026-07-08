#!/usr/bin/env python3
"""
Run TTA and ensemble validation on labeled dataset.

This validates the created labels using:
1. Test-Time Augmentation (TTA) - checks if predictions are consistent under transformations
2. Ensemble agreement - checks agreement between SAM and Florence

Usage:
    python scripts/run_validation.py [--method tta|ensemble|both]
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    SAM_LABELS_DIR,
    FLORENCE_LABELS_DIR,
    FLICKR_IMAGES_DIR,
    COMPARISONS_DIR,
)
from src.utils.dataset_utils import BoundingBox, save_yolo_labels
from src.validation.ensemble import EnsembleValidator, create_ensemble_dataset


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
    parser = argparse.ArgumentParser(description="Validate created labels")
    parser.add_argument("--method", type=str, default="ensemble",
                        choices=["tta", "ensemble", "both"],
                        help="Validation method")
    parser.add_argument("--min-agreement", type=float, default=0.5,
                        help="Minimum agreement score for ensemble (0-1)")
    parser.add_argument("--output-ensemble", action="store_true",
                        help="Output combined ensemble labels")
    parser.add_argument("--sam-dir", type=str, default=None)
    parser.add_argument("--florence-dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Setup paths
    sam_labels_dir = Path(args.sam_dir) if args.sam_dir else SAM_LABELS_DIR / "labels"
    florence_labels_dir = Path(args.florence_dir) if args.florence_dir else FLORENCE_LABELS_DIR / "labels"

    # Load labels
    logger.info("Loading labels...")
    sam_labels = load_labels_from_dir(sam_labels_dir) if sam_labels_dir.exists() else {}
    florence_labels = load_labels_from_dir(florence_labels_dir) if florence_labels_dir.exists() else {}

    if not sam_labels and not florence_labels:
        logger.error("No labels found!")
        sys.exit(1)

    logger.info(f"Loaded {len(sam_labels)} SAM labels, {len(florence_labels)} Florence labels")

    # ========================================
    # Ensemble Validation
    # ========================================
    if args.method in ["ensemble", "both"]:
        logger.info("=" * 60)
        logger.info("Running Ensemble Validation")
        logger.info("=" * 60)

        if not sam_labels or not florence_labels:
            logger.warning("Both SAM and Florence labels required for ensemble validation")
        else:
            validator = EnsembleValidator()
            results = validator.validate_dataset(sam_labels, florence_labels)

            # Print results
            print("\nEnsemble Validation Results:")
            print("-" * 40)
            print(f"Total images: {results['total_images']}")
            print(f"Agreement boxes: {results['total_agreement_boxes']}")
            print(f"SAM-only boxes: {results['total_sam_only_boxes']}")
            print(f"Florence-only boxes: {results['total_florence_only_boxes']}")
            print(f"Average agreement score: {results['average_agreement_score']:.4f}")
            print(f"Overall agreement rate: {results['agreement_rate']:.4f}")
            print(f"High agreement images (>=0.8): {results['high_agreement_images']}")
            print(f"Low agreement images (<0.3): {results['low_agreement_images']}")

            # Output combined labels if requested
            if args.output_ensemble:
                logger.info("Creating ensemble dataset...")
                ensemble_labels = create_ensemble_dataset(
                    sam_labels,
                    florence_labels,
                    method="wbf",
                    min_agreement=args.min_agreement,
                )

                output_dir = COMPARISONS_DIR / "ensemble_labels"
                save_yolo_labels(ensemble_labels, output_dir)
                logger.info(f"Saved {len(ensemble_labels)} ensemble labels to {output_dir}")

    # ========================================
    # TTA Validation (requires model)
    # ========================================
    if args.method in ["tta", "both"]:
        logger.info("=" * 60)
        logger.info("TTA Validation")
        logger.info("=" * 60)

        logger.info("TTA validation requires a loaded model.")
        logger.info("Use run_tta_validation.py with --labeler flag to run TTA.")

        # TTA would require loading the model and re-running predictions
        # This is expensive, so we provide a separate script for it
        print("\nTo run TTA validation:")
        print("  python scripts/run_tta_validation.py --labeler sam --sample-size 100")
        print("  python scripts/run_tta_validation.py --labeler florence --sample-size 100")

    logger.info("\nValidation complete!")


if __name__ == "__main__":
    main()
