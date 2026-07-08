#!/usr/bin/env python3
"""
Label the Roboflow dataset with SAM and/or Florence for ground truth evaluation.

This runs the labelers on the Roboflow train set so we can compare
their predictions against the human-annotated ground truth.

Usage:
    python scripts/label_roboflow_dataset.py --labeler both
    python scripts/label_roboflow_dataset.py --labeler sam --split train
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ROBOFLOW_DATASET_DIR, OUTPUTS_DIR
from src.labeling.sam_labeler import create_sam_labeler
from src.labeling.florence_labeler import create_florence_labeler


def main():
    parser = argparse.ArgumentParser(description="Label Roboflow dataset for ground truth evaluation")
    parser.add_argument("--labeler", type=str, default="both",
                        choices=["sam", "florence", "both"],
                        help="Which labeler to use")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test", "valid"],
                        help="Dataset split to label (default: train)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Setup paths
    images_dir = ROBOFLOW_DATASET_DIR / args.split / "images"

    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        sys.exit(1)

    # Count images
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    logger.info(f"Found {len(image_files)} images in {images_dir}")

    # Run SAM labeling
    if args.labeler in ["sam", "both"]:
        logger.info("=" * 60)
        logger.info("Running SAM labeling on Roboflow dataset")
        logger.info("=" * 60)

        sam_output_dir = OUTPUTS_DIR / f"roboflow_sam_labels"

        labeler = create_sam_labeler(
            output_dir=sam_output_dir,
            confidence_threshold=args.confidence,
            device=args.device,
        )
        labeler.config.batch_size = args.batch_size

        try:
            results = labeler.process_dataset(
                images_dir=images_dir,
                resume=True,
            )

            stats = labeler.get_stats()
            logger.info(f"SAM: Processed {stats['total_images']} images, {stats['total_detections']} detections")
        except Exception as e:
            logger.error(f"SAM labeling failed: {e}")
            raise

    # Run Florence labeling
    if args.labeler in ["florence", "both"]:
        logger.info("=" * 60)
        logger.info("Running Florence labeling on Roboflow dataset")
        logger.info("=" * 60)

        florence_output_dir = OUTPUTS_DIR / f"roboflow_florence_labels"

        labeler = create_florence_labeler(
            output_dir=florence_output_dir,
            confidence_threshold=args.confidence,
            device=args.device,
        )
        labeler.config.batch_size = args.batch_size

        try:
            results = labeler.process_dataset(
                images_dir=images_dir,
                resume=True,
            )

            stats = labeler.get_stats()
            logger.info(f"Florence: Processed {stats['total_images']} images, {stats['total_detections']} detections")
        except Exception as e:
            logger.error(f"Florence labeling failed: {e}")
            raise

    logger.info("=" * 60)
    logger.info("Roboflow labeling complete!")
    logger.info("=" * 60)
    logger.info(f"Next step: Run ground truth evaluation:")
    logger.info(f"  python scripts/run_comparison.py --eval-ground-truth")


if __name__ == "__main__":
    main()
