#!/usr/bin/env python3
"""
Compare SAM and Florence labels and evaluate against ground truth.

Usage:
    python scripts/run_comparison.py [--visualize] [--num-samples N]
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
    ROBOFLOW_DATASET_DIR,
    COMPARISONS_DIR,
    OUTPUTS_DIR,
)
from src.evaluation.compare_labels import LabelComparator
from src.evaluation.ground_truth_eval import GroundTruthEvaluator
from src.utils.dataset_utils import BoundingBox


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
    parser = argparse.ArgumentParser(description="Compare SAM and Florence labels")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples to visualize")
    parser.add_argument("--eval-ground-truth", action="store_true", help="Evaluate against ground truth")
    parser.add_argument("--sam-dir", type=str, default=None, help="SAM labels directory")
    parser.add_argument("--florence-dir", type=str, default=None, help="Florence labels directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Setup paths
    sam_labels_dir = Path(args.sam_dir) if args.sam_dir else SAM_LABELS_DIR / "labels"
    florence_labels_dir = Path(args.florence_dir) if args.florence_dir else FLORENCE_LABELS_DIR / "labels"
    output_dir = COMPARISONS_DIR

    # Check if label directories exist
    if not sam_labels_dir.exists():
        logger.warning(f"SAM labels not found at {sam_labels_dir}")
    if not florence_labels_dir.exists():
        logger.warning(f"Florence labels not found at {florence_labels_dir}")

    # ========================================
    # Part 1: Compare SAM vs Florence
    # ========================================
    if sam_labels_dir.exists() and florence_labels_dir.exists():
        logger.info("=" * 60)
        logger.info("Comparing SAM vs Florence Labels")
        logger.info("=" * 60)

        comparator = LabelComparator(
            sam_labels_dir=sam_labels_dir,
            florence_labels_dir=florence_labels_dir,
            output_dir=output_dir,
        )

        # Run comparison
        results = comparator.compare()

        # Print report
        report = comparator.generate_report()
        print(report)

        # Visualize disagreements
        if args.visualize:
            logger.info("Creating disagreement visualizations...")
            comparator.visualize_disagreements(
                images_dir=FLICKR_IMAGES_DIR,
                num_samples=args.num_samples,
            )

    # ========================================
    # Part 2: Evaluate against Ground Truth
    # ========================================
    if args.eval_ground_truth:
        logger.info("=" * 60)
        logger.info("Evaluating Against Ground Truth")
        logger.info("=" * 60)

        gt_dir = ROBOFLOW_DATASET_DIR
        if not gt_dir.exists():
            logger.error(f"Ground truth dataset not found: {gt_dir}")
            return

        evaluator = GroundTruthEvaluator(gt_dir, output_dir)
        evaluator.load_ground_truth(split="train")

        # Load SAM and Florence predictions for ROBOFLOW images (not Flickr)
        roboflow_sam_dir = OUTPUTS_DIR / "roboflow_sam_labels" / "labels"
        roboflow_florence_dir = OUTPUTS_DIR / "roboflow_florence_labels" / "labels"

        logger.info(f"Loading Roboflow SAM labels from: {roboflow_sam_dir}")
        logger.info(f"Loading Roboflow Florence labels from: {roboflow_florence_dir}")

        sam_preds = load_labels_from_dir(roboflow_sam_dir) if roboflow_sam_dir.exists() else {}
        florence_preds = load_labels_from_dir(roboflow_florence_dir) if roboflow_florence_dir.exists() else {}

        logger.info(f"Loaded {len(sam_preds)} SAM predictions, {len(florence_preds)} Florence predictions")

        if sam_preds and florence_preds:
            # Compare both
            report = evaluator.generate_report(sam_preds, florence_preds)
            print(report)
        elif sam_preds:
            results = evaluator.evaluate(sam_preds, "sam")
            logger.info(f"SAM mAP@0.5: {results.get('map_results', {}).get('mAP@0.50', 0):.4f}")
        elif florence_preds:
            results = evaluator.evaluate(florence_preds, "florence")
            logger.info(f"Florence mAP@0.5: {results.get('map_results', {}).get('mAP@0.50', 0):.4f}")

    logger.info("Comparison complete. Results saved to: " + str(output_dir))


if __name__ == "__main__":
    main()
