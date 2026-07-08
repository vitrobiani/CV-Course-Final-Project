#!/usr/bin/env python3
"""
Run Florence-2 labeling on the Flickr dataset.

Usage:
    python scripts/run_florence_labeling.py [--max-images N] [--no-resume] [--batch-size N]
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import FLICKR_IMAGES_DIR, FLORENCE_LABELS_DIR, COMPARISONS_DIR
from src.labeling.florence_labeler import FlorenceLabeler, create_florence_labeler
from src.validation.tta_validator import TTAValidator
from src.utils.dataset_utils import BoundingBox


def main():
    parser = argparse.ArgumentParser(description="Run Florence-2 labeling on Flickr dataset")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum images to process")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (don't resume)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--model", type=str, default="microsoft/Florence-2-large",
                        help="Model name (Florence-2-base or Florence-2-large)")
    parser.add_argument("--images-dir", type=str, default=None, help="Input images directory (default: Flickr)")
    parser.add_argument("--validate", action="store_true", help="Run TTA validation after labeling")
    parser.add_argument("--validate-sample", type=int, default=100, help="Sample size for TTA validation")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Setup
    output_dir = Path(args.output_dir) if args.output_dir else FLORENCE_LABELS_DIR
    images_dir = Path(args.images_dir) if args.images_dir else FLICKR_IMAGES_DIR

    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        sys.exit(1)

    logger.info(f"Starting Florence-2 labeling")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model: {args.model}")

    # Create labeler
    labeler = create_florence_labeler(
        output_dir=output_dir,
        confidence_threshold=args.confidence,
        device=args.device,
        model_name=args.model,
    )
    labeler.config.batch_size = args.batch_size

    # Run labeling
    try:
        results = labeler.process_dataset(
            images_dir=images_dir,
            resume=not args.no_resume,
            max_images=args.max_images,
        )

        # Print stats
        stats = labeler.get_stats()
        logger.info("=" * 50)
        logger.info("Florence-2 Labeling Complete")
        logger.info("=" * 50)
        logger.info(f"Total images: {stats['total_images']}")
        logger.info(f"Total detections: {stats['total_detections']}")
        logger.info(f"Images with detections: {stats['images_with_detections']}")
        logger.info(f"Processing time: {stats['processing_time']:.2f}s")
        logger.info(f"Avg detections/image: {stats['total_detections'] / max(1, stats['total_images']):.2f}")

        # Run TTA validation if requested
        if args.validate:
            logger.info("")
            logger.info("=" * 50)
            logger.info("Running TTA Validation")
            logger.info("=" * 50)

            # Load labels
            labels_dir = output_dir / "labels"
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

            # Run TTA
            validator = TTAValidator(labeler)
            tta_results = validator.validate_dataset(
                labels=labels,
                images_dir=images_dir,
                sample_size=args.validate_sample,
            )

            # Print TTA results
            logger.info(f"Images validated: {tta_results['images_validated']}")
            logger.info(f"Average consistency: {tta_results['average_consistency']:.4f}")
            logger.info(f"Stability rate: {tta_results['stability_rate']:.4f}")
            logger.info(f"Stable boxes: {tta_results['total_stable_boxes']}")
            logger.info(f"Unstable boxes: {tta_results['total_unstable_boxes']}")

            if tta_results['low_consistency_images']:
                logger.info(f"Low consistency images: {len(tta_results['low_consistency_images'])}")

            # Save TTA results
            import json
            COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
            tta_output = COMPARISONS_DIR / "tta_validation_florence.json"
            with open(tta_output, "w") as f:
                json.dump(tta_results, f, indent=2)
            logger.info(f"TTA results saved to: {tta_output}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Progress saved in checkpoint.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during labeling: {e}")
        raise


if __name__ == "__main__":
    main()
