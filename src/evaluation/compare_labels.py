"""
Compare labels from different labeling sources (SAM vs Florence).
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

from ..utils.dataset_utils import BoundingBox, load_yolo_dataset
from ..utils.visualization import visualize_comparison, save_sample_grid
from .metrics import compute_agreement_metrics, match_boxes, compute_iou

logger = logging.getLogger(__name__)


class LabelComparator:
    """
    Compare labels from different sources (e.g., SAM vs Florence).

    Provides metrics on agreement, divergence, and identifies
    images where the labelers disagree significantly.
    """

    def __init__(
        self,
        sam_labels_dir: Path,
        florence_labels_dir: Path,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the comparator.

        Args:
            sam_labels_dir: Directory containing SAM labels
            florence_labels_dir: Directory containing Florence labels
            output_dir: Directory for comparison outputs
        """
        self.sam_labels_dir = Path(sam_labels_dir)
        self.florence_labels_dir = Path(florence_labels_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/comparisons")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sam_labels: Dict[str, List[BoundingBox]] = {}
        self.florence_labels: Dict[str, List[BoundingBox]] = {}

    def load_labels(self) -> None:
        """Load labels from both sources."""
        # Load SAM labels
        sam_label_files = list(self.sam_labels_dir.glob("*.txt"))
        for label_file in sam_label_files:
            img_name = label_file.stem
            boxes = []
            with open(label_file, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            box = BoundingBox.from_yolo(line.strip())
                            box.class_name = "person" if box.class_id == 0 else "vehicle"
                            boxes.append(box)
                        except Exception as e:
                            logger.warning(f"Failed to parse line in {label_file}: {e}")
            self.sam_labels[img_name] = boxes

        # Load Florence labels
        florence_label_files = list(self.florence_labels_dir.glob("*.txt"))
        for label_file in florence_label_files:
            img_name = label_file.stem
            boxes = []
            with open(label_file, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            box = BoundingBox.from_yolo(line.strip())
                            box.class_name = "person" if box.class_id == 0 else "vehicle"
                            boxes.append(box)
                        except Exception as e:
                            logger.warning(f"Failed to parse line in {label_file}: {e}")
            self.florence_labels[img_name] = boxes

        logger.info(f"Loaded {len(self.sam_labels)} SAM labels and {len(self.florence_labels)} Florence labels")

    def compare(self, iou_threshold: float = 0.5) -> Dict:
        """
        Compare SAM and Florence labels.

        Args:
            iou_threshold: IoU threshold for matching

        Returns:
            Comparison results dictionary
        """
        if not self.sam_labels or not self.florence_labels:
            self.load_labels()

        # Compute agreement metrics
        agreement = compute_agreement_metrics(
            self.sam_labels,
            self.florence_labels,
            iou_threshold=iou_threshold,
        )

        # Per-class analysis
        class_stats = self._compute_class_stats()

        # Find disagreement images
        disagreements = self._find_disagreements(iou_threshold)

        results = {
            "agreement": agreement,
            "class_stats": class_stats,
            "high_disagreement_images": disagreements[:50],  # Top 50
            "summary": {
                "sam_total_images": len(self.sam_labels),
                "florence_total_images": len(self.florence_labels),
                "common_images": len(set(self.sam_labels.keys()) & set(self.florence_labels.keys())),
                "sam_total_detections": sum(len(boxes) for boxes in self.sam_labels.values()),
                "florence_total_detections": sum(len(boxes) for boxes in self.florence_labels.values()),
            },
        }

        # Save results
        self._save_results(results)

        return results

    def _compute_class_stats(self) -> Dict:
        """Compute per-class statistics."""
        sam_class_counts = {0: 0, 1: 0}
        florence_class_counts = {0: 0, 1: 0}

        for boxes in self.sam_labels.values():
            for box in boxes:
                sam_class_counts[box.class_id] = sam_class_counts.get(box.class_id, 0) + 1

        for boxes in self.florence_labels.values():
            for box in boxes:
                florence_class_counts[box.class_id] = florence_class_counts.get(box.class_id, 0) + 1

        return {
            "sam": {
                "person": sam_class_counts.get(0, 0),
                "vehicle": sam_class_counts.get(1, 0),
            },
            "florence": {
                "person": florence_class_counts.get(0, 0),
                "vehicle": florence_class_counts.get(1, 0),
            },
        }

    def _find_disagreements(self, iou_threshold: float) -> List[Tuple[str, Dict]]:
        """Find images with significant disagreements."""
        disagreements = []

        common_images = set(self.sam_labels.keys()) & set(self.florence_labels.keys())

        for img_name in common_images:
            sam_boxes = self.sam_labels[img_name]
            florence_boxes = self.florence_labels[img_name]

            matches, sam_only, florence_only = match_boxes(
                sam_boxes, florence_boxes, iou_threshold
            )

            # Calculate disagreement score
            total = len(sam_boxes) + len(florence_boxes)
            if total == 0:
                continue

            disagreement_score = (len(sam_only) + len(florence_only)) / total

            if disagreement_score > 0:
                disagreements.append((img_name, {
                    "sam_count": len(sam_boxes),
                    "florence_count": len(florence_boxes),
                    "matches": len(matches),
                    "sam_only": len(sam_only),
                    "florence_only": len(florence_only),
                    "disagreement_score": disagreement_score,
                }))

        # Sort by disagreement score
        disagreements.sort(key=lambda x: x[1]["disagreement_score"], reverse=True)

        return disagreements

    def _save_results(self, results: Dict) -> None:
        """Save comparison results."""
        output_file = self.output_dir / "sam_vs_florence_comparison.json"

        # Convert to JSON-serializable format
        json_results = json.loads(json.dumps(results, default=str))

        with open(output_file, "w") as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"Comparison results saved to {output_file}")

    def visualize_disagreements(
        self,
        images_dir: Path,
        num_samples: int = 10,
    ) -> None:
        """
        Create visualizations for images with high disagreement.

        Args:
            images_dir: Directory containing original images
            num_samples: Number of samples to visualize
        """
        if not self.sam_labels or not self.florence_labels:
            self.load_labels()

        disagreements = self._find_disagreements(iou_threshold=0.5)

        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        for i, (img_name, stats) in enumerate(disagreements[:num_samples]):
            # Find image file
            image_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = images_dir / (img_name + ext)
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                logger.warning(f"Image not found: {img_name}")
                continue

            sam_boxes = self.sam_labels.get(img_name, [])
            florence_boxes = self.florence_labels.get(img_name, [])

            output_path = vis_dir / f"disagreement_{i:03d}_{img_name}.jpg"

            visualize_comparison(
                image_path=image_path,
                sam_boxes=sam_boxes,
                florence_boxes=florence_boxes,
                output_path=output_path,
            )

        logger.info(f"Saved {min(num_samples, len(disagreements))} disagreement visualizations")

    def generate_report(self) -> str:
        """Generate a text report of the comparison."""
        results = self.compare()

        report_lines = [
            "=" * 60,
            "SAM vs Florence Label Comparison Report",
            "=" * 60,
            "",
            "Summary:",
            f"  SAM labeled images: {results['summary']['sam_total_images']}",
            f"  Florence labeled images: {results['summary']['florence_total_images']}",
            f"  Common images: {results['summary']['common_images']}",
            f"  SAM total detections: {results['summary']['sam_total_detections']}",
            f"  Florence total detections: {results['summary']['florence_total_detections']}",
            "",
            "Agreement Metrics:",
            f"  Total matches: {results['agreement']['total_matches']}",
            f"  Only in SAM: {results['agreement']['only_in_first']}",
            f"  Only in Florence: {results['agreement']['only_in_second']}",
            f"  Agreement rate: {results['agreement']['agreement_rate']:.2%}",
            f"  Average IoU (matched): {results['agreement']['average_iou']:.4f}",
            "",
            "Per-Class Statistics:",
            "  SAM:",
            f"    Person: {results['class_stats']['sam']['person']}",
            f"    Vehicle: {results['class_stats']['sam']['vehicle']}",
            "  Florence:",
            f"    Person: {results['class_stats']['florence']['person']}",
            f"    Vehicle: {results['class_stats']['florence']['vehicle']}",
            "",
            "Top 10 Disagreement Images:",
        ]

        for img_name, stats in results["high_disagreement_images"][:10]:
            report_lines.append(
                f"  {img_name}: SAM={stats['sam_count']}, Florence={stats['florence_count']}, "
                f"Matches={stats['matches']}, Score={stats['disagreement_score']:.2f}"
            )

        report_lines.append("")
        report_lines.append("=" * 60)

        report = "\n".join(report_lines)

        # Save report
        report_file = self.output_dir / "comparison_report.txt"
        with open(report_file, "w") as f:
            f.write(report)

        logger.info(f"Report saved to {report_file}")

        return report


def compare_labels(
    sam_labels_dir: Path,
    florence_labels_dir: Path,
    images_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    visualize: bool = True,
) -> Dict:
    """
    Convenience function to compare SAM and Florence labels.

    Args:
        sam_labels_dir: Directory with SAM labels
        florence_labels_dir: Directory with Florence labels
        images_dir: Directory with original images (for visualization)
        output_dir: Output directory
        visualize: Whether to create visualizations

    Returns:
        Comparison results
    """
    comparator = LabelComparator(
        sam_labels_dir=sam_labels_dir,
        florence_labels_dir=florence_labels_dir,
        output_dir=output_dir,
    )

    results = comparator.compare()
    print(comparator.generate_report())

    if visualize and images_dir:
        comparator.visualize_disagreements(images_dir)

    return results


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python compare_labels.py <sam_labels_dir> <florence_labels_dir> [images_dir]")
        sys.exit(1)

    sam_dir = Path(sys.argv[1])
    florence_dir = Path(sys.argv[2])
    images_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    compare_labels(sam_dir, florence_dir, images_dir)
