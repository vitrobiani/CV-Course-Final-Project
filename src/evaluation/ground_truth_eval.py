"""
Evaluate labeling results against ground truth (Roboflow dataset).
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import yaml

from ..utils.dataset_utils import BoundingBox, load_yolo_dataset
from ..utils.visualization import visualize_comparison
from .metrics import compute_map, compute_precision_recall, match_boxes, compute_agreement_metrics

logger = logging.getLogger(__name__)


class GroundTruthEvaluator:
    """
    Evaluate SAM and Florence labels against ground truth labels.

    Uses the Roboflow Persons_and_cars dataset as ground truth.
    """

    def __init__(
        self,
        ground_truth_dir: Path,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            ground_truth_dir: Path to ground truth dataset (YOLO format)
            output_dir: Directory for evaluation outputs
        """
        self.gt_dir = Path(ground_truth_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/comparisons")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ground_truth: Dict[str, List[BoundingBox]] = {}
        self.class_mapping = self._load_class_mapping()

    def _load_class_mapping(self) -> Dict[str, int]:
        """Load class mapping from data.yaml."""
        yaml_path = self.gt_dir / "data.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
                names = data.get("names", [])
                # Map Roboflow classes to our classes
                # Roboflow: 0=car, 1=person
                # Our format: 0=person, 1=vehicle
                return {
                    "car": 1,      # Map car -> vehicle (1)
                    "person": 0,   # Map person -> person (0)
                }
        return {"car": 1, "person": 0}

    def load_ground_truth(self, split: str = "test") -> None:
        """
        Load ground truth labels.

        Args:
            split: Dataset split to use (train, valid, test)
        """
        labels_dir = self.gt_dir / split / "labels"
        images_dir = self.gt_dir / split / "images"

        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        for label_file in labels_dir.glob("*.txt"):
            img_name = label_file.stem
            boxes = []

            with open(label_file, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            parts = line.strip().split()
                            orig_class_id = int(parts[0])

                            # Remap class ID
                            # Roboflow: 0=car, 1=person -> Our: 0=person, 1=vehicle
                            if orig_class_id == 0:  # car in Roboflow
                                new_class_id = 1    # vehicle in our format
                                class_name = "vehicle"
                            else:  # person in Roboflow
                                new_class_id = 0    # person in our format
                                class_name = "person"

                            box = BoundingBox(
                                x_center=float(parts[1]),
                                y_center=float(parts[2]),
                                width=float(parts[3]),
                                height=float(parts[4]),
                                class_id=new_class_id,
                                class_name=class_name,
                            )
                            boxes.append(box)
                        except Exception as e:
                            logger.warning(f"Failed to parse line in {label_file}: {e}")

            self.ground_truth[img_name] = boxes

        logger.info(f"Loaded {len(self.ground_truth)} ground truth labels from {split} split")

    def evaluate(
        self,
        predictions: Dict[str, List[BoundingBox]],
        source_name: str,
        iou_thresholds: Optional[List[float]] = None,
    ) -> Dict:
        """
        Evaluate predictions against ground truth.

        Args:
            predictions: Dictionary mapping image names to predicted boxes
            source_name: Name of the prediction source (sam or florence)
            iou_thresholds: IoU thresholds for mAP calculation

        Returns:
            Evaluation results dictionary
        """
        if not self.ground_truth:
            self.load_ground_truth()

        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        # Find common images
        common_images = set(predictions.keys()) & set(self.ground_truth.keys())

        if len(common_images) == 0:
            logger.warning("No common images found between predictions and ground truth")
            return {"error": "No common images"}

        # Filter to common images
        pred_filtered = {k: predictions[k] for k in common_images}
        gt_filtered = {k: self.ground_truth[k] for k in common_images}

        # Compute mAP
        map_results = compute_map(
            pred_filtered,
            gt_filtered,
            iou_thresholds=iou_thresholds,
            num_classes=2,
        )

        # Compute per-image metrics
        per_image_metrics = []
        for img_name in common_images:
            pred_boxes = pred_filtered[img_name]
            gt_boxes = gt_filtered[img_name]

            metrics = compute_precision_recall(pred_boxes, gt_boxes, iou_threshold=0.5)
            metrics["image"] = img_name
            per_image_metrics.append(metrics)

        # Aggregate metrics
        total_tp = sum(m["true_positives"] for m in per_image_metrics)
        total_fp = sum(m["false_positives"] for m in per_image_metrics)
        total_fn = sum(m["false_negatives"] for m in per_image_metrics)

        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

        # Per-class statistics
        class_stats = self._compute_class_metrics(pred_filtered, gt_filtered)

        results = {
            "source": source_name,
            "num_images": len(common_images),
            "overall": {
                "precision": overall_precision,
                "recall": overall_recall,
                "f1_score": overall_f1,
                "true_positives": total_tp,
                "false_positives": total_fp,
                "false_negatives": total_fn,
            },
            "map_results": map_results,
            "class_metrics": class_stats,
            "per_image_sample": per_image_metrics[:10],  # Sample
        }

        # Save results
        self._save_results(results, source_name)

        return results

    def _compute_class_metrics(
        self,
        predictions: Dict[str, List[BoundingBox]],
        ground_truth: Dict[str, List[BoundingBox]],
    ) -> Dict:
        """Compute per-class metrics."""
        class_names = {0: "person", 1: "vehicle"}
        class_metrics = {}

        for class_id, class_name in class_names.items():
            pred_count = sum(
                len([b for b in boxes if b.class_id == class_id])
                for boxes in predictions.values()
            )
            gt_count = sum(
                len([b for b in boxes if b.class_id == class_id])
                for boxes in ground_truth.values()
            )

            # Filter to specific class
            pred_class = {
                k: [b for b in v if b.class_id == class_id]
                for k, v in predictions.items()
            }
            gt_class = {
                k: [b for b in v if b.class_id == class_id]
                for k, v in ground_truth.items()
            }

            # Compute metrics for this class
            total_tp = 0
            total_fp = 0
            total_fn = 0

            for img_name in predictions.keys():
                pred_boxes = pred_class.get(img_name, [])
                gt_boxes = gt_class.get(img_name, [])

                matches, fps, fns = match_boxes(pred_boxes, gt_boxes, 0.5)
                total_tp += len(matches)
                total_fp += len(fps)
                total_fn += len(fns)

            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

            class_metrics[class_name] = {
                "predictions": pred_count,
                "ground_truth": gt_count,
                "precision": precision,
                "recall": recall,
            }

        return class_metrics

    def _save_results(self, results: Dict, source_name: str) -> None:
        """Save evaluation results."""
        output_file = self.output_dir / f"{source_name}_vs_ground_truth.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Evaluation results saved to {output_file}")

    def compare_both(
        self,
        sam_predictions: Dict[str, List[BoundingBox]],
        florence_predictions: Dict[str, List[BoundingBox]],
    ) -> Dict:
        """
        Evaluate both SAM and Florence against ground truth and compare.

        Args:
            sam_predictions: SAM predictions
            florence_predictions: Florence predictions

        Returns:
            Combined comparison results
        """
        sam_results = self.evaluate(sam_predictions, "sam")
        florence_results = self.evaluate(florence_predictions, "florence")

        comparison = {
            "sam": sam_results,
            "florence": florence_results,
            "comparison": {
                "mAP@0.5": {
                    "sam": sam_results.get("map_results", {}).get("mAP@0.50", 0),
                    "florence": florence_results.get("map_results", {}).get("mAP@0.50", 0),
                },
                "precision": {
                    "sam": sam_results.get("overall", {}).get("precision", 0),
                    "florence": florence_results.get("overall", {}).get("precision", 0),
                },
                "recall": {
                    "sam": sam_results.get("overall", {}).get("recall", 0),
                    "florence": florence_results.get("overall", {}).get("recall", 0),
                },
            },
        }

        # Determine winner for each metric
        for metric in ["mAP@0.5", "precision", "recall"]:
            sam_val = comparison["comparison"][metric]["sam"]
            florence_val = comparison["comparison"][metric]["florence"]
            if sam_val > florence_val:
                comparison["comparison"][metric]["winner"] = "sam"
            elif florence_val > sam_val:
                comparison["comparison"][metric]["winner"] = "florence"
            else:
                comparison["comparison"][metric]["winner"] = "tie"

        # Save combined results
        output_file = self.output_dir / "ground_truth_comparison.json"
        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2, default=str)

        return comparison

    def generate_report(
        self,
        sam_predictions: Dict[str, List[BoundingBox]],
        florence_predictions: Dict[str, List[BoundingBox]],
    ) -> str:
        """Generate a text report comparing both labelers to ground truth."""
        comparison = self.compare_both(sam_predictions, florence_predictions)

        report_lines = [
            "=" * 70,
            "Ground Truth Evaluation Report",
            "=" * 70,
            "",
            f"Ground truth dataset: {self.gt_dir}",
            f"Ground truth images: {len(self.ground_truth)}",
            "",
            "-" * 70,
            "SAM Results:",
            "-" * 70,
            f"  Images evaluated: {comparison['sam'].get('num_images', 0)}",
            f"  Precision: {comparison['sam'].get('overall', {}).get('precision', 0):.4f}",
            f"  Recall: {comparison['sam'].get('overall', {}).get('recall', 0):.4f}",
            f"  F1 Score: {comparison['sam'].get('overall', {}).get('f1_score', 0):.4f}",
            f"  mAP@0.5: {comparison['sam'].get('map_results', {}).get('mAP@0.50', 0):.4f}",
            "",
            "-" * 70,
            "Florence Results:",
            "-" * 70,
            f"  Images evaluated: {comparison['florence'].get('num_images', 0)}",
            f"  Precision: {comparison['florence'].get('overall', {}).get('precision', 0):.4f}",
            f"  Recall: {comparison['florence'].get('overall', {}).get('recall', 0):.4f}",
            f"  F1 Score: {comparison['florence'].get('overall', {}).get('f1_score', 0):.4f}",
            f"  mAP@0.5: {comparison['florence'].get('map_results', {}).get('mAP@0.50', 0):.4f}",
            "",
            "-" * 70,
            "Comparison Summary:",
            "-" * 70,
        ]

        for metric in ["mAP@0.5", "precision", "recall"]:
            sam_val = comparison["comparison"][metric]["sam"]
            florence_val = comparison["comparison"][metric]["florence"]
            winner = comparison["comparison"][metric]["winner"]
            report_lines.append(
                f"  {metric}: SAM={sam_val:.4f}, Florence={florence_val:.4f} -> Winner: {winner.upper()}"
            )

        report_lines.extend([
            "",
            "=" * 70,
        ])

        report = "\n".join(report_lines)

        # Save report
        report_file = self.output_dir / "ground_truth_report.txt"
        with open(report_file, "w") as f:
            f.write(report)

        return report


def evaluate_against_ground_truth(
    ground_truth_dir: Path,
    sam_labels_dir: Optional[Path] = None,
    florence_labels_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Convenience function to evaluate labels against ground truth.

    Args:
        ground_truth_dir: Path to ground truth dataset
        sam_labels_dir: Directory with SAM labels
        florence_labels_dir: Directory with Florence labels
        output_dir: Output directory

    Returns:
        Evaluation results
    """
    from ..utils.dataset_utils import BoundingBox

    evaluator = GroundTruthEvaluator(ground_truth_dir, output_dir)
    evaluator.load_ground_truth()

    results = {}

    # Load and evaluate SAM predictions
    if sam_labels_dir and Path(sam_labels_dir).exists():
        sam_preds = {}
        for label_file in Path(sam_labels_dir).glob("*.txt"):
            img_name = label_file.stem
            boxes = []
            with open(label_file, "r") as f:
                for line in f:
                    if line.strip():
                        box = BoundingBox.from_yolo(line.strip())
                        boxes.append(box)
            sam_preds[img_name] = boxes

        results["sam"] = evaluator.evaluate(sam_preds, "sam")

    # Load and evaluate Florence predictions
    if florence_labels_dir and Path(florence_labels_dir).exists():
        florence_preds = {}
        for label_file in Path(florence_labels_dir).glob("*.txt"):
            img_name = label_file.stem
            boxes = []
            with open(label_file, "r") as f:
                for line in f:
                    if line.strip():
                        box = BoundingBox.from_yolo(line.strip())
                        boxes.append(box)
            florence_preds[img_name] = boxes

        results["florence"] = evaluator.evaluate(florence_preds, "florence")

    return results


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python ground_truth_eval.py <ground_truth_dir> [sam_labels_dir] [florence_labels_dir]")
        sys.exit(1)

    gt_dir = Path(sys.argv[1])
    sam_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    florence_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    evaluate_against_ground_truth(gt_dir, sam_dir, florence_dir)
