"""
Ensemble methods for combining and validating labels from multiple sources.

Uses techniques like Weighted Box Fusion (WBF) and voting to combine
predictions from SAM and Florence, identifying high-confidence labels.
"""
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

import numpy as np

from ..config import TTAConfig, DEFAULT_TTA_CONFIG
from ..utils.dataset_utils import BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Result from ensemble combination."""
    image_name: str
    combined_boxes: List[BoundingBox]
    agreement_boxes: List[BoundingBox]  # Boxes both sources agree on
    sam_only_boxes: List[BoundingBox]
    florence_only_boxes: List[BoundingBox]
    agreement_score: float


class EnsembleValidator:
    """
    Combine and validate labels from multiple sources using ensemble methods.

    Supports:
    - Voting: Keep boxes that appear in multiple sources
    - NMS: Non-Maximum Suppression to merge overlapping boxes
    - WBF: Weighted Box Fusion for smooth combination
    """

    def __init__(self, config: Optional[TTAConfig] = None):
        """
        Initialize ensemble validator.

        Args:
            config: Configuration for ensemble methods
        """
        self.config = config or DEFAULT_TTA_CONFIG

    def _compute_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Compute IoU between two boxes."""
        x1_1 = box1.x_center - box1.width / 2
        y1_1 = box1.y_center - box1.height / 2
        x2_1 = box1.x_center + box1.width / 2
        y2_1 = box1.y_center + box1.height / 2

        x1_2 = box2.x_center - box2.width / 2
        y1_2 = box2.y_center - box2.height / 2
        x2_2 = box2.x_center + box2.width / 2
        y2_2 = box2.y_center + box2.height / 2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def voting_ensemble(
        self,
        sam_boxes: List[BoundingBox],
        florence_boxes: List[BoundingBox],
        iou_threshold: float = 0.5,
    ) -> Tuple[List[BoundingBox], List[BoundingBox], List[BoundingBox], List[BoundingBox]]:
        """
        Simple voting ensemble - keep boxes that appear in both sources.

        Args:
            sam_boxes: Boxes from SAM
            florence_boxes: Boxes from Florence
            iou_threshold: IoU threshold for matching

        Returns:
            Tuple of (combined, agreement, sam_only, florence_only)
        """
        agreement_boxes = []
        sam_only = list(sam_boxes)
        florence_only = list(florence_boxes)

        matched_sam = set()
        matched_florence = set()

        # Find matching boxes
        for i, sam_box in enumerate(sam_boxes):
            for j, flor_box in enumerate(florence_boxes):
                if j in matched_florence:
                    continue

                if sam_box.class_id != flor_box.class_id:
                    continue

                iou = self._compute_iou(sam_box, flor_box)
                if iou >= iou_threshold:
                    # Average the boxes
                    avg_box = BoundingBox(
                        x_center=(sam_box.x_center + flor_box.x_center) / 2,
                        y_center=(sam_box.y_center + flor_box.y_center) / 2,
                        width=(sam_box.width + flor_box.width) / 2,
                        height=(sam_box.height + flor_box.height) / 2,
                        class_id=sam_box.class_id,
                        confidence=(sam_box.confidence + flor_box.confidence) / 2,
                        class_name=sam_box.class_name,
                    )
                    agreement_boxes.append(avg_box)
                    matched_sam.add(i)
                    matched_florence.add(j)
                    break

        # Get unmatched boxes
        sam_only = [b for i, b in enumerate(sam_boxes) if i not in matched_sam]
        florence_only = [b for j, b in enumerate(florence_boxes) if j not in matched_florence]

        # Combined = agreement + unique from each source (with lower confidence)
        combined = list(agreement_boxes)
        for box in sam_only:
            reduced_conf_box = BoundingBox(
                x_center=box.x_center,
                y_center=box.y_center,
                width=box.width,
                height=box.height,
                class_id=box.class_id,
                confidence=box.confidence * 0.7,  # Reduce confidence for single-source
                class_name=box.class_name,
            )
            combined.append(reduced_conf_box)

        for box in florence_only:
            reduced_conf_box = BoundingBox(
                x_center=box.x_center,
                y_center=box.y_center,
                width=box.width,
                height=box.height,
                class_id=box.class_id,
                confidence=box.confidence * 0.7,
                class_name=box.class_name,
            )
            combined.append(reduced_conf_box)

        return combined, agreement_boxes, sam_only, florence_only

    def weighted_box_fusion(
        self,
        sam_boxes: List[BoundingBox],
        florence_boxes: List[BoundingBox],
        iou_threshold: float = 0.5,
        weights: Tuple[float, float] = (1.0, 1.0),
    ) -> List[BoundingBox]:
        """
        Weighted Box Fusion for combining predictions.

        Args:
            sam_boxes: Boxes from SAM
            florence_boxes: Boxes from Florence
            iou_threshold: IoU threshold for fusion
            weights: Weights for (SAM, Florence)

        Returns:
            Fused boxes
        """
        # Combine all boxes with source weights
        all_boxes = []
        for box in sam_boxes:
            all_boxes.append((box, weights[0], "sam"))
        for box in florence_boxes:
            all_boxes.append((box, weights[1], "florence"))

        if not all_boxes:
            return []

        # Sort by confidence * weight
        all_boxes.sort(key=lambda x: x[0].confidence * x[1], reverse=True)

        # Group boxes by class
        class_groups = {}
        for box, weight, source in all_boxes:
            if box.class_id not in class_groups:
                class_groups[box.class_id] = []
            class_groups[box.class_id].append((box, weight, source))

        fused_boxes = []

        for class_id, boxes_with_weights in class_groups.items():
            clusters = []

            for box, weight, source in boxes_with_weights:
                matched = False

                for cluster in clusters:
                    # Check if box belongs to this cluster
                    cluster_box = cluster["fused"]
                    iou = self._compute_iou(box, cluster_box)

                    if iou >= iou_threshold:
                        # Add to cluster
                        cluster["boxes"].append((box, weight))

                        # Recompute fused box
                        total_weight = sum(w for _, w in cluster["boxes"])
                        x_center = sum(b.x_center * w for b, w in cluster["boxes"]) / total_weight
                        y_center = sum(b.y_center * w for b, w in cluster["boxes"]) / total_weight
                        width = sum(b.width * w for b, w in cluster["boxes"]) / total_weight
                        height = sum(b.height * w for b, w in cluster["boxes"]) / total_weight
                        confidence = sum(b.confidence * w for b, w in cluster["boxes"]) / total_weight

                        cluster["fused"] = BoundingBox(
                            x_center=x_center,
                            y_center=y_center,
                            width=width,
                            height=height,
                            class_id=class_id,
                            confidence=min(confidence, 1.0),
                            class_name=box.class_name,
                        )
                        matched = True
                        break

                if not matched:
                    # Create new cluster
                    clusters.append({
                        "boxes": [(box, weight)],
                        "fused": box,
                    })

            # Extract fused boxes from clusters
            for cluster in clusters:
                fused_boxes.append(cluster["fused"])

        return fused_boxes

    def combine_labels(
        self,
        sam_labels: Dict[str, List[BoundingBox]],
        florence_labels: Dict[str, List[BoundingBox]],
        method: str = "wbf",
    ) -> Dict[str, EnsembleResult]:
        """
        Combine labels from SAM and Florence for all images.

        Args:
            sam_labels: SAM labels per image
            florence_labels: Florence labels per image
            method: Combination method ("voting", "wbf")

        Returns:
            Dictionary mapping image names to EnsembleResult
        """
        all_images = set(sam_labels.keys()) | set(florence_labels.keys())
        results = {}

        for img_name in all_images:
            sam_boxes = sam_labels.get(img_name, [])
            florence_boxes = florence_labels.get(img_name, [])

            if method == "voting":
                combined, agreement, sam_only, flor_only = self.voting_ensemble(
                    sam_boxes, florence_boxes
                )
            else:  # wbf
                combined = self.weighted_box_fusion(sam_boxes, florence_boxes)
                # For WBF, compute agreement separately
                _, agreement, sam_only, flor_only = self.voting_ensemble(
                    sam_boxes, florence_boxes
                )

            total_unique = len(agreement) + len(sam_only) + len(flor_only)
            agreement_score = len(agreement) / total_unique if total_unique > 0 else 0.0

            results[img_name] = EnsembleResult(
                image_name=img_name,
                combined_boxes=combined,
                agreement_boxes=agreement,
                sam_only_boxes=sam_only,
                florence_only_boxes=flor_only,
                agreement_score=agreement_score,
            )

        return results

    def validate_dataset(
        self,
        sam_labels: Dict[str, List[BoundingBox]],
        florence_labels: Dict[str, List[BoundingBox]],
    ) -> Dict:
        """
        Validate dataset using ensemble agreement.

        Args:
            sam_labels: SAM labels
            florence_labels: Florence labels

        Returns:
            Validation statistics
        """
        results = self.combine_labels(sam_labels, florence_labels)

        total_agreement = 0
        total_sam_only = 0
        total_florence_only = 0
        agreement_scores = []

        high_agreement_images = []
        low_agreement_images = []

        for img_name, result in results.items():
            total_agreement += len(result.agreement_boxes)
            total_sam_only += len(result.sam_only_boxes)
            total_florence_only += len(result.florence_only_boxes)
            agreement_scores.append(result.agreement_score)

            if result.agreement_score >= 0.8:
                high_agreement_images.append(img_name)
            elif result.agreement_score < 0.3:
                low_agreement_images.append(img_name)

        avg_agreement = np.mean(agreement_scores) if agreement_scores else 0.0

        return {
            "total_images": len(results),
            "total_agreement_boxes": total_agreement,
            "total_sam_only_boxes": total_sam_only,
            "total_florence_only_boxes": total_florence_only,
            "average_agreement_score": float(avg_agreement),
            "high_agreement_images": len(high_agreement_images),
            "low_agreement_images": len(low_agreement_images),
            "agreement_rate": total_agreement / (total_agreement + total_sam_only + total_florence_only) if (total_agreement + total_sam_only + total_florence_only) > 0 else 0.0,
        }

    def get_high_confidence_labels(
        self,
        sam_labels: Dict[str, List[BoundingBox]],
        florence_labels: Dict[str, List[BoundingBox]],
        min_agreement_score: float = 0.7,
    ) -> Dict[str, List[BoundingBox]]:
        """
        Get labels only from images with high agreement.

        Args:
            sam_labels: SAM labels
            florence_labels: Florence labels
            min_agreement_score: Minimum agreement score

        Returns:
            High-confidence labels
        """
        results = self.combine_labels(sam_labels, florence_labels, method="wbf")

        high_conf_labels = {}
        for img_name, result in results.items():
            if result.agreement_score >= min_agreement_score:
                high_conf_labels[img_name] = result.combined_boxes

        logger.info(
            f"Selected {len(high_conf_labels)} images with agreement >= {min_agreement_score}"
        )

        return high_conf_labels


def create_ensemble_dataset(
    sam_labels: Dict[str, List[BoundingBox]],
    florence_labels: Dict[str, List[BoundingBox]],
    method: str = "wbf",
    min_agreement: float = 0.0,
) -> Dict[str, List[BoundingBox]]:
    """
    Create a combined dataset from SAM and Florence labels.

    Args:
        sam_labels: SAM labels
        florence_labels: Florence labels
        method: Combination method
        min_agreement: Minimum agreement score to include image

    Returns:
        Combined labels
    """
    validator = EnsembleValidator()
    results = validator.combine_labels(sam_labels, florence_labels, method)

    combined = {}
    for img_name, result in results.items():
        if result.agreement_score >= min_agreement:
            combined[img_name] = result.combined_boxes

    return combined
