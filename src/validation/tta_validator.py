"""
Test-Time Augmentation (TTA) for validating label consistency.

TTA applies various transformations to images and checks if the model
produces consistent predictions across augmented versions. This helps
identify unreliable labels.
"""
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..config import TTAConfig, DEFAULT_TTA_CONFIG
from ..utils.dataset_utils import BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class TTAResult:
    """Result from TTA validation."""
    image_name: str
    original_boxes: List[BoundingBox]
    augmented_results: List[Tuple[str, List[BoundingBox]]]
    consistency_score: float
    stable_boxes: List[BoundingBox]
    unstable_boxes: List[BoundingBox]


class TTAValidator:
    """
    Test-Time Augmentation validator for label consistency.

    Applies transformations to images and checks if predictions
    remain consistent, helping identify unreliable labels.
    """

    def __init__(
        self,
        labeler: "BaseLabeler",  # Forward reference
        config: Optional[TTAConfig] = None,
    ):
        """
        Initialize TTA validator.

        Args:
            labeler: The labeler to use for predictions
            config: TTA configuration
        """
        self.labeler = labeler
        self.config = config or DEFAULT_TTA_CONFIG

    def _horizontal_flip(self, image: Image.Image) -> Image.Image:
        """Apply horizontal flip."""
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    def _vertical_flip(self, image: Image.Image) -> Image.Image:
        """Apply vertical flip."""
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    def _rotate(self, image: Image.Image, angle: int) -> Image.Image:
        """Apply rotation."""
        return image.rotate(angle, expand=True, resample=Image.Resampling.BILINEAR)

    def _scale(self, image: Image.Image, scale: float) -> Image.Image:
        """Apply scaling."""
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _flip_boxes_horizontal(
        self,
        boxes: List[BoundingBox],
    ) -> List[BoundingBox]:
        """Flip box coordinates horizontally."""
        flipped = []
        for box in boxes:
            flipped_box = BoundingBox(
                x_center=1.0 - box.x_center,
                y_center=box.y_center,
                width=box.width,
                height=box.height,
                class_id=box.class_id,
                confidence=box.confidence,
                class_name=box.class_name,
            )
            flipped.append(flipped_box)
        return flipped

    def _flip_boxes_vertical(
        self,
        boxes: List[BoundingBox],
    ) -> List[BoundingBox]:
        """Flip box coordinates vertically."""
        flipped = []
        for box in boxes:
            flipped_box = BoundingBox(
                x_center=box.x_center,
                y_center=1.0 - box.y_center,
                width=box.width,
                height=box.height,
                class_id=box.class_id,
                confidence=box.confidence,
                class_name=box.class_name,
            )
            flipped.append(flipped_box)
        return flipped

    def _compute_box_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
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

    def validate_image(
        self,
        image_path: Path,
        original_boxes: Optional[List[BoundingBox]] = None,
    ) -> TTAResult:
        """
        Validate labels for a single image using TTA.

        Args:
            image_path: Path to the image
            original_boxes: Original boxes (will predict if not provided)

        Returns:
            TTAResult with consistency metrics
        """
        image = Image.open(image_path).convert("RGB")
        temp_dir = Path("/tmp/tta_temp")
        temp_dir.mkdir(exist_ok=True)

        # Get original predictions if not provided
        if original_boxes is None:
            original_boxes = self.labeler.predict_single(image_path)

        augmented_results = []

        # Apply horizontal flip
        if self.config.horizontal_flip:
            flipped_img = self._horizontal_flip(image)
            temp_path = temp_dir / f"hflip_{image_path.name}"
            flipped_img.save(temp_path)

            flipped_boxes = self.labeler.predict_single(temp_path)
            # Transform boxes back to original coordinates
            transformed_boxes = self._flip_boxes_horizontal(flipped_boxes)
            augmented_results.append(("horizontal_flip", transformed_boxes))

        # Apply vertical flip
        if self.config.vertical_flip:
            flipped_img = self._vertical_flip(image)
            temp_path = temp_dir / f"vflip_{image_path.name}"
            flipped_img.save(temp_path)

            flipped_boxes = self.labeler.predict_single(temp_path)
            transformed_boxes = self._flip_boxes_vertical(flipped_boxes)
            augmented_results.append(("vertical_flip", transformed_boxes))

        # Compute consistency
        consistency_score, stable_boxes, unstable_boxes = self._compute_consistency(
            original_boxes, augmented_results
        )

        return TTAResult(
            image_name=image_path.name,
            original_boxes=original_boxes,
            augmented_results=augmented_results,
            consistency_score=consistency_score,
            stable_boxes=stable_boxes,
            unstable_boxes=unstable_boxes,
        )

    def _compute_consistency(
        self,
        original_boxes: List[BoundingBox],
        augmented_results: List[Tuple[str, List[BoundingBox]]],
    ) -> Tuple[float, List[BoundingBox], List[BoundingBox]]:
        """
        Compute consistency between original and augmented predictions.

        Args:
            original_boxes: Original predictions
            augmented_results: List of (aug_name, boxes) tuples

        Returns:
            Tuple of (consistency_score, stable_boxes, unstable_boxes)
        """
        if not original_boxes or not augmented_results:
            return 1.0, original_boxes, []

        # For each original box, check if it appears in augmented results
        stable_boxes = []
        unstable_boxes = []

        for orig_box in original_boxes:
            votes = 1  # Original vote

            for aug_name, aug_boxes in augmented_results:
                # Check if any augmented box matches
                best_iou = 0.0
                for aug_box in aug_boxes:
                    if aug_box.class_id == orig_box.class_id:
                        iou = self._compute_box_iou(orig_box, aug_box)
                        best_iou = max(best_iou, iou)

                if best_iou > self.config.nms_iou_threshold:
                    votes += 1

            # Determine stability
            total_views = 1 + len(augmented_results)
            if votes >= self.config.min_votes:
                stable_boxes.append(orig_box)
            else:
                unstable_boxes.append(orig_box)

        # Consistency score
        total_boxes = len(original_boxes)
        consistency_score = len(stable_boxes) / total_boxes if total_boxes > 0 else 1.0

        return consistency_score, stable_boxes, unstable_boxes

    def validate_dataset(
        self,
        labels: Dict[str, List[BoundingBox]],
        images_dir: Path,
        sample_size: Optional[int] = None,
    ) -> Dict:
        """
        Validate a dataset using TTA.

        Args:
            labels: Dictionary mapping image names to boxes
            images_dir: Directory containing images
            sample_size: Number of images to sample (None for all)

        Returns:
            Validation results
        """
        image_names = list(labels.keys())

        if sample_size and sample_size < len(image_names):
            import random
            image_names = random.sample(image_names, sample_size)

        results = []
        total_consistency = 0.0
        total_stable = 0
        total_unstable = 0

        for img_name in image_names:
            # Find image file
            image_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = images_dir / (Path(img_name).stem + ext)
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                continue

            try:
                result = self.validate_image(image_path, labels[img_name])
                results.append(result)

                total_consistency += result.consistency_score
                total_stable += len(result.stable_boxes)
                total_unstable += len(result.unstable_boxes)

            except Exception as e:
                logger.warning(f"Failed to validate {img_name}: {e}")

        avg_consistency = total_consistency / len(results) if results else 0.0

        return {
            "images_validated": len(results),
            "average_consistency": avg_consistency,
            "total_stable_boxes": total_stable,
            "total_unstable_boxes": total_unstable,
            "stability_rate": total_stable / (total_stable + total_unstable) if (total_stable + total_unstable) > 0 else 0.0,
            "low_consistency_images": [
                r.image_name for r in results if r.consistency_score < 0.5
            ],
        }


def validate_with_tta(
    labeler: "BaseLabeler",
    labels: Dict[str, List[BoundingBox]],
    images_dir: Path,
    sample_size: int = 100,
) -> Dict:
    """
    Convenience function for TTA validation.

    Args:
        labeler: Labeler to use
        labels: Labels to validate
        images_dir: Directory with images
        sample_size: Number of images to sample

    Returns:
        Validation results
    """
    validator = TTAValidator(labeler)
    return validator.validate_dataset(labels, images_dir, sample_size)
