"""
SAM 3 (Segment Anything with Concepts) based labeler.

SAM 3 supports text-prompted segmentation, allowing zero-shot detection
of concepts like "person" and "vehicle" without requiring bounding box prompts.
"""
from pathlib import Path
from typing import List, Dict, Optional
import logging

from PIL import Image

from .base_labeler import BaseLabeler
from ..config import SAMConfig, DEFAULT_SAM_CONFIG
from ..utils.dataset_utils import BoundingBox

logger = logging.getLogger(__name__)


def nms_boxes(boxes: List[BoundingBox], iou_threshold: float = 0.5) -> List[BoundingBox]:
    """
    Apply Non-Maximum Suppression to remove duplicate boxes.

    Args:
        boxes: List of bounding boxes
        iou_threshold: IoU threshold for considering boxes as duplicates

    Returns:
        Filtered list of boxes
    """
    if not boxes:
        return []

    # Sort by confidence (descending)
    boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)

    keep = []
    while boxes:
        # Keep the highest confidence box
        best = boxes.pop(0)
        keep.append(best)

        # Remove boxes that overlap too much with the best box
        remaining = []
        for box in boxes:
            # Only compare boxes of the same class
            if box.class_id != best.class_id:
                remaining.append(box)
                continue

            # Compute IoU
            iou = compute_iou(best, box)
            if iou < iou_threshold:
                remaining.append(box)

        boxes = remaining

    return keep


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
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


class SAMLabeler(BaseLabeler):
    """
    SAM 3 based auto-labeler using text prompts.

    Uses SAM 3's Promptable Concept Segmentation (PCS) to detect
    all instances of specified concepts (person, vehicle) in images.
    """

    def __init__(
        self,
        config: Optional[SAMConfig] = None,
        output_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize SAM labeler.

        Args:
            config: SAM configuration
            output_dir: Directory for output labels
            checkpoint_dir: Directory for checkpoints
        """
        config = config or DEFAULT_SAM_CONFIG
        output_dir = output_dir or Path("outputs/sam_labels")

        super().__init__(config, output_dir, checkpoint_dir)

        self.sam_config: SAMConfig = config
        self.predictor = None

    @property
    def labeler_name(self) -> str:
        return "sam"

    def load_model(self) -> None:
        """Load SAM 3 model using Ultralytics."""
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except ImportError:
            raise ImportError(
                "SAM 3 requires ultralytics>=8.3.237. "
                "Install with: pip install -U ultralytics\n"
                "Also ensure you have downloaded sam3.pt from HuggingFace."
            )

        logger.info(f"Loading SAM 3 model: {self.sam_config.model_name}")

        overrides = {
            "conf": self.config.confidence_threshold,
            "task": "segment",
            "mode": "predict",
            "model": self.sam_config.model_name,
            "half": self.sam_config.half_precision,
            "device": self.device,
        }

        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self.model = self.predictor  # For base class compatibility

        logger.info("SAM 3 model loaded successfully")

    def predict_single(self, image_path: Path) -> List[BoundingBox]:
        """
        Run SAM 3 prediction on a single image.

        Args:
            image_path: Path to the image

        Returns:
            List of detected bounding boxes
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Load image and get dimensions
        image = Image.open(image_path)
        img_width, img_height = image.size

        # Set image in predictor
        self.predictor.set_image(str(image_path))

        all_boxes = []

        # Process each class
        for target_class, prompt in self.sam_config.detection_prompts.items():
            try:
                # Get class ID
                class_id = self.config.class_to_id.get(target_class)
                if class_id is None:
                    continue

                # Split comma-separated prompts into list
                # SAM 3 expects text=["person", "man", "woman"] not text=["person, man, woman"]
                prompt_list = [p.strip() for p in prompt.split(",")]

                # Run prediction with text prompts
                # SAM 3 returns masks for all matching instances
                results = self.predictor(text=prompt_list)

                if results is None or len(results) == 0:
                    continue

                # Process results - extract masks and convert to boxes
                for result in results:
                    if hasattr(result, 'masks') and result.masks is not None:
                        masks = result.masks.data.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy() if hasattr(result, 'boxes') and result.boxes is not None else None

                        for i, mask in enumerate(masks):
                            # Convert mask to bounding box
                            box = BoundingBox.from_mask(
                                mask,
                                class_id=class_id,
                                confidence=float(confs[i]) if confs is not None else 1.0,
                                class_name=target_class,
                            )
                            if box is not None:
                                all_boxes.append(box)

                    elif hasattr(result, 'boxes') and result.boxes is not None:
                        # If only boxes are returned (no masks)
                        boxes_data = result.boxes
                        for i in range(len(boxes_data)):
                            xyxy = boxes_data.xyxy[i].cpu().numpy()
                            conf = float(boxes_data.conf[i].cpu())

                            box = BoundingBox.from_xyxy(
                                x1=float(xyxy[0]),
                                y1=float(xyxy[1]),
                                x2=float(xyxy[2]),
                                y2=float(xyxy[3]),
                                img_width=img_width,
                                img_height=img_height,
                                class_id=class_id,
                                confidence=conf,
                                class_name=target_class,
                            )
                            all_boxes.append(box)

            except Exception as e:
                logger.warning(f"Error processing {target_class} in {image_path.name}: {e}")

        # Apply NMS to remove duplicate detections from multiple prompts
        # (e.g., same person detected as "person", "man", and "human")
        all_boxes = nms_boxes(all_boxes, iou_threshold=0.5)

        return all_boxes

    def predict_batch(self, image_paths: List[Path]) -> Dict[str, List[BoundingBox]]:
        """
        Run prediction on a batch of images.

        SAM 3 processes images one at a time but reuses features for multiple prompts.

        Args:
            image_paths: List of image paths

        Returns:
            Dictionary mapping image names to bounding boxes
        """
        results = {}
        for img_path in image_paths:
            try:
                boxes = self.predict_single(img_path)
                results[img_path.name] = boxes
            except Exception as e:
                logger.warning(f"Failed to process {img_path.name}: {e}")
                results[img_path.name] = []
        return results


def create_sam_labeler(
    output_dir: Optional[Path] = None,
    confidence_threshold: float = 0.3,
    device: str = "cuda",
    half_precision: bool = True,
) -> SAMLabeler:
    """
    Factory function to create a configured SAM labeler.

    Args:
        output_dir: Output directory for labels
        confidence_threshold: Minimum confidence threshold
        device: Device to run on
        half_precision: Whether to use FP16

    Returns:
        Configured SAMLabeler instance
    """
    config = SAMConfig(
        confidence_threshold=confidence_threshold,
        device=device,
        half_precision=half_precision,
    )

    return SAMLabeler(config=config, output_dir=output_dir)


if __name__ == "__main__":
    # Simple test
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python sam_labeler.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    labeler = create_sam_labeler()
    labeler.load_model()

    boxes = labeler.predict_single(image_path)
    print(f"Found {len(boxes)} detections:")
    for box in boxes:
        print(f"  {box.class_name}: conf={box.confidence:.3f}, area={box.area():.4f}")
