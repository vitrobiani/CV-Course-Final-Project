"""
Florence-2 based labeler for object detection.

Florence-2 is a vision-language model that can perform various tasks
including object detection with text prompts.
"""
from pathlib import Path
from typing import List, Dict, Optional
import logging

import torch
from PIL import Image

from .base_labeler import BaseLabeler
from ..config import FlorenceConfig, DEFAULT_FLORENCE_CONFIG
from ..utils.dataset_utils import BoundingBox

logger = logging.getLogger(__name__)


class FlorenceLabeler(BaseLabeler):
    """
    Florence-2 based auto-labeler using object detection and phrase grounding.

    Uses Florence-2's object detection task to detect objects,
    then maps detected classes to our target classes (person, vehicle).
    """

    def __init__(
        self,
        config: Optional[FlorenceConfig] = None,
        output_dir: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize Florence labeler.

        Args:
            config: Florence configuration
            output_dir: Directory for output labels
            checkpoint_dir: Directory for checkpoints
        """
        config = config or DEFAULT_FLORENCE_CONFIG
        output_dir = output_dir or Path("outputs/florence_labels")

        super().__init__(config, output_dir, checkpoint_dir)

        self.florence_config: FlorenceConfig = config
        self.processor = None
        self.model = None

    @property
    def labeler_name(self) -> str:
        return "florence"

    def load_model(self) -> None:
        """Load Florence-2 model from HuggingFace."""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "Florence-2 requires transformers. "
                "Install with: pip install 'transformers>=4.44.0,<4.46.0'"
            )

        logger.info(f"Loading Florence-2 model: {self.florence_config.model_name}")

        # Load processor and model
        # Note: Requires transformers 4.44.x-4.45.x for compatibility
        self.processor = AutoProcessor.from_pretrained(
            self.florence_config.model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.florence_config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.florence_config.half_precision else torch.float32,
            attn_implementation="eager",  # Avoid flash_attn requirement
        ).to(self.device)

        self.model.eval()
        logger.info("Florence-2 model loaded successfully")

    def _run_inference(
        self,
        image: Image.Image,
        task_prompt: str,
        text_input: Optional[str] = None,
    ) -> Dict:
        """
        Run Florence-2 inference.

        Args:
            image: PIL Image
            task_prompt: Task prompt (e.g., "<OD>" for object detection)
            text_input: Optional additional text input

        Returns:
            Parsed prediction dictionary
        """
        if text_input:
            prompt = task_prompt + text_input
        else:
            prompt = task_prompt

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device, torch.float16 if self.florence_config.half_precision else torch.float32)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # Parse the response
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )

        return parsed

    def _parse_od_results(
        self,
        results: Dict,
        img_width: int,
        img_height: int,
    ) -> List[BoundingBox]:
        """
        Parse object detection results from Florence-2.

        Args:
            results: Raw results from Florence-2
            img_width: Image width
            img_height: Image height

        Returns:
            List of bounding boxes
        """
        boxes = []

        # Florence-2 OD output format: {'<OD>': {'bboxes': [...], 'labels': [...]}}
        od_key = self.florence_config.task_prompt
        if od_key not in results:
            return boxes

        od_results = results[od_key]
        bboxes = od_results.get("bboxes", [])
        labels = od_results.get("labels", [])

        for bbox, label in zip(bboxes, labels):
            # Map label to our classes
            class_id = self.map_class_to_id(label)
            if class_id is None:
                continue  # Skip classes we don't care about

            # Florence returns [x1, y1, x2, y2] in pixel coordinates
            x1, y1, x2, y2 = bbox

            # Get target class name
            class_name = "person" if class_id == 0 else "vehicle"

            box = BoundingBox.from_xyxy(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                img_width=img_width,
                img_height=img_height,
                class_id=class_id,
                confidence=1.0,  # Florence doesn't return confidence for OD
                class_name=class_name,
            )
            boxes.append(box)

        return boxes

    def _parse_grounding_results(
        self,
        results: Dict,
        img_width: int,
        img_height: int,
        target_class: str,
        class_id: int,
    ) -> List[BoundingBox]:
        """
        Parse phrase grounding results from Florence-2.

        Args:
            results: Raw results from Florence-2
            img_width: Image width
            img_height: Image height
            target_class: Target class name
            class_id: Target class ID

        Returns:
            List of bounding boxes
        """
        boxes = []

        # Phrase grounding output format
        grounding_key = self.florence_config.phrase_grounding_prompt
        if grounding_key not in results:
            return boxes

        grounding_results = results[grounding_key]
        bboxes = grounding_results.get("bboxes", [])

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox

            box = BoundingBox.from_xyxy(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                img_width=img_width,
                img_height=img_height,
                class_id=class_id,
                confidence=1.0,
                class_name=target_class,
            )
            boxes.append(box)

        return boxes

    def _resize_to_square(self, image: Image.Image) -> tuple:
        """
        Resize image to square by padding with black.

        Returns:
            Tuple of (square_image, original_width, original_height, pad_left, pad_top)
        """
        orig_width, orig_height = image.size
        max_dim = max(orig_width, orig_height)

        # Create square black canvas
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))

        # Paste original image centered
        pad_left = (max_dim - orig_width) // 2
        pad_top = (max_dim - orig_height) // 2
        square_img.paste(image, (pad_left, pad_top))

        return square_img, orig_width, orig_height, pad_left, pad_top

    def _adjust_bbox_from_square(
        self,
        bbox: List[float],
        orig_width: int,
        orig_height: int,
        pad_left: int,
        pad_top: int,
        square_size: int
    ) -> List[float]:
        """
        Adjust bounding box coordinates from square image back to original dimensions.
        """
        x1, y1, x2, y2 = bbox

        # Remove padding offset
        x1 = x1 - pad_left
        y1 = y1 - pad_top
        x2 = x2 - pad_left
        y2 = y2 - pad_top

        # Clip to original image bounds
        x1 = max(0, min(x1, orig_width))
        y1 = max(0, min(y1, orig_height))
        x2 = max(0, min(x2, orig_width))
        y2 = max(0, min(y2, orig_height))

        return [x1, y1, x2, y2]

    def predict_single(self, image_path: Path) -> List[BoundingBox]:
        """
        Run Florence-2 prediction on a single image.

        Uses object detection task and maps results to our classes.

        Args:
            image_path: Path to the image

        Returns:
            List of detected bounding boxes
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Load image
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size

        # Resize to square (Florence requires square feature maps)
        square_img, orig_w, orig_h, pad_left, pad_top = self._resize_to_square(image)
        square_size = square_img.size[0]

        all_boxes = []

        try:
            # Method 1: General object detection
            od_results = self._run_inference(
                square_img,
                self.florence_config.task_prompt,
            )
            # Adjust coordinates back to original image space
            od_key = self.florence_config.task_prompt
            if od_key in od_results:
                adjusted_bboxes = []
                for bbox in od_results[od_key].get("bboxes", []):
                    adj_bbox = self._adjust_bbox_from_square(
                        bbox, orig_w, orig_h, pad_left, pad_top, square_size
                    )
                    adjusted_bboxes.append(adj_bbox)
                od_results[od_key]["bboxes"] = adjusted_bboxes

            od_boxes = self._parse_od_results(od_results, img_width, img_height)
            all_boxes.extend(od_boxes)

        except Exception as e:
            logger.warning(f"OD failed for {image_path.name}: {e}")

        # Method 2: Phrase grounding for specific classes (backup/supplement)
        # This can catch objects that generic OD might miss
        for target_class, synonyms in self.config.class_synonyms.items():
            class_id = self.config.class_to_id.get(target_class)
            if class_id is None:
                continue

            # Use primary synonym for grounding
            phrase = synonyms[0] if synonyms else target_class

            try:
                grounding_results = self._run_inference(
                    square_img,
                    self.florence_config.phrase_grounding_prompt,
                    text_input=phrase,
                )
                # Adjust grounding results coordinates
                grounding_key = self.florence_config.phrase_grounding_prompt
                if grounding_key in grounding_results:
                    adjusted_bboxes = []
                    for bbox in grounding_results[grounding_key].get("bboxes", []):
                        adj_bbox = self._adjust_bbox_from_square(
                            bbox, orig_w, orig_h, pad_left, pad_top, square_size
                        )
                        adjusted_bboxes.append(adj_bbox)
                    grounding_results[grounding_key]["bboxes"] = adjusted_bboxes

                grounding_boxes = self._parse_grounding_results(
                    grounding_results, img_width, img_height, target_class, class_id
                )

                # Only add if significantly different from OD results
                # (avoid duplicates)
                for new_box in grounding_boxes:
                    is_duplicate = False
                    for existing_box in all_boxes:
                        if existing_box.class_id == new_box.class_id:
                            iou = self._compute_iou(existing_box, new_box)
                            if iou > 0.5:
                                is_duplicate = True
                                break
                    if not is_duplicate:
                        all_boxes.append(new_box)

            except Exception as e:
                logger.debug(f"Grounding failed for {phrase} in {image_path.name}: {e}")

        return all_boxes

    def _compute_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Compute IoU between two normalized bounding boxes."""
        # Convert to xyxy (still normalized)
        x1_1 = box1.x_center - box1.width / 2
        y1_1 = box1.y_center - box1.height / 2
        x2_1 = box1.x_center + box1.width / 2
        y2_1 = box1.y_center + box1.height / 2

        x1_2 = box2.x_center - box2.width / 2
        y1_2 = box2.y_center - box2.height / 2
        x2_2 = box2.x_center + box2.width / 2
        y2_2 = box2.y_center + box2.height / 2

        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        # Union
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def predict_batch(self, image_paths: List[Path]) -> Dict[str, List[BoundingBox]]:
        """
        Run prediction on a batch of images.

        Florence-2 processes images one at a time.

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


def create_florence_labeler(
    output_dir: Optional[Path] = None,
    confidence_threshold: float = 0.3,
    device: str = "cuda",
    half_precision: bool = True,
    model_name: str = "microsoft/Florence-2-large",
) -> FlorenceLabeler:
    """
    Factory function to create a configured Florence labeler.

    Args:
        output_dir: Output directory for labels
        confidence_threshold: Minimum confidence threshold
        device: Device to run on
        half_precision: Whether to use FP16
        model_name: HuggingFace model name

    Returns:
        Configured FlorenceLabeler instance
    """
    config = FlorenceConfig(
        confidence_threshold=confidence_threshold,
        device=device,
        half_precision=half_precision,
        model_name=model_name,
    )

    return FlorenceLabeler(config=config, output_dir=output_dir)


if __name__ == "__main__":
    # Simple test
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python florence_labeler.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    labeler = create_florence_labeler()
    labeler.load_model()

    boxes = labeler.predict_single(image_path)
    print(f"Found {len(boxes)} detections:")
    for box in boxes:
        print(f"  {box.class_name}: conf={box.confidence:.3f}, area={box.area():.4f}")
