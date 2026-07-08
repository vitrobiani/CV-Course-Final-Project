"""
Base class for auto-labeling implementations.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import asdict
import logging
import time

from tqdm import tqdm

from ..config import LabelingConfig, CHECKPOINTS_DIR
from ..utils.checkpoint import CheckpointManager, LabelCache
from ..utils.dataset_utils import (
    BoundingBox,
    load_flickr_images,
    save_yolo_labels,
    image_batch_generator,
)

logger = logging.getLogger(__name__)


class BaseLabeler(ABC):
    """Abstract base class for image labelers."""

    def __init__(
        self,
        config: LabelingConfig,
        output_dir: Path,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize the labeler.

        Args:
            config: Labeling configuration
            output_dir: Directory to save output labels
            checkpoint_dir: Directory for checkpoints (default: CHECKPOINTS_DIR)
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_dir = checkpoint_dir or CHECKPOINTS_DIR
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir, self.labeler_name
        )
        self.label_cache = LabelCache(checkpoint_dir, self.labeler_name)

        self.model = None
        self.device = config.device
        self.stats = {
            "total_images": 0,
            "total_detections": 0,
            "images_with_detections": 0,
            "processing_time": 0.0,
        }

    @property
    @abstractmethod
    def labeler_name(self) -> str:
        """Name of the labeler (used for checkpoints)."""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Load the model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def predict_single(self, image_path: Path) -> List[BoundingBox]:
        """
        Run prediction on a single image.

        Args:
            image_path: Path to the image

        Returns:
            List of detected bounding boxes
        """
        pass

    def predict_batch(self, image_paths: List[Path]) -> Dict[str, List[BoundingBox]]:
        """
        Run prediction on a batch of images.
        Default implementation processes images one by one.
        Override for batch-optimized implementations.

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

    def filter_boxes(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """
        Filter bounding boxes based on config criteria.

        Args:
            boxes: List of bounding boxes

        Returns:
            Filtered list of bounding boxes
        """
        filtered = []
        for box in boxes:
            # Check confidence
            if box.confidence < self.config.confidence_threshold:
                continue

            # Check size
            if not box.is_valid(
                min_size=self.config.min_box_size,
                max_size=self.config.max_box_size,
            ):
                continue

            filtered.append(box)

        return filtered

    def map_class_to_id(self, class_name: str) -> Optional[int]:
        """
        Map detected class name to our class IDs.

        Args:
            class_name: Detected class name

        Returns:
            Class ID or None if not in our classes
        """
        class_name_lower = class_name.lower()

        # Check direct mapping
        if class_name_lower in self.config.class_to_id:
            return self.config.class_to_id[class_name_lower]

        # Check synonyms
        for target_class, synonyms in self.config.class_synonyms.items():
            if class_name_lower in [s.lower() for s in synonyms]:
                return self.config.class_to_id[target_class]

        return None

    def process_dataset(
        self,
        images_dir: Path,
        resume: bool = True,
        max_images: Optional[int] = None,
        save_interval: Optional[int] = None,
    ) -> Dict[str, List[BoundingBox]]:
        """
        Process all images in a directory.

        Args:
            images_dir: Directory containing images
            resume: Whether to resume from checkpoint
            max_images: Maximum number of images to process
            save_interval: Save checkpoint every N images (default from config)

        Returns:
            Dictionary mapping image names to bounding boxes
        """
        # Load model if not already loaded
        if self.model is None:
            logger.info(f"Loading {self.labeler_name} model...")
            self.load_model()

        # Get list of images
        all_images = load_flickr_images(images_dir)
        if max_images:
            all_images = all_images[:max_images]

        # Handle resume
        all_results = {}
        if resume and self.checkpoint_manager.exists():
            checkpoint = self.checkpoint_manager.load()
            if checkpoint:
                all_images = self.checkpoint_manager.get_remaining_images(
                    [str(p) for p in all_images], checkpoint
                )
                all_images = [Path(p) for p in all_images]
                all_results = self.label_cache.load_all()
                self.stats = checkpoint.stats

        save_interval = save_interval or self.config.checkpoint_interval
        batch_size = self.config.batch_size

        logger.info(f"Processing {len(all_images)} images with {self.labeler_name}")
        start_time = time.time()

        processed_count = len(all_results)
        for batch in tqdm(
            image_batch_generator(all_images, batch_size),
            total=(len(all_images) + batch_size - 1) // batch_size,
            desc=f"{self.labeler_name} labeling",
        ):
            # Process batch
            batch_results = self.predict_batch(batch)

            # Filter and update stats
            for img_name, boxes in batch_results.items():
                filtered_boxes = self.filter_boxes(boxes)
                all_results[img_name] = filtered_boxes

                self.stats["total_images"] += 1
                self.stats["total_detections"] += len(filtered_boxes)
                if filtered_boxes:
                    self.stats["images_with_detections"] += 1

            processed_count += len(batch)

            # Save checkpoint periodically
            if processed_count % save_interval == 0:
                self._save_checkpoint(all_results, processed_count)

        # Final save
        self.stats["processing_time"] = time.time() - start_time
        self._save_checkpoint(all_results, processed_count)

        # Save YOLO format labels
        save_yolo_labels(all_results, self.output_dir / "labels")

        logger.info(
            f"Completed: {self.stats['total_images']} images, "
            f"{self.stats['total_detections']} detections, "
            f"{self.stats['processing_time']:.2f}s"
        )

        return all_results

    def _save_checkpoint(
        self,
        results: Dict[str, List[BoundingBox]],
        current_index: int,
    ) -> None:
        """Save checkpoint and cache."""
        self.label_cache.save_batch(results)
        self.checkpoint_manager.save(
            processed_images=list(results.keys()),
            current_index=current_index,
            config=asdict(self.config),
            stats=self.stats,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()

    def clear_checkpoints(self) -> None:
        """Clear all checkpoints and cache."""
        self.checkpoint_manager.clear()
        self.label_cache.clear()
        logger.info("Cleared all checkpoints and cache")
