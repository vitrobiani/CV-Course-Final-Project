"""
Checkpoint utilities for saving and resuming labeling progress.
"""
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CheckpointData:
    """Data structure for checkpoint information."""

    # Which images have been processed
    processed_images: List[str]

    # Current index in the image list
    current_index: int

    # Labeler type (sam or florence)
    labeler_type: str

    # Configuration used
    config: Dict[str, Any]

    # Timestamp
    timestamp: str

    # Statistics
    stats: Dict[str, Any]

    # Version for compatibility
    version: str = "1.0"


class CheckpointManager:
    """Manages saving and loading checkpoints for the labeling process."""

    def __init__(self, checkpoint_dir: Path, labeler_type: str):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            labeler_type: Type of labeler (sam or florence)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.labeler_type = labeler_type
        self.checkpoint_file = self.checkpoint_dir / f"{labeler_type}_checkpoint.json"
        self.backup_file = self.checkpoint_dir / f"{labeler_type}_checkpoint_backup.json"

    def save(
        self,
        processed_images: List[str],
        current_index: int,
        config: Dict[str, Any],
        stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a checkpoint.

        Args:
            processed_images: List of processed image filenames
            current_index: Current position in the image list
            config: Configuration dictionary
            stats: Optional statistics dictionary
        """
        # Create backup of existing checkpoint
        if self.checkpoint_file.exists():
            self.checkpoint_file.rename(self.backup_file)

        checkpoint = CheckpointData(
            processed_images=processed_images,
            current_index=current_index,
            labeler_type=self.labeler_type,
            config=config,
            timestamp=datetime.now().isoformat(),
            stats=stats or {},
        )

        with open(self.checkpoint_file, "w") as f:
            json.dump(asdict(checkpoint), f, indent=2)

        logger.info(f"Checkpoint saved: {len(processed_images)} images processed")

    def load(self) -> Optional[CheckpointData]:
        """
        Load the latest checkpoint.

        Returns:
            CheckpointData if checkpoint exists, None otherwise
        """
        checkpoint_path = self.checkpoint_file
        if not checkpoint_path.exists():
            checkpoint_path = self.backup_file

        if not checkpoint_path.exists():
            logger.info("No checkpoint found, starting fresh")
            return None

        try:
            with open(checkpoint_path, "r") as f:
                data = json.load(f)

            checkpoint = CheckpointData(**data)
            logger.info(
                f"Checkpoint loaded: {len(checkpoint.processed_images)} images "
                f"already processed (index: {checkpoint.current_index})"
            )
            return checkpoint

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def get_remaining_images(
        self,
        all_images: List[str],
        checkpoint: Optional[CheckpointData] = None,
    ) -> List[str]:
        """
        Get list of images that still need to be processed.

        Args:
            all_images: Complete list of all images
            checkpoint: Optional checkpoint data (will load if not provided)

        Returns:
            List of image paths that haven't been processed
        """
        if checkpoint is None:
            checkpoint = self.load()

        if checkpoint is None:
            return all_images

        processed_set = set(checkpoint.processed_images)
        remaining = [img for img in all_images if img not in processed_set]

        logger.info(
            f"Resuming: {len(remaining)} images remaining "
            f"({len(processed_set)} already done)"
        )
        return remaining

    def clear(self) -> None:
        """Remove all checkpoints."""
        for f in [self.checkpoint_file, self.backup_file]:
            if f.exists():
                f.unlink()
        logger.info("Checkpoints cleared")

    def exists(self) -> bool:
        """Check if a checkpoint exists."""
        return self.checkpoint_file.exists() or self.backup_file.exists()


class LabelCache:
    """Cache for storing labels during processing."""

    def __init__(self, cache_dir: Path, labeler_type: str):
        """
        Initialize the label cache.

        Args:
            cache_dir: Directory to store cached labels
            labeler_type: Type of labeler (sam or florence)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.labeler_type = labeler_type
        self.cache_file = self.cache_dir / f"{labeler_type}_labels_cache.pkl"

    def save_batch(self, labels: Dict[str, List[Dict]]) -> None:
        """
        Save a batch of labels to cache.

        Args:
            labels: Dictionary mapping image names to their label lists
        """
        existing = self.load_all()
        existing.update(labels)

        with open(self.cache_file, "wb") as f:
            pickle.dump(existing, f)

    def load_all(self) -> Dict[str, List[Dict]]:
        """Load all cached labels."""
        if not self.cache_file.exists():
            return {}

        try:
            with open(self.cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load label cache: {e}")
            return {}

    def clear(self) -> None:
        """Clear the cache."""
        if self.cache_file.exists():
            self.cache_file.unlink()
