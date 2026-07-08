"""
Dataset loading and conversion utilities.
"""
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator
import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box representation."""

    x_center: float  # Normalized (0-1)
    y_center: float  # Normalized (0-1)
    width: float  # Normalized (0-1)
    height: float  # Normalized (0-1)
    class_id: int
    confidence: float = 1.0
    class_name: str = ""

    def to_yolo(self) -> str:
        """Convert to YOLO format string."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    def to_yolo_with_conf(self) -> str:
        """Convert to YOLO format string with confidence."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f} {self.confidence:.6f}"

    @classmethod
    def from_yolo(cls, line: str, class_names: Optional[Dict[int, str]] = None) -> "BoundingBox":
        """Parse from YOLO format string."""
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        confidence = float(parts[5]) if len(parts) > 5 else 1.0
        class_name = class_names.get(class_id, "") if class_names else ""

        return cls(
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            class_id=class_id,
            confidence=confidence,
            class_name=class_name,
        )

    @classmethod
    def from_xyxy(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        img_width: int,
        img_height: int,
        class_id: int,
        confidence: float = 1.0,
        class_name: str = "",
    ) -> "BoundingBox":
        """Create from absolute xyxy coordinates."""
        # Normalize coordinates
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        return cls(
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            class_id=class_id,
            confidence=confidence,
            class_name=class_name,
        )

    @classmethod
    def from_mask(
        cls,
        mask: np.ndarray,
        class_id: int,
        confidence: float = 1.0,
        class_name: str = "",
    ) -> Optional["BoundingBox"]:
        """Create bounding box from binary mask."""
        # Find non-zero pixels
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]

        y1, y2 = y_indices[0], y_indices[-1]
        x1, x2 = x_indices[0], x_indices[-1]

        img_height, img_width = mask.shape[:2]

        return cls.from_xyxy(
            x1, y1, x2, y2,
            img_width, img_height,
            class_id, confidence, class_name
        )

    def to_xyxy(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert to absolute xyxy coordinates."""
        x1 = int((self.x_center - self.width / 2) * img_width)
        y1 = int((self.y_center - self.height / 2) * img_height)
        x2 = int((self.x_center + self.width / 2) * img_width)
        y2 = int((self.y_center + self.height / 2) * img_height)
        return (x1, y1, x2, y2)

    def area(self) -> float:
        """Get normalized area."""
        return self.width * self.height

    def is_valid(self, min_size: float = 0.001, max_size: float = 0.99) -> bool:
        """Check if bounding box is valid."""
        area = self.area()
        if area < min_size or area > max_size:
            return False
        if self.width <= 0 or self.height <= 0:
            return False
        if not (0 <= self.x_center <= 1 and 0 <= self.y_center <= 1):
            return False
        return True


def load_flickr_images(images_dir: Path) -> List[Path]:
    """
    Load list of all Flickr images.

    Args:
        images_dir: Path to Flickr images directory

    Returns:
        List of image paths
    """
    images_dir = Path(images_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

    images = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    logger.info(f"Found {len(images)} images in {images_dir}")
    return sorted(images)


def load_flickr_captions(captions_csv: Path) -> Dict[str, List[str]]:
    """
    Load Flickr captions from CSV.

    Args:
        captions_csv: Path to results.csv

    Returns:
        Dictionary mapping image names to list of captions
    """
    captions = {}
    captions_csv = Path(captions_csv)

    with open(captions_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        next(reader)  # Skip header

        for row in reader:
            if len(row) >= 3:
                image_name = row[0].strip()
                caption = row[2].strip()
                if image_name not in captions:
                    captions[image_name] = []
                captions[image_name].append(caption)

    logger.info(f"Loaded captions for {len(captions)} images")
    return captions


def load_yolo_dataset(dataset_dir: Path, split: str = "train") -> Dict[str, List[BoundingBox]]:
    """
    Load YOLO format dataset.

    Args:
        dataset_dir: Path to dataset root
        split: Dataset split (train, valid, test)

    Returns:
        Dictionary mapping image names to list of bounding boxes
    """
    dataset_dir = Path(dataset_dir)
    labels_dir = dataset_dir / split / "labels"
    images_dir = dataset_dir / split / "images"

    # Load class names from data.yaml
    class_names = {}
    yaml_path = dataset_dir / "data.yaml"
    if yaml_path.exists():
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
            names = data.get("names", [])
            class_names = {i: name for i, name in enumerate(names)}

    labels = {}
    for label_file in labels_dir.glob("*.txt"):
        image_name = label_file.stem
        boxes = []

        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        box = BoundingBox.from_yolo(line, class_names)
                        boxes.append(box)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse line in {label_file}: {e}")

        labels[image_name] = boxes

    logger.info(f"Loaded {len(labels)} label files from {split} split")
    return labels


def save_yolo_labels(
    labels: Dict[str, List[BoundingBox]],
    output_dir: Path,
    include_confidence: bool = False,
) -> None:
    """
    Save labels in YOLO format.

    Args:
        labels: Dictionary mapping image names to bounding boxes
        output_dir: Output directory
        include_confidence: Whether to include confidence scores
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_name, boxes in labels.items():
        # Remove extension from image name
        stem = Path(image_name).stem
        label_file = output_dir / f"{stem}.txt"

        with open(label_file, "w") as f:
            for box in boxes:
                if include_confidence:
                    f.write(box.to_yolo_with_conf() + "\n")
                else:
                    f.write(box.to_yolo() + "\n")

    logger.info(f"Saved {len(labels)} label files to {output_dir}")


def image_batch_generator(
    image_paths: List[Path],
    batch_size: int = 8,
) -> Generator[List[Path], None, None]:
    """
    Generate batches of image paths.

    Args:
        image_paths: List of image paths
        batch_size: Number of images per batch

    Yields:
        Batches of image paths
    """
    for i in range(0, len(image_paths), batch_size):
        yield image_paths[i:i + batch_size]


def filter_images_by_caption(
    images: List[Path],
    captions: Dict[str, List[str]],
    keywords: List[str],
) -> List[Path]:
    """
    Filter images that have captions containing specific keywords.

    Args:
        images: List of image paths
        captions: Dictionary of captions
        keywords: Keywords to search for

    Returns:
        Filtered list of images
    """
    keywords_lower = [k.lower() for k in keywords]
    filtered = []

    for img_path in images:
        img_name = img_path.name
        if img_name in captions:
            caption_text = " ".join(captions[img_name]).lower()
            if any(kw in caption_text for kw in keywords_lower):
                filtered.append(img_path)

    logger.info(
        f"Filtered {len(filtered)} images containing keywords: {keywords}"
    )
    return filtered


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """Get image width and height."""
    with Image.open(image_path) as img:
        return img.size  # (width, height)
