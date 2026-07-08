"""
Visualization utilities for bounding boxes and comparisons.
"""
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .dataset_utils import BoundingBox

logger = logging.getLogger(__name__)

# Color palette for different classes and sources
COLORS = {
    "person": (255, 0, 0),      # Red
    "vehicle": (0, 0, 255),     # Blue
    "car": (0, 0, 255),         # Blue (alias)
    "sam": (0, 255, 0),         # Green
    "florence": (255, 165, 0),  # Orange
    "ground_truth": (128, 0, 128),  # Purple
    "default": (255, 255, 0),   # Yellow
}


def get_color(key: str) -> Tuple[int, int, int]:
    """Get color for a given key."""
    return COLORS.get(key.lower(), COLORS["default"])


def draw_boxes_on_image(
    image: Image.Image,
    boxes: List[BoundingBox],
    color_by: str = "class",  # "class" or "source"
    source_name: Optional[str] = None,
    line_width: int = 2,
    show_labels: bool = True,
    show_confidence: bool = True,
) -> Image.Image:
    """
    Draw bounding boxes on an image.

    Args:
        image: PIL Image
        boxes: List of bounding boxes
        color_by: How to color boxes ("class" or "source")
        source_name: Source name for coloring when color_by="source"
        line_width: Line width for boxes
        show_labels: Whether to show class labels
        show_confidence: Whether to show confidence scores

    Returns:
        Image with drawn boxes
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    width, height = image.size

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for box in boxes:
        # Get color
        if color_by == "source" and source_name:
            color = get_color(source_name)
        else:
            color = get_color(box.class_name or str(box.class_id))

        # Convert normalized coords to pixels
        x1, y1, x2, y2 = box.to_xyxy(width, height)

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Draw label
        if show_labels:
            label_parts = []
            if box.class_name:
                label_parts.append(box.class_name)
            else:
                label_parts.append(f"cls_{box.class_id}")

            if show_confidence and box.confidence < 1.0:
                label_parts.append(f"{box.confidence:.2f}")

            label = " ".join(label_parts)

            # Draw label background
            bbox = draw.textbbox((x1, y1 - 15), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1 - 15), label, fill=(255, 255, 255), font=font)

    return img_copy


def visualize_comparison(
    image_path: Path,
    sam_boxes: List[BoundingBox],
    florence_boxes: List[BoundingBox],
    ground_truth_boxes: Optional[List[BoundingBox]] = None,
    output_path: Optional[Path] = None,
) -> Image.Image:
    """
    Create a side-by-side comparison visualization.

    Args:
        image_path: Path to the original image
        sam_boxes: Boxes from SAM
        florence_boxes: Boxes from Florence
        ground_truth_boxes: Optional ground truth boxes
        output_path: Optional path to save the result

    Returns:
        Combined comparison image
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Determine layout based on whether we have ground truth
    if ground_truth_boxes is not None:
        # 3 images side by side
        n_images = 3
        labels = ["SAM", "Florence", "Ground Truth"]
        box_lists = [sam_boxes, florence_boxes, ground_truth_boxes]
        source_names = ["sam", "florence", "ground_truth"]
    else:
        # 2 images side by side
        n_images = 2
        labels = ["SAM", "Florence"]
        box_lists = [sam_boxes, florence_boxes]
        source_names = ["sam", "florence"]

    # Create combined image
    combined_width = width * n_images
    combined = Image.new("RGB", (combined_width, height + 30))

    # Draw each version
    for i, (label, boxes, source) in enumerate(zip(labels, box_lists, source_names)):
        # Draw boxes
        img_with_boxes = draw_boxes_on_image(
            image, boxes,
            color_by="source",
            source_name=source,
            show_labels=True,
        )

        # Paste into combined image
        combined.paste(img_with_boxes, (i * width, 30))

        # Add label at top
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except (IOError, OSError):
            font = ImageFont.load_default()

        color = get_color(source)
        draw.rectangle([i * width, 0, (i + 1) * width, 30], fill=color)
        draw.text((i * width + 10, 5), f"{label} ({len(boxes)} detections)", fill=(255, 255, 255), font=font)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.save(output_path)
        logger.info(f"Saved comparison to {output_path}")

    return combined


def create_metrics_visualization(
    metrics: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str = "Comparison Metrics",
) -> None:
    """
    Create a simple text-based metrics visualization.

    Args:
        metrics: Dictionary of metrics per source/method
        output_path: Path to save the visualization
        title: Title for the visualization
    """
    # Create a simple image with metrics text
    img_width = 600
    img_height = 400

    image = Image.new("RGB", (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    # Draw title
    draw.text((20, 20), title, fill=(0, 0, 0), font=title_font)

    # Draw metrics
    y_offset = 60
    for source, source_metrics in metrics.items():
        color = get_color(source)
        draw.text((20, y_offset), f"{source}:", fill=color, font=title_font)
        y_offset += 30

        for metric_name, value in source_metrics.items():
            if isinstance(value, float):
                text = f"  {metric_name}: {value:.4f}"
            else:
                text = f"  {metric_name}: {value}"
            draw.text((40, y_offset), text, fill=(0, 0, 0), font=text_font)
            y_offset += 25

        y_offset += 10

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    logger.info(f"Saved metrics visualization to {output_path}")


def save_sample_grid(
    images_with_boxes: List[Tuple[Path, List[BoundingBox]]],
    output_path: Path,
    grid_size: Tuple[int, int] = (4, 4),
    cell_size: Tuple[int, int] = (300, 300),
) -> None:
    """
    Create a grid of sample images with boxes.

    Args:
        images_with_boxes: List of (image_path, boxes) tuples
        output_path: Path to save the grid
        grid_size: (columns, rows)
        cell_size: Size of each cell
    """
    cols, rows = grid_size
    cell_w, cell_h = cell_size

    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(255, 255, 255))

    for idx, (img_path, boxes) in enumerate(images_with_boxes[:cols * rows]):
        row = idx // cols
        col = idx % cols

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(cell_size, Image.Resampling.LANCZOS)

            # Scale boxes to cell size
            orig_w, orig_h = Image.open(img_path).size
            scaled_boxes = []
            for box in boxes:
                scaled_boxes.append(box)  # Boxes are normalized, so they scale automatically

            img_with_boxes = draw_boxes_on_image(img, scaled_boxes, show_confidence=False)
            grid.paste(img_with_boxes, (col * cell_w, row * cell_h))

        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    logger.info(f"Saved sample grid to {output_path}")
