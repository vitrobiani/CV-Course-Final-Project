#!/usr/bin/env python3
"""
Visualize labeled images with bounding boxes.

Saves images to folders for easy browsing with any image viewer.
"""
import argparse
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SAM_LABELS_DIR, FLORENCE_LABELS_DIR, FLICKR_IMAGES_DIR, OUTPUTS_DIR


# Colors (RGB)
COLORS = {
    "sam": {
        "person": (0, 255, 0),      # Green
        "vehicle": (0, 200, 0),     # Dark green
    },
    "florence": {
        "person": (255, 165, 0),    # Orange
        "vehicle": (255, 100, 0),   # Dark orange
    },
    "both": {
        "person": (0, 150, 255),    # Blue (for overlay)
        "vehicle": (0, 100, 200),
    }
}

CLASS_NAMES = {0: "person", 1: "vehicle"}


def load_labels(label_path: Path) -> list:
    """Load YOLO format labels."""
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                boxes.append({
                    "class_id": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                })
    return boxes


def draw_boxes(image: Image.Image, boxes: list, source: str, line_width: int = 3) -> Image.Image:
    """Draw bounding boxes on image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for box in boxes:
        class_id = box["class_id"]
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        color = COLORS[source].get(class_name, (255, 255, 255))

        # Convert YOLO format to pixel coordinates
        x_center = box["x_center"] * w
        y_center = box["y_center"] * h
        box_w = box["width"] * w
        box_h = box["height"] * h

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Draw label background and text
        label = class_name
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
        draw.text((x1, y1), label, fill=(0, 0, 0), font=font)

    return img


def create_comparison_image(image: Image.Image, sam_boxes: list, florence_boxes: list,
                            image_name: str) -> Image.Image:
    """Create side-by-side comparison image."""
    w, h = image.size

    # Draw SAM boxes
    sam_img = draw_boxes(image, sam_boxes, "sam")

    # Draw Florence boxes
    florence_img = draw_boxes(image, florence_boxes, "florence")

    # Create side-by-side (with header)
    header_h = 35
    combined = Image.new("RGB", (w * 2, h + header_h), (30, 30, 30))

    # Paste images
    combined.paste(sam_img, (0, header_h))
    combined.paste(florence_img, (w, header_h))

    # Add headers
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = font

    # SAM header
    draw.text((10, 8), f"SAM ({len(sam_boxes)} detections)", fill=(0, 255, 0), font=font)

    # Florence header
    draw.text((w + 10, 8), f"Florence ({len(florence_boxes)} detections)", fill=(255, 165, 0), font=font)

    # Image name in center
    draw.text((w - 80, 10), image_name, fill=(200, 200, 200), font=small_font)

    return combined


def save_single_labeler(images_dir: Path, labels_dir: Path, output_dir: Path,
                        labeler_name: str, max_images: int = None):
    """Save visualizations for a single labeler."""
    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))
    if max_images:
        label_files = label_files[:max_images]

    print(f"Saving {len(label_files)} {labeler_name} visualizations to {output_dir}")

    source = "sam" if "sam" in labeler_name.lower() else "florence"

    for i, label_path in enumerate(label_files):
        stem = label_path.stem

        # Find image file
        image_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                image_path = p
                break

        if image_path is None:
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            continue

        boxes = load_labels(label_path)

        # Draw boxes
        vis_img = draw_boxes(image, boxes, source)

        # Add header with info
        w, h = vis_img.size
        header_h = 30
        final = Image.new("RGB", (w, h + header_h), (30, 30, 30))
        final.paste(vis_img, (0, header_h))

        draw = ImageDraw.Draw(final)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()

        color = (0, 255, 0) if source == "sam" else (255, 165, 0)
        draw.text((10, 6), f"{labeler_name}: {stem} ({len(boxes)} detections)", fill=color, font=font)

        # Save
        out_path = output_dir / f"{stem}.jpg"
        final.save(out_path, quality=90)

        if (i + 1) % 100 == 0:
            print(f"  Saved {i + 1}/{len(label_files)}")

    print(f"Done! Browse: {output_dir}")


def save_comparisons(images_dir: Path, sam_labels_dir: Path, florence_labels_dir: Path,
                     output_dir: Path, filter_type: str = "all", max_images: int = None):
    """Save comparison images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sam_labels = {p.stem for p in sam_labels_dir.glob("*.txt")}
    florence_labels = {p.stem for p in florence_labels_dir.glob("*.txt")}

    if filter_type == "disagree":
        # Find images where detection counts differ
        image_stems = []
        for stem in sorted(sam_labels & florence_labels):
            sam_boxes = load_labels(sam_labels_dir / f"{stem}.txt")
            flo_boxes = load_labels(florence_labels_dir / f"{stem}.txt")
            if len(sam_boxes) != len(flo_boxes):
                image_stems.append((stem, abs(len(sam_boxes) - len(flo_boxes))))
        # Sort by disagreement amount
        image_stems = [s[0] for s in sorted(image_stems, key=lambda x: -x[1])]
    elif filter_type == "both":
        image_stems = sorted(sam_labels & florence_labels)
    else:
        image_stems = sorted(sam_labels | florence_labels)

    if max_images:
        image_stems = image_stems[:max_images]

    print(f"Saving {len(image_stems)} comparison images to {output_dir}")

    for i, stem in enumerate(image_stems):
        # Find image file
        image_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                image_path = p
                break

        if image_path is None:
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            continue

        sam_boxes = load_labels(sam_labels_dir / f"{stem}.txt")
        florence_boxes = load_labels(florence_labels_dir / f"{stem}.txt")

        # Create comparison
        comparison = create_comparison_image(image, sam_boxes, florence_boxes, stem)

        # Save
        out_path = output_dir / f"{stem}.jpg"
        comparison.save(out_path, quality=90)

        if (i + 1) % 100 == 0:
            print(f"  Saved {i + 1}/{len(image_stems)}")

    print(f"Done! Browse: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize labeled images")
    parser.add_argument("--view", choices=["sam", "florence", "compare", "disagree", "all"],
                        default="all", help="What to visualize")
    parser.add_argument("--images-dir", type=str, default=None, help="Images directory")
    parser.add_argument("--sam-labels", type=str, default=None, help="SAM labels directory")
    parser.add_argument("--florence-labels", type=str, default=None, help="Florence labels directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max-images", type=int, default=100, help="Max images to save (default: 100)")
    args = parser.parse_args()

    # Set defaults
    images_dir = Path(args.images_dir) if args.images_dir else FLICKR_IMAGES_DIR
    sam_labels_dir = Path(args.sam_labels) if args.sam_labels else SAM_LABELS_DIR / "labels"
    florence_labels_dir = Path(args.florence_labels) if args.florence_labels else FLORENCE_LABELS_DIR / "labels"

    print(f"Images: {images_dir}")
    print(f"SAM labels: {sam_labels_dir}")
    print(f"Florence labels: {florence_labels_dir}")
    print()

    if args.view == "sam":
        output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / "visualizations" / "sam"
        save_single_labeler(images_dir, sam_labels_dir, output_dir, "SAM", args.max_images)

    elif args.view == "florence":
        output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / "visualizations" / "florence"
        save_single_labeler(images_dir, florence_labels_dir, output_dir, "Florence", args.max_images)

    elif args.view == "compare":
        output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / "visualizations" / "comparison"
        save_comparisons(images_dir, sam_labels_dir, florence_labels_dir, output_dir, "both", args.max_images)

    elif args.view == "disagree":
        output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR / "visualizations" / "disagreements"
        save_comparisons(images_dir, sam_labels_dir, florence_labels_dir, output_dir, "disagree", args.max_images)

    elif args.view == "all":
        # Save all views
        print("=== Saving SAM visualizations ===")
        save_single_labeler(images_dir, sam_labels_dir,
                           OUTPUTS_DIR / "visualizations" / "sam", "SAM", args.max_images)
        print()

        print("=== Saving Florence visualizations ===")
        save_single_labeler(images_dir, florence_labels_dir,
                           OUTPUTS_DIR / "visualizations" / "florence", "Florence", args.max_images)
        print()

        print("=== Saving disagreement comparisons ===")
        save_comparisons(images_dir, sam_labels_dir, florence_labels_dir,
                        OUTPUTS_DIR / "visualizations" / "disagreements", "disagree", args.max_images)

        print()
        print("=" * 60)
        print("All visualizations saved to: outputs/visualizations/")
        print("  - sam/           : SAM labels (green boxes)")
        print("  - florence/      : Florence labels (orange boxes)")
        print("  - disagreements/ : Side-by-side where they disagree")
        print()
        print("Browse with: nautilus outputs/visualizations/")


if __name__ == "__main__":
    main()
