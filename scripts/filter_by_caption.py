#!/usr/bin/env python3
"""
Filter Flickr images by caption keywords.

Selects images whose captions mention persons or vehicles,
creating a focused dataset for the labeling pipeline.

Usage:
    python scripts/filter_by_caption.py
    python scripts/filter_by_caption.py --strategy balanced --person-vehicle-ratio 3.0
    python scripts/filter_by_caption.py --output outputs/filtered_flickr
"""
import argparse
import csv
import random
import shutil
import sys
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import FLICKR_IMAGES_DIR, FLICKR_CAPTIONS_CSV

# Keywords to search for
PERSON_KEYWORDS = [
    "man", "woman", "person", "child", "elder",
    "worker", "workers", "men", "women",
    "baby", "infant", "babies", "infants",
    "kid", "kids", "guy", "girl", "boy",
    "female", "male", "people", "pedestrian"
]

VEHICLE_KEYWORDS = [
    "car", "truck", "motorcycle", "bike", "scooter",
    "van", "jeep", "moped", "tractor", "bus",
    "semitrailer", "vehicle", "automobile", "motorbike",
    "bicycle", "taxi", "cab", "suv", "pickup"
]

ALL_KEYWORDS = PERSON_KEYWORDS + VEHICLE_KEYWORDS


def load_captions(csv_path: Path) -> dict:
    """Load captions from Flickr CSV."""
    captions = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        next(reader, None)  # Skip header

        for row in reader:
            if len(row) >= 3:
                image_name = row[0].strip()
                caption = row[2].strip().lower()
                captions[image_name].append(caption)

    return captions


def filter_images(captions: dict, keywords: list) -> dict:
    """
    Filter images that have captions containing any keyword.

    Returns dict with image_name -> {'captions': [...], 'matched_keywords': [...]}
    """
    filtered = {}
    keywords_lower = [kw.lower() for kw in keywords]

    for image_name, caption_list in captions.items():
        all_text = " ".join(caption_list).lower()

        # Find which keywords matched
        matched = []
        for kw in keywords_lower:
            # Word boundary matching (avoid "woman" matching "workman" etc.)
            import re
            if re.search(r'\b' + re.escape(kw) + r'\b', all_text):
                matched.append(kw)

        if matched:
            filtered[image_name] = {
                'captions': caption_list,
                'matched_keywords': list(set(matched))
            }

    return filtered


def select_balanced_dataset(
    filtered: dict,
    person_images: set,
    vehicle_images: set,
    person_vehicle_ratio: float = 3.0,
    negative_fraction: float = 0.07,
    all_images: set = None,
) -> dict:
    """
    Select a balanced dataset prioritizing vehicle images.

    Strategy:
    1. Take ALL vehicle images (they're rare and precious)
    2. Add person-only images to reach desired ratio
    3. Add some negative images (no person/vehicle)

    Args:
        filtered: Dict of all filtered images
        person_images: Set of images with person keywords
        vehicle_images: Set of images with vehicle keywords
        person_vehicle_ratio: Target ratio of person:vehicle IMAGES (not detections)
        negative_fraction: Fraction of final dataset that should be negatives
        all_images: Set of all available images (for negatives)

    Returns:
        Dict of selected images with metadata
    """
    both_images = person_images & vehicle_images
    person_only = person_images - vehicle_images
    vehicle_only = vehicle_images - person_images

    selected = {}

    # Step 1: Take ALL vehicle images (both vehicle-only and both)
    for img in vehicle_images:
        selected[img] = filtered[img].copy()
        selected[img]['category'] = 'both' if img in both_images else 'vehicle_only'

    print(f"\n[Balanced Selection]")
    print(f"  Step 1: Added all {len(vehicle_images)} vehicle images")
    print(f"          ({len(both_images)} with both classes, {len(vehicle_only)} vehicle-only)")

    # Step 2: Calculate how many person-only images to add
    # Images with both classes count toward person count
    current_person_images = len(both_images)
    target_person_images = int(len(vehicle_images) * person_vehicle_ratio)
    person_only_needed = max(0, target_person_images - current_person_images)

    # Sample from person-only images
    person_only_list = list(person_only)
    random.shuffle(person_only_list)
    person_only_selected = person_only_list[:person_only_needed]

    for img in person_only_selected:
        selected[img] = filtered[img].copy()
        selected[img]['category'] = 'person_only'

    print(f"  Step 2: Added {len(person_only_selected)} person-only images")
    print(f"          (target ratio {person_vehicle_ratio}:1 person:vehicle)")

    # Step 3: Add negative images
    if all_images and negative_fraction > 0:
        # Find images NOT in our filtered set
        negative_candidates = all_images - set(filtered.keys())
        num_negatives = int(len(selected) * negative_fraction / (1 - negative_fraction))

        negative_list = list(negative_candidates)
        random.shuffle(negative_list)
        negatives_selected = negative_list[:num_negatives]

        for img in negatives_selected:
            selected[img] = {
                'captions': [],
                'matched_keywords': [],
                'category': 'negative'
            }

        print(f"  Step 3: Added {len(negatives_selected)} negative images ({negative_fraction*100:.0f}% of dataset)")

    # Summary
    categories = defaultdict(int)
    for data in selected.values():
        categories[data['category']] += 1

    print(f"\n  Final dataset: {len(selected)} images")
    print(f"    - Both (person+vehicle): {categories['both']}")
    print(f"    - Vehicle only: {categories['vehicle_only']}")
    print(f"    - Person only: {categories['person_only']}")
    print(f"    - Negative: {categories['negative']}")

    return selected


def main():
    parser = argparse.ArgumentParser(description="Filter Flickr images by caption keywords")
    parser.add_argument("--output", type=str, default="outputs/filtered_flickr",
                        help="Output directory for filtered dataset")
    parser.add_argument("--captions-csv", type=str, default=None,
                        help="Path to captions CSV (default: from config)")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Path to images directory (default: from config)")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images to output dir (default: create symlinks)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just show stats, don't create dataset")
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["all", "balanced"],
                        help="Selection strategy: 'all' keeps everything, 'balanced' prioritizes vehicles")
    parser.add_argument("--person-vehicle-ratio", type=float, default=3.0,
                        help="Target person:vehicle image ratio for balanced strategy (default: 3.0)")
    parser.add_argument("--negative-fraction", type=float, default=0.07,
                        help="Fraction of dataset that should be negative images (default: 0.07 = 7%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    captions_csv = Path(args.captions_csv) if args.captions_csv else FLICKR_CAPTIONS_CSV
    images_dir = Path(args.images_dir) if args.images_dir else FLICKR_IMAGES_DIR
    output_dir = Path(args.output)

    print(f"Captions CSV: {captions_csv}")
    print(f"Images dir: {images_dir}")
    print(f"Output dir: {output_dir}")
    print()

    # Check paths exist
    if not captions_csv.exists():
        print(f"Error: Captions CSV not found: {captions_csv}")
        sys.exit(1)
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    # Load captions
    print("Loading captions...")
    captions = load_captions(captions_csv)
    print(f"Loaded captions for {len(captions)} images")

    # Filter by keywords
    print(f"\nFiltering by {len(ALL_KEYWORDS)} keywords...")
    filtered = filter_images(captions, ALL_KEYWORDS)

    # Separate by category
    person_images = set()
    vehicle_images = set()

    for img_name, data in filtered.items():
        keywords = data['matched_keywords']
        if any(kw in PERSON_KEYWORDS for kw in keywords):
            person_images.add(img_name)
        if any(kw in VEHICLE_KEYWORDS for kw in keywords):
            vehicle_images.add(img_name)

    both_images = person_images & vehicle_images

    # Print stats
    print()
    print("=" * 60)
    print("FILTERING RESULTS")
    print("=" * 60)
    print(f"Total images with captions: {len(captions)}")
    print(f"Images matching keywords:   {len(filtered)}")
    print()
    print(f"  - Person keywords only:   {len(person_images - vehicle_images)}")
    print(f"  - Vehicle keywords only:  {len(vehicle_images - person_images)}")
    print(f"  - Both person & vehicle:  {len(both_images)}")
    print()
    print(f"  Total with person:        {len(person_images)}")
    print(f"  Total with vehicle:       {len(vehicle_images)}")
    print()

    # Show keyword frequency
    keyword_counts = defaultdict(int)
    for data in filtered.values():
        for kw in data['matched_keywords']:
            keyword_counts[kw] += 1

    print("Top 20 matched keywords:")
    for kw, count in sorted(keyword_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {kw}: {count}")

    # Apply selection strategy
    if args.strategy == "balanced":
        # Get all image names for negative sampling
        all_image_names = set(captions.keys())

        selected = select_balanced_dataset(
            filtered=filtered,
            person_images=person_images,
            vehicle_images=vehicle_images,
            person_vehicle_ratio=args.person_vehicle_ratio,
            negative_fraction=args.negative_fraction,
            all_images=all_image_names,
        )
    else:
        # Take all filtered images
        selected = filtered
        for img in selected:
            if img in (person_images & vehicle_images):
                selected[img]['category'] = 'both'
            elif img in vehicle_images:
                selected[img]['category'] = 'vehicle_only'
            else:
                selected[img]['category'] = 'person_only'

    if args.dry_run:
        print("\n[Dry run - no files created]")
        return

    # Create output directory structure
    print(f"\nCreating filtered dataset in {output_dir}...")
    output_images = output_dir / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    # Save filtered image list
    copied = 0
    missing = 0

    for img_name in selected.keys():
        src = images_dir / img_name
        dst = output_images / img_name

        if not src.exists():
            missing += 1
            continue

        if dst.exists():
            continue

        if args.copy_images:
            shutil.copy2(src, dst)
        else:
            # Create relative symlink
            dst.symlink_to(src.resolve())
        copied += 1

    # Count categories in selected
    cat_counts = defaultdict(int)
    for data in selected.values():
        cat_counts[data.get('category', 'unknown')] += 1

    # Save metadata
    metadata_file = output_dir / "filter_metadata.txt"
    with open(metadata_file, "w") as f:
        f.write("# Filtered Flickr Dataset\n")
        f.write(f"# Strategy: {args.strategy}\n")
        f.write(f"# Total images: {len(selected)}\n")
        f.write(f"#   - Both (person+vehicle): {cat_counts['both']}\n")
        f.write(f"#   - Person only: {cat_counts['person_only']}\n")
        f.write(f"#   - Vehicle only: {cat_counts['vehicle_only']}\n")
        f.write(f"#   - Negative: {cat_counts['negative']}\n")
        f.write("#\n")
        f.write("# Keywords used:\n")
        f.write(f"# Person: {', '.join(PERSON_KEYWORDS)}\n")
        f.write(f"# Vehicle: {', '.join(VEHICLE_KEYWORDS)}\n")
        f.write("#\n")
        f.write("# image_name | category | matched_keywords\n")
        f.write("#" + "=" * 59 + "\n")
        for img_name, data in sorted(selected.items()):
            category = data.get('category', 'unknown')
            keywords = ', '.join(data.get('matched_keywords', []))
            f.write(f"{img_name} | {category} | {keywords}\n")

    # Save just the image list (for easy use)
    list_file = output_dir / "image_list.txt"
    with open(list_file, "w") as f:
        for img_name in sorted(selected.keys()):
            f.write(f"{img_name}\n")

    print(f"\nDone!")
    print(f"  Images linked/copied: {copied}")
    print(f"  Images missing: {missing}")
    print(f"  Metadata saved to: {metadata_file}")
    print(f"  Image list saved to: {list_file}")
    print(f"\nNext steps:")
    print(f"  1. Run labeling: python scripts/run_sam_labeling.py --images-dir {output_images}")
    print(f"  2. Or use with pipeline")


if __name__ == "__main__":
    main()
