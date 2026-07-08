"""
Configuration settings for the dataset labeling pipeline.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT

# Dataset paths
FLICKR_IMAGES_DIR = DATA_ROOT / "flicker_dataset" / "flickr30k_images" / "flickr30k_images"
FLICKR_CAPTIONS_CSV = DATA_ROOT / "flicker_dataset" / "flickr30k_images" / "results.csv"
ROBOFLOW_DATASET_DIR = DATA_ROOT / "Persons_and_cars_yolo26"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAM_LABELS_DIR = OUTPUTS_DIR / "sam_labels"
FLORENCE_LABELS_DIR = OUTPUTS_DIR / "florence_labels"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
COMPARISONS_DIR = OUTPUTS_DIR / "comparisons"


@dataclass
class LabelingConfig:
    """Configuration for the labeling process."""

    # Classes to detect
    classes: List[str] = field(default_factory=lambda: ["person", "vehicle"])

    # Class ID mapping (YOLO format)
    class_to_id: Dict[str, int] = field(default_factory=lambda: {
        "person": 0,
        "vehicle": 1,
    })

    # Synonyms for detection prompts
    class_synonyms: Dict[str, List[str]] = field(default_factory=lambda: {
        "person": ["person", "man", "woman", "child", "people", "human"],
        "vehicle": ["car", "vehicle", "truck", "bus", "motorcycle", "bicycle", "van"],
    })

    # Minimum confidence threshold
    confidence_threshold: float = 0.3

    # Minimum bounding box size (relative to image)
    min_box_size: float = 0.01  # 1% of image area

    # Maximum bounding box size (relative to image)
    max_box_size: float = 0.95  # 95% of image area

    # Batch size for processing
    batch_size: int = 8

    # Save checkpoint every N images
    checkpoint_interval: int = 100

    # Device settings
    device: str = "cuda"
    half_precision: bool = True


@dataclass
class SAMConfig(LabelingConfig):
    """SAM-specific configuration."""

    model_name: str = "sam3.pt"
    # Text prompts for SAM 3 (matching Florence synonyms for fair comparison)
    detection_prompts: Dict[str, str] = field(default_factory=lambda: {
        "person": "person, man, woman, child, people, human",
        "vehicle": "car, truck, bus, motorcycle, vehicle, van, bicycle",
    })


@dataclass
class FlorenceConfig(LabelingConfig):
    """Florence-specific configuration."""

    model_name: str = "microsoft/Florence-2-large"
    # Task prompt for object detection
    task_prompt: str = "<OD>"
    # Alternative: phrase grounding
    phrase_grounding_prompt: str = "<CAPTION_TO_PHRASE_GROUNDING>"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""

    # IoU thresholds for matching
    iou_threshold: float = 0.5
    iou_thresholds_map: List[float] = field(default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

    # Confidence thresholds for PR curves
    conf_thresholds: List[float] = field(default_factory=lambda: [i/100 for i in range(0, 100, 5)])


@dataclass
class TTAConfig:
    """Test-Time Augmentation configuration."""

    # Augmentation types
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotations: List[int] = field(default_factory=lambda: [0])  # degrees
    scales: List[float] = field(default_factory=lambda: [1.0])

    # Ensemble method for combining predictions
    # Options: "nms", "soft_nms", "wbf" (weighted box fusion)
    ensemble_method: str = "wbf"

    # IoU threshold for NMS/WBF
    nms_iou_threshold: float = 0.5

    # Minimum votes required (for voting ensemble)
    min_votes: int = 2


# Default instances
DEFAULT_LABELING_CONFIG = LabelingConfig()
DEFAULT_SAM_CONFIG = SAMConfig()
DEFAULT_FLORENCE_CONFIG = FlorenceConfig()
DEFAULT_EVAL_CONFIG = EvaluationConfig()
DEFAULT_TTA_CONFIG = TTAConfig()
