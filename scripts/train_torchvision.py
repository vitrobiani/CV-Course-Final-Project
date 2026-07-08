#!/usr/bin/env python3
"""
Torchvision Object Detection Training Pipeline.

Supports:
- RetinaNet (with Focal Loss - good for imbalanced data)
- MobileNetV3-SSDLite (edge-friendly)
- Faster R-CNN (for comparison)

Based on torchvision.models.detection
"""
import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.utils.data
from torch.utils.data import DataLoader
from PIL import Image
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OUTPUTS_DIR


class YOLODataset(torch.utils.data.Dataset):
    """Dataset that reads YOLO format labels for Torchvision models."""

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        transforms=None,
        num_classes: int = 2,
        max_size: int = 640,  # Resize images to save VRAM
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self.num_classes = num_classes
        self.max_size = max_size

        # Find all images with corresponding labels
        self.samples = []
        for img_path in sorted(self.images_dir.glob("*")):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                label_path = self.labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    self.samples.append((img_path, label_path))

        print(f"Found {len(self.samples)} samples in {images_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_path, label_path = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size

        # Resize image to save VRAM
        scale = 1.0
        if self.max_size and max(orig_width, orig_height) > self.max_size:
            scale = self.max_size / max(orig_width, orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            image = image.resize((new_width, new_height), Image.BILINEAR)

        img_width, img_height = image.size

        # Load labels (YOLO format: class x_center y_center width height)
        boxes = []
        labels = []

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height

                    # Convert to xyxy format
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2

                    # Clamp to image boundaries
                    x1 = max(0, min(x1, img_width))
                    y1 = max(0, min(y1, img_height))
                    x2 = max(0, min(x2, img_width))
                    y2 = max(0, min(y2, img_height))

                    # Skip invalid boxes
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        # Torchvision expects 1-indexed labels (0 is background)
                        labels.append(class_id + 1)

        # Handle empty annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Create target dict
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        # Convert image to tensor
        import torchvision.transforms.functional as F
        image = F.to_tensor(image)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def collate_fn(batch):
    """Custom collate function for detection."""
    return tuple(zip(*batch))


def create_mobilenetv4_detector(num_classes: int, pretrained: bool = True,
                                 min_size: int = 480, max_size: int = 640):
    """
    Create object detector with MobileNetV4 backbone.

    This is the "advanced work" mentioned in the assignment:
    integrating MobileNetV4 backbone with a detection head.

    Uses timm for MobileNetV4 and torchvision's FPN + detection head.
    """
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required for MobileNetV4. Install with: pip install timm")

    from torchvision.models.detection.backbone_utils import BackboneWithFPN
    from torchvision.models.detection import RetinaNet
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import misc as misc_nn_ops
    import torch.nn as nn

    # Load MobileNetV4 backbone from timm
    # Available variants: mobilenetv4_conv_small, mobilenetv4_conv_medium, mobilenetv4_conv_large
    backbone_name = "mobilenetv4_conv_small"  # Smallest for edge deployment

    print(f"Loading MobileNetV4 backbone: {backbone_name}")
    mobilenetv4 = timm.create_model(
        backbone_name,
        pretrained=pretrained,
        features_only=True,  # Return intermediate features for FPN
        out_indices=(1, 2, 3, 4),  # Feature levels to extract
    )

    # Get feature channel sizes
    feature_info = mobilenetv4.feature_info.channels()
    print(f"MobileNetV4 feature channels: {feature_info}")

    # Create a wrapper that works with torchvision's detection framework
    class MobileNetV4Backbone(nn.Module):
        def __init__(self, model, out_channels=256):
            super().__init__()
            self.model = model
            self.out_channels = out_channels

            # Get input channel sizes from the backbone
            in_channels_list = model.feature_info.channels()

            # Create lateral convolutions for FPN
            self.fpn_lateral = nn.ModuleList([
                nn.Conv2d(in_ch, out_channels, 1)
                for in_ch in in_channels_list
            ])

            # Create output convolutions for FPN
            self.fpn_output = nn.ModuleList([
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
                for _ in in_channels_list
            ])

        def forward(self, x):
            # Get features from backbone
            features = self.model(x)

            # Apply FPN
            results = []
            last_inner = None

            for idx in range(len(features) - 1, -1, -1):
                inner = self.fpn_lateral[idx](features[idx])

                if last_inner is not None:
                    # Upsample and add
                    inner = inner + nn.functional.interpolate(
                        last_inner, size=inner.shape[-2:], mode='nearest'
                    )

                results.insert(0, self.fpn_output[idx](inner))
                last_inner = inner

            # Return as ordered dict for torchvision compatibility
            from collections import OrderedDict
            out = OrderedDict()
            for i, feat in enumerate(results):
                out[str(i)] = feat

            return out

    # Create backbone with FPN
    backbone = MobileNetV4Backbone(mobilenetv4, out_channels=256)
    backbone.out_channels = 256

    # Create anchor generator
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
                         for x in [32, 64, 128, 256])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    # Create RetinaNet with MobileNetV4 backbone
    # RetinaNet uses Focal Loss which helps with class imbalance
    model = RetinaNet(
        backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        # Use smaller image sizes to fit in 8GB VRAM
        min_size=min_size,
        max_size=max_size,
        # Use smaller head for edge deployment
        head=None,  # Will use default
    )

    return model


def get_model(model_name: str, num_classes: int, pretrained: bool = True,
              min_size: int = 480, max_size: int = 640):
    """Get a detection model with configurable image size to save VRAM."""
    import torchvision.models.detection as detection

    # Add 1 for background class
    num_classes_with_bg = num_classes + 1

    if model_name == "mobilenetv4":
        # MobileNetV4 with custom detection head (advanced work)
        return create_mobilenetv4_detector(num_classes_with_bg, pretrained, min_size=min_size, max_size=max_size)

    elif model_name == "retinanet":
        # RetinaNet with ResNet50-FPN backbone
        # Use smaller image sizes to fit in 8GB VRAM
        if pretrained:
            model = detection.retinanet_resnet50_fpn_v2(
                weights="DEFAULT",
                min_size=min_size,
                max_size=max_size,
            )
            # Replace the classification head
            in_features = model.head.classification_head.conv[0][0].in_channels
            num_anchors = model.head.classification_head.num_anchors
            model.head.classification_head.num_classes = num_classes_with_bg
            model.head.classification_head.cls_logits = torch.nn.Conv2d(
                in_features, num_anchors * num_classes_with_bg, kernel_size=3, stride=1, padding=1
            )
        else:
            model = detection.retinanet_resnet50_fpn_v2(
                weights=None,
                num_classes=num_classes_with_bg,
                min_size=min_size,
                max_size=max_size,
            )

    elif model_name == "mobilenet_ssd" or model_name == "ssdlite":
        # MobileNetV3-Large SSDLite
        if pretrained:
            model = detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")
            # Replace the classification head
            in_channels = model.head.classification_head.in_channels
            num_anchors = model.head.classification_head.num_anchors
            model.head.classification_head.num_classes = num_classes_with_bg
            # Recreate classification head
            from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
            model.head.classification_head = SSDLiteClassificationHead(
                in_channels, num_anchors, num_classes_with_bg
            )
        else:
            model = detection.ssdlite320_mobilenet_v3_large(
                weights=None, num_classes=num_classes_with_bg
            )

    elif model_name == "fasterrcnn":
        # Faster R-CNN with ResNet50-FPN backbone
        if pretrained:
            model = detection.fasterrcnn_resnet50_fpn_v2(
                weights="DEFAULT",
                min_size=min_size,
                max_size=max_size,
            )
            # Replace the box predictor
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes_with_bg
            )
        else:
            model = detection.fasterrcnn_resnet50_fpn_v2(
                weights=None,
                num_classes=num_classes_with_bg,
                min_size=min_size,
                max_size=max_size,
            )

    elif model_name == "fasterrcnn_mobilenet":
        # Faster R-CNN with MobileNetV3-Large-FPN backbone (lighter)
        if pretrained:
            model = detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes_with_bg
            )
        else:
            model = detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=None, num_classes=num_classes_with_bg
            )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    num_batches = 0

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        num_batches += 1

        if (i + 1) % print_freq == 0:
            print(f"  Epoch {epoch}, Batch {i+1}/{len(data_loader)}, "
                  f"Loss: {losses.item():.4f}")

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate the model."""
    model.eval()

    all_predictions = []
    all_targets = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]

        predictions = model(images)

        all_predictions.extend(predictions)
        all_targets.extend(targets)

    # Calculate simple metrics (precision, recall at IoU=0.5)
    tp, fp, fn = 0, 0, 0

    for pred, target in zip(all_predictions, all_targets):
        pred_boxes = pred["boxes"].cpu()
        pred_labels = pred["labels"].cpu()
        pred_scores = pred["scores"].cpu()

        target_boxes = target["boxes"]
        target_labels = target["labels"]

        # Filter by confidence
        mask = pred_scores > 0.5
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]

        # Match predictions to targets
        matched = set()
        for pb, pl in zip(pred_boxes, pred_labels):
            best_iou = 0
            best_idx = -1
            for idx, (tb, tl) in enumerate(zip(target_boxes, target_labels)):
                if idx in matched:
                    continue
                if pl != tl:
                    continue
                iou = box_iou(pb.unsqueeze(0), tb.unsqueeze(0))[0, 0].item()
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= 0.5:
                tp += 1
                matched.add(best_idx)
            else:
                fp += 1

        fn += len(target_boxes) - len(matched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def box_iou(boxes1, boxes2):
    """Calculate IoU between two sets of boxes."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter

    return inter / union


def train_model(
    model_name: str,
    data_yaml: str,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 0.005,
    device: str = "cuda",
    num_workers: int = 4,
    eval_freq: int = 5,
    max_size: int = 512,
):
    """Train a Torchvision detection model."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data config
    with open(data_yaml, "r") as f:
        data_config = yaml.safe_load(f)

    data_root = Path(data_config["path"])
    num_classes = data_config["nc"]
    class_names = data_config["names"]

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Classes: {class_names}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Device: {device}")

    # Create datasets
    train_dataset = YOLODataset(
        images_dir=data_root / "train" / "images",
        labels_dir=data_root / "train" / "labels",
        num_classes=num_classes,
        max_size=max_size,
    )

    val_dataset = YOLODataset(
        images_dir=data_root / "val" / "images",
        labels_dir=data_root / "val" / "labels",
        num_classes=num_classes,
        max_size=max_size,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create model
    print(f"\nLoading {model_name} model...")
    # Use smaller image sizes (min_size, max_size) to fit in 8GB VRAM
    # Default torchvision uses 800-1333 which needs 16GB+
    model = get_model(model_name, num_classes, pretrained=True,
                      min_size=max_size - 128, max_size=max_size)
    model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    results_log = {
        "model": model_name,
        "epochs": [],
        "best_f1": 0,
        "best_epoch": 0,
    }

    best_f1 = 0

    print(f"\nStarting training...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        print(f"  Train loss: {train_loss:.4f}")

        # Update learning rate
        lr_scheduler.step()

        # Evaluate periodically
        if epoch % eval_freq == 0 or epoch == epochs:
            print("  Evaluating...")
            metrics = evaluate(model, val_loader, device)
            print(f"  Val Precision: {metrics['precision']:.4f}")
            print(f"  Val Recall: {metrics['recall']:.4f}")
            print(f"  Val F1: {metrics['f1']:.4f}")

            results_log["epochs"].append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
                "val_f1": metrics["f1"],
            })

            # Save best model
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                results_log["best_f1"] = best_f1
                results_log["best_epoch"] = epoch

                best_path = output_dir / f"{model_name}_best.pt"
                torch.save(model.state_dict(), best_path)
                print(f"  New best model saved: {best_path}")

    # Save final model
    final_path = output_dir / f"{model_name}_final.pt"
    torch.save(model.state_dict(), final_path)

    # Save results
    total_time = time.time() - start_time
    results_log["total_time_seconds"] = total_time
    results_log["total_time_minutes"] = total_time / 60

    results_path = output_dir / f"{model_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results_log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best F1: {best_f1:.4f} (epoch {results_log['best_epoch']})")
    print(f"Results saved to: {results_path}")
    print(f"Best model: {output_dir / f'{model_name}_best.pt'}")

    return results_log


def main():
    parser = argparse.ArgumentParser(description="Train Torchvision Detection Models")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="retinanet",
                        choices=["retinanet", "mobilenetv4", "mobilenet_ssd", "ssdlite",
                                 "fasterrcnn", "fasterrcnn_mobilenet"],
                        help="Model to train (mobilenetv4 requires 'pip install timm')")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Data loader workers")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--eval-freq", type=int, default=5,
                        help="Evaluation frequency (epochs)")
    parser.add_argument("--imgsz", type=int, default=512,
                        help="Max image size (reduces VRAM usage)")
    args = parser.parse_args()

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = OUTPUTS_DIR / "training_torchvision" / datetime.now().strftime("%Y%m%d_%H%M%S")

    # Detect device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    train_model(
        model_name=args.model,
        data_yaml=args.data,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device,
        num_workers=args.workers,
        eval_freq=args.eval_freq,
        max_size=args.imgsz,
    )


if __name__ == "__main__":
    main()
