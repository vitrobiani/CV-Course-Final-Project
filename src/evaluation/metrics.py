"""
Evaluation metrics for object detection.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np

from ..utils.dataset_utils import BoundingBox


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """
    Compute Intersection over Union between two bounding boxes.

    Args:
        box1: First bounding box
        box2: Second bounding box

    Returns:
        IoU value between 0 and 1
    """
    # Convert to xyxy (normalized)
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


def match_boxes(
    pred_boxes: List[BoundingBox],
    gt_boxes: List[BoundingBox],
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Match predicted boxes to ground truth boxes using Hungarian matching.

    Args:
        pred_boxes: List of predicted bounding boxes
        gt_boxes: List of ground truth bounding boxes
        iou_threshold: Minimum IoU for a match
        class_agnostic: If True, ignore class when matching

    Returns:
        Tuple of:
        - List of (pred_idx, gt_idx, iou) for matched pairs
        - List of unmatched prediction indices (false positives)
        - List of unmatched ground truth indices (false negatives)
    """
    if len(pred_boxes) == 0:
        return [], [], list(range(len(gt_boxes)))

    if len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), []

    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            # Check class match if not class agnostic
            if not class_agnostic and pred.class_id != gt.class_id:
                iou_matrix[i, j] = 0
            else:
                iou_matrix[i, j] = compute_iou(pred, gt)

    # Greedy matching (simple approach)
    matches = []
    matched_pred = set()
    matched_gt = set()

    # Sort by IoU descending
    while True:
        # Find highest IoU
        if len(matched_pred) == len(pred_boxes) or len(matched_gt) == len(gt_boxes):
            break

        max_iou = 0
        max_i, max_j = -1, -1

        for i in range(len(pred_boxes)):
            if i in matched_pred:
                continue
            for j in range(len(gt_boxes)):
                if j in matched_gt:
                    continue
                if iou_matrix[i, j] > max_iou:
                    max_iou = iou_matrix[i, j]
                    max_i, max_j = i, j

        if max_iou < iou_threshold:
            break

        matches.append((max_i, max_j, max_iou))
        matched_pred.add(max_i)
        matched_gt.add(max_j)

    # Find unmatched
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    unmatched_gt = [j for j in range(len(gt_boxes)) if j not in matched_gt]

    return matches, unmatched_pred, unmatched_gt


def compute_precision_recall(
    pred_boxes: List[BoundingBox],
    gt_boxes: List[BoundingBox],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute precision and recall for a single image.

    Args:
        pred_boxes: Predicted boxes
        gt_boxes: Ground truth boxes
        iou_threshold: IoU threshold for matching

    Returns:
        Dictionary with precision, recall, and other metrics
    """
    matches, fps, fns = match_boxes(pred_boxes, gt_boxes, iou_threshold)

    tp = len(matches)
    fp = len(fps)
    fn = len(fns)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def compute_ap(
    all_predictions: List[Tuple[float, bool]],
) -> float:
    """
    Compute Average Precision from sorted predictions.

    Args:
        all_predictions: List of (confidence, is_true_positive) tuples,
                        sorted by confidence descending

    Returns:
        Average Precision value
    """
    if not all_predictions:
        return 0.0

    # Sort by confidence descending
    sorted_preds = sorted(all_predictions, key=lambda x: x[0], reverse=True)

    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    total_gt = sum(1 for _, is_tp in sorted_preds if is_tp)

    for conf, is_tp in sorted_preds:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_gt if total_gt > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        precisions_above_t = [p for p, r in zip(precisions, recalls) if r >= t]
        if precisions_above_t:
            ap += max(precisions_above_t)
    ap /= 11

    return ap


def compute_map(
    predictions: Dict[str, List[BoundingBox]],
    ground_truths: Dict[str, List[BoundingBox]],
    iou_thresholds: Optional[List[float]] = None,
    num_classes: int = 2,
) -> Dict[str, float]:
    """
    Compute mean Average Precision across all images and classes.

    Args:
        predictions: Dict mapping image names to predicted boxes
        ground_truths: Dict mapping image names to ground truth boxes
        iou_thresholds: List of IoU thresholds (default: [0.5])
        num_classes: Number of classes

    Returns:
        Dictionary with mAP and per-class AP
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    results = {}

    for iou_thresh in iou_thresholds:
        # Collect all predictions per class
        class_predictions = {c: [] for c in range(num_classes)}
        class_gt_counts = {c: 0 for c in range(num_classes)}

        for img_name in set(predictions.keys()) | set(ground_truths.keys()):
            pred_boxes = predictions.get(img_name, [])
            gt_boxes = ground_truths.get(img_name, [])

            # Count ground truths per class
            for gt in gt_boxes:
                class_gt_counts[gt.class_id] += 1

            # Match predictions
            matches, fps, _ = match_boxes(pred_boxes, gt_boxes, iou_thresh)
            matched_pred = {m[0] for m in matches}

            for i, pred in enumerate(pred_boxes):
                is_tp = i in matched_pred
                class_predictions[pred.class_id].append((pred.confidence, is_tp))

        # Compute AP per class
        aps = []
        for class_id in range(num_classes):
            if class_gt_counts[class_id] == 0:
                continue

            # Sort by confidence
            preds = sorted(class_predictions[class_id], key=lambda x: x[0], reverse=True)

            if not preds:
                aps.append(0.0)
                continue

            # Compute precision-recall curve
            tp_cumsum = 0
            fp_cumsum = 0
            precisions = []
            recalls = []

            for conf, is_tp in preds:
                if is_tp:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1

                precision = tp_cumsum / (tp_cumsum + fp_cumsum)
                recall = tp_cumsum / class_gt_counts[class_id]

                precisions.append(precision)
                recalls.append(recall)

            # Compute AP using all-point interpolation
            ap = 0.0
            prev_recall = 0.0
            for i in range(len(recalls) - 1, -1, -1):
                if i > 0:
                    precisions[i - 1] = max(precisions[i - 1], precisions[i])

            for i in range(len(recalls)):
                if recalls[i] != prev_recall:
                    ap += (recalls[i] - prev_recall) * precisions[i]
                    prev_recall = recalls[i]

            aps.append(ap)
            results[f"AP@{iou_thresh:.2f}_class{class_id}"] = ap

        if aps:
            results[f"mAP@{iou_thresh:.2f}"] = np.mean(aps)
        else:
            results[f"mAP@{iou_thresh:.2f}"] = 0.0

    # Compute mAP@0.5:0.95 if multiple thresholds
    if len(iou_thresholds) > 1:
        maps = [results[f"mAP@{t:.2f}"] for t in iou_thresholds]
        results["mAP@0.5:0.95"] = np.mean(maps)

    return results


def compute_agreement_metrics(
    labels1: Dict[str, List[BoundingBox]],
    labels2: Dict[str, List[BoundingBox]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute agreement metrics between two sets of labels.

    Args:
        labels1: First set of labels
        labels2: Second set of labels
        iou_threshold: IoU threshold for matching

    Returns:
        Dictionary with agreement metrics
    """
    total_matches = 0
    total_only_1 = 0
    total_only_2 = 0
    iou_sum = 0.0

    common_images = set(labels1.keys()) & set(labels2.keys())

    for img_name in common_images:
        boxes1 = labels1[img_name]
        boxes2 = labels2[img_name]

        matches, unmatched_1, unmatched_2 = match_boxes(
            boxes1, boxes2, iou_threshold, class_agnostic=False
        )

        total_matches += len(matches)
        total_only_1 += len(unmatched_1)
        total_only_2 += len(unmatched_2)
        iou_sum += sum(m[2] for m in matches)

    total_detections = total_matches + total_only_1 + total_only_2

    agreement_rate = total_matches / total_detections if total_detections > 0 else 0.0
    avg_iou = iou_sum / total_matches if total_matches > 0 else 0.0

    return {
        "total_matches": total_matches,
        "only_in_first": total_only_1,
        "only_in_second": total_only_2,
        "agreement_rate": agreement_rate,
        "average_iou": avg_iou,
        "images_compared": len(common_images),
    }
