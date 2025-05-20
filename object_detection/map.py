

import pandas as pd
import pandas.api.types
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict
# NOTE: Mostly AI generated code 

class ParticipantVisibleError(Exception):
    pass

def iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculates the Intersection over Union (IoU) between two bounding boxes."""
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea == 0:
        return 0.0

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the union area
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea <= 1e-6: # Avoid division by zero or near-zero
        return 0.0

    # Compute the IoU
    iou_val = interArea / unionArea
    return iou_val

def calculate_ap_at_iou_threshold(
    gt_class: pd.DataFrame,
    pred_class: pd.DataFrame,
    iou_threshold: float = 0.5
) -> float:
    """
    Calculates Average Precision (AP) for a single class at a specific IoU threshold.

    Args:
        gt_class: DataFrame containing ground truth boxes for a single class.
                  Must include 'image_id' and box coordinates. Uses index for matching.
        pred_class: DataFrame containing predictions for a single class.
                    Must include 'image_id', 'confidence', and box coordinates.
        iou_threshold: The IoU threshold to consider a match (e.g., 0.5).

    Returns:
        The Average Precision (AP) score for the class (float).
    """
    # --- Preparation ---
    # If no predictions for this class, AP is 0
    if pred_class.empty:
        return 0.0

    # Sort predictions by confidence score (highest first)
    pred_class = pred_class.sort_values('confidence', ascending=False).reset_index(drop=True)

    n_pred = len(pred_class)
    tp = np.zeros(n_pred) # 1 if prediction is TP, 0 otherwise
    fp = np.zeros(n_pred) # 1 if prediction is FP, 0 otherwise

    # Group ground truths by image_id for faster lookup
    gt_grouped = gt_class.groupby('image_id')
    # Dictionary to keep track of matched ground truth indices per image
    # key: image_id, value: set of gt indices (original index from gt_class) matched in this image
    gt_matched_in_image = defaultdict(set)

    total_gt_boxes_for_class = len(gt_class)
    # If there are no ground truths for this class, all predictions are FPs, AP is 0
    if total_gt_boxes_for_class == 0:
        return 0.0

    # --- Matching Loop ---
    for i, pred in pred_class.iterrows():
        pred_image_id = pred['image_id']
        pred_box = [pred['x_min'], pred['y_min'], pred['x_max'], pred['y_max']]

        best_iou = 0.0
        best_gt_match_idx = -1 # Original index of the best matching GT box

        # Check if there are any ground truths in the same image
        if pred_image_id in gt_grouped.groups:
            gts_in_image = gt_grouped.get_group(pred_image_id)

            for gt_idx, gt in gts_in_image.iterrows():
                # Check if this GT has already been matched *in this image*
                if gt_idx in gt_matched_in_image[pred_image_id]:
                    continue

                gt_box = [gt['x_min'], gt['y_min'], gt['x_max'], gt['y_max']]
                current_iou = iou(pred_box, gt_box)

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_match_idx = gt_idx # Store original index

        # --- Assign TP or FP ---
        if best_iou >= iou_threshold and best_gt_match_idx != -1:
             # Check again if the best match index found is actually available (it should be, but double-check)
             if best_gt_match_idx not in gt_matched_in_image[pred_image_id]:
                 tp[i] = 1
                 gt_matched_in_image[pred_image_id].add(best_gt_match_idx) # Mark GT as used for this image
             else:
                 # This case should ideally not happen with the logic above,
                 # but if it did, it means the chosen GT was already matched by a higher confidence pred.
                 fp[i] = 1
        else:
            # No match found above threshold, or no GTs in image
            fp[i] = 1

    # --- Calculate Precision, Recall, and AP ---
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    # Recall = TP / (Total GTs for class)
    recall = cumulative_tp / total_gt_boxes_for_class # Denominator is fixed

    # Precision = TP / (TP + FP) = TP / (Total Predictions Processed So Far)
    precision = cumulative_tp / (cumulative_tp + cumulative_fp)

    # Calculate AP using 11-point interpolation method (PASCAL VOC style)
    # Or using the area under the precision-recall curve directly (more accurate)

    # Method 2: Area under PR curve (All-point interpolation) - More common now
    # Add sentinel values for recall=0 and recall=1
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])

    # Find indices where recall changes
    recall_change_indices = np.where(recall[1:] != recall[:-1])[0]

    # Calculate area using trapezoidal rule sums where recall changes
    ap = np.sum((recall[recall_change_indices + 1] - recall[recall_change_indices]) * precision[recall_change_indices + 1])

    return ap


def score(preds, targets) -> float:
    """
    Calculates the mean Average Precision (mAP) at an IoU threshold of 0.5.

    Args:
        preds   : iterable [e.g. list | np.ndarray | pd.DataFrame]  
                   Each element → ['class_name', confidence, x_min, y_min, x_max, y_max]
        targets : iterable [e.g. list | np.ndarray | pd.DataFrame]  
                   Each element → ['class_name', x_min, y_min, x_max, y_max]

    Returns:
        mAP@0.5 as a float.
    """
    if not isinstance(preds, pd.DataFrame):
        preds = pd.DataFrame(
            preds,
            columns=["class_name", "confidence", "x_min", "y_min", "x_max", "y_max"],
        )

    if not isinstance(targets, pd.DataFrame):
        targets = pd.DataFrame(
            targets,
            columns=["class_name", "x_min", "y_min", "x_max", "y_max"],
        )
    
    class_names = sorted(targets["class_name"].unique())

    # --- Scoring Logic ---
    iou_threshold = 0.5
    average_precisions = []
    
    # --- Calculate AP per class ---
    for class_name in class_names:
        
        gt_class = targets[targets["class_name"] == class_name].copy()
        pred_class = preds[preds["class_name"] == class_name].copy()

        # If there are no GT boxes, AP is undefined – treat as zero
        if gt_class.empty:
            average_precisions.append(0.0)
            continue

        # Sort predictions by confidence (high → low)
        if "confidence" in pred_class.columns:
            pred_class = pred_class.sort_values("confidence", ascending=False)

        pred_class = pred_class.reset_index(drop=True)


        # Pass DataFrames with original GT indices preserved
        ap = calculate_ap_at_iou_threshold(gt_class, pred_class, iou_threshold)
        average_precisions.append(ap)

    # --- Final Score (mAP) ---
    mean_ap = np.mean(average_precisions) if average_precisions else 0.0

    # Ensure score is a non-null float
    if not np.isfinite(mean_ap):
        print(f"Warning: Calculated mAP score is not finite ({mean_ap}). Returning 0.0.")
        return 0.0

    return float(mean_ap)
