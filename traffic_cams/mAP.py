

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


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Calculates the mean Average Precision (mAP) at an IoU threshold of 0.5.

    Requires confidence scores in the submission DataFrame.

    Args:
        solution: DataFrame with ground truth annotations. Expected columns:
                  'image_id', 'class_name', 'x_min', 'y_min', 'x_max', 'y_max'.
        submission: DataFrame with predicted annotations. Expected columns:
                    'image_id', 'class_name', 'confidence',
                    'x_min', 'y_min', 'x_max', 'y_max'.
        row_id_column_name: Name of the column identifying unique rows (ignored).

    Returns:
        The mAP@0.5 score as a single float.
    """
    required_sol_cols = {"image_id", "class_name", "x_min", "y_min", "x_max", "y_max"}
    required_sub_cols = {"image_id", "class_name", "confidence", "x_min", "y_min", "x_max", "y_max"}
    numeric_cols = ["x_min", "y_min", "x_max", "y_max", "confidence"] # Add confidence

    # --- Data Validation ---
    if row_id_column_name in solution.columns:
        solution = solution.drop(columns=[row_id_column_name])
    if row_id_column_name in submission.columns:
        submission = submission.drop(columns=[row_id_column_name])

    missing_sol_cols = required_sol_cols - set(solution.columns)
    if missing_sol_cols:
        raise ParticipantVisibleError(f"Solution DataFrame missing required columns: {missing_sol_cols}")

    missing_sub_cols = required_sub_cols - set(submission.columns)
    if missing_sub_cols:
        raise ParticipantVisibleError(f"Submission DataFrame missing required columns: {missing_sub_cols}")

    # Copy to avoid modifying originals and ensure float types
    sol_copy = solution.copy()
    sub_copy = submission.copy()


    
    # Convert solution numeric columns
    for col in ["x_min", "y_min", "x_max", "y_max"]:
         try:
             sol_copy[col] = sol_copy[col].astype(float)
         except ValueError as e:
              raise ParticipantVisibleError(f"Solution column '{col}' conversion failed: {e}")

    # Convert submission numeric columns
    for col in numeric_cols:
        try:
            sub_copy[col] = sub_copy[col].astype(float)
        except ValueError as e:
             raise ParticipantVisibleError(f"Submission column '{col}' conversion failed: {e}")

    # Check for invalid boxes (optional, but good practice)
    invalid_sol_boxes = (sol_copy['x_min'] >= sol_copy['x_max']) | (sol_copy['y_min'] >= sol_copy['y_max'])
    if invalid_sol_boxes.any():
        print(f"Warning: Solution contains {invalid_sol_boxes.sum()} invalid boxes. These might affect scoring.")
        # Optionally filter: sol_copy = sol_copy[~invalid_sol_boxes]

    invalid_sub_boxes = (sub_copy['x_min'] >= sub_copy['x_max']) | (sub_copy['y_min'] >= sub_copy['y_max'])
    if invalid_sub_boxes.any():
        print(f"Warning: Submission contains {invalid_sub_boxes.sum()} invalid boxes. These will likely count as False Positives.")
        # Optionally filter: sub_copy = sub_copy[~invalid_sub_boxes]

    
    

    # --- Scoring Logic ---
    iou_threshold = 0.5
    average_precisions = []

    # Use classes present in the solution as the basis for evaluation
    # Important: Keep original index for GT matching within AP function
    sol_copy = sol_copy.reset_index().rename(columns={'index': 'gt_original_index'})

    class_names = sorted(sol_copy["class_name"].unique())
    exclude_classes = ['utility_vehicle']
    class_names = [cls for cls in class_names if cls not in exclude_classes]

    # Ensuring only private or public predicitions get evaluated
    sol_img_ids = list(sol_copy['image_id'].unique())
    used_imgs_mask = sub_copy['image_id'].isin(sol_img_ids)
    sub_copy = sub_copy[used_imgs_mask]
    
    if not class_names:
        return 0.0 # No classes in solution to evaluate
    
    # --- Calculate AP per class ---
    for class_name in class_names:
        gt_class = sol_copy[sol_copy["class_name"] == class_name]
        pred_class = sub_copy[sub_copy["class_name"] == class_name]

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

if __name__ == '__main__':
    subm_fname = 'traffic_cams/baseline_submission.csv'
    gt_fname = 'traffic_cams/datasets/to_kaggle/test_ground_truth.csv'

    subm_df = pd.read_csv(subm_fname)
    gt_df = pd.read_csv(gt_fname)

    print(f'Final score is {score(gt_df, subm_df, '')}')
