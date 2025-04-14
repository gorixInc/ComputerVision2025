import numpy as np
import torch # Assuming torch is used in targets_to_results
from itertools import product # Assuming product is used

# --- Assume your targets_to_results function exists ---
# (Including a dummy version here for testing, replace with your actual one)
def target_offsets_to_bbox(tx, ty, tw, th, aw, ah, stride, i, j):
    # Dummy conversion - REPLACE with your actual logic
    # This should convert grid-relative offsets back to image coords
    # Using stride and anchor dims
    grid_x = j
    grid_y = i
    pred_cx = (torch.sigmoid(tx) + grid_x) * stride
    pred_cy = (torch.sigmoid(ty) + grid_y) * stride
    pred_w = torch.exp(tw) * aw
    pred_h = torch.exp(th) * ah
    # Return xywh
    return pred_cx, pred_cy, pred_w, pred_h

def targets_to_results(targets, anchors, K, model_h, model_w, H, W):
    # Targets shape: grid_h, grid_w, n_anchors, K+5
    # H, W: original image height, width
    # model_h, model_w: model input height, width
    stride_h = model_h / targets.shape[0] # Calculate stride
    stride_w = model_w / targets.shape[1] # Calculate stride (assuming square grid cells for simplicity)
    stride = stride_w # Use one stride value

    ijn = product(range(targets.shape[0]),
                  range(targets.shape[1]),
                  range(targets.shape[2]))
    class_ids, confidences, bboxes = [], [], []

    for i, j, n in ijn:
        cell_preds = targets[i, j, n]
        # Assuming format: [class_probs..., tx, ty, tw, th, objectness_conf]
        tx, ty, tw, th = cell_preds[K:K+4]
        pred_objectness = torch.sigmoid(cell_preds[K+4]) # Apply sigmoid to objectness

        pred_class_probs = torch.softmax(cell_preds[:K], dim=0)
        pred_class_confidence, pred_class_id_tensor = torch.max(pred_class_probs, dim=0)

        final_confidence = pred_objectness * pred_class_confidence

        pred_class_id = pred_class_id_tensor.item() # Get Python int

        # Assuming anchors are relative to model input size (model_h, model_w)
        # Adjust if anchors are defined differently (e.g., relative to stride)
        # Example: anchors shape might be (num_anchor_sets, num_anchors_per_set, 2)
        # Need to map i,j to the correct anchor set if multiple strides/feature maps
        # For simplicity, assume single feature map and anchors[n] is relevant
        # Also assume anchors store (w, h) relative to model input size
        aw = anchors[n][0] # Anchor width (relative to model input size)
        ah = anchors[n][1] # Anchor height (relative to model input size)

        x, y, w, h = target_offsets_to_bbox(tx, ty, tw, th, aw, ah, stride, i, j) # x,y,w,h (center coords, relative to model input size)

        # Scale to original image dimensions (W, H)
        scale_w = W / model_w
        scale_h = H / model_h
        x_orig = x * scale_w
        y_orig = y * scale_h
        w_orig = w * scale_w
        h_orig = h * scale_h

        result = (pred_class_id, final_confidence, x_orig, y_orig, w_orig, h_orig)
        # Convert tensors to numpy if they aren't already scalars
        result_np = []
        for el in result:
            if isinstance(el, torch.Tensor):
                result_np.append(el.cpu().numpy())
            else:
                result_np.append(el) # Already a Python number

        pred_class_id_np, final_confidence_np, x_np, y_np, w_np, h_np = result_np
        class_ids.append(int(pred_class_id_np))
        confidences.append(final_confidence_np)
        bboxes.append([x_np, y_np, w_np, h_np]) # Store as xywh

    return np.array(class_ids), np.array(confidences), np.array(bboxes)
# --- End of assumed targets_to_results ---


# --- Helper Functions ---
def xywh_to_xyxy(boxes_xywh):
    """Converts boxes from [x_center, y_center, width, height] to [x_min, y_min, x_max, y_max]."""
    if boxes_xywh.shape[0] == 0:
        return np.array([]).reshape(0, 4)
    x_center, y_center, width, height = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x_min = x_center - width / 2.0
    y_min = y_center - height / 2.0
    x_max = x_center + width / 2.0
    y_max = y_center + height / 2.0
    return np.stack([x_min, y_min, x_max, y_max], axis=1)

def calculate_iou_xyxy(box, other_boxes):
    """
    Calculates Intersection over Union (IoU) between a single box [xmin, ymin, xmax, ymax]
    and multiple other boxes [N, 4].
    """
    box = np.asarray(box)
    other_boxes = np.asarray(other_boxes)

    x_inter_min = np.maximum(box[0], other_boxes[:, 0])
    y_inter_min = np.maximum(box[1], other_boxes[:, 1])
    x_inter_max = np.minimum(box[2], other_boxes[:, 2])
    y_inter_max = np.minimum(box[3], other_boxes[:, 3])

    inter_width = np.maximum(0.0, x_inter_max - x_inter_min)
    inter_height = np.maximum(0.0, y_inter_max - y_inter_min)
    intersection_area = inter_width * inter_height

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    other_boxes_area = (other_boxes[:, 2] - other_boxes[:, 0]) * \
                       (other_boxes[:, 3] - other_boxes[:, 1])

    union_area = box_area + other_boxes_area - intersection_area
    iou = intersection_area / (union_area + 1e-6)
    return iou

# --- NMS Function ---
def nms_xywh(class_ids, confidences, bboxes_xywh, iou_threshold=0.5, score_threshold=0.25):
    """
    Performs Non-Maximum Suppression (NMS) on detections.

    Args:
        class_ids (np.ndarray): Array of class IDs for each detection. Shape (N,).
        confidences (np.ndarray): Array of confidence scores for each detection. Shape (N,).
        bboxes_xywh (np.ndarray): Array of bounding boxes in [x_center, y_center, width, height]
                                  format. Shape (N, 4).
        iou_threshold (float): The IoU threshold for suppression.
        score_threshold (float): The minimum confidence score to consider a box.

    Returns:
        tuple: A tuple containing:
            - final_class_ids (np.ndarray): Class IDs after NMS. Shape (M,).
            - final_confidences (np.ndarray): Confidences after NMS. Shape (M,).
            - final_bboxes_xywh (np.ndarray): Bounding boxes (xywh) after NMS. Shape (M, 4).
            Where M is the number of detections remaining after NMS.
    """
    if len(class_ids) == 0:
        return np.array([]), np.array([]), np.array([])

    # List to store the indices of the elements to keep
    keep_indices = []
    unique_classes = np.unique(class_ids)

    for cls in unique_classes:
        # Get indices for the current class
        class_mask = (class_ids == cls)
        class_confidences = confidences[class_mask]
        class_bboxes_xywh = bboxes_xywh[class_mask]
        original_indices = np.where(class_mask)[0] # Store original indices

        # --- 1. Apply Score Threshold ---
        score_mask = class_confidences >= score_threshold
        if not np.any(score_mask):
            continue # No boxes left for this class after score filtering

        class_confidences = class_confidences[score_mask]
        class_bboxes_xywh = class_bboxes_xywh[score_mask]
        original_indices = original_indices[score_mask] # Keep track

        # --- 2. Convert boxes to xyxy for IoU calculation ---
        class_bboxes_xyxy = xywh_to_xyxy(class_bboxes_xywh)

        # --- 3. Sort by Confidence (Descending) ---
        sorted_indices = np.argsort(class_confidences)[::-1]
        class_confidences = class_confidences[sorted_indices]
        class_bboxes_xyxy = class_bboxes_xyxy[sorted_indices]
        # Keep original indices and xywh boxes aligned with sorting
        original_indices = original_indices[sorted_indices]
        class_bboxes_xywh_sorted = class_bboxes_xywh[sorted_indices] # Store sorted xywh

        # --- 4. Perform NMS Loop ---
        current_keep_indices_cls = [] # Indices (relative to the sorted list for this class)
        suppressed = np.zeros(len(class_confidences), dtype=bool)

        for i in range(len(class_confidences)):
            if suppressed[i]:
                continue

            # Keep the current box (index i)
            current_keep_indices_cls.append(i)

            # Get current box and compare with subsequent boxes
            current_box_xyxy = class_bboxes_xyxy[i]
            compare_boxes_xyxy = class_bboxes_xyxy[i+1:]

            if len(compare_boxes_xyxy) == 0:
                break # No more boxes to compare

            # Calculate IoU
            ious = calculate_iou_xyxy(current_box_xyxy, compare_boxes_xyxy)

            # Find indices (relative to compare_boxes_xyxy) to suppress
            suppress_mask = ious > iou_threshold

            # Mark corresponding boxes in the original sorted list as suppressed
            # Need to map suppress_mask indices back to the full sorted list indices (i+1 onwards)
            indices_to_suppress = np.where(suppress_mask)[0] + (i + 1)
            suppressed[indices_to_suppress] = True


        # --- 5. Store final indices for this class ---
        # Use the indices kept relative to the sorted list (current_keep_indices_cls)
        # to retrieve the original indices of the boxes we want to keep globally
        kept_original_indices_for_class = original_indices[current_keep_indices_cls]
        keep_indices.extend(kept_original_indices_for_class)


    # --- 6. Filter original inputs using kept indices ---
    if not keep_indices: # Check if list is empty
         return np.array([]), np.array([]), np.array([])

    final_class_ids = class_ids[keep_indices]
    final_confidences = confidences[keep_indices]
    final_bboxes_xywh = bboxes_xywh[keep_indices] # Select from original xywh bboxes

    return final_class_ids, final_confidences, final_bboxes_xywh