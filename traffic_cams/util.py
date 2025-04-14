import numpy as np
import torch

def bbox_to_target_offsets(bbox_x, bbox_y, bbox_w, bbox_h, anchor_w, anchor_h, stride, grid_i, grid_j):
    
    # bbox_x = grid_i * stride + sigmoid(t_x) * stride -> sigmoid(t_x) = (bbox_x - gird_i * stride)/stride
    # bbox_y = grid_j * stride + sigmoid(t_y) * stride -> sigmoid(t_y) = (bbox_y - grid_j * stride)/stride
    # bbox_w = anchor_w * exp(t_h) -> exp(t_h) = bbox_w/anchor_w
    # bbox_h = anchor_h * exp(t_w)

    t_x = bbox_x/stride - grid_j
    t_y = bbox_y/stride - grid_i
    t_w = bbox_w/anchor_w
    t_h = bbox_h/anchor_h

    return [t_x, t_y, t_w, t_h]  # We will apply sigmoid and exp to the outputs of the model


def target_offsets_to_bbox(t_x, t_y, t_w, t_h, anchor_w, anchor_h, stride, grid_i, grid_j):
    bbox_x = grid_j * stride + t_x * stride
    bbox_y = grid_i * stride + t_y * stride
    bbox_h = anchor_h * t_h
    bbox_w = anchor_w * t_w
    return [bbox_x, bbox_y, bbox_w, bbox_h] # We will apply sigmoid and exp to the outputs of the model

def iou(anchor_w, anchor_h, bbox_w, bbox_h):

    # Intersection dimensions
    inter_w = min(anchor_w, bbox_w)
    inter_h = min(anchor_h, bbox_h)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    inter_area = inter_w * inter_h
    anchor_area = anchor_w * anchor_h
    bbox_area   = bbox_w * bbox_h
    union_area  = anchor_area + bbox_area - inter_area
    return inter_area / union_area

from itertools import product

def targets_to_results(targets, anchors, K, stride, model_h, model_w, H, W): 
    # Targets shape: H, W, n_anchors, K+5
    ijn = product(range(targets.shape[0]), 
                  range(targets.shape[1]),
                  range(targets.shape[2]))
    class_ids, confidences, bboxes = [], [], []
    for i,j, n in ijn:
        #cell_anchor_preds = targets[i, j]
        #anchor_confidences = cell_anchor_preds[:, K+4]
        #n = torch.argmax(anchor_confidences)

        cell_preds = targets[i, j, n]
        tx, ty, tw, th = cell_preds[K:K+4] 
        pred_confidence = cell_preds[K+4]

        pred_class_probs = torch.softmax(cell_preds[:K], dim=0)
        pred_class_id = np.argmax(pred_class_probs)
        pred_class_confidence = pred_class_probs[pred_class_id]
        final_confidence = pred_confidence * pred_class_confidence

        ah, aw = anchors[pred_class_id][n][0]*H, anchors[pred_class_id][n][1]*W

        x, y, w, h = target_offsets_to_bbox(tx, ty, tw, th, aw, ah, stride, i, j) # x,y,w,h
        x = x*(W/model_w)
        y = y*(H/model_h)
        result = (pred_class_id, final_confidence, x, y, w, h)
        result = [el.cpu().numpy() for el in result]
        pred_class_id, final_confidence, x, y, w, h = result
        class_ids.append(int(pred_class_id))
        confidences.append(final_confidence)
        bboxes.append([x, y, w, h])
        
    return np.array(class_ids), np.array(confidences), np.array(bboxes)

def draw_bbox(ax, bx, by, bw, bh, color='red'):
    top_left_x = bx - bw/2
    top_left_y = by - bh/2

    top_left = (top_left_x, top_left_y)
    bottom_right = (top_left_x + bw, top_left_y + bh)

    # Plotting each edge of the rectangle on the specified axes
    ax.plot([top_left[0], top_left[0]], [top_left[1], bottom_right[1]], '-', color=color)  # Left edge
    ax.plot([bottom_right[0], bottom_right[0]], [top_left[1], bottom_right[1]], '-', color=color)  # Right edge
    ax.plot([top_left[0], bottom_right[0]], [top_left[1], top_left[1]], '-', color=color)  # Top edge
    ax.plot([top_left[0], bottom_right[0]], [bottom_right[1], bottom_right[1]], '-', color=color)  # Bottom edge

def plot_results(ax, class_ids, confidences, bboxes):
    for i in range(len(class_ids)):
        pred_class_id = class_ids[i]
        bx, by, bw, bh = list(bboxes[i])
        colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:brown', 'tab:purple']
        draw_bbox(ax, bx, by, bw, bh, color=colors[pred_class_id])

def draw_rectangle(ax, top_left_x, top_left_y, width, height, color='red'):
    top_left = (top_left_x, top_left_y)
    bottom_right = (top_left_x + width, top_left_y + height)

    # Plotting each edge of the rectangle on the specified axes
    ax.plot([top_left[0], top_left[0]], [top_left[1], bottom_right[1]], '-', color=color)  # Left edge
    ax.plot([bottom_right[0], bottom_right[0]], [top_left[1], bottom_right[1]], '-', color=color)  # Right edge
    ax.plot([top_left[0], bottom_right[0]], [top_left[1], top_left[1]], '-', color=color)  # Top edge
    ax.plot([top_left[0], bottom_right[0]], [bottom_right[1], bottom_right[1]], '-', color=color)  # Bottom edge