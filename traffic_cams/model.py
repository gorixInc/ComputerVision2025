from torchvision.models import get_model
from torch import nn
import torch
import numpy as np
from traffic_cams.non_max_suppression import nms_xywh
from traffic_cams.util import targets_to_results
import pandas as pd
from traffic_cams.mAP import score
class AdvancedObjectDetector(nn.Module):
  def __init__(self, K, num_anchors, backbone='efficientnet_b3', weights="IMAGENET1K_V1"):
    super(AdvancedObjectDetector, self).__init__()
    self.K = K
    self.n_anchors = num_anchors

    self.efficentnet = get_model(backbone, weights=weights)
    self.backbone = self.efficentnet.features
    
    backbone_out_channels = self.efficentnet.features[-1][0].out_channels
    self.head = nn.Conv2d(
          in_channels=backbone_out_channels,  
          out_channels=num_anchors*(5 + K),
          kernel_size=1
      )


  def get_stride(self):
    h, w = 640, 640
    x = torch.randn(1, 3, h, w)
    outputs = self.backbone(x)
    f_h = outputs.shape[2]
    f_w = outputs.shape[3]
    stride_y = h/f_h
    stride_x = w/f_w
    assert stride_x == stride_y
    assert int(stride_x) == stride_x
    return int(stride_x)

  def forward(self, x):

    x = self.backbone(x)
    x = self.head(x)

    # Reshape to [B, H, W, n_anchors, K+4+1]
    #print(x.shape) # B, C, H, W
    x = x.view(x.shape[0], self.n_anchors, self.K + 5, x.shape[-2], x.shape[-1]) # B, N, K+5, H, W
    x = x.permute(0, 3, 4, 1, 2).contiguous()  # B, H, W, N, K+5
    # Softmax if we had to use it would have been here:
    # x[..., 0:self.K] = torch.So(x[..., 0:self.K]) # first K are responsible for K classes

    # Apply sigmoid to object center coordinates (assumed to be the next two values after class scores)
    x[..., :, self.K:self.K+2] = torch.sigmoid(x[...,:, self.K:self.K+2])

    # Apply exp to width, height (the next three values)
    x[...,:, self.K+2:self.K+4] = torch.exp(x[...,:, self.K+2:self.K+4])

    # Apply sigmoid to confidence
    x[...,:, self.K+4:self.K+5] = torch.sigmoid(x[...,:, self.K+4:self.K+5])
    
    return x



def get_basic_loss(K=6, lambda_coord=5, lambda_noobj=0.5, verbal=False):
    classification_loss = nn.CrossEntropyLoss() # For classes
    bbox_loss = nn.SmoothL1Loss() # For bounding box coordinates and sizes
    confidence_loss = nn.BCELoss()  # For object confidence scores

    def compute_loss(predictions, targets, K=6, lambda_coord=5, lambda_noobj=0.5, verbal=False):
        
        # Extract predictions
        pred_confidence = predictions[..., K+4]  # Confidence scores

        pred_classes = predictions[..., :K]  # Assuming class scores are the first K channels
        pred_coords = predictions[..., K:K+2]  # Next two channels for coordinates
        pred_sizes = predictions[..., K+2:K+4]  # Sizes (width and height)
        

        
        # Similar extraction needs to be done for targets based on our dataset structure
        # Extract targets
        targets_confidence = targets[..., K+4]
        object_mask = targets_confidence == 1
        targets_classes = targets[..., :K]
        targets_coords = targets[..., K:K+2]
        targets_sizes = targets[..., K+2:K+4]
        negative_mask = ~object_mask
        loss_classes = (classification_loss(pred_classes[object_mask], targets_classes[object_mask])
                        if object_mask.sum() > 0 else 0.0)
        
        # loss_classes_neg = (classification_loss(pred_classes[negative_mask], targets_classes[negative_mask])
        #             if negative_mask.sum() > 0 else 0.0)
        # loss_classes = loss_classes_pos + lambda_noobj*loss_classes_neg

        # Compute the localization (bbox coordinate) losses only on positive anchors.
        loss_coords = (bbox_loss(pred_coords[object_mask], targets_coords[object_mask])
                    if object_mask.sum() > 0 else 0.0)
        
        loss_sizes = (bbox_loss(pred_sizes[object_mask], targets_sizes[object_mask])
                    if object_mask.sum() > 0 else 0.0)


        loss_conf_obj = (confidence_loss(pred_confidence[object_mask], 
                                        targets_confidence[object_mask])
                        if object_mask.sum() > 0 else 0.0)
        

        loss_conf_noobj = (confidence_loss(pred_confidence[negative_mask], 
                                        targets_confidence[negative_mask])
                        if negative_mask.sum() > 0 else 0.0)
        
        loss_confidence = loss_conf_obj + lambda_noobj * loss_conf_noobj
        #print(targets_classes.shape, pred_classes.shape)

        # loss_classes = classification_loss(pred_classes, targets_classes)
        # loss_coords = bbox_loss(pred_coords, targets_coords)
        # loss_sizes = bbox_loss(pred_sizes, targets_sizes)
        # loss_confidence = confidence_loss(pred_confidence, targets_confidence)

        if verbal:
            print(f'Classes loss: {loss_classes}; Coordinates loss: {loss_coords}; Sizes loss: {loss_sizes}; Confidence loss: {loss_confidence}')

        # Combine losses
        total_loss = lambda_coord * loss_coords + lambda_coord * loss_sizes + loss_confidence + loss_classes

        return total_loss
    return compute_loss

def fit(model, 
        optimizer, 
        loss_func, 
        train_loader, 
        val_loader, 
        n_epochs, 
        anchors,
        stride,
        model_img_height,
        model_img_width,
        classes,
        checkpoint_dir='checkpoints', 
        model_name='', 
        history=None,
        device='cuda'):

  if history is None:
    history = {'loss': [], 'val_loss': [], 'map@05':[]}

  for epoch in range(n_epochs):
    # initialise losses for logging
    epoch_loss, val_epoch_loss = 0.0, 0.0

    model.train()
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()   # reseting gradients

        # Forward pass
        outputs = model(images)

        loss = loss_func(outputs, targets)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    model.eval()
    pred_val_targets = []
    pred_img_ids = []
    with torch.inference_mode():
      for images, targets, img_ids in val_loader:
          images, targets, img_ids = images.to(device), targets.to(device), img_ids

          # Forward pass only
          outputs = model(images)
          loss = loss_func(outputs, targets)
          
          val_epoch_loss += loss.item()
          
          for i in range(len(outputs)):
             pred_val_targets.append(outputs[i])
             pred_img_ids.append(img_ids[i])
        
    
    # Badly programmed mAP calculation -------------
    out_data = {'image_id':[], 
                'confidence':[],
                'class_name':[],
                'x_min':[],
                'y_min':[],
                'x_max':[],
                'y_max':[]}
    
    for i in range(len(pred_val_targets)):
        pred_targets = pred_val_targets[i]
        img_id = pred_img_ids[i]
        pred_targets = pred_targets.cpu().detach()
        #print(pred_targets.shape)
        #print(pred_targets)
        class_ids, confidences, bboxes = targets_to_results(pred_targets, anchors, len(classes), stride=stride,
                                model_h=model_img_height, model_w=model_img_width, H=720, W=1280)
        class_ids, confidences, bboxes = nms_xywh(class_ids, confidences, bboxes, score_threshold=0.5, iou_threshold=0.2)
        #print(class_ids, confidences, bboxes)
        for i in range(len(class_ids)):
            class_id = class_ids[i]
            conf = confidences[i]
            x, y, w, h = bboxes[i]
            class_id = int(class_id)
            out_data['image_id'].append(img_id)
            out_data['confidence'].append(conf)
            out_data['class_name'].append(classes[class_id])
            out_data['x_min'].append(x-w/2)
            out_data['y_min'].append(y-h/2)
            out_data['x_max'].append(x+w/2)
            out_data['y_max'].append(y+h/2)

    val_pred_data = pd.DataFrame(out_data)
    val_pred_data.to_csv("traffic_cams/val_pred_data.csv")
    gt_data = pd.read_csv('traffic_cams/val_gt.csv')
    map_score = score(gt_data, val_pred_data, '')
    # END badly programmed mAP calculation -------------

    val_loss = val_epoch_loss/len(val_loader)
    history['loss'].append(epoch_loss/len(train_loader))
    history['val_loss'].append(val_loss)
    history['map@05'].append(map_score)
    if map_score >= max(history['map@05']):
      checkpoint = {
        'history': history,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
      }
      print(f'Saving checkpoint! mAP={map_score}')

      torch.save(checkpoint, f'{checkpoint_dir}/best_{model_name}_{len(history['loss'])}ep.pth')
      
    print(f"Epoch {epoch + 1}, Loss: {history['loss'][-1]}, Val loss: {history['val_loss'][-1]}, 'Val MAP@0.5: {map_score}'")

  return history