from torch.utils.data import Dataset, DataLoader # abstract primitives for handling data in pytorch
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes
import xmltodict
from glob import glob
import torch
import numpy as np
import os
from PIL import Image
from pathlib import Path
from traffic_cams.util import bbox_to_target_offsets, iou


class ObjectDetectionDataset(Dataset):
    def __init__(self, dataset_dir, resize_h, resize_w, anchors, classes, train_transforms=False,
                 img_ids_to_use = None, return_img_id=False):
      super(ObjectDetectionDataset, self).__init__()
      self.dataset_dir = dataset_dir
      self.classes = classes 
      self.K = len(classes)
      self.anchors = anchors  # (K, n_anchros, 2)
      self.n_anchors = self.anchors.shape[1]
      self.resize_h = resize_h
      self.resize_w = resize_w
      self.train_transforms=train_transforms
      self.return_img_id=return_img_id
      
      self.stride = None
      self.grid_height = None
      self.grid_width = None
      self.cell_width = None
      self.cell_height = None

      self.images, self.bboxes, self.class_ids, self.img_ids = self.read_dataset()

      if img_ids_to_use is not None: 
        self.images = [img for i, img in enumerate(self.images) if i in img_ids_to_use]
        self.bboxes = [bbox for i, bbox in enumerate(self.bboxes) if i in img_ids_to_use]
        self.class_ids = [cls_id for i, cls_id in enumerate(self.class_ids) if i in img_ids_to_use]
      

      transforms_l = []
      transforms_l += [v2.Resize((resize_h, resize_w)),
                v2.ToTensor()]
      if train_transforms:
        #  transforms_l.append(v2.ColorJitter(brightness=(0.4, 1.4),
        #                         contrast = (0.6, 1.4),
        #                         saturation = (0.6, 1.4),
        #                         hue=(-0.2, 0.2)))
        #  transforms_l.append(v2.RandomPerspective(distortion_scale=0.4, p=0.6))
        transforms_l.append(v2.ColorJitter(brightness=(0.3, 1.4),
                                contrast = (0.5, 1.4),
                                saturation = (0.5, 1.4),
                                hue=(-0.3, 0.3)))
        transforms_l.append(v2.RandomPerspective(distortion_scale=0.5, p=0.6))
         
         
      self.horizontal_flip = v2.RandomHorizontalFlip(p=1)
      self.transform = v2.Compose(transforms_l)
      

    def set_stride(self, stride):
      self.stride = stride
      assert self.resize_h % stride == 0
      assert self.resize_w % stride == 0
      self.grid_height = int(self.resize_h / stride)
      self.grid_width = int(self.resize_w / stride)

      #self.cell_width = np.ceil(self.resize_w / self.grid_width)
      #self.cell_height = np.ceil(self.resize_h / self.grid_height)

    def read_dataset(self): 
      img_paths = glob(f"{self.dataset_dir}/images/*")
      annt_paths = [f'{self.dataset_dir}/Annotations/{Path(img_path).stem}.xml' for img_path in img_paths]
      annt_datas = []
      for path in annt_paths:
        if not os.path.isfile(path):
           annt_datas.append(None)
           continue
        with open(path) as f:
          annt_data = f.read()
          annt_data = xmltodict.parse(annt_data)['annotation']
          annt_datas.append(annt_data)

      images = [Image.open(file_name) for file_name in img_paths]
      img_ids = [Path(path).stem for path in img_paths]
      annotations = [self.parse_annotation(annt) for annt in annt_datas]
      bboxes = [annt[0] for annt in annotations]
      class_ids = [annt[1] for annt in annotations]

      return images, bboxes, class_ids, img_ids


    def parse_annotation(self, annt_data):
      """
      Parse annotations for a single image and
      get image pixel coordinates for bbox center, width and height
      """
      
      if annt_data is None: 
         return [], []
      
      img_bboxes, img_classes = [], []
      for i in range(len(annt_data['object'])):
        if isinstance(annt_data['object'], list):
            object_data = annt_data['object'][i]
        else: 
            object_data = annt_data['object']
          
        bbox_data = object_data['bndbox']
        cls_name = object_data['name']
        class_id = self.classes.index(cls_name)

        H, W = annt_data['size']['height'], annt_data['size']['width']

        #scale_factor_x = self.resize_w/float(W)
        #scale_factor_y = self.resize_h/float(H)

        lx = float(bbox_data['xmin'])# * scale_factor_x  # converting to scaled image coords
        ly = float(bbox_data['ymin'])# * scale_factor_y
        hx = float(bbox_data['xmax'])# * scale_factor_x
        hy = float(bbox_data['ymax'])# * scale_factor_y
        center_x = ((lx + hx)/2)
        center_y = ((ly + hy)/2)
        height = (hy - ly)
        width = (hx - lx)
        if center_x >= 1280:
           print(center_x)
        bbox = [np.round(center_x), np.round(center_y),  np.round(width, 2), np.round(height, 2)]
        img_classes.append(class_id)
        img_bboxes.append(bbox)
      return img_bboxes, img_classes

    
    def create_targets_tensor(self, image, bboxes, class_ids):
      """
      Turn bbox annotations into learning targets
      param annotations: List of [class_id, x_center, y_center, width, height]
      param image: 
      """

      targets = torch.zeros((self.grid_height, self.grid_width, self.n_anchors, self.K + 5))
      
      for i in range(len(class_ids)):
          class_id = class_ids[i]
          bbox_x, bbox_y, bbox_w, bbox_h = bboxes[i]
          anchor_ious = []
          width, height = image.size
          # scale_factor_x = self.resize_w/width
          # scale_factor_y = self.resize_h/height
          for n in range(self.n_anchors):
            anchor_h = self.anchors[class_id][n][0] * self.resize_h
            anchor_w = self.anchors[class_id][n][1] * self.resize_w
            this_iou = iou(anchor_w,anchor_h, bbox_h, bbox_w)
            anchor_ious.append(this_iou)
            grid_j = int(bbox_x // self.stride)
            grid_i = int(bbox_y // self.stride)
            target_offsets = bbox_to_target_offsets(bbox_x, bbox_y, bbox_w, bbox_h, 
                                                    anchor_w, anchor_h,
                                                    self.stride, grid_i, grid_j)  # Size (4, )

            if targets[grid_i, grid_j, n, self.K+4] == 1:
              break # if there is already an object in the same grid

            targets[grid_i, grid_j, n, class_id] = 1  # Class score
            targets[grid_i, grid_j, n, self.K:self.K+4] = torch.tensor(target_offsets)

          best_anchor_n = np.argmax(anchor_ious)

          targets[grid_i, grid_j, best_anchor_n, self.K+4] = 1  # High confidence only for the best anchor
               
      return targets.view(self.grid_height, self.grid_width, self.n_anchors, (self.K + 5))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        bboxes = self.bboxes[idx]
        class_ids = self.class_ids[idx]
        width, height = image.size
        
        if len(class_ids)!= 0:
           bboxes = BoundingBoxes(bboxes, format='XYWH', canvas_size=(height, width))
           trans_img, bboxes, class_ids = self.transform(image, bboxes, class_ids)
           #trans_img, bboxes, class_ids = self.flip_transform(trans_img, bboxes, class_ids)
        else:
           trans_img = self.transform(image)


        # Native horizontal flip does not work with bboxes for some reason
        # implementing our own
        if np.random.random() > 0.5 and self.train_transforms: 
          trans_img = self.horizontal_flip(trans_img)
          if len(class_ids) != 0:
            H, W = bboxes.canvas_size
            for i in range(len(bboxes)):
              bbox = bboxes[i]
              bbox[0] = W - bbox[0]
              bboxes[i] = bbox

        targets = self.create_targets_tensor(image, bboxes, class_ids)
        if self.return_img_id:
            return trans_img, targets, self.img_ids[idx]
        return trans_img, targets