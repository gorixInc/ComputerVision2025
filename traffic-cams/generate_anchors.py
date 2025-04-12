import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.cluster import KMeans

def parse_annotation_xml(xml_file):
    """
    Parse a PASCAL VOC XML file and return list of (width, height)
    for each bounding box.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    size = root.find('size')
    if size is None:
        return boxes

    # The actual image dimensions
    w_img = float(size.find('width').text)
    h_img = float(size.find('height').text)

    # For each object/annotation, read xmin, ymin, xmax, ymax
    for obj in root.iter('object'):
        bbox = obj.find('bndbox')
        if bbox is None:
            continue

        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        box_w = xmax - xmin
        box_h = ymax - ymin
        boxes.append((box_w, box_h))

    return boxes, (w_img, h_img)

def load_dataset_annotations(annotations_dir):
    """
    Load all bounding boxes (width, height) from a folder of VOC .xml files.
    Returns:
        a list of tuples (box_w, box_h) for every bounding box in the dataset
    """
    all_boxes = []
    xml_files = glob.glob(os.path.join(annotations_dir, "*.xml"))
    for xml_file in xml_files:
        boxes_info = parse_annotation_xml(xml_file)
        # parse_annotation_xml returns (boxes, (w_img, h_img)), so unpack
        if boxes_info:
            boxes, (w_img, h_img) = boxes_info
            for (bw, bh) in boxes:
                # Collect raw widths/heights
                all_boxes.append((bw, bh))

    return all_boxes

def kmeans_anchors(all_boxes, input_size=(416, 416), num_anchors=9):
    """
    Run KMeans on bounding box widths/heights to generate anchor boxes.

    :param all_boxes: List of (width, height) for all bounding boxes.
    :param input_size: Tuple (img_width, img_height) to which boxes are scaled.
    :param num_anchors: Number of anchors (clusters) to generate.
    :return: KMeans model (with cluster centers) and scaled anchor boxes.
    """
    if not all_boxes:
        raise ValueError("No bounding boxes found in annotations.")

    img_w, img_h = input_size
    # Convert all_boxes to a numpy array, scale them up or down
    # so that they are relative to the chosen input size
    boxes_array = []
    # Optionally, you might store the original image size for each box
    # and then do an exact scale. For simplicity, we assume the dataset
    # is consistent or we approximate with one scale factor.
    # Alternatively, if you want each bounding box to be scaled
    # individually by the original image size, do so within the
    # parse_annotation_xml function. For now:
    for (bw, bh) in all_boxes:
        # We'll assume the original Pascal VOC images can be different, 
        # so you'd want to handle that carefully. Here is the naive approach:
        #scale = [bw / 1280, bh / 720]
        # For demonstration, let's just treat these widths and heights "as is"
        boxes_array.append([bw/1280, bh/720])

    boxes_array = np.array(boxes_array)

    # Run K-means
    kmeans = KMeans(n_clusters=num_anchors, n_init=10, random_state=42)
    kmeans.fit(boxes_array)

    # The cluster centers are our anchor boxes in (width, height)
    anchors = kmeans.cluster_centers_

    # Sort anchors by area (w*h)
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]

    return kmeans, anchors

def average_iou(boxes, anchors):
    """
    Compute the average IoU between a set of boxes and anchor boxes.
    boxes: N x 2 (w, h)
    anchors: K x 2 (w, h)
    Returns average IoU across all boxes.
    """
    def iou(box, anchor):
        # box: (bw, bh), anchor: (aw, ah)
        intersect_w = np.minimum(box[0], anchor[0])
        intersect_h = np.minimum(box[1], anchor[1])
        intersect = intersect_w * intersect_h
        union = (box[0]*box[1]) + (anchor[0]*anchor[1]) - intersect
        return intersect / (union + 1e-6)

    iou_sum = 0.0
    for b in boxes:
        # for each box, find the anchor that yields the best IoU
        ious = [iou(b, a) for a in anchors]
        best_iou = max(ious)
        iou_sum += best_iou

    return iou_sum / len(boxes)

def main():
    # 1. Define your PASCAL VOC Annotations directory
    annotations_dir = "Project/dataset_resized_split/train/Annotations"  # Change this path

    # 2. Load bounding boxes
    all_boxes = load_dataset_annotations(annotations_dir)
    print(f"Loaded {len(all_boxes)} bounding boxes.")

    # 3. Generate anchors with k-means
    num_anchors = 6
    input_shape = (1, 1)  # Typical YOLO input resolution
    kmeans_model, anchors = kmeans_anchors(all_boxes, input_size=input_shape, num_anchors=num_anchors)

    # 4. Compute average IoU
    boxes_array = np.array(all_boxes)
    avg_iou = average_iou(boxes_array, anchors)

    # 5. Print results
    print(f"\nAnchors (width, height) for input size {input_shape}:")
    for i, anchor in enumerate(anchors):
        w, h = anchor
        print(f"[{h:.4f}, {w:.4f}]")

    print(f"\nAverage IoU: {avg_iou:.4f}")

if __name__ == "__main__":
    main()