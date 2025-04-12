import os
import glob
import shutil
import cv2
import xml.etree.ElementTree as ET

def resize_image(img_path, out_path, new_size=(1280, 720)):
    """
    Read an image, resize to (width=1280, height=720),
    and save to out_path. Returns (orig_w, orig_h).
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    orig_h, orig_w = img.shape[:2]
    
    # If already the correct size, just copy instead of re-encoding
    if (orig_h, orig_w) == (new_size[1], new_size[0]):
        shutil.copy(img_path, out_path)
    else:
        resized_img = cv2.resize(img, new_size)  # (width, height)
        cv2.imwrite(out_path, resized_img)
        
    return orig_w, orig_h

def resize_voc_annotation(xml_in, xml_out, orig_w, orig_h, new_w=1280, new_h=720):
    """
    Read PASCAL VOC .xml annotation, scale bounding boxes 
    from (orig_w, orig_h) -> (new_w, new_h), and save new .xml.
    """
    tree = ET.parse(xml_in)
    root = tree.getroot()
    
    # Update <width>, <height> in <size>
    size_tag = root.find('size')
    if size_tag is not None:
        w_tag = size_tag.find('width')
        h_tag = size_tag.find('height')
        if w_tag is not None:
            w_tag.text = str(new_w)
        if h_tag is not None:
            h_tag.text = str(new_h)
    
    # Scale bounding boxes
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        if bndbox is not None:
            xmin = bndbox.find('xmin')
            ymin = bndbox.find('ymin')
            xmax = bndbox.find('xmax')
            ymax = bndbox.find('ymax')
            
            if None not in (xmin, ymin, xmax, ymax):
                # Convert to float
                x1 = float(xmin.text)
                y1 = float(ymin.text)
                x2 = float(xmax.text)
                y2 = float(ymax.text)
                
                # Scale coords
                scale_x = new_w / float(orig_w)
                scale_y = new_h / float(orig_h)
                
                new_x1 = int(round(x1 * scale_x))
                new_y1 = int(round(y1 * scale_y))
                new_x2 = int(round(x2 * scale_x))
                new_y2 = int(round(y2 * scale_y))
                
                # Update XML
                xmin.text = str(new_x1)
                ymin.text = str(new_y1)
                xmax.text = str(new_x2)
                ymax.text = str(new_y2)
    
    # Write out new XML
    tree.write(xml_out)

def main():
    # --- USER-DEFINED PATHS ---
    images_dir = "Project/test-set/images"  # original images
    annots_dir = "Project/test-set/Annotations" # original annotations
    
    # Output folders (will be created if they don't exist)
    out_images_dir = "Project/test_resized/images"
    out_annots_dir = "Project/test_resized/Annotations"
    
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_annots_dir, exist_ok=True)
    
    # Gather all image files (assuming .jpg)
    image_paths = glob.glob(os.path.join(images_dir, "*.jpg"))
    
    # For each image, we expect a matching .xml annotation with the same base name
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]
        
        # Source XML
        xml_path = os.path.join(annots_dir, basename + ".xml")

        
        # Destination paths
        out_img_path = os.path.join(out_images_dir, filename)
        out_xml_path = os.path.join(out_annots_dir, basename + ".xml")
        
        # 1) Resize/copy image
        try:
            orig_w, orig_h = resize_image(img_path, out_img_path, new_size=(1280, 720))
        except ValueError as e:
            print(e)
            continue
        
        # 2) Resize annotation
        # If the original is already 1280x720, we just copy annotation 
        #   (although copying does not harm if we re-write bounding box 
        #    coords that are the same).
        if not os.path.isfile(xml_path):
            # If there's no matching XML, skip or handle differently
            print(f"[Warning] No XML annotation for {img_path}, skipping.")
            continue

        if (orig_h, orig_w) == (720, 1280):
            shutil.copy(xml_path, out_xml_path)
        else:
            resize_voc_annotation(xml_path, out_xml_path, orig_w, orig_h,
                                  new_w=1280, new_h=720)
        
        print(f"Processed: {filename}")
    
    print("Done resizing dataset!")

if __name__ == "__main__":
    main()