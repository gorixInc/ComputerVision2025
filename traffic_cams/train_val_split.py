import os
import glob
import shutil
import random

## AI GENERATED CODE

def create_dirs(dirs_list):
    """Helper to create directories if they don't exist."""
    for d in dirs_list:
        os.makedirs(d, exist_ok=True)

def main():
    # -------------------------- USER SETTINGS --------------------------
    images_dir = "traffic_cams/main_dataset/images"
    annots_dir = "traffic_cams/main_dataset/Annotations"

    # Output folders for train/val
    base_output_dir = "traffic_cams/dataset_resized_split/"  # Everything will be inside here
    train_images_dir = os.path.join(base_output_dir, "train", "images")
    train_annots_dir = os.path.join(base_output_dir, "train", "Annotations")
    val_images_dir   = os.path.join(base_output_dir, "val",   "images")
    val_annots_dir   = os.path.join(base_output_dir, "val",   "Annotations")

    # Create all needed directories
    create_dirs([
        train_images_dir, train_annots_dir,
        val_images_dir,   val_annots_dir
    ])

    # Ratio for train/val split
    train_ratio = 0.8
    random_seed = 42  # for reproducibility
    
    # ----------------------- GATHER IMAGE PATHS ------------------------
    # We'll assume images are .jpg (adjust if needed)
    image_paths = glob.glob(os.path.join(images_dir, "*.jpg"))
    image_paths.sort()  # optional: sort so the shuffle is consistent

    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    # Shuffle the list for random split
    random.seed(random_seed)
    random.shuffle(image_paths)

    # Determine train/val cutoff
    train_count = int(len(image_paths) * train_ratio)
    train_images = image_paths[:train_count]
    val_images   = image_paths[train_count:]

    print(f"Total images: {len(image_paths)}")
    print(f"Train images: {len(train_images)}")
    print(f"Val images:   {len(val_images)}")

    # ------------------------ COPY TO TRAIN ----------------------------
    for img_path in train_images:
        img_name = os.path.basename(img_path)
        # copy image
        shutil.copy(img_path, os.path.join(train_images_dir, img_name))
        
        # see if there's an XML file
        base_name, _ = os.path.splitext(img_name)
        xml_name = base_name + ".xml"
        xml_path = os.path.join(annots_dir, xml_name)
        
        # if it exists, copy it
        if os.path.exists(xml_path):
            shutil.copy(xml_path, os.path.join(train_annots_dir, xml_name))

    # ------------------------- COPY TO VAL -----------------------------
    for img_path in val_images:
        img_name = os.path.basename(img_path)
        # copy image
        shutil.copy(img_path, os.path.join(val_images_dir, img_name))
        
        # see if there's an XML file
        base_name, _ = os.path.splitext(img_name)
        xml_name = base_name + ".xml"
        xml_path = os.path.join(annots_dir, xml_name)
        
        # if it exists, copy it
        if os.path.exists(xml_path):
            shutil.copy(xml_path, os.path.join(val_annots_dir, xml_name))

    print("Train/Val split completed!")

if __name__ == "__main__":
    main()
