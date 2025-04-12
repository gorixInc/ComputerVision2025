import xml.etree.ElementTree as ET
import pandas as pd
import os
import glob
## AI GENERATED CODE

# --- Configuration ---

# Directory containing the image files (e.g., .jpg, .png)
IMAGE_DIR = 'Project/test_resized/images'
# Directory containing the PASCAL VOC XML annotation files
# (Can be the same as IMAGE_DIR if they are stored together)
VOC_XML_DIR = 'Project/test_resized/Annotations'
# Path to save the output ground truth CSV file
OUTPUT_CSV_FILE = 'Project/to_kaggle/test_ground_truth.csv'
# List of image file extensions to look for
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

# --- NO LONGER NEEDED FOR CONVERSION, but good for validation/documentation ---
# Define the list of expected class names found in your XMLs.
# This helps catch unexpected class names during processing.
EXPECTED_CLASS_NAMES = {
    'passenger_car',
    'pedestrian',
    'bus',
    'lorry',
    'utility_vehicle',
    'tram'
}
# ---

# --- Helper Function ---
def parse_xml_annotation_with_names(xml_file_path, expected_classes):
    """Parses a single PASCAL VOC XML file and returns annotations with class names."""
    annotations = []
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Extract image ID (filename without extension) - Useful for verification
        filename_tag = root.find('filename')
        xml_image_filename = filename_tag.text if filename_tag is not None else "Unknown"

        # Process each object annotation in the XML
        for obj in root.findall('object'):
            class_name_tag = obj.find('name')
            if class_name_tag is None:
                 print(f"Warning: Skipping object in {xml_file_path} - missing <name> tag.")
                 continue
            class_name = class_name_tag.text.strip() # Use strip() to remove leading/trailing whitespace

            # Optional Validation: Check if the class name is expected
            if expected_classes and class_name not in expected_classes:
                print(f"Warning: Found unexpected class name '{class_name}' in {xml_file_path}. Skipping object or check EXPECTED_CLASS_NAMES.")
                # Depending on strictness, you might want to 'continue' here to skip it
                # continue # Uncomment this line to strictly enforce only expected classes

            bndbox = obj.find('bndbox')
            if bndbox is None:
                print(f"Warning: Skipping object '{class_name}' in {xml_file_path} - missing <bndbox> tag.")
                continue

            # Extract bounding box coordinates
            try:
                # Using float for potentially more precision from some annotation tools
                x_min = float(bndbox.find('xmin').text)
                y_min = float(bndbox.find('ymin').text)
                x_max = float(bndbox.find('xmax').text)
                y_max = float(bndbox.find('ymax').text)
                # Optional: Convert to int if pixel coordinates are always integers
                # x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            except (ValueError, TypeError, AttributeError) as e:
                 print(f"Warning: Skipping object '{class_name}' in {xml_file_path} - error parsing bbox: {e}")
                 continue

            # Append data for this object (using class_name directly)
            annotations.append({
                # 'image_id' will be added in the main loop based on the image file
                'class_name': class_name,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
            })
    except ET.ParseError:
        print(f"Error parsing XML file: {xml_file_path}")
    except Exception as e:
        print(f"An unexpected error occurred processing {xml_file_path}: {e}")

    return annotations

# --- Main Conversion Logic ---
all_ground_truths = []
processed_image_ids = set()
images_without_xml = []

print(f"Scanning for images in: {IMAGE_DIR}")
# Find all image files based on extensions
image_files = []
for ext in IMAGE_EXTENSIONS:
    image_files.extend(glob.glob(os.path.join(IMAGE_DIR, '*' + ext)))
    image_files.extend(glob.glob(os.path.join(IMAGE_DIR, '*' + ext.upper()))) # Include uppercase extensions

print(f"Found {len(image_files)} potential image files.")

if not image_files:
    print("Error: No image files found. Please check IMAGE_DIR and IMAGE_EXTENSIONS.")
    exit()

for img_path in image_files:
    # Extract image ID (filename without extension)
    img_filename = os.path.basename(img_path)
    image_id = os.path.splitext(img_filename)[0]

    if image_id in processed_image_ids:
        continue # Avoid processing duplicates
    processed_image_ids.add(image_id)

    # Construct the expected path for the corresponding XML file
    xml_filename = image_id + '.xml'
    xml_path = os.path.join(VOC_XML_DIR, xml_filename)

    if os.path.exists(xml_path):
        # XML exists, parse annotations using the updated function
        annotations = parse_xml_annotation_with_names(xml_path, EXPECTED_CLASS_NAMES)
        if annotations:
            # Add the image_id to each annotation dictionary
            for ann in annotations:
                ann['image_id'] = image_id
                all_ground_truths.append(ann)
        else:
             # XML existed but had parsing errors or no valid objects
             print(f"Processed '{img_filename}': XML found but yielded no valid annotations.")
    else:
        # XML does not exist for this image - Treat as image with no objects
        images_without_xml.append(image_id)

# --- Create DataFrame and Save ---
if all_ground_truths:
    df_gt = pd.DataFrame(all_ground_truths)
    # Define column order explicitly (NO class_id)
    column_order = ['image_id', 'class_name', 'x_min', 'y_min', 'x_max', 'y_max']
    # Ensure all columns exist
    for col in column_order:
        if col not in df_gt.columns:
            df_gt[col] = None # Or appropriate default like pd.NA
    df_gt = df_gt[column_order]

    df_gt.to_csv(OUTPUT_CSV_FILE, index=False)
    print("-" * 30)
    print(f"Successfully created ground truth CSV: {OUTPUT_CSV_FILE}")
    print(f"Total annotations written: {len(df_gt)}")
    print(f"Total unique images processed: {len(processed_image_ids)}")
    print(f"Images processed with annotations: {len(df_gt['image_id'].unique())}")
    print(f"Images processed without annotations (no XML found): {len(images_without_xml)}")
else:
    print("-" * 30)
    print("Warning: No valid annotations were found in any XML files.")
    print(f"Total unique images processed: {len(processed_image_ids)}")
    print(f"Images processed without annotations (no XML found): {len(images_without_xml)}")
    print("Output CSV file was NOT created as it would be empty.")

print("-" * 30)
print("IMPORTANT:")
print("1. This script outputs CLASS NAMES directly into the CSV.")
print("2. Ensure the class names in your XML files are consistent (case, spelling).")
print(f"3. Provide the list of valid class names (like {list(EXPECTED_CLASS_NAMES)[:5]}...) to competition participants.")
print("4. Your Kaggle evaluation script MUST be adapted to work with class names (string comparison).")
print(f"5. The evaluation script must know the complete list of image IDs being scored (total: {len(processed_image_ids)} images processed).")
print("6. The script assumes images listed in `images_without_xml` have zero ground truth objects.")

    