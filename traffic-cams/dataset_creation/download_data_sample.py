import os
import time
import requests
from datetime import datetime

# URL of the image
BASE_URL = "https://ristmikud.tallinn.ee/last/"
# CAMERA_LIST = ['cam092',
#                'cam141',
#                'cam057',
#                'cam149',
#                'cam202',
#                'cam163',
#                'cam073',
#                'cam210',
#                'cam046',
#                'cam246',
#                'cam103',
#                'cam163',
#                'cam081',]

CAMERA_LIST = ['cam056',  # Test SET
               'cam256',
               'cam142',
               'cam251',
               'cam087',
               'cam238',    
               'cam070']


#CAMERA_LIST = ['cam081',]
# Directory to save images
BASE_DIR = "downloaded_images_test" 
# Interval in seconds (e.g., 60 seconds = 1 minute)
INTERVAL = 60  


def get_save_directory():
    """Returns the directory path for today's date and creates it if necessary."""
    today = datetime.now().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
    save_dir = os.path.join(BASE_DIR, today)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    return save_dir

def download_image(cam_name):
    """Downloads the image and saves it with a timestamp."""
    try:
        url = BASE_URL + cam_name + '.jpg'
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(get_save_directory(), f"{cam_name}_{timestamp}.jpg")

        # Save the image
        with open(filename, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        
        print(f"[{timestamp}] Image saved: {filename}")
    
    except requests.RequestException as e:
        print(f"Error downloading image: {e}")

def main():
    """Continuously downloads images at the specified interval."""
    print("Starting image downloader...")
    for _ in range(10):
        for cam_name in CAMERA_LIST:
            download_image(cam_name)
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
