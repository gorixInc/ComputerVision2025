#!/usr/bin/env python3

import glob
import os
import datetime
import matplotlib.pyplot as plt
from collections import Counter

def plot_jpegs_distribution(dir_path):
    # Gather .jpg and .jpeg files
    jpg_files = glob.glob(os.path.join(dir_path, '*.jpg'))
    jpg_files += glob.glob(os.path.join(dir_path, '*.jpeg'))

    hours = []

    for filepath in jpg_files:
        # Example filename: cam046_20250325_233844.jpg
        filename = os.path.basename(filepath)  # cam046_20250325_233844.jpg
        base_name = os.path.splitext(filename)[0]  # cam046_20250325_233844
        
        # Split on underscores: ['cam046', '20250325', '233844']
        parts = base_name.split('_')
        if len(parts) == 3:
            # Extract the HHMMSS part
            time_str = parts[2]  # e.g. '233844'
            
            # Convert HHMMSS to a time object
            try:
                time_obj = datetime.datetime.strptime(time_str, '%H%M%S').time()
                hours.append(time_obj.hour)
            except ValueError:
                # If parsing fails, skip this file
                pass

    # Count occurrences per hour (0–23)
    hour_counts = Counter(hours)

    # Prepare data for plotting
    x_values = list(range(24))
    y_values = [hour_counts.get(h, 0) for h in x_values]

    # Plot the distribution
    plt.bar(x_values, y_values)
    plt.xlabel("Hour of Day (0-23)")
    plt.ylabel("Number of Images")
    plt.title("Distribution of JPEG Images by Hour of the Day")
    plt.xticks(x_values)
    plt.show()

if __name__ == "__main__":
    # Prompt user for directory path
    directory = input("Enter the directory path containing the images: ")
    plot_jpegs_distribution(directory)
