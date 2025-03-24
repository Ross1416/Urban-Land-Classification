# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 17:19:37 2025

@author: Chris
"""

import os
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set dataset path
data_dir = "C:/Users/Chris/Desktop/EuroSAT/EuroSAT_MS"
output_stats_path = "band_stats_before.csv"
hist_dir = "band_histograms"
img_size = (64, 64)
num_bands = 12  # Including B10
band_to_remove = 10  # Index of Band 10

# Create histogram output directory
os.makedirs(hist_dir, exist_ok=True)


# Function to load, resize, remove Band 10, and clip pixel values
def load_image(path):
    with rasterio.open(path) as src:
        img = src.read()  # Shape: (bands, height, width)
        img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        img = np.delete(img, band_to_remove, axis=2)  # Remove B10
        img = tf.image.resize(img, img_size).numpy()

        # Count clipped pixels
        num_below_0 = np.sum(img < 0)
        num_above_5000 = np.sum(img > 5000)

        # Clip pixel values
        img = np.clip(img, 0, 5000)

        return img.astype(np.float32), num_below_0, num_above_5000


# Load all image paths
print("🔍 Scanning dataset...")
image_paths = []
class_names = sorted(os.listdir(data_dir))

for class_name in class_names:
    print(class_name)
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        for img_file in os.listdir(class_path):
            full_path = os.path.join(class_path, img_file)
            image_paths.append(full_path)

print(f"📸 Found {len(image_paths)} images in total.")

# Pre-allocate stats lists
band_mins, band_maxs = [], []
band_means, band_stds = [], []

# Process all images
all_images = []
total_clipped_below_0 = 0
total_clipped_above_5000 = 0

print("📥 Loading and processing images...")
for path in tqdm(image_paths):
    try:
        img, clipped_below, clipped_above = load_image(
            path
        )  # Shape: (H, W, 11)
        all_images.append(img)
        total_clipped_below_0 += clipped_below
        total_clipped_above_5000 += clipped_above
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")


# Convert to NumPy array
all_images = np.array(all_images)  # Shape: (N, H, W, 11)
print(f"✅ Loaded {all_images.shape[0]} images, shape: {all_images.shape}")

# Compute statistics and plot histograms
print("📊 Computing statistics and plotting histograms...")
for i in range(all_images.shape[-1]):
    band_index = i if i < band_to_remove else i + 1
    band_name = f"B{band_index}"

    band = all_images[:, :, :, i].flatten()

    # Stats
    band_mins.append(np.min(band))
    band_maxs.append(np.max(band))
    band_means.append(np.mean(band))
    band_stds.append(np.std(band))

    # Plot histogram
    plt.figure(figsize=(6, 4))
    plt.hist(band, bins=100, color="steelblue", edgecolor="black")
    plt.title(f"{band_name} - Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(hist_dir, f"{band_name}_hist.png"))
    plt.close()

print(f"📁 Histograms saved to: {hist_dir}")

# Save stats to CSV
print("💾 Saving band statistics...")
stats_df = pd.DataFrame(
    {
        "Band": [
            f"B{i if i < band_to_remove else i + 1}"
            for i in range(all_images.shape[-1])
        ],
        "Min": band_mins,
        "Max": band_maxs,
        "Mean": band_means,
        "Std": band_stds,
    }
)

stats_df.to_csv(output_stats_path, index=False)
print(f"✅ Band statistics saved to: {output_stats_path}")
