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
from tqdm import tqdm

# Set dataset path
data_dir = "C:/Users/Chris/Desktop/EuroSAT/EuroSAT_MS"
output_stats_path = "band_stats_before_after.csv"
img_size = (64, 64)
num_bands = 12  # Including B10
band_to_remove = 10  # Index of Band 10


# Function to load and process a single image
def load_image(path):
    with rasterio.open(path) as src:
        img = src.read()  # Shape: (bands, height, width)
        img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        img = np.delete(img, band_to_remove, axis=2)  # Remove B10
        img = tf.image.resize(img, img_size).numpy()
        return img.astype(np.float32)


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
band_mins_before, band_maxs_before = [], []
band_means_before, band_stds_before = [], []

band_mins_after, band_maxs_after = [], []
band_means_after, band_stds_after = [], []

# Process all images
all_images = []
print("📥 Loading and processing images...")
for path in tqdm(image_paths):
    try:
        img = load_image(path)  # Shape: (H, W, 11)
        all_images.append(img)
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")

# Convert to NumPy array
all_images = np.array(all_images)  # Shape: (N, H, W, 11)
print(f"✅ Loaded {all_images.shape[0]} images, shape: {all_images.shape}")

# Compute pre-normalization stats
print("📊 Computing statistics before normalization...")
for i in range(all_images.shape[-1]):
    band = all_images[:, :, :, i]
    band_flat = band.flatten()
    band_mins_before.append(np.min(band_flat))
    band_maxs_before.append(np.max(band_flat))
    band_means_before.append(np.mean(band_flat))
    band_stds_before.append(np.std(band_flat))

# Normalize each band by dividing by its max
print("🧪 Normalizing each band by its max value...")
normalized_images = np.empty_like(all_images)
for i in range(all_images.shape[-1]):
    max_val = band_maxs_before[i]
    normalized_images[:, :, :, i] = all_images[:, :, :, i] / (
        max_val if max_val != 0 else 1
    )

# Compute post-normalization stats
print("📊 Computing statistics after normalization...")
for i in range(normalized_images.shape[-1]):
    band = normalized_images[:, :, :, i]
    band_flat = band.flatten()
    band_mins_after.append(np.min(band_flat))
    band_maxs_after.append(np.max(band_flat))
    band_means_after.append(np.mean(band_flat))
    band_stds_after.append(np.std(band_flat))

# Save stats to CSV
print("💾 Saving band statistics...")
stats_df = pd.DataFrame(
    {
        "Band": [
            f"B{i if i < band_to_remove else i + 1}"
            for i in range(all_images.shape[-1])
        ],
        "Min_Before": band_mins_before,
        "Max_Before": band_maxs_before,
        "Mean_Before": band_means_before,
        "Std_Before": band_stds_before,
        "Min_After": band_mins_after,
        "Max_After": band_maxs_after,
        "Mean_After": band_means_after,
        "Std_After": band_stds_after,
    }
)

stats_df.to_csv(output_stats_path, index=False)
print(f"✅ Band statistics saved to: {output_stats_path}")
