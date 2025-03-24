# Libraries
import numpy as np
import os
import time
import gc
import rasterio
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# Fix the seed
def set_global_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"✅ Global seed set to {seed}")


# Define Normalisation Function
def normalise_band(band, mean, std):
    band = np.nan_to_num(band, nan=0)
    band = ((band - np.mean(band)) / (np.std(band) + 1e-8)) * std + mean
    band = np.clip(band, 0, 1)
    return band


# Set seed
set_global_seed()

# Parameters
optimizer_algorithm = Adam(learning_rate=0.00003)
number_epoch = 200
batch_length = 16
show_inter_results = 1
num_rows, num_cols, num_bands = (
    64,
    64,
    12,
)  # 12 bands for MS data (excluding B10)

# Regularization Parameters
use_l1, use_l2 = True, True
l1_reg, l2_reg = 0.0002, 0.001

# Define Data Path
data_dir = "C:/Users/Chris/Desktop/EuroSAT/EuroSAT_MS"
band_stats_path = "band_statistics.csv"
max_images_per_class = 250  # Limit dataset size


# Load Band Statistics
band_stats_df = pd.read_csv(band_stats_path).drop_duplicates(subset="Band")
band_stats_df = band_stats_df.sort_values("Band").reset_index(drop=True)
band_means = band_stats_df["Mean"].values
band_stds = band_stats_df["Std"].values

# Measure data loading time
start_data_time = time.time()

# Load Image Paths & Labels
print("Loading image paths and labels...")
class_names = sorted(os.listdir(data_dir))
data, labels = [], []

for class_name in class_names:
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        img_files = os.listdir(class_path)
        selected_files = random.sample(
            img_files, min(len(img_files), max_images_per_class)
        )
        print(
            f"Loading {len(selected_files)} images for class '{class_name}'..."
        )
        for img_file in selected_files:
            img_path = os.path.join(class_path, img_file)
            try:
                with rasterio.open(img_path) as img:
                    image_array = img.read()
                    image_array = np.transpose(
                        image_array, (1, 2, 0)
                    )  # (H, W, C)
                    image_array = np.delete(
                        image_array, 10, axis=2
                    )  # Remove B10 (index 10)
                image_array = tf.image.resize(
                    image_array, (num_rows, num_cols)
                ).numpy()
                data.append(image_array)
                labels.append(class_name)
            except Exception as e:
                print(f"Could not process image {img_path}: {e}")

print("Finished loading images.")

# Convert to numpy arrays
data = np.array(data, dtype=np.float32)
labels = np.array(labels)

# Normalising using per-band stats
print("Normalising...")
for band in range(data.shape[-1]):
    max_val = np.max(data[:, :, :, band])
    data[:, :, :, band] /= max_val
    if max_val <= 0:
        print(f"WARNING: max_val of {band} is <=0 ")

data = data / np.max(data)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
y = to_categorical(labels)

print(f"Dataset size after sampling: {data.shape[0]} images")
print(f"Data shape: {data.shape}, Labels shape: {y.shape}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, random_state=1, stratify=labels
)

# Learning Rate Scheduler
lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-6
)

# Early Stopping
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
)

# Build Model
print("Building the CNN model...")
regularizer = l1_l2(l1=l1_reg if use_l1 else 0.0, l2=l2_reg if use_l2 else 0.0)

model = Sequential(
    [
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(
                num_rows,
                num_cols,
                num_bands,
            ),  # One less due to B10
            kernel_regularizer=regularizer,
        ),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizer),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu", kernel_regularizer=regularizer),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(len(class_names), activation="softmax"),
    ]
)

print("Model built")

# Compile Model
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer_algorithm,
    metrics=["accuracy"],
)
print("Model compiled")

# Train Model
print("🚀 Starting training...")
start_train_time = time.time()

history = model.fit(
    X_train,
    y_train,
    epochs=number_epoch,
    batch_size=batch_length,
    verbose=show_inter_results,
    validation_data=(X_test, y_test),
    callbacks=[lr_scheduler, early_stopping],
)

training_time = time.time() - start_train_time
print(f"Training completed in {training_time:.2f} seconds.")

# Save Model
print("Saving the trained model...")
model.save("eurosat_ms_model_v3.keras")
print("Model saved successfully.")

# Evaluate Model
print("Evaluating model on test dataset...")
start_eval_time = time.time()
scores = model.evaluate(X_test, y_test, verbose=1)
eval_time = time.time() - start_eval_time

print(f"Test accuracy: {scores[1] * 100:.2f}%")
print(f"Evaluation completed in {eval_time:.2f} seconds.")

# Generate Predictions for Confusion Matrix
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Display Classification Report
print(
    "Classification Report:\n",
    classification_report(y_true, y_pred, target_names=class_names),
)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Plot Training & Validation Loss and Accuracy
plt.figure(figsize=(12, 5))

# Loss Curve
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

# Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.show()
