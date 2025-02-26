import numpy as np
import os
import gc
import rasterio
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random

print("All modules imported successfully.")

# Set parameters
np.random.seed(1)
optimizer_loss_fun = "categorical_crossentropy"
optimizer_algorithm = Adam(learning_rate=0.001)
number_epoch = 50  # Reduced epochs to save training time
batch_length = 32  # Increased batch size for faster training
show_inter_results = 1

# Define dataset path
data_dir = "C:/Users/Chris/Desktop/EuroSAT/EuroSAT_MS"
img_size = (256, 256)
max_images_per_class = 50  # Limit dataset size (Adjust as needed)

# Load images and labels (Subset Selection)
print("Starting to load images and labels...")
class_names = sorted(os.listdir(data_dir))
data = []
labels = []

# Load a subset of images per class
for class_name in class_names:
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        img_files = os.listdir(class_path)

        # Select a random subset of images (Limit per class)
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
                    )  # Convert to (height, width, bands)

                if image_array.shape[:2] != img_size:
                    image_array = tf.image.resize(
                        image_array, img_size
                    ).numpy()

                data.append(image_array)
                labels.append(class_name)
            except Exception as e:
                print(f"Could not process image {img_path}: {e}")

print("Finished loading images.")

# Convert to numpy arrays
print("Converting data and labels to numpy arrays...")
data = np.array(data, dtype=np.float32)
labels = np.array(labels)

# Normalize image data
print("Normalizing image data...")
data = data / np.max(data)

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
y = to_categorical(labels)

print(f"Dataset size after sampling: {data.shape[0]} images")
print(f"Data shape: {data.shape}, Labels shape: {y.shape}")

# Train-test split
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, random_state=1, stratify=labels
)

# Build model
print("Building the model...")
model = Sequential(
    [
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(img_size[0], img_size[1], 13),
        ),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(len(class_names), activation="softmax"),
    ]
)

model.compile(
    loss=optimizer_loss_fun,
    optimizer=optimizer_algorithm,
    metrics=["accuracy"],
)

# Train model
print("Training the model...")
history = model.fit(
    X_train,
    y_train,
    epochs=number_epoch,
    batch_size=batch_length,
    verbose=show_inter_results,
    validation_data=(X_test, y_test),
)

# Evaluate model
print("Evaluating the model...")
scores = model.evaluate(X_test, y_test, verbose=1)
print(f"Test accuracy: {scores[1] * 100:.2f}%")

# Plot results
print("Plotting the loss and accuracy curves...")
plt.figure(figsize=(12, 5))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
