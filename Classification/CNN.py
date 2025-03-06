<<<<<<< Updated upstream
# Import Tensorflow, keras, scikit-learn, matplotlib - latest versions
# Python3.12
# Download eurosat dataset - change filepath line 33
try:
    import numpy as np
    import os
    import gc
    import tensorflow
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
    from tensorflow.keras import backend as krs
    from PIL import Image
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt

    print("All modules imported successfully.")
except ImportError as e:
    print(f"An error occurred: {e}")


# Parameters
np.random.seed(1)
optimizer_loss_fun = "categorical_crossentropy"
optimizer_algorithm = Adam(learning_rate=0.001)
number_epoch = 100
batch_length = 20
show_inter_results = 1

# Define data path
data_dir = "EuroSAT/EuroSAT_RGB"  # Update to the correct path
img_size = (256, 256)  # Resize images to 256x256 if needed

# Load images and labels
print("Starting to load images and labels...")
class_names = sorted(os.listdir(data_dir))  # Get all class folder names
data = []
labels = []

# Loop through each class folder
for class_index, class_name in enumerate(class_names):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        print(f"Loading images for class '{class_name}'...")
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                img = Image.open(img_path).convert(
                    "RGB"
                )  # Load image and convert to RGB
                img = img.resize(img_size)  # Resize image to target dimensions
                data.append(np.array(img))
                labels.append(class_name)
            except Exception as e:
                print(f"Could not process image {img_path}: {e}")
        print(
            f"Finished loading {len(os.listdir(class_path))} images for class '{class_name}'."
        )

# Convert data and labels to numpy arrays
print("Converting data and labels to numpy arrays...")
data = np.array(data)
labels = np.array(labels)

# Normalize data
print("Normalizing image data...")
data = data.astype("float32") / 255.0

# Encode labels as integers, then one-hot encode
print("Encoding labels...")
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
y = to_categorical(labels)
print(f"Data shape: {data.shape}, Labels shape: {y.shape}")

# Single train-test split
print("Splitting data into training and testing sets...")
=======
# Libraries
import numpy as np
import os
import time
import gc
import rasterio
import random
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
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

print("Hardware: ", tf.config.list_physical_devices("GPU"))
print("Starting script...")

# Start measuring total execution time
total_start_time = time.time()

# Enable Apple Silicon Acceleration
tf.keras.mixed_precision.set_global_policy("mixed_float16")
print("Enabled Mixed Precision Training")

# Parameters
np.random.seed(1)
optimizer_algorithm = Adam(learning_rate=0.0003)
number_epoch = 100
batch_length = 64
show_inter_results = 1
num_rows, num_cols, num_bands = 64, 64, 13  # 13 bands for MS data

# Regularization Parameters
use_l1, use_l2 = True, True
l1_reg, l2_reg = 0.0002, 0.001

# Define Data Path
data_dir = "C:/Users/Chris/Desktop/EuroSAT/EuroSAT_MS"
max_images_per_class = 200  # Limit dataset size

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
                    )  # Convert to (height, width, bands)
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

# Normalize image data
data = data / np.max(data)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
y = to_categorical(labels)

print(f"Dataset size after sampling: {data.shape[0]} images")
print(f"Data shape: {data.shape}, Labels shape: {y.shape}")

# Train-Test Split
>>>>>>> Stashed changes
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, random_state=1, stratify=labels
)

<<<<<<< Updated upstream
# Build the model
print("Building the model...")
model = Sequential(
    [
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(img_size[0], img_size[1], 3),
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

# Train the model and save history
print("Training the model...")
history = model.fit(
    X_train,
    y_train,
    epochs=number_epoch,
    batch_size=batch_length,
    verbose=show_inter_results,
    validation_data=(X_test, y_test),
)

# Evaluate the model
print("Evaluating the model...")
scores = model.evaluate(X_test, y_test, verbose=1)
print(f"Test accuracy: {scores[1] * 100:.2f}%")

# Plot the loss and accuracy curves
print("Plotting the loss and accuracy curves...")
plt.figure(figsize=(12, 5))
=======
# Learning Rate Scheduler
lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6
)
>>>>>>> Stashed changes

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

<<<<<<< Updated upstream
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
=======
model = Sequential(
    [
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(num_rows, num_cols, num_bands),
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
)

training_time = time.time() - start_train_time
print(f"Training completed in {training_time:.2f} seconds.")

# Save Model
print("Saving the trained model...")
model.save("eurosat_ms_model_2.keras")
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
>>>>>>> Stashed changes
