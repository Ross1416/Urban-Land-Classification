# Libraries
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.mixed_precision import set_global_policy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

print("Hardware: ", tf.config.list_physical_devices('GPU'))

print("Starting script...")

# Start measuring total execution time
total_start_time = time.time()

# Enable Apple Silicon Acceleration
set_global_policy('mixed_float16')
print("Enabled Mixed Precision Training")

# Parameters
np.random.seed(1)
optimizer_algorithm = Adam(learning_rate=0.0001)
number_epoch = 100
batch_length = 20

# Define Data Path
data_dir = '/Users/andrewferguson/EuroSAT/EuroSAT_RGB'  # Update this path if needed
img_size = (256, 256)

# Measure data loading time
start_data_time = time.time()

# Load Image Paths & Labels
print("Loading image paths and labels...")
class_names = sorted(os.listdir(data_dir))  # Get class names
image_paths = []
labels = []

for class_index, class_name in enumerate(class_names):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        for img_file in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, img_file))
            labels.append(class_index)

print(f"Loaded {len(image_paths)} images across {len(class_names)} classes.")

# Convert labels to categorical format
labels = to_categorical(labels, num_classes=len(class_names))

# Train-Test Split
print("Splitting data into training and testing sets...")
X_train_paths, X_test_paths, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=1, stratify=labels)
print(f"Training set: {len(X_train_paths)} images, Testing set: {len(X_test_paths)} images")

# Data Loading with tf.data Pipeline
def parse_image(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)  # Change to decode_png if needed
    img = tf.image.resize(img, [256, 256])  # Resize images
    img = img / 255.0  # Normalize
    return img, label

def build_dataset(image_paths, labels, batch_size=20):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(len(image_paths))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

print("Building TensorFlow datasets...")
train_dataset = build_dataset(X_train_paths, y_train, batch_length)
test_dataset = build_dataset(X_test_paths, y_test, batch_length)

data_loading_time = time.time() - start_data_time
print(f"Data loading completed in {data_loading_time:.2f} seconds.")

# Build Model
print("Building the CNN model...")
model = Sequential([
    SeparableConv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    SeparableConv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    SeparableConv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

print("Model built")

# Compile Model
print("Compiling model...")
model.compile(loss='categorical_crossentropy', optimizer=optimizer_algorithm, metrics=['accuracy'])
print("Model compiled")

# Early Stopping to Reduce Wasted Epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Measure training time
print("🚀 Starting training...")
start_train_time = time.time()

history = model.fit(train_dataset, epochs=number_epoch, validation_data=test_dataset, verbose=1, callbacks=[early_stopping])

training_time = time.time() - start_train_time
print(f"Training completed in {training_time:.2f} seconds.")

# Evaluate Model
print("Evaluating model on test dataset...")
start_eval_time = time.time()
scores = model.evaluate(test_dataset, verbose=1)
eval_time = time.time() - start_eval_time

print(f"Test accuracy: {scores[1] * 100:.2f}%")
print(f"Evaluation completed in {eval_time:.2f} seconds.")

# Generate Predictions for Confusion Matrix
print("Generating predictions for confusion matrix...")
y_pred_probs = model.predict(test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Print Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Plot Training History
print("Plotting training history...")
plt.figure(figsize=(12, 5))

# Loss Curve
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Total Execution Time
total_execution_time = time.time() - total_start_time
print(f"Total execution time: {total_execution_time:.2f} seconds.")
print("Script finished successfully")
