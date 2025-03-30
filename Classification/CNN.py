# ======================================= Libraries ==========================================
import numpy as np
import os
import random
import time
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ======================================= init ==========================================

print("Hardware: ", tf.config.list_physical_devices('GPU'))

print("Starting script...")

# Start measuring total execution time
total_start_time = time.time()

# Enable Apple Silicon Acceleration
set_global_policy('mixed_float16')
print("Enabled Mixed Precision Training")


# Set random seeds
def set_global_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    print("Seed set ")


set_global_seed(42)

# Hyperparameters
optimizer_algorithm = Adam(learning_rate=0.0001)
loss_function = CategoricalCrossentropy(label_smoothing=0.1)
number_epoch = 100
batch_length = 16
show_inter_results = 1
num_rows = 64
num_cols = 64

# Regularization Parameters
use_l1 = True  # Set to True to enable L1 regularization
use_l2 = True  # Set to True to enable L2 regularization
l1_reg = 0.0001  # L1 regularization strength
l2_reg = 0.00075  # L2 regularization strength


# ======================================= Data Loading ==========================================

# Define Data Path
data_dir = '/Users/andrewferguson/EuroSAT/EuroSAT_RGB'  # Update this path if needed

# Measure data loading time
start_data_time = time.time()

# Load Image Paths & Labels
print("Loading image paths and labels...")
class_names = sorted([d for d in os.listdir(data_dir) if not d.startswith('.')])
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
X_train_paths, X_temp_paths, y_train, y_temp = train_test_split(image_paths, labels, test_size=0.2, random_state=42, stratify=labels)
labels_temp = np.argmax(y_temp, axis=1)
X_test_paths, X_val_paths, y_test, y_val = train_test_split(X_temp_paths, y_temp, test_size=0.5, random_state=42, stratify=labels_temp)
print(f"Training set: {len(X_train_paths)} images, Validation set: {len(X_val_paths)} images, Test set: {len(X_test_paths)} images")

# Function to read in and normalized images
def parse_image(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [num_rows, num_cols])
    img = img / 255.0  # Normalize
    return img, label

# Function to augment data
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.05)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

# Build dataset
def build_dataset(image_paths, labels, batch_size=20, augment=True):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

print("Building TensorFlow datasets...")
train_dataset = build_dataset(X_train_paths, y_train, batch_length)
val_dataset = build_dataset(X_val_paths, y_val, batch_length, augment=False)
test_dataset = build_dataset(X_test_paths, y_test, batch_length, augment=False)
print("BuiltTensorFlow datasets")

# ======================================= Example Input Images ==========================================

# Fetch a batch of augmented images from the dataset
sample_batch = next(iter(train_dataset))

# Extract images and labels
sample_images, sample_labels = sample_batch

# Convert labels from one-hot encoding to class indices
sample_labels = np.argmax(sample_labels.numpy(), axis=1)

# Plot the images with their labels
num_images = min(9, len(sample_images))  # Show up to 9 images
plt.figure(figsize=(10, 10))

for i in range(num_images):
    plt.subplot(3, 3, i + 1)
    plt.imshow(sample_images[i].numpy())  # Convert tensor to NumPy for visualization
    plt.title(f"Label: {class_names[sample_labels[i]]}")
    plt.axis("off")

plt.tight_layout()
plt.savefig('Augmented Data.png')

data_loading_time = time.time() - start_data_time
print(f"Data loading completed in {data_loading_time:.2f} seconds.")

# ======================================= Build model ==========================================

# Build Model
print("Building the CNN model...")

regularizer = l1_l2(l1=l1_reg if use_l1 else 0.0, l2=l2_reg if use_l2 else 0.0)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

model = Sequential([

    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(num_rows, num_cols, 3), kernel_regularizer=regularizer),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizer),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizer),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizer),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizer),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizer),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    GlobalAveragePooling2D(),
    Dense(1024, activation='relu', kernel_regularizer=regularizer),
    Dropout(0.4),
    Dense(512, activation='relu', kernel_regularizer=regularizer),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])

print("Model built")

# Compile Model
print("Compiling model...")
model.compile(loss=loss_function, optimizer=optimizer_algorithm, metrics=['accuracy'])
print("Model compiled")

# ======================================= Training ==========================================

# Measure training time
print("🚀 Starting training...")
start_train_time = time.time()

history = model.fit(train_dataset,
                    epochs=number_epoch,
                    verbose=show_inter_results,
                    validation_data=val_dataset,
                    callbacks=[lr_scheduler, early_stopping])

training_time = time.time() - start_train_time
print(f"Training completed in {training_time:.2f} seconds.")

print("Saving the trained model...")
model.save("eurosat_model_augmented_report.keras")
print("Model saved successfully.")

# ======================================= Testing ==========================================
# Evaluate Model
print("Evaluating model on test dataset...")
start_eval_time = time.time()
scores = model.evaluate(test_dataset, verbose=1)
eval_time = time.time() - start_eval_time

print(f"Test accuracy: {scores[1] * 100: .2f}%")
print(f"Evaluation completed in {eval_time: .2f} seconds.")

# Generate Predictions for Confusion Matrix
y_true = []
for image_batch, label_batch in test_dataset:
    y_true.extend(np.argmax(label_batch.numpy(), axis=1))  # Convert one-hot to class indices
y_true = np.array(y_true)

y_pred_probs = model.predict(test_dataset)  # Get probability outputs
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class predictions

# Compute Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Save Classification Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [accuracy, precision, recall, f1]
})

metrics_df.to_csv('metrics.csv', index=False)
print("Saved accuracy, precision, recall, and F1 score to metrics.csv")

# Generate and save classification report
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv('classification_report.csv', index=True)
print("Saved classification report to classification_report.csv")

# Plot Confusion Matrix
plt.figure(figsize=(16, 14))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig('Confusion_Matrix.png')

# Save plots
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('accuracy_plot.png')

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('loss_plot.png')

# Print Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))