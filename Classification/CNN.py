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
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
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
optimizer_loss_fun = 'categorical_crossentropy'
optimizer_algorithm = Adam(learning_rate=0.001)
number_epoch = 100
batch_length = 20
show_inter_results = 1

# Define data path
data_dir = '/Users/andrewferguson/EuroSAT/EuroSAT_RGB'  # Update to the correct path
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
                img = Image.open(img_path).convert('RGB')  # Load image and convert to RGB
                img = img.resize(img_size)  # Resize image to target dimensions
                data.append(np.array(img))
                labels.append(class_name)
            except Exception as e:
                print(f"Could not process image {img_path}: {e}")
        print(f"Finished loading {len(os.listdir(class_path))} images for class '{class_name}'.")

# Convert data and labels to numpy arrays
print("Converting data and labels to numpy arrays...")
data = np.array(data)
labels = np.array(labels)

# Normalize data
print("Normalizing image data...")
data = data.astype('float32') / 255.0

# Encode labels as integers, then one-hot encode
print("Encoding labels...")
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
y = to_categorical(labels)
print(f"Data shape: {data.shape}, Labels shape: {y.shape}")

# Single train-test split
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=1, stratify=labels)

# Build the model
print("Building the model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(loss=optimizer_loss_fun, optimizer=optimizer_algorithm, metrics=['accuracy'])

# Train the model and save history
print("Training the model...")
history = model.fit(X_train, y_train, epochs=number_epoch, batch_size=batch_length, verbose=show_inter_results,
                    validation_data=(X_test, y_test))

# Evaluate the model
print("Evaluating the model...")
scores = model.evaluate(X_test, y_test, verbose=1)
print(f"Test accuracy: {scores[1] * 100:.2f}%")

# Plot the loss and accuracy curves
print("Plotting the loss and accuracy curves...")
plt.figure(figsize=(12, 5))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
