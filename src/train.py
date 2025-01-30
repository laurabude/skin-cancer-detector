import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model import create_model

IMG_SIZE = 128  # Image size

# Load metadata
def load_metadata(metadata_path, image_dir):
    metadata = pd.read_csv(metadata_path)
    metadata['path'] = metadata['image'].map(lambda x: os.path.join(image_dir, f"{x}.jpg"))
    return metadata

# Preprocess images using PIL
def preprocess_images(paths):
    images = []
    for path in paths:
        img = Image.open(path)
        img = img.resize((IMG_SIZE, IMG_SIZE))  # Resize image
        img = np.array(img) / 255.0  # Normalize
        images.append(img)
    return np.array(images)

# Preprocess labels (no need for one-hot encoding here for sparse categorical crossentropy)
def preprocess_labels(metadata):
    labels = metadata[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']].values
    return labels

# Manual image augmentation (rotation, zoom, flip)
def augment_images(images, rotation_range=20, zoom_range=0.2, horizontal_flip=True):
    augmented_images = []

    for img in images:
        # Rotation
        if rotation_range > 0:
            angle = np.random.uniform(-rotation_range, rotation_range)
            img = Image.fromarray((img * 255).astype(np.uint8))
            img = img.rotate(angle)

        # Zoom
        if zoom_range > 0:
            zoom_factor = np.random.uniform(1 - zoom_range, 1 + zoom_range)
            img = img.resize((int(IMG_SIZE * zoom_factor), int(IMG_SIZE * zoom_factor)))
            img = img.crop((0, 0, IMG_SIZE, IMG_SIZE))  # Crop to desired size

        # Horizontal flip
        if horizontal_flip and np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        augmented_images.append(np.array(img) / 255.0)  # Normalize

    return np.array(augmented_images)

# Paths
metadata_path = 'data/HAM10000_metadata.csv'
image_dir = 'data/images/'
model_path = 'results/skin_cancer_detector.h5'

# Load and process data
metadata = load_metadata(metadata_path, image_dir)
X = preprocess_images(metadata['path'])  # Preprocess images
y = preprocess_labels(metadata)  # Preprocess labels

# Encode labels for sparse_categorical_crossentropy (no one-hot encoding)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.argmax(y, axis=1))  # Encode labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Apply augmentation on the training set
X_train_augmented = augment_images(X_train)

# Create model (see model.py for create_model function)
input_shape = (IMG_SIZE, IMG_SIZE, 3)
num_classes = len(np.unique(y_encoded))  # Number of classes
model = create_model(input_shape, num_classes)

# Train the model with augmented images
history = model.fit(X_train_augmented, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Save the model
model.save(f"{model_path}.keras")