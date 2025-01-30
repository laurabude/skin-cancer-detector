import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from preprocess_data import load_metadata, preprocess_images, preprocess_labels

# Paths
model_path = 'results/skin_cancer_detector.h5.keras'
metadata_path = 'data/HAM10000_metadata.csv'
image_dir = 'data/images/'

# Load model and data
model = tf.keras.models.load_model(model_path)
metadata = load_metadata(metadata_path, image_dir)

# Preprocess images
X_test = preprocess_images(metadata['path'])

# Generate labels using the columns MEL, NV, BCC, AKIEC, BKL, DF, VASC
y_test = preprocess_labels(metadata)

# Evaluate
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Display results
print(classification_report(np.argmax(y_test, axis=1), predicted_classes))
