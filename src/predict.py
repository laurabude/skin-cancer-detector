import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 128

model_path = 'results/skin_cancer_detector.h5.keras'
model = tf.keras.models.load_model(model_path)

# Setul de imagini de test
test_images = ['data/test/benign.jpg', 'data/test/melanoma.jpg']
class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# Realizeazm predicțiile pentru fiecare imagine
for img_path in test_images:
    # Preprocesare
    img = Image.open(img_path)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0  # Normalizare
    img_array = np.expand_dims(img, axis=0)

    predictions = model.predict(img_array)

    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_probability = np.max(predictions)

    print(f"Imaginea: {img_path}")
    print(f"Clasa prezisă: {class_names[predicted_class]}")
    print(f"Probabilitate pentru clasa prezisă: {predicted_probability * 100:.2f}%")

    print("Probabilități pentru fiecare clasă:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {predictions[0][i] * 100:.2f}%")
    print("\n")
