import numpy as np
import pandas as pd
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 128

def load_metadata(metadata_path, image_dir):
    metadata = pd.read_csv(metadata_path)
    metadata['path'] = metadata['image'].map(lambda x: os.path.join(image_dir, f"{x}.jpg"))
    return metadata

transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Flip orizontal
    A.RandomBrightnessContrast(p=0.2),  # Ajustare luminozitate/contrast
    A.Rotate(limit=20, p=0.5),  # Rotire random între -20 și +20 grade
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),  # Shift/scale/rotate
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalizare pentru rețele CNN
    ToTensorV2()  # Conversie în tensor
])

# Preprocesarea imaginilor folosind Albumentations
def preprocess_images(paths):
    images = []
    for path in paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertim din BGR (OpenCV) în RGB
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Redimensionăm imaginea
        img = transform(image=img)['image']  # Aplicăm augmentarea
        img = img.numpy()  # Conversie din torch.Tensor în numpy
        img = np.transpose(img, (1, 2, 0))  # Rearanjare (C, H, W) -> (H, W, C)
        images.append(img)
    return np.array(images)

# Preprocesarea etichetelor
def preprocess_labels(metadata):
    labels = metadata[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']].values
    return labels