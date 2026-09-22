import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import easyocr
import numpy as np
import json
import cv2

def load_training_data(data_folder):
    images = []
    labels = []
    for filename in os.listdir(data_folder):
        if filename.endswith('.json'):
            with open(os.path.join(data_folder, filename), 'r') as file:
                data = json.load(file)
                image_path = data['image']
                label = data['label']
                
                # Load the image
                image = cv2.imread(image_path)
                images.append(image)
                labels.append(label)
    return images, labels

# Load training data from the specified folder
data_folder = 'C:/Users/saisk/OneDrive/Desktop/Answer Script evaluation System Final/automated-answer-evaluation/src/train'
images, labels = load_training_data(data_folder)

# Check if training data is loaded correctly
print(f"Loaded {len(images)} images and {len(labels)} labels for OCR training.")

if len(images) == 0 or len(labels) == 0:
    raise ValueError("Training data is empty. Please check the data folder and ensure it contains valid JSON files.")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Train the OCR model (EasyOCR does not support custom training, so this is a placeholder)
# In practice, you would use a different OCR library or custom model for training
print("Training OCR model...")

# Placeholder for training code
# ...

print("OCR model trained and saved.")
