import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
import json
from text_extraction.extract_text import extract_text_from_image
from feature_extraction.extract_features import extract_features

class ScorePredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
        self.model = LinearRegression()

    def predict(self, features):
        features = np.array(list(features.values())).reshape(1, -1)
        score = self.model.predict(features)
        return score[0]

    def train(self, X_train, y_train):
        # Ensure X_train is a 2D array
        X_train = np.array(X_train).reshape(-1, len(X_train[0]))
        y_train = np.array(y_train)
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, 'model.pkl')

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        return mse

def load_training_data(data_folder):
    X_train = []
    y_train = []
    for filename in os.listdir(data_folder):
        if filename.endswith('.json'):
            with open(os.path.join(data_folder, filename), 'r') as file:
                data = json.load(file)
                print(f"Loaded data from {filename}: {data}")  # Debugging statement
                image_path = data['image']
                score = data['score']
                
                # Extract text from the image
                typed_text, handwritten_text = extract_text_from_image(image_path)
                
                # Extract features from text
                typed_features = extract_features(typed_text)
                handwritten_features = extract_features(handwritten_text)
                
                # Combine features
                combined_features = {**typed_features, **handwritten_features}
                
                X_train.append(list(combined_features.values()))
                y_train.append(score)
    return np.array(X_train), np.array(y_train)

# Load training data from the specified folder
data_folder = 'C:/Users/saisk/OneDrive/Desktop/Answer Script evaluation System Final/automated-answer-evaluation/src/train'
X_train, y_train = load_training_data(data_folder)

# Check if training data is loaded correctly
print(f"X_train: {X_train}")  # Debugging statement
print(f"y_train: {y_train}")  # Debugging statement
if X_train.size == 0 or y_train.size == 0:
    raise ValueError("Training data is empty. Please check the data folder and ensure it contains valid JSON files.")

# Train and save the model
predictor = ScorePredictor()
predictor.train(X_train, y_train)
print("Model trained and saved as model.pkl")
