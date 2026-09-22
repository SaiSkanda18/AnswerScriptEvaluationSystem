from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
import os

class ScorePredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
        self.model = joblib.load(model_path)

    def predict(self, features):
        features = np.array(list(features.values())).reshape(1, -1)
        score = self.model.predict(features)
        return score[0]

    def train(self, X_train, y_train):
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, 'model.pkl')

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        return mse