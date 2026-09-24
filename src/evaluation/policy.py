import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression


class EvaluationPolicy:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'policy_v1.pkl')
        self.model_path = model_path
        self.version = "v1"
        # Safe: model_path points to a locally saved sklearn model produced by this project's train().
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = LinearRegression()

    def predict(self, features_dict):
        # Expected feature keys from feature_extraction module
        feature_keys = ['word_count', 'avg_word_length', 'unique_word_count']
        feature_array = np.array([[features_dict.get(k, 0) for k in feature_keys]])
        score = self.model.predict(feature_array)
        return float(score[0])

    def train(self, X_train, y_train, weights=None):
        self.model = LinearRegression()
        X_train = np.array(X_train)
        if weights is not None:
            weights = np.array(weights)
        self.model.fit(X_train, y_train, sample_weight=weights)
        os.makedirs(os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else ".", exist_ok=True)
        joblib.dump(self.model, self.model_path)
        self.version = self._next_version()
        print(f"Policy updated: {self.model_path} (version={self.version})")

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(np.array(X_test))
        mse = np.mean((predictions - np.array(y_test)) ** 2)
        return mse

    def _next_version(self):
        base_path = self.model_path
        version_str = base_path.split("policy_v")[-1].replace(".pkl", "")
        try:
            ver_num = int(version_str)
        except ValueError:
            ver_num = 1
        return f"v{ver_num + 1}"
