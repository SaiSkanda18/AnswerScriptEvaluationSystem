import sqlite3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.policy import EvaluationPolicy
from feature_extraction.extract_features import extract_features
from evaluation.reward import calculate_reward

DB_PATH = os.getenv("DATABASE_PATH", "database/feedback.db")


def load_training_examples():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT evaluation_id, student_text, original_score, corrected_score, teacher_action FROM evaluations WHERE corrected_score IS NOT NULL OR original_score IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()
    return rows


def retrain():
    rows = load_training_examples()
    if not rows:
        print("No training examples found in database. Nothing to retrain.")
        return

    X_train = []
    y_train = []
    weights = []

    for row in rows:
        _, student_text, original_score, corrected_score, action = row
        if student_text is None:
            continue
        features = extract_features(student_text)
        feature_array = [features.get('word_count', 0), features.get('avg_word_length', 0), features.get('unique_word_count', 0)]

        target_score = corrected_score if corrected_score is not None else original_score
        X_train.append(feature_array)
        y_train.append(float(target_score))

        # Calculate reward for weighting
        reward = calculate_reward(original_score, corrected_score or original_score, "")
        weights.append(reward + 0.01)  # minimum weight to avoid zero

    if not X_train:
        print("No valid training data after filtering.")
        return

    policy = EvaluationPolicy()
    policy.train(X_train, y_train, weights=weights)
    print(f"RL policy retrained using {len(X_train)} accumulated evaluations with reward-weighted updates.")


if __name__ == "__main__":
    retrain()
