import sys
import os
import tempfile
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, render_template
from text_extraction.extract_text import extract_text_from_image, extract_text_from_pdf
from feature_extraction.extract_features import extract_features
from score_prediction.predict_score import ScorePredictor
from utils.helpers import calculate_similarity, calculate_metrics

app = Flask(__name__)
app.config['DEBUG'] = True  # Enable debug mode

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    if 'file' not in request.files or 'answer_key' not in request.files:
        return "No file part", 400
    file = request.files['file']
    answer_key = request.files['answer_key']
    if file.filename == '' or answer_key.filename == '':
        return "No selected file", 400

    try:
        # Save the uploaded files to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            file.save(temp_file.name)
            student_text = extract_text_from_pdf(temp_file.name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_answer_key:
            answer_key.save(temp_answer_key.name)
            answer_key_text = extract_text_from_pdf(temp_answer_key.name)
        
        # Extract features from the texts
        print("Extracting features from the student text...")
        student_features = extract_features(student_text)
        print("Extracted student features:", student_features)
        
        print("Extracting features from the answer key text...")
        answer_key_features = extract_features(answer_key_text)
        print("Extracted answer key features:", answer_key_features)
        
        # Predict score based on features
        print("Predicting score based on features...")
        predictor = ScorePredictor()
        score = predictor.predict(student_features)
        print("Predicted score:", score)
        
        # Calculate similarity and metrics
        print("Calculating similarity and metrics...")
        similarity = calculate_similarity(student_text, answer_key_text)
        precision, recall, f1_score = calculate_metrics(student_text, answer_key_text)
        print(f"Similarity: {similarity}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
        
        # Log the metrics (not shown in the web application)
        print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
        
        return render_template('index.html', score=score)
    except Exception as e:
        print(f"Error: {e}")
        return "Internal Server Error", 500

if __name__ == '__main__':
    app.run(debug=True)
