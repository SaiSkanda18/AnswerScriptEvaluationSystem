import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.retrieve import retrieve, init_rag
from evaluation.policy import EvaluationPolicy
from feature_extraction.extract_features import extract_features


def evaluate_answer(student_text, question_text=""):
    # Initialize RAG
    try:
        init_rag()
    except FileNotFoundError:
        # If index not built yet, retrieve empty (graceful fallback)
        reference_context = []
    else:
        query = question_text + " " + student_text if question_text else student_text
        reference_context = retrieve(query, k=3)

    # Extract basic NLP features (existing module)
    features = extract_features(student_text)

    # Predict score with RL-capable policy
    policy = EvaluationPolicy()
    score = policy.predict(features)

    explanation = f"Score: {score:.1f}/100 (reference chunks retrieved: {len(reference_context)})"
    if reference_context:
        explanation += " | Reference context used for evaluation."

    return {
        "score": score,
        "explanation": explanation,
        "reference_context": reference_context,
        "features": features
    }
