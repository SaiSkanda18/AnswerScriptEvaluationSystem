import sys
import os
import tempfile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, render_template, redirect, url_for
from text_extraction.extract_text import extract_text_from_pdf
from rag.retrieve import retrieve, init_rag
from evaluation.inference import evaluate_answer
from evaluation.reward import calculate_reward
from database.init_db import init_db
import sqlite3

app = Flask(__name__)
app.config['DEBUG'] = True

DB_PATH = os.getenv("DATABASE_PATH", "database/feedback.db")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/evaluate', methods=['POST'])
def evaluate():
    init_db()  # ensure DB exists
    if 'file' not in request.files or 'answer_key' not in request.files:
        return "No file part", 400
    file = request.files['file']
    answer_key = request.files['answer_key']
    if file.filename == '' or answer_key.filename == '':
        return "No selected file", 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            file.save(temp_file.name)
            student_text = extract_text_from_pdf(temp_file.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_answer_key:
            answer_key.save(temp_answer_key.name)
            # Answer key text is reference; not evaluated directly as student text

        # Try to retrieve reference context via RAG
        try:
            init_rag()
            query_text = student_text[:500]
            reference_context = retrieve(query_text, k=3)
        except Exception:
            reference_context = []

        # Evaluate using the RL-capable policy
        result = evaluate_answer(student_text, question_text="")
        original_score = result["score"]

        # Store initial evaluation in SQLite
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO evaluations (student_text, question, original_score, rag_reference_ids, policy_version, teacher_action, reward) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (student_text, "", original_score, str(reference_context)[:200], result.get("policy_version", "v1"), "pending", 0.0)
        )
        evaluation_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return render_template('index.html', score=original_score, evaluation_id=evaluation_id, explanation=result["explanation"])
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Internal Server Error: {e}", 500


@app.route('/review/<int:evaluation_id>', methods=['GET', 'POST'])
def review(evaluation_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evaluations WHERE evaluation_id = ?", (evaluation_id,))
    evaluation = cursor.fetchone()

    if request.method == 'POST':
        corrected_score = request.form.get('corrected_score', type=float)
        feedback_text = request.form.get('feedback_text', '').strip()
        action = 'corrected' if corrected_score is not None else 'accepted'

        # Calculate reward from correction
        original_score = evaluation['original_score'] if evaluation['original_score'] is not None else 0
        corrected = corrected_score if corrected_score is not None else original_score
        reward = calculate_reward(original_score, corrected, feedback_text)

        cursor.execute(
            "UPDATE evaluations SET corrected_score = ?, teacher_action = ?, reward = ? WHERE evaluation_id = ?",
            (corrected, action, reward, evaluation_id)
        )
        if feedback_text:
            cursor.execute(
                "INSERT INTO feedback (evaluation_id, feedback_text) VALUES (?, ?)",
                (evaluation_id, feedback_text)
            )
        conn.commit()
        conn.close()
        return redirect(url_for('home'))

    conn.close()
    return render_template('review.html', evaluation=evaluation)


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    init_db()
    evaluation_id = request.form.get('evaluation_id', type=int)
    feedback_text = request.form.get('feedback_text', '').strip()
    if not evaluation_id:
        return "Missing evaluation ID", 400

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT original_score, corrected_score FROM evaluations WHERE evaluation_id = ?", (evaluation_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return "Evaluation not found", 404
    original, corrected = row
    corrected = corrected if corrected is not None else original
    reward = calculate_reward(original, corrected, feedback_text)
    cursor.execute(
        "UPDATE evaluations SET reward = ? WHERE evaluation_id = ?",
        (reward, evaluation_id)
    )
    if feedback_text:
        cursor.execute(
            "INSERT INTO feedback (evaluation_id, feedback_text) VALUES (?, ?)",
            (evaluation_id, feedback_text)
        )
    conn.commit()
    conn.close()
    return "Feedback submitted. Thank you."


if __name__ == '__main__':
    app.run(debug=True)
