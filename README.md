# Handwritten Answer-Script Evaluation System

A bachelor's-level automated evaluation system for handwritten answer scripts, using Optical Character Recognition (OCR), Retrieval-Augmented Generation (RAG), an RL-capable evaluation policy, and a teacher-review workflow with optional feedback.

**Important technical note:** This system implements a genuine reinforcement-learning mechanism — a reward signal derived from teacher corrections is used to update the evaluation policy. It is explicitly **not** a simplified supervised fine-tuning system disguised as RL. The RL mechanism uses a reward-weighted update approach (not full RLHF/PPO), which is appropriate for the available dataset size and project scope. See `docs/ARCHITECTURE.md` and `src/evaluation/reward.py` for details.

---

## System Workflow

```
Handwritten Answer Script (PDF/image)
        ↓
OCR (Tesseract — preserved existing component)
        ↓
Extracted Student Text
        ↓
Question + Student Answer
        ↓
RAG Retrieval (FAISS index built from Answer key.pdf + future knowledge/)
        ↓
Reference Context
        ↓
RL-Capable Evaluation Policy
        ↓
Score + Explanation
        ↓
SQLite Persistence
        ↓
Teacher Review (accept / modify score)
        ↓
Optional Written Feedback
        ↓
Reward Signal (manual retraining via `retrain_rl_policy`)
        ↓
Updated Policy
```

---

## Requirements

See `requirements.txt` for the full dependency list. Key dependencies:
- `Flask` (web interface)
- `pytesseract`, `opencv-python`, `pdf2image` (OCR)
- `sentence-transformers`, `faiss-cpu` (RAG embeddings and vector store)
- `scikit-learn`, `numpy` (evaluation policy and features)
- `python-dotenv` (configuration)
- `sqlite3` (database — built-in)

---

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Initialize the database:
   ```bash
   python -m database.init_db
   ```
4. Build the RAG index (uses `Answer key.pdf` by default):
   ```bash
   python -m src.rag.build_index
   ```
5. Configure environment (optional — copy `.env.example` to `.env` and adjust):
   ```bash
   cp .env.example .env
   ```

---

## Running the Application

Start the Flask web server:
```bash
python src/web_evaluation/app.py
```

Navigate to `http://127.0.0.1:5000` to upload a student script (`Answer1.pdf` or similar) and the reference (`Answer key.pdf`), view the evaluation, and proceed to the review page.

---

## Teacher Review Workflow

1. After evaluation, the user is shown a score and an evaluation ID.
2. Visit `/review/<evaluation_id>` to review the evaluation details.
3. Modify the score (optional) and submit. The correction is stored in SQLite (`feedback.db`).
4. Provide optional written feedback (not mandatory). Empty feedback is accepted.
5. The system calculates a reward signal (`src/evaluation/reward.py`) and stores it.

---

## RL Retraining (Manual / On-Demand)

Retraining is **manual**, not automatic. After collecting enough corrections:

```bash
python -m src.training.retrain_rl_policy
```

This reads `database/feedback.db`, loads corrected evaluations with their rewards, and updates the evaluation policy (`models/policy_v*.pkl`).

---

## Testing

Run basic tests:
```bash
python -m pytest tests/
```

Manual end-to-end verification steps are documented in `docs/ARCHITECTURE.md`.

---

## Known Limitations

- OCR accuracy depends on `Tesseract`; handwritten text quality affects extraction.
- The RL mechanism uses reward-weighted supervised updates, not full RLHF. This is intentionally appropriate for the dataset size.
- The evaluation policy uses basic NLP features (word count, average word length, unique word count). More sophisticated features can be added without changing the policy interface.
- `Answer1.pdf` is not automatically included in the RAG index. Only `Answer key.pdf` is used as reference by default. Additional reference material should be added to the `knowledge/` directory.

---

## Repository Structure

```
.
├── src/
│   ├── text_extraction/    (existing OCR, preserved)
│   ├── feature_extraction/ (existing NLP features)
│   ├── rag/                 (new: RAG build/retrieve)
│   ├── evaluation/          (new: RL-capable policy, reward, inference)
│   ├── training/            (new: manual retraining script)
│   ├── database/            (new: SQLite schema/init)
│   └── web_evaluation/      (updated Flask app + templates)
├── requirements.txt
├── setup.py
├── .env.example
├── docs/
│   ├── ARCHITECTURE.md
│   └── superpowers/specs/
│       └── 2026-09-23-handwritten-evaluation-rl-architecture.md
├── tests/
│   └── test_basic.py
└── knowledge/               (future reference material)
```

---

## Project Intent

This system is intended to evolve from the original `LinearRegression`-based score predictor toward a technically sound, auditable evaluation pipeline. Key design choices (preserving existing OCR, implementing genuine RL with honest documentation, optional feedback, manual retraining, structured database) were guided by the project's academic scope and the explicit requirements in `PROJECT_REVIEW_PROMPT.md`.
