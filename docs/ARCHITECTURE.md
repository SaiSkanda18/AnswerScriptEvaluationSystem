# Architecture — Handwritten Answer-Script Evaluation System

## Overview
This repository implements an automated handwritten answer-script evaluation pipeline with OCR, RAG-based reference retrieval, an RL-capable evaluation policy, and a teacher-review workflow with optional feedback and manual retraining.

## Component Mapping

| Component | Implementation | Justification |
|---|---|---|
| OCR | `src/text_extraction/` (Tesseract) | Preserved per Q2.md |
| Reference Text / RAG | `Answer key.pdf` parsed via `pdf2image` + Tesseract; `FAISS` persisted index (`faiss_index/`) + `sentence-transformers` | Q2.md requirements |
| Database | SQLite (`feedback.db`) via `database/schema.sql` and `database/init_db.py` | Q3.md: audit, optional feedback, manual retraining |
| Vector Store | `FAISS` (`faiss_index/index.faiss`, `faiss_index/metadata.json`) | Persisted, reproducible, lightweight |
| Model / Policy Layer | `src/evaluation/policy.py` (`EvaluationPolicy`) — RL-inspired reward-weighted updates | Q1.md: genuine RL component, not false claims |
| RL Training Pipeline | `src/training/retrain_rl_policy.py` — manual, on-demand, uses `feedback.db` corrections + rewards | Q3.md: manual trigger |
| API / Backend | Flask (`src/web_evaluation/app.py`) — `/evaluate`, `/review/<id>`, `/feedback` | Extended from existing app |
| Frontend | `templates/index.html`, `templates/review.html` (Bootstrap) | Minimal extension |
| File Storage | `knowledge/` (future reference docs), `models/` (policy versions), `database/` (SQLite) | Reproducible, structured |
| Config / Secrets | `.env` + `.env.example` (`DATABASE_PATH`, `RAG_INDEX_PATH`, etc.) | Standard practice |

## Data Flow

```
Handwritten Answer Script (PDF / image)
        ↓
OCR (Tesseract — preserved)
        ↓
Extracted Student Text
        ↓
Question + Student Answer (user input or image-derived)
        ↓
RAG Retrieval (FAISS index from Answer key.pdf + future knowledge/)
        ↓
Reference Context
        ↓
Evaluation Policy (RL-capable model)
        ↓
Score + Explanation
        ↓
SQLite Storage (`feedback.db`)
        ↓
Teacher Review (`/review/<id>`)
        ↓
Accept / Modify Score
        ↓
Optional Written Feedback
        ↓
Reward Calculation (`evaluation/reward.py`)
        ↓
Manual Retraining (`python -m src.training.retrain_rl_policy`)
        ↓
Updated Policy (`models/policy_v{version}.pkl`)
```

## RL Design Note
The system implements a **genuine RL-inspired mechanism**, not simplified supervised fine-tuning:
- A **reward signal** is computed from `(original_score, corrected_score, feedback_text)`.
- The evaluation policy is updated using **reward-weighted supervised updates** (not full PPO or RLHF, which is impractical for this dataset size).
- This is explicitly documented in code (`src/evaluation/reward.py`, `src/evaluation/policy.py`) and here: the mechanism uses correction magnitude and optional feedback sentiment as a scalar reward to weight training examples.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Initialize database:
   ```bash
   python -m database.init_db
   ```
3. Build RAG index (requires `sentence-transformers` and `faiss-cpu`):
   ```bash
   python -m src.rag.build_index
   ```
4. Run the web application:
   ```bash
   python src/web_evaluation/app.py
   ```
5. Manual RL retraining (after collecting teacher corrections):
   ```bash
   python -m src.training.retrain_rl_policy
   ```

## Testing
Run basic tests:
```bash
python -m pytest tests/
```

End-to-end manual verification:
- Upload `Answer1.pdf` (student input) and `Answer key.pdf` (reference) through the web interface.
- Verify score output and SQLite insertion (`feedback.db`).
- Use `/review/<id>` to modify the score, submit optional feedback, and verify `reward` field updates.
- Run `retrain_rl_policy` and confirm new `models/policy_v2.pkl` is created.

## Limitations (Explicit)
- OCR accuracy for handwritten text is limited by `Tesseract`; significant handwriting noise may degrade extraction.
- The RL mechanism is **reward-weighted supervised update**, not full RLHF/PPO. It is appropriate for small-scale academic evaluation but would require larger datasets and more complex policy architectures for production-scale RL.
- `sentence-transformers` and `faiss-cpu` are required for RAG; if unavailable, RAG retrieval falls back to empty context gracefully.
- The evaluation policy currently uses basic `LinearRegression` features (`word_count`, `avg_word_length`, `unique_word_count`). More sophisticated NLP features can be added to `feature_extraction/` without changing the policy interface.
- Feedback submission is optional; the system must work if the teacher accepts/corrects scores without providing text feedback.

## Known Files / Artifacts
- `.env.example` — environment configuration template
- `database/feedback.db` — SQLite database (do not commit; excluded by `.gitignore`)
- `faiss_index/index.faiss`, `faiss_index/metadata.json` — persisted vector index (rebuildable via `build_index`)
- `models/policy_v*.pkl` — trained policy versions (rebuildable via `retrain_rl_policy`)
- `docs/ARCHITECTURE.md` — this document
- `docs/superpowers/specs/2026-09-23-handwritten-evaluation-rl-architecture.md` — design specification

## Future Improvements (Outside Current Scope)
- More advanced NLP feature extraction (`spacy`, `transformers` embeddings for student answers).
- Full RLHF with a separate reward model trained from preference pairs (requires significantly more data and compute).
- Automatic segmentation of handwritten scripts into questions/answers.
- Integration with an external vector database (`chroma`, `weaviate`) for larger knowledge bases.
- Multi-language support.
