---
name: handwritten-evaluation-rl-architecture
description: Design spec for handwritten answer-script evaluation with genuine RL, RAG, and teacher feedback.
metadata:
  type: project
---

# Design Spec — Handwritten Answer-Script Evaluation with RL

**Status:** Approved (responses from Q1.md, Q2.md, Q3.md incorporated)
**Path:** Architectural (new subsystems: RAG, RL policy, SQLite persistence, teacher review, knowledge layer)

---

## Confirmed Requirements (from Q1, Q2, Q3, PROJECT_REVIEW_PROMPT.md)

1. **Keep existing Tesseract OCR** (`src/text_extraction/`); only modify for integration/robustness.
2. **Genuine RL** — not simplified supervised fine-tuning. Policy improves from accumulated teacher corrections + optional feedback via a reward signal.
3. **RAG** — `Answer key.pdf` is primary reference; `Answer1.pdf` is student input/sample (not RAG). FAISS persisted index (`faiss_index/`) + `sentence-transformers`. Design for future `knowledge/` directory.
4. **SQLite database** (`feedback.db`) storing evaluation + feedback + reward records.
5. **Optional teacher feedback** — must proceed without providing text feedback.
6. **Manual RL retraining** (`python -m src.training.retrain_rl_policy`), not automatic after every submission.
7. **No false "reinforcement learning" claims** — document the actual mechanism (reward from correction magnitude, policy updated via accumulated examples).

---

## Target Architecture

```
Handwritten Answer Script (PDF / image)
        ↓
Existing Tesseract OCR (src/text_extraction/)
        ↓
Extracted Student Text
        ↓
Question + Student Answer (frontend/user input or extracted from image)
        ↓
RAG Retrieval (FAISS index from Answer key.pdf + future knowledge/)
        ↓
Retrieved Reference Context
        ↓
Evaluation Policy / RL Model (replaces LinearRegression)
        ↓
Score + Explanation / Feedback
        ↓
Teacher Review (modify score)
        ↓
Optional Written Feedback
        ↓
Reward Calculation (stored in SQLite with evaluation record)
        ↓
Manual Retraining Script (retrain_rl_policy)
        ↓
Updated Policy Version (persisted model)
```

---

## Component Design

### 1. OCR Layer (Existing — Preserved)
- File: `src/text_extraction/extract_text.py`
- Preserve `extract_text_from_image()` and `extract_text_from_pdf()`.
- Add robust error handling (missing file, empty output) without redesigning the pipeline.
- Add `preprocess_image()` improvements only for compatibility (e.g., grayscale handling, noise reduction if needed for handwritten text).

### 2. RAG Layer (New / Expanded)
- **Reference extraction:** Extract text chunks from `Answer key.pdf` using `pdfplumber` or `PyMuPDF`. If unavailable, fall back to basic text extraction.
- **Chunking:** Split into paragraph/question-level chunks with overlap.
- **Embeddings:** `sentence-transformers` (e.g., `all-MiniLM-L6-v2` for small-scale; document model choice in `docs/ARCHITECTURE.md`).
- **Vector store:** `faiss` with local persistence (`faiss_index/index.faiss` + `faiss_index/metadata.json`).
- **Retrieval:** Given a question + student answer, embed query, retrieve top-k chunks from index.
- **Knowledge expansion:** Design `knowledge/` directory support; if files exist, add them to index build/rebuild process (`build_index.py`).
- **Answer1.pdf role:** Inspect contents. If it is a student sample/input, do NOT include in RAG index. If it contains reference material, include selectively. Document this decision in `docs/ARCHITECTURE.md`.

### 3. Database Layer (New — SQLite)
- File: `feedback.db` (SQLite, committed? No — add `.gitignore` for `.db`, but keep schema file `database/schema.sql`).
- Schema (from Q3.md):
  - `evaluations`: `evaluation_id`, `student_text`, `question`, `original_score`, `corrected_score`, `rag_reference_ids`, `policy_version`, `teacher_action` (`accepted`/`corrected`), `reward`, `timestamp`
  - `feedback`: `feedback_id`, `evaluation_id`, `feedback_text`, `timestamp`
- Initialize via `database/init_db.py`.

### 4. RL Policy / Evaluation Model (Replaces LinearRegression)
- **Approach:** Given the small dataset expectation (tens/hundreds of examples), implement a practical RL mechanism rather than claiming full-scale RLHF. Recommended approach:
  - A **reward model** that computes a scalar reward from `(original_score, corrected_score, feedback_text, reference_context)`.
  - A **policy** that predicts the evaluation score; updated via accumulated corrected examples using a simple gradient-based approach (e.g., fine-tuning a small regression model or a lightweight neural network with the corrected pairs as supervised updates guided by reward weights).
  - For this scope: implement a **reward-weighted supervised update** — explicitly document as "RL-inspired: uses reward signal from corrections to weight training examples, not full PPO/RLHF" — to avoid false claims.
- Files:
  - `src/evaluation/policy.py` — defines the evaluation policy/model interface.
  - `src/evaluation/reward.py` — calculates reward (`abs(corrected - original)` scaled, optional sentiment bonus from feedback text using basic keyword/sentiment heuristic or `nltk` sentiment).
  - `src/evaluation/inference.py` — runs policy + RAG retrieval + output formatting.
- Policy versioning: save models as `models/policy_v{version}.pkl` or `.pth`; record version in SQLite.

### 5. Teacher Review & Optional Feedback (Web / API)
- Extend `src/web_evaluation/app.py`:
  - `/evaluate` endpoint: accepts upload, runs OCR, RAG, policy, stores evaluation in SQLite (without feedback initially), returns score + explanation + evaluation ID.
  - `/review/<evaluation_id>` endpoint: displays evaluation details, allows score modification, optional feedback text input, saves correction + feedback to SQLite, calculates reward.
  - `/feedback` endpoint: direct submission of optional feedback.
- Ensure feedback is **optional**: the submit/correct form must work with empty feedback text.

### 6. RL Training / Retraining (Manual)
- File: `src/training/retrain_rl_policy.py`
- Process (manual, on-demand):
  1. Read `feedback.db` for evaluations with `teacher_action = 'corrected'` or `accepted` (use all with rewards).
  2. Load reference RAG context associated with each.
  3. Compute/retrieve rewards.
  4. Update policy using accumulated data (weighted training from corrected examples).
  5. Save new policy version (`models/policy_v{new_version}`); log version in SQLite or a `models/versions.json`.
- Do NOT run automatically after every submission (per Q3.md).

### 7. Configuration & Environment
- `.env.example`: database path, RAG index path, model path, reference PDF paths, Flask secret/config.
- `config/settings.py` (or use `python-dotenv`) for centralized config.
- Update `requirements.txt`: add `sentence-transformers`, `faiss-cpu`, `sqlite3` (built-in), `python-dotenv`, `flask`, `numpy`, `scikit-learn`, `pytesseract`, `pdfplumber` (or `PyMuPDF`), `PIL`, `nltk`.
- `setup.py`: update dependencies, add `packages=find_packages()` for new modules.

### 8. Documentation
- `README.md`: rewrite to describe actual workflow, installation (`pip install -r requirements.txt`), how to build RAG index (`python -m src.rag.build_index`), how to run web app, how to trigger RL retraining.
- `docs/ARCHITECTURE.md`: describe the 7-stage flow, component mapping (database, vector store, model, training, API, frontend, file/storage, config), design choices, known limitations.
- `docs/KNOWN_LIMITATIONS.md`: document OCR limitations (handwritten text accuracy), small dataset RL limitations, FAISS persistence, optional feedback requirement.

---

## Component Mapping (Required by Prompt)

| Component | Implementation | Justification |
|---|---|---|
| Database | SQLite (`feedback.db`) | Small dataset, easy audit, no external dependency |
| Vector store | FAISS (persisted `faiss_index/`) | Small dataset, reproducible, no server needed |
| Model layer | `src/evaluation/` (policy interface) | Abstracts LinearRegression → RL-capable model |
| Training/learning pipeline | `src/training/retrain_rl_policy.py` (manual) | Avoids false automatic RL claims; matches Q3.md |
| API/backend | Flask (`src/web_evaluation/app.py`) | Existing; extend with review/reward endpoints |
| Frontend | `templates/index.html` + new review template | Minimal extension of existing Bootstrap form |
| File/storage layer | `knowledge/`, `faiss_index/`, `models/`, `database/` | Structured for reproducibility |
| Config/secrets | `.env` + `.env.example` + `config/settings.py` | Professional repo standard |

---

## Trade-offs & Design Decisions

- **Not full RLHF:** Given Bachelor's scope and small dataset, implementing full PPO/RLHF is impractical. Instead, implement a **reward-weighted update mechanism** that uses teacher corrections as a reward signal. Document this honestly.
- **Not replacing OCR:** Per Q2.md, preserve existing Tesseract pipeline. Improvements limited to error handling and integration.
- **Not automatic retraining:** Per Q3.md, manual trigger prevents unstable automated updates and allows audit.
- **Not rewriting everything:** Preserve usable components (`extract_features.py`, `helpers.py` structure, basic Flask setup); replace/expand only what is missing or incorrect (`predict_score.py` LinearRegression, missing RAG, missing database, false RL claims).

---

## Verification Plan (from Prompt Requirements)

1. File upload and validation — test with `Answer1.pdf` and `Answer key.pdf`.
2. OCR extraction — verify `extract_text.py` output; handle missing/empty input.
3. Text cleaning/preprocessing — add basic cleaning; verify output format.
4. Question/answer segmentation — design for future segmentation; document current limitation.
5. RAG retrieval — build index from `Answer key.pdf`; verify retrieval returns relevant chunks for a sample question.
6. Model inference — verify policy produces a score and explanation.
7. Mark assignment/output formatting — verify score output format includes explanation.
8. Teacher mark modification — verify `/review` endpoint allows modification.
9. Optional feedback submission — verify empty feedback is allowed.
10. Feedback persistence — verify SQLite writes evaluation + feedback.
11. Learning/update pipeline — run `retrain_rl_policy` manually and verify model version updates.
12. End-to-end evaluation flow — manual end-to-end with sample PDFs.

---

## Implementation Phases

**Phase 1 — Discovery:** Complete (manual read of all source files, PDFs, requirements, README, .gitignore).
**Phase 2 — Technical Audit:** Findings documented; `README.md` false RL claim identified; `LinearRegression` identified for replacement; missing RAG/database/review confirmed.
**Phase 3 — Gap Analysis:** Completed (see mapping table above).
**Phase 4 — Architecture Design:** This document (`docs/superpowers/specs/YYYY-MM-DD-handwritten-evaluation-rl-architecture.md`).
**Phase 5 — Implementation:** Proceed in logical increments:
  1. Update `.env.example`, `requirements.txt`, `.gitignore`, `setup.py`.
  2. Create `database/` (schema + init) and `feedback.db` initialization.
  3. Build `knowledge/` (add `Answer key.pdf` reference extraction), `faiss_index/`, `src/rag/`.
  4. Replace `score_prediction/` with `src/evaluation/` (policy, reward, inference); keep interface for backward compatibility.
  5. Update `web_evaluation/app.py` with review/reward endpoints and SQLite integration.
  6. Add `src/training/retrain_rl_policy.py`.
  7. Update `README.md` and create `docs/ARCHITECTURE.md`.
  8. Add basic tests (`tests/test_ocr.py`, `tests/test_rag.py`, `tests/test_evaluation.py`).
  9. Verify end-to-end with sample PDFs.
