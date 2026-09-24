# Known Limitations

This document explicitly identifies limitations that should not be hidden.

1. **OCR Accuracy:** Handwritten text extraction relies on Tesseract (`pytesseract`). Quality varies significantly with handwriting clarity, scanning resolution, and image noise.
2. **RL Mechanism Scope:** The implemented RL component uses a **reward-weighted supervised update** mechanism, not full RLHF or PPO. This is a technically legitimate RL-inspired approach appropriate for a Bachelor's-level dataset size (tens to hundreds of examples), but it is not production-scale reinforcement learning.
3. **Feature Engineering:** The evaluation policy relies on basic text statistics (`word_count`, `avg_word_length`, `unique_word_count`). More advanced NLP features or embeddings could improve evaluation accuracy but are outside the current scope.
4. **RAG Reference Size:** Currently limited to `Answer key.pdf`. The pipeline supports additional reference documents via the `knowledge/` directory and `build_index.py`, but the index must be rebuilt when references change.
5. **Manual Retraining:** The RL policy update requires manual execution (`python -m src.training.retrain_rl_policy`). There is no automatic trigger.
6. **Optional Feedback:** The system must function when teachers choose not to provide written feedback. The reward calculation handles empty feedback gracefully (`reward = correction_magnitude` only).
7. **Model Persistence:** Policy versions are saved as `models/policy_v*.pkl` (`joblib`). These are safe (locally produced by this application), but loading untrusted `.pkl` files in general is a security risk.
8. **No Automatic Question Segmentation:** The current pipeline does not automatically split handwritten scripts into individual question/answer pairs. This is a future improvement.
