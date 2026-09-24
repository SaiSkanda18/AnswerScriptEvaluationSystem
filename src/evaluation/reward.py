def calculate_reward(original_score, corrected_score, feedback_text=""):
    """Calculate a scalar reward signal from teacher interaction.
    Not full RLHF; this is a reward-weighted update mechanism that uses
    correction magnitude and optional feedback sentiment.
    """
    correction_magnitude = abs(float(corrected_score) - float(original_score))
    reward = correction_magnitude
    # Optional positive sentiment bonus from feedback text (basic heuristic)
    positive_words = {"good", "excellent", "correct", "well", "great", "improved"}
    negative_words = {"wrong", "incorrect", "poor", "bad", "mistake", "error"}
    if feedback_text:
        words = set(feedback_text.lower().split())
        if words & positive_words:
            reward += 0.5
        elif words & negative_words:
            reward += 0.1  # small positive reward for any feedback (engagement signal)
    return reward
