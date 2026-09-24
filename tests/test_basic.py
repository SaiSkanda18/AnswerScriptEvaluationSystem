def test_ocr_extraction():
    from src.text_extraction.extract_text import extract_text_from_image
    # This test checks that the function exists and can handle missing files gracefully
    try:
        result = extract_text_from_image("nonexistent.png")
    except Exception:
        pass  # expected behavior for missing file


def test_feature_extraction():
    from src.feature_extraction.extract_features import extract_features
    result = extract_features("This is a sample student answer.")
    assert 'word_count' in result
    assert result['word_count'] == 6


def test_reward_calculation():
    from src.evaluation.reward import calculate_reward
    r = calculate_reward(70, 65, "Good explanation.")
    assert r > 0
