def extract_features(text):
    # Tokenization
    tokens = text.split()
    
    # Example feature extraction: word count
    word_count = len(tokens)
    
    # Example feature extraction: average word length
    avg_word_length = sum(len(token) for token in tokens) / word_count if word_count > 0 else 0
    
    # Example feature extraction: unique word count
    unique_word_count = len(set(tokens))
    
    features = {
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'unique_word_count': unique_word_count,
    }
    
    return features

def extract_features_from_texts(texts):
    features_list = []
    for text in texts:
        features = extract_features(text)
        features_list.append(features)
    return features_list