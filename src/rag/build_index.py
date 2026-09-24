import os
import json

# Conditional import to allow environment without full ML stack installed
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import faiss
except ImportError:
    faiss = None


def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return chunks


def build_index(text, index_path="faiss_index/index.faiss", meta_path="faiss_index/metadata.json"):
    if SentenceTransformer is None or faiss is None:
        raise ImportError("sentence-transformers and faiss-cpu are required. Install: pip install sentence-transformers faiss-cpu")

    chunks = chunk_text(text)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    faiss.write_index(index, index_path)

    with open(meta_path, "w") as f:
        json.dump({"chunks": chunks, "count": len(chunks)}, f)

    print(f"FAISS index built: {len(chunks)} chunks -> {index_path}")


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from rag.extract_reference_text import extract_reference_text
    text = extract_reference_text()
    build_index(text)
