import os
import json
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
try:
    import faiss
except ImportError:
    faiss = None


model = None
index = None
metadata = None


def init_rag(index_path="faiss_index/index.faiss", meta_path="faiss_index/metadata.json"):
    global model, index, metadata
    if SentenceTransformer is None or faiss is None:
        raise ImportError("sentence-transformers and faiss-cpu are required.")
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}. Run build_index.py first.")
    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    return index


def retrieve(query_text, k=3):
    global model, index, metadata
    if index is None:
        init_rag()
    query_embedding = model.encode([query_text], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    results = [metadata["chunks"][i] for i in indices[0] if i < len(metadata["chunks"])]
    return results
