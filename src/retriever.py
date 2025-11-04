import numpy as np
from src.vector_store import load_faiss_index
from src.embedder import get_embeddings
from src.config import load_config

def retrieve_top_k(query, embedder_model, chunks, k=3):
    """Return top-k most similar chunks for a given query."""
    config = load_config()
    index = load_faiss_index()
    query_embedding = embedder_model.encode([query])
    D, I = index.search(np.array(query_embedding, dtype=np.float32), k)
    retrieved = [chunks[i] for i in I[0]]
    return retrieved

