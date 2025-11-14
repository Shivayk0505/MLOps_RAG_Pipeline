import faiss
import numpy as np
import os
from .config import load_config
from src.logger import get_logger   # <-- added

log = get_logger()                  # <-- added

def save_faiss_index(embeddings, vector_store_path):
    os.makedirs(vector_store_path, exist_ok=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)

    index.add(embeddings)

    index_path = os.path.join(vector_store_path, "faiss.index")
    faiss.write_index(index, index_path)

    log.info(f"FAISS index saved → {index_path} | vectors: {embeddings.shape[0]} dim: {dim}")   # <-- added

def load_faiss_index():
    config = load_config()
    path = os.path.join(config["vector_store_path"], "faiss.index")

    if not os.path.exists(path):
        log.warning("FAISS index not found, returning None")  # <-- added
        return None

    log.info(f"FAISS index loaded from → {path}")             # <-- added
    return faiss.read_index(path)
