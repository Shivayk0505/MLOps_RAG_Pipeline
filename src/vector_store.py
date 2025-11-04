import faiss
import numpy as np
import os
from .config import load_config

def save_faiss_index(embeddings, vector_store_path):
    os.makedirs(vector_store_path, exist_ok=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(vector_store_path, "faiss.index"))
    print(" FAISS index saved successfully!")

def load_faiss_index():
    config = load_config()
    path = os.path.join(config["vector_store_path"], "faiss.index")
    return faiss.read_index(path)
