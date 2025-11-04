from sentence_transformers import SentenceTransformer
from .config import load_config

def get_embeddings(chunks):
    config = load_config()
    model = SentenceTransformer(config["embedding_model"])
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings, model
