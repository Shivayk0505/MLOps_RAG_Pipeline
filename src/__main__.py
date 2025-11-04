from .data_loader import load_text_files
from .chunker import chunk_documents
from .embedder import get_embeddings
from .vector_store import save_faiss_index
from .config import load_config

def main():
    config = load_config()
    texts = load_text_files(config["data_path"])
    chunks = chunk_documents(texts)
    print(f"Chunks created: {len(chunks)}")

    embeddings, model = get_embeddings(chunks)
    print(f"Embeddings shape: {embeddings.shape}")

    save_faiss_index(embeddings, config["vector_store_path"])

if __name__ == "__main__":
    main()
