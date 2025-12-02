import os
from src.pdf_to_txt import extract_text_from_pdfs
from src.data_loader import load_text_files
from src.chunker import chunk_documents
from src.embedder import get_embeddings
from src.vector_store import save_faiss_index
from src.config import load_config

# orchestrates text extraction + embeddings + FAISS storage

def process_pdf_and_build_index(pdf_path):
    """
    Takes a single PDF path → extracts text → creates chunks → embeddings → FAISS index
    """
    config = load_config()
    pdf_folder = os.path.dirname(pdf_path)
    txt_folder = config["data_path"]

    #  Extract text
    print(f"🔹 Extracting text from {pdf_path}")
    extract_text_from_pdfs(pdf_folder=pdf_folder, txt_folder=txt_folder)

    #  Load extracted text
    print("🔹 Loading extracted text...")
    texts = load_text_files(txt_folder)

    #  Chunk and embed
    print("🔹 Chunking and embedding...")
    chunks = chunk_documents(texts)
    embeddings, _ = get_embeddings(chunks)

    #  Store in FAISS
    print("🔹 Saving FAISS index...")
    save_faiss_index(embeddings, config["vector_store_path"])

    print(" PDF processed and stored successfully!")
