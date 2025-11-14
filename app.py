import streamlit as st
import os
import time
from src.pdf_to_txt_plumber import extract_text_from_pdfs
from src.pipeline import process_pdf_and_build_index
from src.data_loader import load_text_files
from src.chunker import chunk_documents
from src.rag_service import RAGService
from src.config import load_config

# Configure Streamlit
st.set_page_config(page_title="📘 RAG PDF Assistant", layout="wide")
st.title("📘 Retrieval-Augmented Generation (RAG) Assistant")

# Load Config and Models
config = load_config()

@st.cache_resource
def load_rag_service():
    return RAGService()

rag = load_rag_service()

# Section 1: Upload PDFs
st.header("1️⃣ Upload your PDF files")
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    os.makedirs("data/pdfs", exist_ok=True)
    for file in uploaded_files:
        pdf_path = os.path.join("data/pdfs", file.name)
        with open(pdf_path, "wb") as f:
            f.write(file.getbuffer())
        st.success(f"Uploaded: {file.name}")

    if st.button("📚 Process PDFs"):
        with st.spinner("Extracting text and building FAISS index..."):
            start_time = time.time()
            extract_text_from_pdfs(pdf_folder="data/pdfs", txt_folder="data/docs")
            process_pdf_and_build_index("data/pdfs")
            st.success(f"✅ All PDFs processed in {round(time.time() - start_time, 2)}s")

# Section 2: Ask Questions
st.header("2️⃣ Ask a question about your documents")
query = st.text_input("💬 Type your question here:")
if query:
    with st.spinner("Retrieving context and generating answer..."):
        texts = load_text_files(config["data_path"])
        chunks = chunk_documents(texts)
        answer = rag.ask(query, chunks)
        st.markdown("### 💡 **Answer:**")
        st.write(answer)
