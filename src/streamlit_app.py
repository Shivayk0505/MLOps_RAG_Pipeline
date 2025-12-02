import streamlit as st
import os
import subprocess
import webbrowser
import time
import sys
import socket

from src.pdf_to_txt import extract_text_from_pdf
from src.chunker import chunk_documents
from src.embedder import get_embeddings
from src.vector_store import save_faiss_index
from src.rag_service import RAGService
from src.config import load_config
from src.logger import get_logger
from src.data_loader import load_text_files

# Initialize logger
log = get_logger()

# -------------------------------------------------------------
#  Streamlit UI Setup
# -------------------------------------------------------------
st.set_page_config(page_title="RAG PDF Assistant", page_icon="🤖", layout="wide")
st.title(" RAG-based PDF Assistant")
st.write("Upload one or more PDFs, index them, and ask questions about their contents!")

config = load_config()
rag = None

# -------------------------------------------------------------
#  MLflow Dashboard Launcher
# -------------------------------------------------------------
st.sidebar.markdown("###  MLflow Experiment Tracking")

def find_free_port(start_port=5000, max_tries=10):
    """Find the first available port starting from `start_port`."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
    return start_port  # fallback

if st.sidebar.button(" Open MLflow Dashboard"):
    try:
        python_exec = sys.executable
        port = find_free_port(5000)

        subprocess.Popen(
            [python_exec, "-m", "mlflow", "ui", "--backend-store-uri", "./mlruns", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        time.sleep(3)
        webbrowser.open(f"http://localhost:{port}")
        st.sidebar.success(f"MLflow UI launched successfully on port {port} 🌐")
        log.info(f"MLflow UI launched on http://localhost:{port}")
    except Exception as e:
        st.sidebar.error(f"Failed to start MLflow UI: {e}")
        log.error(f"Failed to launch MLflow UI: {e}")

# -------------------------------------------------------------
#  File Upload Section (supports multiple PDFs)
# -------------------------------------------------------------
uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
pdf_dir = "data/pdfs"
os.makedirs(pdf_dir, exist_ok=True)

if uploaded_files:
    # Extract all PDFs (new or existing)
    for uploaded_file in uploaded_files:
        pdf_path = os.path.join(pdf_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        st.sidebar.success(f" Uploaded: {uploaded_file.name}")
        log.info(f"PDF uploaded: {uploaded_file.name}")

        # Extract text for this specific file
        extract_text_from_pdf(pdf_path, txt_folder=config["data_path"])

    st.sidebar.info(" PDF text extraction completed for all uploads.")
    log.info("Text extraction completed for all PDFs.")

    # -------------------------------------------------------------
    #  FAISS & Embeddings Handling
    # -------------------------------------------------------------
    faiss_index_path = os.path.join(config["vector_store_path"], "faiss.index")

    if not os.path.exists(faiss_index_path):
        st.sidebar.warning("No FAISS index found — creating embeddings now...")
        log.info("No FAISS found. Creating embeddings fresh...")

        texts = load_text_files(config["data_path"])
        chunks = chunk_documents(texts)
        embeddings, _ = get_embeddings(chunks)
        save_faiss_index(embeddings, config["vector_store_path"])

        st.sidebar.success(" Embeddings & FAISS index created successfully!")
        log.info("Embeddings & FAISS index created successfully.")
    else:
        st.sidebar.info("FAISS index already exists — skipping embedding step.")
        log.info("Skipped embedding — FAISS already exists.")

    # Initialize RAG
    rag = RAGService()

# -------------------------------------------------------------
#  Q&A Interface
# -------------------------------------------------------------
if rag:
    st.subheader("Ask your document(s) a question ")
    query = st.text_input("Enter your question:")
    top_k = st.slider("Number of chunks to retrieve", 1, 10, 3)

    if st.button("Ask"):
        with st.spinner("Thinking..."):
            result = rag.ask(query, k=top_k)
            log.info(f"User query: {query}")

            st.markdown("### 💬 Answer")
            st.write(result["answer"])

            #  Show context chunks used (transparency)
            if "chunks" in result and len(result["chunks"]) > 0:
                with st.expander("📚 Context used"):
                    for i, c in enumerate(result["chunks"], start=1):
                        if isinstance(c, dict):
                            chunk_text = c.get("chunk", str(c))[:500]
                            score = c.get("score", 0.0)
                            # st.markdown(f"**Chunk {i} (score: {score:.4f})**")
                            st.write(chunk_text + "...")
                        else:
                            st.markdown(f"**Chunk {i}**")
                            st.write(str(c)[:500] + "...")
else:
    st.info("Upload one or more PDFs in the sidebar to start.")
