import streamlit as st
import os
from src.pdf_to_txt import extract_text_from_pdfs
from src.chunker import chunk_documents
from src.embedder import get_embeddings
from src.vector_store import save_faiss_index
from src.rag_service import RAGService
from src.config import load_config

st.set_page_config(page_title="RAG PDF Assistant", page_icon="🤖", layout="wide")
st.title("📄 RAG-based PDF Assistant")
st.write("Upload a PDF, index it, and ask questions about its contents!")

config = load_config()
rag = None

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    pdf_dir = "data/pdfs"
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.sidebar.success("PDF uploaded successfully!")

    extract_text_from_pdfs(pdf_folder=pdf_dir, txt_folder=config["data_path"])
    st.sidebar.info("Extracted text and saved as .txt")

    from src.data_loader import load_text_files
    texts = load_text_files(config["data_path"])
    chunks = chunk_documents(texts)
    embeddings, _ = get_embeddings(chunks)
    save_faiss_index(embeddings, config["vector_store_path"])
    st.sidebar.success("Embeddings & FAISS index updated!")

    rag = RAGService()

if rag:
    st.subheader("Ask your document a question 👇")
    query = st.text_input("Enter your question:")
    top_k = st.slider("Top K context chunks", 1, 10, 3)

    if st.button("Ask"):
        with st.spinner("Thinking..."):
            result = rag.ask(query, k=top_k)

            st.markdown("### 💬 Answer")
            st.write(result["answer"])   # ONLY show answer, nothing else
else:
    st.info("Upload a PDF in the sidebar to start.")
