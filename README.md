<!-- # RAG MLOps MVP

A minimal **Retrieval-Augmented Generation (RAG)** pipeline with MLOps integration.

## Features
- Vector storage using ChromaDB  
- Text embedding with Sentence Transformers  
- Generation via open-source Mistral model  
- Logging and tracking with MLflow  
- Deployable via FastAPI + Docker  

## Setup
```bash
git clone <your-repo>
cd Mlops_project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

### To test the RAG inference pipeline
Run the following commands:

```bash
python src/pdf_to_txt.py
python -m src.__main__
python -m src.rag_pipeline
``` -->

# 📘 RAG PDF Assistant

This Space allows you to upload PDFs, automatically extract text, store embeddings in FAISS, and ask questions using an open-source Mistral LLM.

**Pipeline:**
1. Upload → PDF → Extract text (pdfplumber + OCR)
2. Chunk & embed (Sentence Transformers)
3. Store embeddings in FAISS
4. Query → Retrieve relevant chunks → Mistral LLM generates answer

---
**How to use:**
- Upload one or more PDF files  
- Click **Process PDFs**  
- Type a question about their content  
- Get your answer!

---
Built with ❤️ using **Streamlit**, **Mistral**, and **Hugging Face Spaces**.
