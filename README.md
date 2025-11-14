

📘 RAG PDF Question Answering System (MLOps Integrated)

A Retrieval-Augmented Generation system using Mistral LLM, FAISS Vector Search, and Streamlit UI

🚀 Overview

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline that allows users to:

✔️ Upload one or multiple PDF files
✔️ Automatically extract text (pdfplumber + OCR for scanned PDFs)
✔️ Chunk & embed text using Sentence Transformers
✔️ Store embeddings in a FAISS vector database
✔️ Retrieve relevant chunks using cosine similarity
✔️ Generate accurate answers using Mistral LLM
✔️ Interact through a clean Streamlit web interface

This system follows MLOps best practices, making it modular, reusable, and production-ready.

🧱 Architecture

PDF Upload
   ↓
PDF Text Extraction (pdfplumber + OCR)
   ↓
Text Chunking
   ↓
Embedding Generation (SentenceTransformers)
   ↓
FAISS Vector Store (Cosine Similarity Search)
   ↓
Retriever (Top-k Relevant Chunks)
   ↓
Mistral LLM Answer Generation
   ↓
Streamlit UI

🧰 Features
🔹 PDF Processing

Extracts text from digital PDFs using pdfplumber

Uses Tesseract OCR for scanned PDFs

Stores extracted text in data/docs/

🔹 Chunking

Splits long documents into 300–500 character segments

Ensures meaningful, context-preserving retrieval

🔹 Embedding Generation

Uses sentence-transformers/all-MiniLM-L6-v2

Creates 384-dimensional semantic embeddings

🔹 FAISS Vector Database

Fast vector similarity search

Stores all embeddings

Performs cosine similarity search

🔹 RAG Pipeline

Embeds user query

Retrieves top-k relevant chunks

Passes them to Mistral LLM

Generates accurate, context-grounded answers

🔹 Streamlit Interface

Upload PDFs

Process documents

Ask natural-language questions

View generated answers

🛠️ Installation
1. Clone repo
   git clone https://github.com/<your-username>/<repo>.git
cd <repo>
python3 -m venv .mlops_env
source .mlops_env/bin/activate
pip install -r requirements.txt
streamlit run app.py

How to Use
1️⃣ Upload PDFs

Supports multiple files.

2️⃣ Process PDFs

This step performs:

Text extraction

OCR (if scanned)

Chunking

Embeddings

FAISS index creation

3️⃣ Ask Questions

Ask any question related to the uploaded PDF contents.

Example:

What is the conclusion of the document?

🔍 Retrieval Score Used

We use cosine similarity for retrieval.

FAISS performs inner-product search on normalized vectors:

cosine_similarity=q⋅d
cosine_similarity=q⋅d

Top-k most similar chunks are retrieved and passed to the LLM.

📊 MLOps Components
✔️ Loguru Logging

Tracks:

PDF → text extraction

Embeddings generation

Vector indexing

Query processing

✔️ MLflow Tracking (Optional)

Logs:

Chunk count

Processing time

Embedding model

LLM model used

✔️ Modular Pipeline

Each stage (extraction, chunking, embeddings, retrieval, generation) is isolated and reusable.

Future Enhancements

Incremental FAISS indexing

Metadata-based search

Cloud deployment (Hugging Face / AWS)

Cross-encoder re-ranking

GPU inference

Chat memory

🏁 Conclusion

This project demonstrates a complete, production-ready RAG system with:

PDF ingestion

Text & OCR extraction

Embedding generation

Vector search

LLM-based question answering

Streamlit UI

MLOps-ready architecture

It is modular, scalable, and can be deployed or extended easily.
