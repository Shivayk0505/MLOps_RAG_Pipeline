RAG PDF Question Answering System — Instructions Summary
What This System Does

Upload one or multiple PDFs

Extract text automatically (pdfplumber + OCR)

Chunk the text

Generate embeddings

Store vectors in FAISS

Retrieve relevant chunks using cosine similarity

Pass retrieved context to Qwen LLM

Generate accurate answers

Interact through Streamlit UI

✅ How the System Works (Pipeline)

Upload PDF(s)

Extract text

pdfplumber → digital PDFs

Tesseract → scanned PDFs

Chunk text into 300–500 character segments

Generate embeddings using MiniLM

Store embeddings in FAISS vector DB

Retrieve top-k similar chunks using cosine similarity

Generate answer using Qwen LLM

Display answer via Streamlit UI

✅ Technologies Used

pdfplumber → extract text

Tesseract OCR → scanned PDFs

SentenceTransformers → embeddings

FAISS → vector search

Qwen LLM → answer generation

Streamlit → frontend

Loguru → logging

MLflow → pipeline tracking (optional)
