import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from src.pipeline import process_pdf_and_build_index
from src.data_loader import load_text_files
from src.chunker import chunk_documents
from src.rag_service import RAGService
from src.config import load_config

app = FastAPI(title="RAG System with PDF Upload", version="2.0")
config = load_config()
rag = RAGService()

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, extract text, build embeddings, store in FAISS."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_path = os.path.join("data/pdfs", file.filename)
    os.makedirs("data/pdfs", exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    process_pdf_and_build_index(pdf_path)
    return {"message": f"✅ PDF {file.filename} processed and indexed successfully."}

@app.post("/ask")
async def ask_question(query: str):
    """Ask a question based on the latest indexed PDF."""
    texts = load_text_files(config["data_path"])
    chunks = chunk_documents(texts)
    answer = rag.ask(query, chunks)
    return {"query": query, "answer": answer}

@app.get("/")
def home():
    return {"message": "Welcome to the RAG PDF API. Use /upload_pdf and then /ask."}
