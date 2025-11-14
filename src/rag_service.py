import time
import numpy as np
from src.logger import get_logger
from src.config import load_config
from src.data_loader import load_text_files
from src.chunker import chunk_documents
from src.rag_pipeline import generate_answer
from src.global_objects import get_llm, get_embedder, get_faiss  # cached singletons
from src.mlflow_logger import log_rag_run

log = get_logger()


class RAGService:
    def __init__(self):
        log.info("🔹 Initializing RAG Service...")
        self.config = load_config()

        # --- Load and chunk documents ---
        texts = load_text_files(self.config["data_path"])
        self.chunks = chunk_documents(texts)
        log.info(f"📘 Loaded {len(self.chunks)} chunks from documents")

        # --- Cached global models ---
        self.embedder = get_embedder()
        log.info("🧩 Embedder cached + ready")

        self.faiss_index = get_faiss()
        log.info("📚 FAISS index cached + ready")

        self.tokenizer, self.model = get_llm()
        log.info(f"🧠 LLM cached + ready: {self.config['llm_model']}")

    # ------------------------------------------------------------------
    # 🔍 Core query handler
    # ------------------------------------------------------------------
    def ask(self, query: str, k: int = 3):
        start = time.perf_counter()
        log.info(f"User query received: {query}")

        # --- Step 1: Encode the query ---
        q_emb = self.embedder.encode([query])
        log.debug(f"Query embedding shape: {np.array(q_emb).shape}")

        # --- Step 2: Retrieve top-k chunks using FAISS ---
        D, I = self.faiss_index.search(np.array(q_emb).astype(np.float32), k)
        retrieved = [
            {"chunk": self.chunks[idx], "score": float(score)}
            for idx, score in zip(I[0], D[0])
        ]
        log.info(f"Top-{k} chunks retrieved with scores")

        # --- Step 3: Generate refined LLM answer ---
        chunks_text = [r["chunk"] for r in retrieved]
        answer = generate_answer(query, chunks_text, self.tokenizer, self.model)

        # --- Step 4: Track latency ---
        latency = (time.perf_counter() - start) * 1000
        log.info(f"🕒 RAG answer generated in {latency:.2f} ms")

        # --- Step 5: MLflow experiment tracking ---
        try:
            log_rag_run(
                pdf_name="uploaded_docs",
                chunks_count=len(retrieved),
                embedding_model="sentence-transformer",
                llm_model=self.config["llm_model"],
                latency_ms=latency
            )
            log.debug("RAG run logged to MLflow.")
        except Exception as e:
            log.warning(f"⚠️ MLflow logging failed: {e}")

        # --- Step 6: Return final structured result ---
        return {
            "answer": answer,
            "chunks": retrieved,   # now contains both text + score
            "latency_ms": latency
        }
