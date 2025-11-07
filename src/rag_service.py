from src.embedder import get_embeddings
from src.retriever import retrieve_top_k
from src.config import load_config
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.chunker import chunk_documents
from src.data_loader import load_text_files
from src.rag_pipeline import generate_answer  # <-- important

class RAGService:
    def __init__(self):
        print("🔹 Initializing RAG Service...")
        self.config = load_config()

        # Load documents & chunk
        texts = load_text_files(self.config["data_path"])
        self.chunks = chunk_documents(texts)

        # Load embedder only once
        _, self.embedder_model = get_embeddings(["warmup"])

        # Load LLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["llm_model"])
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["llm_model"],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def ask(self, query, k=3):
        # Retrieve chunks (silent, no printing)
        retrieved = retrieve_top_k(query, self.embedder_model, self.chunks, k)

        # Generate final refined answer from our good generate_answer()
        answer = generate_answer(query, retrieved, self.tokenizer, self.model)

        return {"answer": answer}
