from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.embedder import get_embeddings
from src.vector_store import load_faiss_index
from src.config import load_config
import torch

@lru_cache(maxsize=1)
def get_llm():
    config = load_config()
    tokenizer = AutoTokenizer.from_pretrained(config["llm_model"])
    model = AutoModelForCausalLM.from_pretrained(
        config["llm_model"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

@lru_cache(maxsize=1)
def get_embedder():
    _, embedder = get_embeddings(["warmup"])
    return embedder

@lru_cache(maxsize=1)
def get_faiss():
    return load_faiss_index()
