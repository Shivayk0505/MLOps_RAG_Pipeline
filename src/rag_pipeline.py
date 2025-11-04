import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.embedder import get_embeddings
from src.retriever import retrieve_top_k
from src.chunker import chunk_documents
from src.data_loader import load_text_files
from src.config import load_config

# def load_llm(model_name):
#     print("🔹 Loading Mistral model...")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
#     return tokenizer, model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llm(model_name, fallback_model="microsoft/Phi-3-mini-4k-instruct", hf_token=None):
    """
    Load the primary LLM; if it fails, fall back to a smaller public model.
    """
    print(f"🔹 Attempting to load model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token
        )
        print(f"Successfully loaded: {model_name}")
        return tokenizer, model

    except Exception as e:
        print(f"Failed to load '{model_name}': {e}")
        print(f" Falling back to public model: {fallback_model}")

        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        model = AutoModelForCausalLM.from_pretrained(
            fallback_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        print(f"Loaded fallback model: {fallback_model}")
        return tokenizer, model

def generate_answer(query, chunks, tokenizer, model):
    context = "\n".join(chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    config = load_config()

    # Load docs and re-use embedding model
    texts = load_text_files(config["data_path"])
    chunks = chunk_documents(texts)
    _, embedder_model = get_embeddings(["test"])  # load model only once

    # Load LLM
    tokenizer, model = load_llm(config["llm_model"])

    # Example query
    query = "What is Artificial Intelligence?"
    retrieved = retrieve_top_k(query, embedder_model, chunks, k=3)
    print("\n🔹 Retrieved Context:")
    for i, c in enumerate(retrieved):
        print(f"\n--- Chunk {i+1} ---\n{c[:200]}...")

    # Generate RAG answer
    answer = generate_answer(query, retrieved, tokenizer, model)
    print("\n💡 Generated Answer:\n", answer)

if __name__ == "__main__":
    main()
