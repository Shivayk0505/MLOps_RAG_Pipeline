import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.embedder import get_embeddings
from src.retriever import retrieve_top_k
from src.chunker import chunk_documents
from src.data_loader import load_text_files
from src.config import load_config

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llm(model_name, fallback_model="microsoft/Phi-3-mini-4k-instruct", hf_token=None):
    print(f"🔹 Attempting to load model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

        # ✅ important fix for truncated outputs
        tokenizer.pad_token = tokenizer.eos_token              # <-- added

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

        # ✅ also set inside fallback
        tokenizer.pad_token = tokenizer.eos_token              # <-- added

        model = AutoModelForCausalLM.from_pretrained(
            fallback_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        print(f"Loaded fallback model: {fallback_model}")
        return tokenizer, model



# def generate_answer(query, chunks, tokenizer, model):
#     context = "\n".join(chunks)
#     prompt = (
#         "You are a helpful AI assistant. Answer ONLY based on the given context.\n\n"
#         f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
#     )
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(**inputs, max_new_tokens=256)
#     text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # return only answer part
#     if "Answer:" in text:
#         text = text.split("Answer:")[-1].strip()
#     return text

def generate_answer(query, chunks, tokenizer, model):
    context = "\n\n".join(f"[Doc {i+1}]\n{c}" for i, c in enumerate(chunks))

    messages = [
        {"role": "system", "content":
         "You are a precise RAG assistant. Use only the provided context. "
         "Give complete answers. End with [END]."},
        {"role": "user", "content":
         f"Context:\n{context}\n\nQuestion:\n{query}\n\n"
         "Respond with a thorough, complete answer.\nEnd with [END]."}
    ]

    # 👉 QWEN recommended way (this builds prompt + ATTENTION MASK automatically)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True
    )

    return decoded.split("[END]")[0].strip()


def main():
    config = load_config()

    texts = load_text_files(config["data_path"])
    chunks = chunk_documents(texts)
    _, embedder_model = get_embeddings(["test"])  # load embedding model once

    tokenizer, model = load_llm(config["llm_model"])

    print("\n✅ RAG Pipeline ready. Type your questions (or 'exit').")

    while True:
        query = input("\n❓ Enter your question: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Exiting RAG system.")
            break

        retrieved = retrieve_top_k(query, embedder_model, chunks, k=3)
        answer = generate_answer(query, retrieved, tokenizer, model)

        print("\n💡 Answer:", answer)


if __name__ == "__main__":
    main()
