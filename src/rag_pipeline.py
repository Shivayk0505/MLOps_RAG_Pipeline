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

        # important fix for truncated outputs
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

        # also set inside fallback
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
    # limit context chunk length to reduce repetition feedback
    context_text = " ".join([c[:800] for c in chunks])

    system_prompt = (
        "You are a precise RAG assistant. "
        "Use the internal context ONLY for reference and never reveal it. "
        "Give a single, self-contained answer that is concise and easy to read. "
        "Use 3–5 short bullet points if that improves clarity; otherwise a short paragraph. "
        "Do NOT repeat the same idea or sentence more than once. "
        "If the context itself is repetitive, summarize the repeated idea only ONCE. "
        "Do not ask new questions or add unrelated details."
    )

    prompt = (
        f"<SYS>\n{system_prompt}\n"
        f"[INTERNAL_CONTEXT]: {context_text}\n</SYS>\n\n"
        f"USER: {query}\nASSISTANT:"
    )

    model_max = getattr(model.config, "max_position_embeddings", 4096)
    max_new = 256
    max_input_tokens = max(256, model_max - max_new - 32)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.25   # STRONGER anti-loop
        )

    gen_ids = outputs[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def _finish_neatly(s: str) -> str:
        s = s.strip()
        if s.endswith((".", "!", "?")):
            return s
        for sep in [".", "!", "?", "\n- ", "\n• "]:
            idx = s.rfind(sep)
            if idx != -1 and idx >= len(s) * 0.6:
                cut = s[: idx + (1 if sep in {".", "!", "?"} else 0)].strip()
                if cut:
                    return cut
        return s

    return _finish_neatly(text)




def main():
    config = load_config()

    texts = load_text_files(config["data_path"])
    chunks = chunk_documents(texts)
    _, embedder_model = get_embeddings(["test"])  # load embedding model once

    tokenizer, model = load_llm(config["llm_model"])

    print("\n RAG Pipeline ready. Type your questions (or 'exit').")

    while True:
        query = input("\n Enter your question: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print(" Exiting RAG system.")
            break

        retrieved = retrieve_top_k(query, embedder_model, chunks, k=3)
        answer = generate_answer(query, retrieved, tokenizer, model)

        print("\n Answer:", answer)


if __name__ == "__main__":
    main()
