# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import load_config
from .data_loader import load_text_files

def chunk_documents(texts):
    config = load_config()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )
    chunks = []
    for doc in texts:
        chunks.extend(text_splitter.split_text(doc))
    return chunks

if __name__ == "__main__":
    config = load_config()
    texts = load_text_files(config["data_path"])
    chunks = chunk_documents(texts)
    print(f"Total chunks created: {len(chunks)}")
    print(chunks[0][:500])
