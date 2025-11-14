from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import load_config
from src.data_loader import load_text_files
from src.logger import get_logger     # <-- added

log = get_logger()                    # <-- added

def chunk_documents(texts):
    config = load_config()
    log.info("Starting document chunking...")   # <-- added

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )
    chunks = []
    for doc in texts:
        chunks.extend(text_splitter.split_text(doc))

    log.info(f"Chunking completed — total chunks created: {len(chunks)}")  # <-- added
    return chunks

if __name__ == "__main__":
    config = load_config()
    texts = load_text_files(config["data_path"])
    chunks = chunk_documents(texts)
    print(f"Total chunks created: {len(chunks)}")
    print(chunks[0][:500])
