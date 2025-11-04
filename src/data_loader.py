import os

def load_text_files(data_dir):
    docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

if __name__ == "__main__":
    from config import load_config
    config = load_config()
    data = load_text_files(config["data_path"])
    print(f"Loaded {len(data)} documents.")
    print(data[0][:500])
