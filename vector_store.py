import json
import numpy as np


CHUNKS_FILE = "storage/chunks.json"
EMBEDDINGS_FILE = "storage/embeddings.npy"


def save_vector_store(chunks, embeddings):
    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    np.save(EMBEDDINGS_FILE, np.array(embeddings, dtype=np.float32))


def load_vector_store():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    embeddings = np.load(EMBEDDINGS_FILE)

    return chunks, embeddings