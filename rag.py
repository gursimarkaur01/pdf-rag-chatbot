import numpy as np


def create_chunks(pages, chunk_size=1000, overlap=150):
    chunks = []

    for page in pages:
        text = page["text"]

        start = 0

        while start < len(text):
            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page["page"]
                })

            start += chunk_size - overlap

    return chunks


def cosine_similarity(query_vector, document_vectors):
    query_vector = np.array(query_vector, dtype=np.float32)
    document_vectors = np.array(
        document_vectors,
        dtype=np.float32
    )

    query_norm = np.linalg.norm(query_vector)

    document_norms = np.linalg.norm(
        document_vectors,
        axis=1
    )

    similarities = (
        document_vectors @ query_vector
    ) / (
        document_norms * query_norm + 1e-10
    )

    return similarities


def retrieve(
    query_vector,
    chunks,
    embeddings,
    top_k=5,
    min_score=0.35
):
    scores = cosine_similarity(
        query_vector,
        embeddings
    )

    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []

    for index in top_indices:

        score = float(scores[index])

        if score < min_score:
            continue

        results.append({
            "text": chunks[index]["text"],
            "page": chunks[index]["page"],
            "score": score
        })

    return results