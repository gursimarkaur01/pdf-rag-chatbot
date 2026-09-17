import os

from pdf_processor import load_pdf
from rag import create_chunks
from embeddings import embed_documents
from vector_store import save_vector_store


PDF_PATH = "data/document.pdf"


def main():
    print("Loading PDF...")

    pages = load_pdf(PDF_PATH)

    print(f"Extracted {len(pages)} pages.")

    print("Creating chunks...")

    chunks = create_chunks(pages)

    print(f"Created {len(chunks)} chunks.")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Generating embeddings...")

    embeddings = embed_documents(texts)

    os.makedirs("storage", exist_ok=True)

    save_vector_store(
        chunks,
        embeddings
    )

    print("Vector store created successfully.")


if __name__ == "__main__":
    main()