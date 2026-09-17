import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="This is a test sentence for my RAG chatbot.",
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=768
    )
)


embedding = result.embeddings[0].values

print("Embedding generated successfully.")
print("Number of dimensions:", len(embedding))
print("First 5 values:", embedding[:5])