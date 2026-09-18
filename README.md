# PDF RAG Chatbot

A lightweight conversational **Retrieval-Augmented Generation (RAG) chatbot** that allows users to ask questions about a single PDF document.

The application extracts information from the PDF, converts document chunks into embeddings, retrieves the most relevant sections for a user's question, and uses the Gemini API to generate an answer based only on the retrieved PDF context.

The project is designed to run locally on a **CPU-only machine** without requiring a local generative AI model or GPU.

---

## Features

- Ask natural-language questions about a PDF
- Retrieval-Augmented Generation (RAG) pipeline
- Semantic search using Gemini embeddings
- Cosine similarity-based document retrieval
- Conversational follow-up questions
- PDF page references for retrieved information
- Responses grounded in the provided PDF context
- Explicit response when relevant information cannot be found
- Lightweight local vector storage using NumPy
- Simple Flask web interface
- API key stored securely using environment variables
- No GPU or locally hosted LLM required

---

## How It Works

The chatbot follows this pipeline:

```text
                 PDF Document
                      |
                      v
              Extract PDF Text
                      |
                      v
                 Clean Text
                      |
                      v
                 Create Chunks
                      |
                      v
          Gemini Embedding Model
                      |
                      v
          Store Embeddings Locally
                      |
                      |
              User asks a question
                      |
                      v
            Create Query Embedding
                      |
                      v
            Cosine Similarity Search
                      |
                      v
            Retrieve Top-K Chunks
                      |
                      v
        PDF Context + Conversation
                      |
                      v
             Gemini LLM Response
                      |
                      v
                   Answer
```
# Why RAG?
A general-purpose language model may not know the contents of a specific document provided by a user.

RAG solves this by first searching the document for relevant information and then providing that information to the language model as context.

This project therefore follows:

User Question
      ↓
Retrieve relevant PDF content
      ↓
Give retrieved content to Gemini
      ↓
Generate grounded answer
