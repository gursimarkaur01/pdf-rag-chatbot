import os
import traceback

from flask import Flask, request, jsonify, render_template
from google import genai
from dotenv import load_dotenv

from embeddings import embed_query
from rag import retrieve
from vector_store import load_vector_store


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )


# --------------------------------------------------
# Flask application
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# Gemini client
# --------------------------------------------------

client = genai.Client(
    api_key=api_key
)

MODEL = "gemini-3.5-flash-lite"


# --------------------------------------------------
# Load our PDF vector database
# --------------------------------------------------

chunks, embeddings = load_vector_store()

print()
print("Vector store loaded successfully.")
print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print()


# --------------------------------------------------
# Conversation memory
# --------------------------------------------------

conversation_history = []


# --------------------------------------------------
# Generate answer using Gemini
# --------------------------------------------------

def generate_answer(question, context, conversation):

    prompt = f"""
You are a PDF question-answering assistant.

Your job is to answer the user's question ONLY
using the information provided in the PDF CONTEXT.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not guess.
3. Do not invent information.
4. If the answer cannot be found in the PDF context,
   say exactly:

"I couldn't find that information in the PDF."

5. Use the conversation history only to understand
   follow-up questions.

PDF CONTEXT:
----------------
{context}
----------------

CONVERSATION HISTORY:
----------------
{conversation}
----------------

CURRENT USER QUESTION:
----------------
{question}
----------------

Answer clearly and naturally.
"""


    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Chat endpoint
# --------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat():

    try:

        # ------------------------------------------
        # Get user's question
        # ------------------------------------------

        data = request.get_json()

        question = data.get("question", "").strip()

        print()
        print("User question:", question)


        if not question:

            return jsonify({
                "error": "Question cannot be empty."
            }), 400


        # ------------------------------------------
        # Convert question into embedding
        # ------------------------------------------

        print("Generating query embedding...")

        query_vector = embed_query(question)

        print("Query embedding generated.")


        # ------------------------------------------
        # Retrieve relevant PDF chunks
        # ------------------------------------------

        print("Searching PDF...")

        results = retrieve(
            query_vector=query_vector,
            chunks=chunks,
            embeddings=embeddings,
            top_k=5,
            min_score=0.35
        )


        print("Retrieved chunks:", len(results))


        # ------------------------------------------
        # No relevant information
        # ------------------------------------------

        if not results:

            answer = (
                "I couldn't find that information in the PDF."
            )

            conversation_history.append({
                "user": question,
                "assistant": answer
            })

            return jsonify({
                "answer": answer,
                "sources": []
            })


        # ------------------------------------------
        # Build context
        # ------------------------------------------

        context = "\n\n".join(
            f"[Page {result['page']}]\n"
            f"{result['text']}"
            for result in results
        )


        # ------------------------------------------
        # Build conversation history
        # ------------------------------------------

        conversation = "\n".join(
            f"User: {item['user']}\n"
            f"Assistant: {item['assistant']}"
            for item in conversation_history[-6:]
        )


        # ------------------------------------------
        # Generate Gemini answer
        # ------------------------------------------

        print("Calling Gemini...")

        answer = generate_answer(
            question=question,
            context=context,
            conversation=conversation
        )

        print("Gemini response received.")


        # ------------------------------------------
        # Save conversation
        # ------------------------------------------

        conversation_history.append({
            "user": question,
            "assistant": answer
        })


        # ------------------------------------------
        # Send response to browser
        # ------------------------------------------

        return jsonify({
            "answer": answer,
            "sources": [
                {
                    "page": result["page"],
                    "score": result["score"]
                }
                for result in results
            ]
        })


    except Exception as error:

        print()
        print("========== ERROR ==========")
        print(str(error))
        traceback.print_exc()
        print("===========================")
        print()


        return jsonify({
            "error": str(error)
        }), 500


# --------------------------------------------------
# Run Flask
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)