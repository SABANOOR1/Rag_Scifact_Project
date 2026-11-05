import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # Reads GEMINI_API_KEY from your .env file

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def generate_answer(query: str, retrieved_docs: list) -> str:
    """
    Constructs a RAG prompt and queries the Gemini Flash model.
    """
    if not GEMINI_API_KEY:
        return (
            f"[Simulated Answer - No GEMINI_API_KEY found] "
            f"Based on doc {retrieved_docs[0]['id']}, the claim is supported."
        )

    # Initialize client with your key
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Format the evidence blocks
    context_blocks = [
        f"Document ID: {doc['id']}\nTitle: {doc['title']}\nAbstract: {doc['text']}"
        for doc in retrieved_docs
    ]
    context_str = "\n\n---\n\n".join(context_blocks)

    system_instruction = (
        "You are an objective scientific question-answering assistant. "
        "Answer the user question strictly using the provided scientific abstracts. "
        "Every claim you state must cite the source Document ID in brackets (e.g. [10423]). "
        "If the evidence does not contain enough information, state that clearly."
    )

    prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

    try:
        # backend/generator.py
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1
            )
        )
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {str(e)}"