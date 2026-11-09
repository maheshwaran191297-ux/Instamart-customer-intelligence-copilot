import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from partD_retrieval import retrieve
import mock_llm

# 1. RAG answer function
def answer(question):

    # Retrieve the top 3 relevant policy chunks
    retrieved_chunks = retrieve(question, k=3)

    # Build context in the required format
    context = "\n".join(
        f"[{chunk['source']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )

    # System instructions
    system_message = {
        "role": "system",
        "content": (
            "POLICY ASSISTANT: You are an Instamart policy assistant. "
            "Answer only using the supplied policy context. "
            "Do not invent information. "
            "If the answer is unavailable, your response must start "
            "with 'I could not find'."
        )
    }

    # User question and retrieved context
    user_message = {
        "role": "user",
        "content": (
            f"CONTEXT:\n{context}\n"
            f"QUESTION:\n{question}"
        )
    }

    # Call the offline mock LLM
    response = mock_llm.chat(
        messages=[system_message, user_message]
    )

    answer_text = response.choices[0].message.content

    # Return answer and sources
    sources = list(dict.fromkeys(
        chunk["source"] for chunk in retrieved_chunks
    ))

    return {
        "answer": answer_text,
        "sources": sources
    }


# 2. Test the RAG assistant
if __name__ == "__main__":

    question = "Within how many working days are approved refunds credited to the original payment method?"

    result = answer(question)

    print("\nUSER QUESTION:")
    print(question)

    print("\nRAG ASSISTANT ANSWER:")
    print(result["answer"])

    print("\nSOURCES USED:")
    for source in result["sources"]:
        print(source)
