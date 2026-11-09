
import sys
import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Locate project root
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import mock_llm
from partD_policy_loader import documents


# 1. Load evaluation questions
EVAL_FILE = PROJECT_DIR / "data" / "raw" / "policy_eval.csv"

with open(EVAL_FILE, "r", encoding="utf-8") as file:
    questions = list(csv.DictReader(file))


# 2. Create chunks with configurable overlap
def create_chunks(overlap):

    chunk_list = []

    for doc in documents:

        words = doc["content"].split()
        start = 0

        while start < len(words):

            chunk_words = words[start:start + 40]

            chunk_list.append({
                "source": doc["source"],
                "text": " ".join(chunk_words)
            })

            if start + 40 >= len(words):
                break

            start += 40 - overlap

    return chunk_list


# 3. Build TF-IDF index
def build_index(chunks):

    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix


# 4. Retrieve top 3 chunks
def retrieve_chunks(query, chunks, vectorizer, matrix):

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector, matrix
    ).flatten()

    top_indices = scores.argsort()[::-1][:3]

    return [chunks[i] for i in top_indices]


# 5. Generate an answer using offline mock LLM
def generate_answer(question, retrieved_chunks):

    context = "\n".join(
        f"[{chunk['source']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )

    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT: Answer only using the supplied "
                "policy context. Do not invent information. "
                "If the answer is unavailable, start your response "
                "with 'I could not find'."
            )
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n"
                f"QUESTION:\n{question}"
            )
        }
    ]

    response = mock_llm.chat(messages=messages)

    return response.choices[0].message.content


# 6. Evaluate one overlap configuration
def evaluate(overlap):

    chunks = create_chunks(overlap)

    vectorizer, matrix = build_index(chunks)

    document_correct = 0
    fact_correct = 0
    fallback_correct = 0

    answerable_count = 0
    unanswerable_count = 0

    failures = []

    print("\n" + "=" * 60)
    print(f"EVALUATION: CHUNK OVERLAP = {overlap}")
    print("=" * 60)

    for item in questions:

        question = item["question"]
        expected_doc = item["expected_doc"]
        expected_fact = item["expected_fact"]

        retrieved = retrieve_chunks(
            question, chunks, vectorizer, matrix
        )

        answer = generate_answer(question, retrieved)

        retrieved_sources = [
            chunk["source"] for chunk in retrieved
        ]

        print("\nQuestion:", question)
        print("Answer:", answer)

        if expected_doc != "NONE":

            answerable_count += 1

            doc_ok = expected_doc in retrieved_sources

            fact_ok = (
                bool(expected_fact)
                and expected_fact in answer
            )

            if doc_ok:
                document_correct += 1

            if fact_ok:
                fact_correct += 1

            if not doc_ok or not fact_ok:

                failures.append({
                    "question": question,
                    "expected_doc": expected_doc,
                    "expected_fact": expected_fact,
                    "answer": answer,
                    "document_correct": doc_ok,
                    "fact_correct": fact_ok
                })

        else:

            unanswerable_count += 1

            fallback_ok = answer.startswith("I could not find")

            if fallback_ok:
                fallback_correct += 1

            if not fallback_ok:

                failures.append({
                    "question": question,
                    "expected_doc": "NONE",
                    "expected_fact": "",
                    "answer": answer,
                    "document_correct": None,
                    "fact_correct": fallback_ok
                })

    print("\n--- EVALUATION SUMMARY ---")

    print(
        f"Expected document retrieved: "
        f"{document_correct}/{answerable_count}"
    )

    print(
        f"Expected fact found: "
        f"{fact_correct}/{answerable_count}"
    )

    print(
        f"Unanswerable fallback correct: "
        f"{fallback_correct}/{unanswerable_count}"
    )

    print("\n--- FAILED QUESTIONS ---")

    if not failures:
        print("No failures.")
    else:
        for failure in failures:
            print("\nQuestion:", failure["question"])
            print("Expected document:", failure["expected_doc"])
            print("Expected fact:", failure["expected_fact"])
            print("Actual answer:", failure["answer"])

    return {
        "overlap": overlap,
        "document_correct": document_correct,
        "fact_correct": fact_correct,
        "fallback_correct": fallback_correct,
        "failures": failures
    }


# 7. Run both experiments
if __name__ == "__main__":

    overlap_results = evaluate(overlap=10)

    zero_overlap_results = evaluate(overlap=0)

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print("\n10-word overlap:")
    print("Document retrieval:",
          overlap_results["document_correct"], "/ 12")
    print("Expected facts:",
          overlap_results["fact_correct"], "/ 12")
    print("Fallback:",
          overlap_results["fallback_correct"], "/ 2")

    print("\nZero overlap:")
    print("Document retrieval:",
          zero_overlap_results["document_correct"], "/ 12")
    print("Expected facts:",
          zero_overlap_results["fact_correct"], "/ 12")
    print("Fallback:",
          zero_overlap_results["fallback_correct"], "/ 2")
