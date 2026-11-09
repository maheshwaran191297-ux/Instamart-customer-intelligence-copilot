
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from partD_policy_loader import chunks


# STEP 1: Extract text from all policy chunks
chunk_texts = [
    chunk["content"] for chunk in chunks
]


# STEP 2: Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit TF-IDF on policy chunk texts
tfidf_matrix = vectorizer.fit_transform(chunk_texts)


# STEP 3: Define retrieval function
def retrieve(query, k=3):

    # Convert customer query into TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    # Get indexes of top k matching chunks
    top_indices = similarity_scores.argsort()[::-1][:k]

    # Store retrieved results
    results = []

    for idx in top_indices:

        results.append({
            "source": chunks[idx]["source"],
            "score": float(similarity_scores[idx]),
            "text": chunks[idx]["content"]
        })

    return results


# STEP 4: Test retrieval
if __name__ == "__main__":

    question = "refund policy late delivery"

    retrieved_results = retrieve(question, k=3)

    print("\nCustomer Question:", question)

    print("\nTop Retrieved Policy Documents:")

    for result in retrieved_results:

        print("\nSource:", result["source"])
        print("Score:", round(result["score"], 4))
        print("Content:", result["text"])
