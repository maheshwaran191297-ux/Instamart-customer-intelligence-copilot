
from pathlib import Path

# 1. Define project and policy folder paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
POLICY_DIR = PROJECT_DIR / "data" / "policies"

# 2. Load policy documents
documents = []

for file_path in POLICY_DIR.glob("*.txt"):

    text = file_path.read_text(encoding="utf-8")

    documents.append({
        "source": file_path.name,
        "content": text
    })

print("Total policy documents:", len(documents))


# 3. Define word-based chunking parameters
CHUNK_SIZE = 40
CHUNK_OVERLAP = 10

chunks = []

# 4. Create chunks for each document
for doc in documents:

    words = doc["content"].split()

    start = 0

    while start < len(words):

        # Select 40 words
        chunk_words = words[start:start + CHUNK_SIZE]

        # Convert words back into text
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "source": doc["source"],
            "chunk_id": len(chunks) + 1,
            "content": chunk_text
        })
                # Stop if this chunk already reaches the end
        if start + CHUNK_SIZE >= len(words):
            break

        # Move forward by 30 words (40 - 10 overlap)
        start += CHUNK_SIZE - CHUNK_OVERLAP


# 5. Display chunking results
print("\nTotal chunks created:", len(chunks))

for chunk in chunks:

    print("\nChunk ID:", chunk["chunk_id"])
    print("Source:", chunk["source"])
    print("Word count:", len(chunk["content"].split()))
    print("Content:", chunk["content"])
