
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# STEP 1: Load policy documents
POLICY_DIR = Path("data/policies")

documents = []

for file_path in POLICY_DIR.glob("*.txt"):
    content = file_path.read_text(encoding="utf-8")

    documents.append({
        "source": file_path.name,
        "content": content
    })


# STEP 2: Create chunks
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

chunks = []

for doc in documents:

    content = doc["content"]
    start = 0

    while start < len(content):

        end = start + CHUNK_SIZE

        chunks.append({
            "source": doc["source"],
            "chunk_id": len(chunks) + 1,
            "content": content[start:end]
        })

        start += CHUNK_SIZE - CHUNK_OVERLAP


# STEP 3: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# STEP 4: Extract text from chunks
texts = [chunk["content"] for chunk in chunks]


# STEP 5: Generate embeddings
embeddings = model.encode(
    texts,
    convert_to_numpy=True
)

print("Embedding shape:", embeddings.shape)


# STEP 6: Convert embeddings to float32
embeddings = np.asarray(
    embeddings,
    dtype="float32"
)


# STEP 7: Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)


# STEP 8: Add embeddings to FAISS
index.add(embeddings)


# STEP 9: Save FAISS index
Path("outputs").mkdir(exist_ok=True)

faiss.write_index(
    index,
    "outputs/policy_index.faiss"
)


# STEP 10: Save chunk metadata
with open("outputs/policy_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=4, ensure_ascii=False)


print("Total documents:", len(documents))
print("Total chunks:", len(chunks))
print("Total vectors:", index.ntotal)

print("FAISS vector database created successfully!")
