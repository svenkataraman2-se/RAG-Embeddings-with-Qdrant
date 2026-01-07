import fitz
import numpy as np
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import uuid


PDF_PATH = r"datacontext\WorkFromAnyWherePolicy.pdf"
INDEX_DIR = r"index"
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.txt")
TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
MAX_CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

os.makedirs(INDEX_DIR, exist_ok=True)
print("📄 Processing PDF and generating embeddings...")

# Extract text
doc = fitz.open(PDF_PATH)
text = "\n".join([page.get_text("text") for page in doc])

# Paragraph chunking
paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
chunks = []
print("Number of paragraphs extracted:", len(paragraphs))
for para in paragraphs:
    chunks.append(para)
    # if len(para) <= MAX_CHUNK_SIZE:
    #     print("Adding paragraph of size", len(para))
    #     chunks.append(para)
    # else:
    #     start = 0
    #     while start < len(para):
    #         end = start + MAX_CHUNK_SIZE
    #         chunks.append(para[start:end].strip())
    #         start += MAX_CHUNK_SIZE - CHUNK_OVERLAP

# Generate embeddings
# Using a sentence-transformer model to generate the embeddings
model = SentenceTransformer(TRANSFORMER_MODEL)
embeddings = []
for chunk in chunks:
    emb = model.encode(chunk, convert_to_numpy=True)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm  # normalize
    embeddings.append(emb)
embeddings = np.array(embeddings, dtype="float32")


# Save embeddings in Vector DB (Qdrant)
client = QdrantClient(url="http://localhost:6333")
client.recreate_collection(
    collection_name="test_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

points = []
for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    points.append(
        PointStruct(
            id=idx,
            vector=embedding.tolist(), # Convert numpy array to list
            payload={"text": chunk}    # Store the original text as payload
        )
    )

# Example of adding an extra point of vector search data
# conversionInfoStr = "Conversion of AUD dollars into INR rupees is 61.0 as of today"
# points.append(PointStruct(
#             id=len(chunks),
#             vector=model.encode(conversionInfoStr, convert_to_numpy=True).tolist(), # Convert numpy array to list
#             payload={"text": conversionInfoStr}    # Store the original text as payload
#         ))
    # Upsert the points into the collection
operation_info = client.upsert(
    collection_name="test_collection",
    points=points,
)

print(f"Ingested {len(chunks)} chunks into Qdrant collection.")


# Build FAISS index
# dim = embeddings.shape[1]
# index = faiss.IndexFlatL2(dim)
# index.add(embeddings)
# faiss.write_index(index, FAISS_INDEX_PATH)

# Save chunks
with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk.replace("\n", " ") + "\n---\n")