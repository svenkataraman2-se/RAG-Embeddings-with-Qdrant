from httpcore import stream
import numpy as np
import os
from qdrant_client import QdrantClient, models
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import ollama

# ----------------------------
# CONFIG
# ----------------------------
TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
LLAMA_MODEL = "llama3.1:latest"
TOP_K = 5

INDEX_DIR = r"index"
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.txt")
EMBEDDINGS_PATH = os.path.join(INDEX_DIR, "embeddings.npy")

os.makedirs(INDEX_DIR, exist_ok=True)

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def load_chunks(chunks_path):
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = f.read()
    return [chunk.strip() for chunk in data.split("\n---\n") if chunk.strip()]

def load_embeddings(embeddings_path):
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    return np.load(embeddings_path).astype("float32")

def embed_text(text):
    model = SentenceTransformer(TRANSFORMER_MODEL)
    emb = model.encode(text, convert_to_numpy=True)
    #emb = np.array(emb, dtype="float32")
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm  # normalize for cosine similarity
    return emb

def search_cosine(index, query_emb, top_k=TOP_K):
    query_emb = np.array([query_emb], dtype="float32")
    D, I = index.search(query_emb, top_k)
    cosine_sim = 1 - D  # approximate cosine similarity
    return cosine_sim[0], I[0]

def main():
    user_input = input("Enter your query: ").strip()
    print("🧠 Embedding the input query...")
    query_emb = embed_text(user_input)
    print("✅ Query embedded." + str(query_emb))

    client = QdrantClient(host="localhost", port=6333)

    results = client.query_points(collection_name="test_collection", query=query_emb.tolist(), 
                        limit=TOP_K, with_payload=True)
    
    # for point in results.points:
    #     print("----" + point.score.__str__() + "----")

    print()
    print(results.points[0].score)

    # print("📂 Loading chunks and embeddings...")
    # chunks = load_chunks(CHUNKS_PATH)
    # embeddings = load_embeddings(EMBEDDINGS_PATH)

    # print(f"🔍 Searching top {TOP_K} similar chunks...")
    # similarities, indices = search_cosine(index, query_emb, top_k=TOP_K)

    # print("✅ Top 5 matches:")
    # for i, idx in enumerate(indices):
    #     print(f"Rank {i+1} | Cosine similarity: {similarities[i]:.4f}")
    #     print(chunks[idx][:200], "...\n---\n")

    # # Select top-1 chunk for LLaMA
    top1_chunk = results.points[0].payload.get("text")
    print(top1_chunk)

    print("🤖 Generating response using LLaMA 3.1...")

    # # Correct Ollama API usage
    messages = [
    {
        "role": "Human Resource assistant",
        "content": "You are a Human Resource assistant. ensure the response is not more 200 words. ensure the response is in bullet points"
    },
    {
        "role": "user",
        "content": (
            f"Use the following text to answer the query:\n\n"
            f"Text: {top1_chunk}\n\n"
            f"Query: {user_input}"
        )
    }
    ]

    stream = response = ollama.chat(LLAMA_MODEL, messages=messages, stream=True)
    content = ''
    thinking = ''
    for response in stream:
        if response.message.thinking:
            print(response.message.thinking, end='', flush=True)
            thinking += response.message.thinking
        elif response.message.content:
            print(response.message.content, end='', flush=True)
    # accumulate the partial content
            content += response.message.content
    print("\n💬 LLaMA 3.1 Response:")
    print(response.message['content'])

if __name__ == "__main__":
    main()
