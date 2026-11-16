import os
from dotenv import load_dotenv
import chromadb

load_dotenv()

client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

collection = client.get_or_create_collection("getting-started")

documents = [
    "Chroma is a vector database for building AI applications.",
    "Vector databases store embeddings and allow similarity search.",
    "Python is a popular language for machine learning and data science.",
    "Embeddings are numerical representations of text, images, or other data.",
    "Retrieval-augmented generation (RAG) uses vector search to ground LLM responses.",
]
ids = [f"doc{i}" for i in range(len(documents))]

collection.upsert(ids=ids, documents=documents)

print("=== Ingested Documents ===")
for doc_id, doc in zip(ids, documents):
    print(f"  [{doc_id}] {doc}")

query = "How does semantic search work with embeddings?"
print(f"\n=== Query ===")
print(f"  {query}")

results = collection.query(query_texts=[query], n_results=3)

print("\n=== Results ===")
for rank, (doc, distance) in enumerate(
    zip(results["documents"][0], results["distances"][0]), start=1
):
    print(f"  {rank}. (distance={distance:.4f}) {doc}")
