import chromadb
import json
from pathlib import Path
from src.txt_processing import embed_open_ai

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


chroma_client = chromadb.PersistentClient(path=DATA_DIR / "chroma_db")

collection = chroma_client.get_or_create_collection(
    name="surgery_text_collection",
    metadata={"hnsw:space": "cosine"}
)


def add_embeddings_to_collection():
    with open(DATA_DIR / "c_chunks_with_embeddings.json", "r", encoding="utf-8") as f:
        embeddings = json.load(f)

    collection.add(
        embeddings = [c['embedding'] for c in embeddings],
        documents = [c['text'] for c in embeddings],
        ids = [f"chunk_{i}" for i in range(len(embeddings))]
    )

def retrieve(query, n_results = 5):
    query_embedding = embed_open_ai(query)
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = n_results
    )
    return results['documents'][0]