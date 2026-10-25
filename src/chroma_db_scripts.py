import chromadb
import json
from pathlib import Path
from src.txt_processing import embed_openai

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"



def get_collection():
    chroma_client = chromadb.PersistentClient(path=DATA_DIR / "chroma_db")
    collection = chroma_client.get_collection("surgery_text_collection")

    return collection

def retrieve(query, n_results = 5):
    collection = get_collection()
    query_embedding = embed_openai(query)
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = n_results
    )
    return results['documents'][0]


def test_retrieval(query, n_results = 5):
    collection = get_collection()
    query_embedding = embed_openai(query)
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = n_results
    )

    return results['documents'][0]

if __name__ == "__main__":
    results = test_retrieval("what is the screening protocol for someone with esophageal varices")
    for i, chunk in enumerate(results):
        print(f"Result {i+1}: {chunk[:200]}\n")