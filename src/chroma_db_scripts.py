import chromadb
import json
from pathlib import Path
from src.txt_processing import embed_openai

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"



def get_surgery_text_collection():
    chroma_client = chromadb.PersistentClient(path=DATA_DIR / "chroma_db")
    collection = chroma_client.get_collection("surgery_text_collection")

    return collection

def get_cochrane_collection():
    chroma_client = chromadb.PersistentClient(path=DATA_DIR / "chroma_db")
    collection = chroma_client.get_collection("cochrane_reviews")
    return collection

def retrieve(query, n_results = 5):
    collection = get_surgery_text_collection()
    query_embedding = embed_openai(query)
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = n_results
    )
    return results['documents'][0]


def test_text_retrieval(query, n_results = 5):
    collection = get_surgery_text_collection()
    query_embedding = embed_openai(query)
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = n_results,
    )

    return results['documents'][0]

def test_cochrane_retrieval(query, n_results=5):
    collection = get_cochrane_collection()
    query_embedding = embed_openai(query)
    results = collection.query(
        query_embeddings = query_embedding,
        n_results = n_results,
        include = ['documents', 'metadatas']
    )

    return list(zip(results["documents"][0], results['metadatas'][0]))

if __name__ == "__main__":
    results = test_cochrane_retrieval("what is the screening protocol for someone with esophageal varices")
    for doc, meta in results:
        print(f"\nTitle: {meta['title']}")
        print(f"Year: {meta['year']}")
        print(f"Abstract: {doc[:200]}...")
    
    
    
    #surgery text retrieval
    #for i, chunk in enumerate(results):
    #    print(f"Result {i+1}: {chunk[:200]}\n")