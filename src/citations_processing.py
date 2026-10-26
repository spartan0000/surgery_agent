import pandas as pd
import json
import time
from pathlib import Path
from openai import OpenAI
import chromadb
import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CITATIONS_DIR = BASE_DIR / "data" / "c_surgical_citations" / "citation-export.csv"
CHROMA_PATH = BASE_DIR / "data" / "chroma_db"

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path = CHROMA_PATH)

df = pd.read_csv(CITATIONS_DIR)

#drop rows without an abstract
df = df[df['Abstract'].notna()].reset_index(drop=True)

def process_cochrane(row):
    return {
        "text": str(row['Abstract']), #the actual text that gets the embedding
        "metadata": { 
            "title": str(row.get('Title', '')),
            "year": str(row.get('Year', '')),
            "doi": str(row.get('DOI', '')),
            "authors": str(row.get('Author(s)', '')),
            "keywords": str(row.get('Keywords', '')),
            #"review_id": str(row.get('Cochrane Review ID', '')) #use this as the ids in the vector database
        }
    }

def get_citations_collection():
    collection = chroma_client.get_or_create_collection(
        name = "cochrane_reviews",
        metadata={"hnsw:space": "cosine"}
    )                    
    return collection


collection = get_citations_collection()


def embed_cochrane():
    batch_size = 50
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        processed = [process_cochrane(row) for _, row in batch.iterrows()]
        texts = [p['text'] for p in processed]
        metadatas = [p['metadata'] for p in processed]
        response = openai_client.embeddings.create(
            model = "text-embedding-3-small",
            input = texts
        )

        embeddings = [r.embedding for r in response.data]

        ids = [str(row['Cochrane Review ID']) for _, row in batch.iterrows()]

        

        collection.add(
            embeddings = embeddings,
            documents = texts,
            ids = ids,
            metadatas = metadatas
        )

        if i % 500 == 0:
            print(f"{i} / {len(df)} embedded")
        time.sleep(1)

    print(f"Done - {collection.count()} reviews embedded")


if __name__ == "__main__":
    embed_cochrane()
