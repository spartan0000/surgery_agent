#script for text processing
from pathlib import Path
import json
from openai import OpenAI
import time
import os

from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def combine_text_files():

    combined = ""

    for txt_file in Path("data/c_surgical_text").glob("*.txt"):
        chapter_name = txt_file.stem
        text = txt_file.read_text(encoding="latin-1")
        combined += f"\n\n{chapter_name}\n{text}"

    with open("data/c_combined_text.txt", "w", encoding = "utf-8") as f:
        f.write(combined)


    print(f"Total words: {len(combined.split())}")


def chunk_text_file(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        chunk_words = words[start:start + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append({
            "text": chunk,
            "start_index": start, 
            "end_index": start + len(chunk_words)
        }
    )
        start += chunk_size - overlap    
    with open("data/c_chunks.json", "w", encoding = "utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunk size: {sum(len(chunk['text'].split()) for chunk in chunks) / len(chunks)} words")
    print(f"\nSample chunk: {chunks[100]['text']}")


#used one time to get embeddings for the entire text corpus
def embed_text_corpus():
    with open("data/c_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")


    for i, chunk in enumerate(chunks):
            
        response = client.embeddings.create(
            model = "text-embedding-3-small",
            input = chunk["text"]
        )
        chunk["embedding"] = response.data[0].embedding

        if i % 100 == 0:
            print(f"Processed {i} / {len(chunks)} chunks")
            time.sleep(1)

    with open("data/c_chunks_with_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print("Done with embeddings")


#for reuse when getting embeddings for queries

def embed_openai(text):
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = text
    )
    return response.data[0].embedding

if __name__ == "__main__":
    text = Path("data/c_combined_text.txt").read_text(encoding = "latin-1")
    #chunk_text_file(text)
    