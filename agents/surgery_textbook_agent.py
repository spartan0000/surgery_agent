from anthropic import Anthropic, beta_tool
import json
import os
import chromadb
from openai import OpenAI
from pathlib import Path
from src.chroma_db_scripts import get_collection

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


claude = Anthropic(
    api_key = os.getenv("ANTHROPIC_API_KEY")
)
openai_client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path = BASE_DIR / "data" / "chroma_db")
collection = get_collection()

def retrieve(query, n_results = 5):
    embeddings = openai_client.embeddings.create(
        model = "text-embedding-3-small",
        input = query
    ).data[0].embedding

    results = collection.query(
        query_embeddings = embeddings,
        n_results = n_results
    )
    return results['documents'][0]

@beta_tool
def search_surgical_textbook(query:str) -> str:
    """Search the surgical textbook for information about surgical procedures, anatomy, pathophysiology
    diagnostic tools or tests, indications for intervention, technique, recovery, and possible complications.
    Use this for specific surgical knowledge questions"""
    chunks = retrieve(query)
    return json.dumps({"results": chunks})

@beta_tool
def calculate_bmi(height_cm:float, weight_kg:float, height_units:str = "cm", weight_units:str = "kg") -> float:
    """calculate BMI from height in centimeters and weight in kilograms.
    Accepts metric (cm, kg) or imperial (inches (in), pounds (lbs))
    Specify height_units as 'cm' or 'inches' and weight_units as 'kg or 'pounds'
    'in' is an abbreviation for 'inches'.  'lb' or 'lbs' is an abbreviation for 'pounds'
    """

    if height_units == 'inches':
        height = height * 2.54
    if weight_units == 'pounds':
        weight = weight / 2.2

    return weight / (height / 100)**2


SYSTEM = """You are a helpful assistant with access to a surgical textbook.
Rules:
1. Always search the textbook before answering surgical questions.
2. Base your answers on what the textbook says, not on your own knowledge.
3. If the textbook doesn't contain information on the query, say, I could not find information on that specific topic in the surgical textbook
4. Do not provide medical advice or recommendations
5. Always recommend expert clinical judgement and consulting a qualified healthcare professional for any patient decisions"""

def ask(question: str) -> str:
    response = claude.beta.messages.tool_runner(
        model = "claude-sonnet-4-6",
        max_tokens = 2048,
        system = SYSTEM,
        tools = [search_surgical_textbook, calculate_bmi],
        messages = [{'role': 'user', 'content': question}]
    )
    return response.content[0].text