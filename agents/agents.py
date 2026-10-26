from anthropic import Anthropic, beta_tool
import json
import os
import chromadb
from openai import OpenAI
from pathlib import Path
from src.chroma_db_scripts import get_surgery_text_collection

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


claude = Anthropic(
    api_key = os.getenv("ANTHROPIC_API_KEY")
)
openai_client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path = BASE_DIR / "data" / "chroma_db")
collection = get_surgery_text_collection()

### Retrieval functions
def retrieve_surgical_text(query, n_results = 5):
    embeddings = openai_client.embeddings.create(
        model = "text-embedding-3-small",
        input = query
    ).data[0].embedding

    results = collection.query(
        query_embeddings = embeddings,
        n_results = n_results
    )
    return results['documents'][0]

def retrieve_cochrane(query, n_results = 5):
    embeddings = openai_client.embeddings.create(
        model = 'text-embedding-3-small',
        input = query
    ).data[0].embedding

    results = collection.query(
        query_embeddings = embeddings,
        n_results = n_results,
        include = ['documents', 'metadatas']
    )

    return list(zip(results['documents'][0], results['metadatas'][0]))
    

### Tools
@beta_tool
def search_surgical_textbook(query:str) -> str:
    """Search the surgical textbook for information about surgical procedures, anatomy, pathophysiology
    diagnostic tools or tests, indications for intervention, technique, recovery, and possible complications.
    Use this for specific surgical knowledge questions"""
    print(f"[TEXTBOOK TOOL] query: {query}")

    chunks = retrieve_surgical_text(query)
    
    print(f"[TEXTBOOK TOOL] returned {len(chunks)} chunks")

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

@beta_tool
def search_cochrane_citations(query: str) -> str:
    """Search Cochrane systematic reviews for high quality synthesized clinical evidence.
    Use for questions about treatment effectiveness, comparative outcomes, evidenced based treatment options,
    evidence based diagnostic studies"""
    print(f"[COCHRANE TOOL] query: {query}")

    results = retrieve_cochrane(query)

    print(f"[COCHRANE TOOL] returned {len(results)} chunks")

    return json.dumps({'results': results, 'source': 'cochrane'})






### Agent

SYSTEM = """You are a helpful assistant with access to a surgical textbook.
Rules:
1. Always search the textbook before answering surgical questions.
2. Base your answers on what the textbook says, not on your own knowledge.
3. If the textbook doesn't contain information on the query, say, I could not find information on that specific topic in the surgical textbook
4. Do not provide medical advice or recommendations
5. Always recommend expert clinical judgement and consulting a qualified healthcare professional for any patient decisions"""

def ask(question: str) -> str:
    runner = claude.beta.messages.tool_runner(
        model = "claude-sonnet-4-6",
        max_tokens = 2048,
        system = SYSTEM,
        tools = [search_surgical_textbook, calculate_bmi, search_cochrane_citations],
        messages = [{'role': 'user', 'content': question}]
    )
    response = runner.until_done()
    return response.content[0].text




###
### Testing a conversation functionality
class SurgicalChatbot:
    def __init__(self):
        self.messages = []
    
    def ask(self, question: str) -> str:
        # Add user message to history
        self.messages.append({
            "role": "user",
            "content": question
        })
        
        runner = claude.beta.messages.tool_runner(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM,
            tools=[search_surgical_textbook, search_cochrane_citations, calculate_bmi],
            messages=self.messages
        )
        
        result = runner.until_done()
        answer = result.content[0].text
        
        #Add assistant response to history
        self.messages.append({
            "role": "assistant",
            "content": answer
        })
        
        return answer
    
    def reset(self):
        """Start a new conversation"""
        self.messages = []
        print("Conversation reset")
    
    def history(self):
        """Print conversation so far"""
        for msg in self.messages:
            role = "You" if msg["role"] == "user" else "Assistant"
            print(f"\n{role}: {msg['content'][:200]}...")








if __name__ == "__main__":
    res = ask("what is the best treatment option for recurrent melanoma")
    print(res) #pulling the text response so that we can use the raw response below
                                #will have to change this at some point

    