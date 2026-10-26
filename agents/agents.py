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

SYSTEM = """You are a surgical education assistant with access to a surgical textbook and Cochrane systematic reviews.

Knowledge sources:
- Surgical textbook: operative techniques, anatomy, procedures, complications, surgical decision making
- Cochrane reviews: high quality synthesized clinical evidence, treatment outcomes, comparative effectiveness

Rules:
1. Always search the appropriate source before answering:
   - Surgical questions → search the textbook
   - Evidence or outcomes questions → search Cochrane reviews
   - Questions involving both → search both sources
2. Ground your answers in the retrieved content but use your knowledge to explain and synthesize clearly
3. Always cite your source — textbook for operative knowledge, Cochrane review title and year for evidence
4. If neither source contains relevant information, say so clearly
5. Never provide specific clinical advice for individual patients
6. Always recommend clinical judgment and consultation with a qualified surgeon for patient decisions
7. When Cochrane evidence is used, note that systematic reviews represent the highest level of evidence but still require clinical judgment for individual patient application"""


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

    