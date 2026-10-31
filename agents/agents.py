from anthropic import Anthropic, beta_tool
import json
import os
import chromadb
from openai import OpenAI
from pathlib import Path
from src.chroma_db_scripts import get_surgery_text_collection, get_cochrane_collection
from agents.agent_tools import tool_definitions
from agents.calculators import run_calculator

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


claude = Anthropic(
    api_key = os.getenv("ANTHROPIC_API_KEY")
)
openai_client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path = BASE_DIR / "data" / "chroma_db")
text_collection = get_surgery_text_collection()
cochrane_collection = get_cochrane_collection()

### Retrieval functions
def retrieve_surgical_text(query, n_results = 5):
    embeddings = openai_client.embeddings.create(
        model = "text-embedding-3-small",
        input = query
    ).data[0].embedding

    results = text_collection.query(
        query_embeddings = embeddings,
        n_results = n_results
    )
    return results['documents'][0]

def retrieve_cochrane(query, n_results = 5):
    embeddings = openai_client.embeddings.create(
        model = 'text-embedding-3-small',
        input = query
    ).data[0].embedding

    results = cochrane_collection.query(
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
def medical_calculator(calculator: str, parameters: dict) -> str:
    """Calculate medical risk scores and measurements.
    
    Available calculators:
    
    bmi: Body mass index
        height (number), weight (number)
        height_unit (cm or inches, default cm)
        weight_unit (kg or pounds, default kg)
    
    meld: MELD score for liver disease severity
        creatinine (mg/dL), bilirubin (mg/dL),
        inr, sodium (optional), dialysis (bool, optional)

    chads_vasc: CHA2DS2-VASc stroke risk score in atrial fibrillation
        Useful for perioperative anticoagulation decisions
        heart_failure (bool), hypertension (bool),
        age (years, integer), diabetes (bool),
        stroke_tia (bool), vascular_disease (bool),
        female (bool)
    """
    print(f"\n[CALCULATOR] {calculator}: {parameters}")
    result = run_calculator(calculator, parameters)
    return json.dumps(result)

@beta_tool
def search_cochrane_citations(query: str) -> str:
    """Search Cochrane systematic reviews for high quality synthesized clinical evidence.
    Use for questions about treatment effectiveness, comparative outcomes, evidenced based treatment options,
    evidence based diagnostic studies"""
    print(f"[COCHRANE TOOL] query: {query}")

    results = retrieve_cochrane(query)

    print(f"[COCHRANE TOOL] returned {len(results)} chunks")

    return json.dumps({'results': results, 'source': 'cochrane'})






### Agents

SYSTEM = """You are a surgical education assistant with access to a surgical textbook and Cochrane systematic reviews.

Knowledge sources:
- Surgical textbook: operative techniques, anatomy, procedures, complications, surgical decision making
- Cochrane reviews: high quality synthesized clinical evidence, treatment outcomes, comparative effectiveness
- A medical calculator to calculate risk scores and other values to help make surgical decisions

Rules:
1. Always search the appropriate source before answering:
   - Surgical questions - search the textbook
   - Evidence or outcomes questions - search Cochrane reviews
   - Questions involving both - search both sources
   - If a numerical score is required and the specific calculator is available, use that calculator
2. Ground your answers in the retrieved content but use your knowledge to explain and synthesize clearly
3. Always cite your source - textbook for operative knowledge, Cochrane review title and year for evidence
4. If neither source contains relevant information, say so clearly
5. Never provide specific clinical advice for individual patients
6. Always recommend clinical judgment and consultation with a qualified surgeon for patient decisions
7. When Cochrane evidence is used, note that systematic reviews represent the highest level of evidence but still require clinical judgment for individual patient application"""


def ask(question: str) -> str:
    runner = claude.beta.messages.tool_runner(
        model = "claude-sonnet-4-6",
        max_tokens = 2048,
        system = SYSTEM,
        tools = [search_surgical_textbook, medical_calculator, search_cochrane_citations],
        messages = [{'role': 'user', 'content': question}]
    )
    response = runner.until_done()
    return response.content[0].text




###
### Testing a conversation functionality - non streaming agent
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
            tools=[search_surgical_textbook, search_cochrane_citations, medical_calculator],
            messages=self.messages
        )
        
        result = runner.until_done()

        full_response = ""
        for block in result.content:
            if hasattr(block, 'text') and block.text:
                full_response += block.text

        print(full_response)
        #Add assistant response to history
        self.messages.append({
            "role": "assistant",
            "content": full_response
        })
        
        return full_response

       
    def reset(self):
        """Start a new conversation"""
        self.messages = []
        print("Conversation reset")
    
    def history(self):
        """Print conversation so far"""
        if not self.messages:
            print("No conversation history yet")
            return
        
        for msg in self.messages:
            role = "You" if msg["role"] == "user" else "Assistant"
            print(f"\n{role}: {msg['content'][:200]}")
            if len(msg['content']) > 200:
                print("...")


###
#Streaming agent - lots more code unfortunately...
#the @beta_tool decorator does a lot under the hood apparently - extracts the schema in tool definitions automatically
#since we're manually running the tools here, need the tool definitions in agent_tools.py so that the 
#anthropic api knows what we're doing.  
#tool runner also executes the tool
#here we have an execute_tool method to manually execute the tools


#this is the function to get human confirmation of calculator in the terminal
#for a browser, will need a different method to do this in an HTTP request - two separate requests with
#stored state in between

NEEDS_CONFIRMATION = {'medical_calculator'}

def request_confirmation(self, tool_input: dict):
    calc = tool_input["calculator"]
    params = dict(tool_input["parameters"])
    print(f"\n[CONFIRM {calc.upper()}]")
    for k, v in params.items():
        print(f"  {k}: {v}")
    choice = input("Confirm? (y / edit / n): ").strip().lower()

    if choice == "y":
        return params
    if choice == "edit":
        for k in params:
            new = input(f"  {k} [{params[k]}]: ").strip()
            if new:
                params[k] = new
        return params
    return None  # cancelled


class SurgicalChatbotStreaming:
    def __init__(self):
        self.messages = []
    
    def ask(self, question: str) -> str:
        self.messages.append({
            "role": "user",
            "content": question
        })
        
        full_response = ""
        
        while True:
            with claude.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=SYSTEM,
                tools=tool_definitions,  # JSON schemas not decorators
                messages=self.messages
            ) as stream:
                
                response = stream.get_final_message()
                
                # Tool call — execute and loop
                if response.stop_reason == "tool_use":
                    self.messages.append({"role": "assistant", "content": response.content})

                    tool_results = []
                    for block in response.content:
                        if block.type != "tool_use":
                            continue

                        print(f"\n[Calling {block.name}...]", flush=True)

                        if block.name == "medical_calculator":
                            confirmed = self.request_confirmation(block.input)
                            if confirmed is None:
                                content = json.dumps({"status": "cancelled_by_user"})
                            else:
                                content = json.dumps(
                                    run_calculator(block.input["calculator"], confirmed)
                                )
                        else:
                            content = execute_tool(block.name, block.input)

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": content,
                        })

                    self.messages.append({"role": "user", "content": tool_results})
                    continue
                
                # Final answer
                if response.stop_reason == "end_turn":
                    for block in response.content:
                        if hasattr(block, 'text') and block.text:
                            # Stream word by word
                            for char in block.text:
                                print(char, end="", flush=True)
                            full_response += block.text
                    break
        
        print()
        
        self.messages.append({
            "role": "assistant",
            "content": full_response
        })
        
        return full_response
    
    def history(self):
        if not self.messages:
            print("No conversation history yet")
            return
        print("\n=== Conversation History ===")
        for i, msg in enumerate(self.messages):
            role = "You" if msg["role"] == "user" else "Assistant"
            print(f"\n[{i+1}] {role}:")
            content = msg['content']
            if isinstance(content, str):
                print(content[:300])
                if len(content) > 300:
                    print("...")
            else:
                print("[tool calls/results]")
        print("===========================")
    
    def reset(self):
        self.messages = []
        print("Conversation reset")
    
    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.messages, f, indent=2, default=str)
        print(f"Saved to {filepath}")
    
    def load(self, filepath: str):
        with open(filepath) as f:
            self.messages = json.load(f)
        print(f"Loaded {len(self.messages)} messages")


def execute_tool(name: str, args: dict) -> str:
    if name == "search_surgical_textbook":
        print(f"[TEXTBOOK] {args['query']}")
        results = retrieve_surgical_text(args["query"])
        #return {"results": [r[0] for r in results]}
        return results
    
    elif name == "search_cochrane_citations":
        print(f"[COCHRANE] {args['query']}")
        results = retrieve_cochrane(args["query"])
        # return {"results": [{
        #     "text": r[0],
        #     "title": r[1].get("title"),
        #     "year": r[1].get("year"),
        #     "doi": r[1].get("doi")
        # } for r in results]}
        return results
    
    
    return {"error": f"Unknown tool: {name}"}


if __name__ == "__main__":
    bot = SurgicalChatbot()

    print("Surgical assistant ready")
    print("Commands: 'history', 'reset', 'quit' ")

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        elif question.lower() == 'quit':
            break
        elif question.lower() == 'history':
            bot.history()
        elif question.lower() == 'reset':
            bot.reset()
        else:
            print("\nAssistant: ", end="")
            bot.ask(question)

    