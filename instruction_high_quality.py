from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import re
import json
from typing import List
from pydantic import BaseModel
from litellm import completion
from generated_prompt import (
    prompt_temp1, prompt_temp2, prompt_temp3, 
    prompt_temp4, prompt_temp5, prompt_temp6
)
from colorama import Fore
from ollama import chat
import random
PROMPTS = (prompt_temp1, prompt_temp2, prompt_temp3, prompt_temp4, prompt_temp5, prompt_temp6)
class Record(BaseModel):
    question: str
    answer: str

class Response(BaseModel):
    generated: List[Record]

def llm_call(text: str, num_records: int = 15):
    
    prompt_template = random.choice(PROMPTS)
    stream = chat(
        model='glm-5:cloud',
        messages=[{"role": "user", "content": prompt_template(text, num_records)}],
        format=Response.model_json_schema(),
        stream=True,
        options={"num_predict": 2000, "temperature": 0},
        think=False,
    )

    raw_output = ""

    for chunk in stream:
        if 'message' in chunk and chunk['message'].get('content'):
            raw_output += chunk['message']['content']

    # ---------------- CLEAN OUTPUT ----------------
    clean = raw_output.strip()

    # remove markdown fences
    if clean.startswith("```"):
        clean = re.sub(r'^```json\s*|```$', '', clean, flags=re.MULTILINE).strip()

    if not clean:
        raise ValueError("Model returned empty response")

    # ---------------- PARSE SAFELY ----------------
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        print(Fore.RED + "\n⚠ RAW OUTPUT RECEIVED:\n" + Fore.RESET)
        print(raw_output)
        raise ValueError(f"Invalid JSON returned: {e}")

if __name__ == "__main__":
    converter = DocumentConverter()
    doc = converter.convert("jain-2025-ijca-924786.pdf").document
    chunker = HybridChunker()
    chunks = chunker.chunk(dl_doc=doc)
    chunks = list(chunks)
    dataset = {}

    print(Fore.CYAN + f"Total chunks to process: {len(chunks)}" + Fore.RESET)

    for i, chunk in enumerate(chunks):
        print(Fore.YELLOW + f"\n[Chunk {i+1}/{len(chunks)}] Processing..." + Fore.RESET)
        enriched_text = chunker.contextualize(chunk=chunk)
        
        try:
            data = llm_call(enriched_text)
            # Check if 'generated' key exists in the returned dictionary
            if isinstance(data, dict) and "generated" in data:
                dataset[i] = {
                    "generated": data["generated"],
                    "context": enriched_text
                }
                print(Fore.GREEN + f"[Chunk {i+1}/{len(chunks)}] Done — {len(data['generated'])} records generated" + Fore.RESET)
            else:
                # If the LLM returned a list directly instead of a dict
                if isinstance(data, list):
                    dataset[i] = {"generated": data, "context": enriched_text}
                    print(Fore.GREEN + f"[Chunk {i+1}] Done (List detected)" + Fore.RESET)
                else:
                    print(Fore.RED + f"[Chunk {i+1}] Failed — Unexpected data format" + Fore.RESET)
                    
        except Exception as e:
            print(Fore.RED + f"[Chunk {i+1}/{len(chunks)}] Failed — {e}" + Fore.RESET)
            continue
    with open('train_best.json','w') as f: 
        json.dump(dataset, f) 
