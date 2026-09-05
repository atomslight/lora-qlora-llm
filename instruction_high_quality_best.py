from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import re
import json
import os  # <-- Added for checkpointing
from typing import List
from pydantic import BaseModel
from litellm import completion
from generated_prompt import (
    prompt_base, prompt_temp1, prompt_temp2, prompt_temp3, 
    prompt_temp4, prompt_temp5, prompt_temp6
)
from colorama import Fore
from ollama import chat
import random

PROMPTS = (prompt_base, prompt_temp1, prompt_temp2, prompt_temp3, prompt_temp4, prompt_temp5, prompt_temp6)
OUTPUT_FILE = 'train_best.json' # <-- Defined output file as a constant

class Record(BaseModel):
    question: str
    answer: str

class Response(BaseModel):
    generated: List[Record]

def llm_call(prompt_template, text: str, num_records: int = 15):
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
    
    # --- CHECKPOINT LOAD LOGIC ---
    dataset = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                dataset = json.load(f)
            print(Fore.GREEN + f"Loaded checkpoint! Found {len(dataset)} previously processed chunks." + Fore.RESET)
        except json.JSONDecodeError:
            print(Fore.RED + f"Warning: {OUTPUT_FILE} is corrupted or empty. Starting fresh." + Fore.RESET)
            dataset = {}
    
    print(Fore.CYAN + f"Total chunks to process: {len(chunks)}" + Fore.RESET)

    for i, chunk in enumerate(chunks):
        # --- CHECKPOINT SKIP LOGIC ---
        # Note: JSON saves dictionary keys as strings. We check str(i)
        if str(i) in dataset:
            print(Fore.BLUE + f"\n[Chunk {i+1}/{len(chunks)}] Already processed. Skipping..." + Fore.RESET)
            continue
            
        print(Fore.YELLOW + f"\n[Chunk {i+1}/{len(chunks)}] Processing all 7 variations..." + Fore.RESET)
        enriched_text = chunker.contextualize(chunk=chunk)
        
        chunk_aggregated_records = []
        
        for pt_idx, prompt_template in enumerate(PROMPTS):
            try:
                data = llm_call(prompt_template, enriched_text)
                
                if isinstance(data, dict) and "generated" in data:
                    chunk_aggregated_records.extend(data["generated"])
                elif isinstance(data, list):
                    chunk_aggregated_records.extend(data)
                    
            except Exception as e:
                print(Fore.RED + f"  ↳ Prompt variation {pt_idx+1} failed — {e}" + Fore.RESET)
                continue
        
        random.shuffle(chunk_aggregated_records)
        
        if chunk_aggregated_records:
            # Save using a string key to match JSON loading behavior
            dataset[str(i)] = {
                "generated": chunk_aggregated_records,
                "context": enriched_text
            }
            print(Fore.GREEN + f"[Chunk {i+1}/{len(chunks)}] Done — {len(chunk_aggregated_records)} total records generated" + Fore.RESET)
        else:
            print(Fore.RED + f"[Chunk {i+1}] Failed — No records generated from any prompt" + Fore.RESET)

        # --- CHECKPOINT SAVE LOGIC ---
        # We save immediately after every chunk finishes.
        with open(OUTPUT_FILE, 'w') as f: 
            json.dump(dataset, f, indent=4)
