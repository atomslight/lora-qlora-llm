from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from colorama import Fore 

import json
from typing import List 
from pydantic import BaseModel
from litellm import completion
from generated_prompt import prompt_template

class Record(BaseModel):
    question: str
    answer: str

class Response(BaseModel):
    generated: List[Record]

def llm_call(data: str, num_records: int = 10) -> dict:
    stream = completion(
        model="ollama_chat/qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt_template(data, num_records),
            }
        ],
        stream=True,
        think=False,
        options={"num_predict": 2000},
        format=Response.model_json_schema(),
    )
    response_text = ""
    for x in stream: 
        delta = x['choices'][0]["delta"]["content"]
        if delta is not None: 
            print(Fore.LIGHTBLUE_EX+ delta + Fore.RESET, end="") 
            response_text += delta 
    return json.loads(response_text)

if __name__ == "__main__": 
    converter = DocumentConverter()
    doc = converter.convert("jain-2025-ijca-924786.pdf").document
    chunker = HybridChunker()
    chunks = chunker.chunk(dl_doc=doc)

    dataset = {}
    for i, chunk in enumerate(chunks): 
            print(Fore.YELLOW + f"Raw Text:\n{chunk.text[:300]}…" + Fore.RESET)
            enriched_text = chunker.contextualize(chunk=chunk)
            print(Fore.LIGHTMAGENTA_EX + f"Contextualized Tex:\n{enriched_text[:300]}…" + Fore.RESET)

            data = llm_call(
                enriched_text
            )
            dataset[i] = {"generated":data["generated"], "context":enriched_text}
    
    with open('tm1data.json','w') as f: 
        json.dump(dataset, f) 






