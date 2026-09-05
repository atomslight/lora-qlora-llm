import torch
import time
import uvicorn
from fastapi import FastAPI, Request, APIRouter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os

# Updated paths for adapter-based loading
BASE_MODEL = "./local_models/Qwen3-4B"
ADAPTER_PATH = "./final_lora_adapter"

app = FastAPI()

@app.middleware("http")
async def add_ngrok_bypass_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

v1_router = APIRouter(prefix="/v1")

print("Loading base model and LoRA adapter...")

# Create offload directory
os.makedirs("offload", exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    fix_mistral_regex=True
)
if tokenizer.pad_token is None: 
    tokenizer.pad_token = tokenizer.eos_token

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    max_memory={0: "7GB", "cpu": "16GB"}
    
)

# Load LoRA adapter
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    model, 
    ADAPTER_PATH,
    offload_folder="offload",
    offload_state_dict=True
)

print("Model loaded successfully!")

# --- V1 ROUTES ---

@v1_router.get("")
@v1_router.get("/")
async def v1_base():
    return {"status": "ok", "message": "v1 API is active with LoRA adapter"}

@v1_router.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "qwen3-4b-lora-adapter",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local-server",
            "base_model": BASE_MODEL,
            "adapter_path": ADAPTER_PATH
        }]
    }

@v1_router.post("/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        data.get("messages", []), 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=data.get("max_tokens", 256),
            temperature=data.get("temperature", 0.7),
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part (exclude input tokens)
    generated_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "qwen3-4b-lora-adapter",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant", 
                "content": response_text.strip()
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": inputs['input_ids'].shape[1],
            "completion_tokens": len(generated_tokens),
            "total_tokens": inputs['input_ids'].shape[1] + len(generated_tokens)
        }
    }

app.include_router(v1_router)

@app.get("/")
async def root():
    return {
        "message": "LoRA Adapter Server is up", 
        "base_model": BASE_MODEL,
        "adapter": ADAPTER_PATH,
        "endpoints": ["/v1/models", "/v1/chat/completions"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
