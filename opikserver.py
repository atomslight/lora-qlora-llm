import torch
import time
import uvicorn
from fastapi import FastAPI, Request, APIRouter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from fastapi import Request, Response # Add this to your imports
MODEL_PATH = "./merged_model"
app = FastAPI()
@app.middleware("http")
async def add_ngrok_bypass_header(request: Request, call_next):
    response = await call_next(request)
    # This tells ngrok (and Opik) that the request is safe to process
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response
# 1. Create a Router specifically for /v1
v1_router = APIRouter(prefix="/v1")

print("Loading model... (Initial setup)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True,fix_mistral_regex=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map="auto",
    trust_remote_code=True
)

# --- V1 ROUTES ---

@v1_router.get("") # Handles /v1
@v1_router.get("/") # Handles /v1/
async def v1_base():
    return {"status": "ok", "message": "v1 API is active"}

@v1_router.get("/models") # Handles /v1/models
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "qwen3-4b-finetuned-merged",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local-server"
        }]
    }

@v1_router.post("/chat/completions") # Handles /v1/chat/completions
async def chat_completions(request: Request):
    data = await request.json()
    prompt = tokenizer.apply_chat_template(data.get("messages", []), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=data.get("max_tokens", 256))
    
    response_text = tokenizer.decode(output_ids[inputs.input_ids.shape:], skip_special_tokens=True)
    return {
        "choices": [{"message": {"role": "assistant", "content": response_text.strip()}}]
    }

# Include the router in the main app
app.include_router(v1_router)

# Root level check
@app.get("/")
async def root():
    return {"message": "Server is up. Use /v1/models to verify."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
