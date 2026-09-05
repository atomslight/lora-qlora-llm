from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from typing import List, Optional, Dict, Any
import time
import uuid
from opik import track
import opik

# Initialize FastAPI app
app = FastAPI(title="Fine-tuned LLM API for Opik", version="1.0.0")

# OpenAI-compatible models
class Message(BaseModel):
    role: str
    content: str

class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "fine-tuned"

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

# Global variables
model = None
tokenizer = None
MERGED_MODEL_PATH = "./merged_model"
MODEL_NAME = "fine-tuned-model-v1"  # Your model identifier

SYSTEM_PROMPT = """You are a helpful, honest and harmless assistant designed to help engineers.
Think through each question logically and provide clear answers.
If you don't know something, say so honestly."""

def setup_opik():
    """Setup Opik configuration automatically"""
    try:
        opik.configure(use_local=True)
        print("✅ Opik configured successfully for local use")
        return True
    except Exception as e:
        print(f"❌ Failed to setup Opik: {e}")
        return False

@track(project_name="fine-tuned-llm-server")
def generate_response(messages: List[dict], temperature: float = 0.1, max_tokens: int = 512, top_p: float = 0.9) -> tuple[str, int, int]:
    """Generate response using the fine-tuned model with Opik tracking"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format messages for the model
        formatted_prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                formatted_prompt += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                formatted_prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted_prompt += f"Assistant: {msg['content']}\n"
        
        formatted_prompt += "Assistant: "
        
        # Tokenize input
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        prompt_tokens = inputs.input_ids.shape[1]
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        completion_tokens = outputs[0].shape[0] - prompt_tokens
        
        return response, prompt_tokens, completion_tokens
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize everything on startup"""
    global model, tokenizer
    
    print("🚀 Starting Fine-tuned LLM API for Opik Provider Integration...")
    
    # Setup Opik
    setup_opik()
    
    # Load model
    print(f"📦 Loading model from {MERGED_MODEL_PATH}...")
    
    if not os.path.exists(MERGED_MODEL_PATH):
        print(f"❌ Model path {MERGED_MODEL_PATH} does not exist!")
        print("Please ensure your merged model is in the correct location.")
        return
    
    try:
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MERGED_MODEL_PATH,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ Model loaded successfully!")
        print("🌐 Server ready at: http://localhost:8000")
        print("📊 Opik dashboard: http://localhost:5173 (if running locally)")
        print("\n📋 OpenAI-compatible endpoints:")
        print("  • GET /v1/models - List available models")
        print("  • POST /v1/chat/completions - Chat completions")
        print("  • GET /health - Health check")
        print("\n🔧 To add to Opik:")
        print("  1. Go to Workspace Settings > AI Providers")
        print("  2. Click 'Add Configuration'")
        print("  3. Select 'vLLM / Custom provider'")
        print("  4. URL: http://localhost:8000/v1")
        print(f"  5. Models: {MODEL_NAME}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Fine-tuned LLM API", "version": "1.0.0"}

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

# OpenAI-compatible endpoints
@app.get("/v1")
async def v1_root():
    """V1 API root - required for Opik discovery"""
    return {
        "message": "Fine-tuned LLM API v1",
        "endpoints": ["/v1/models", "/v1/chat/completions"]
    }

@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models - required for Opik model discovery"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelsResponse(
        data=[
            ModelInfo(
                id=MODEL_NAME,
                created=int(time.time()),
                owned_by="fine-tuned"
            )
        ]
    )

@app.post("/v1/chat/completions", response_model=OpenAIChatResponse)
async def chat_completions(request: OpenAIChatRequest):
    """OpenAI-compatible chat completions endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert Pydantic models to dicts
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    # Add system prompt if not present
    if not any(msg["role"] == "system" for msg in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    
    try:
        response_text, prompt_tokens, completion_tokens = generate_response(
            messages=messages,
            temperature=request.temperature or 0.1,
            max_tokens=request.max_tokens or 512,
            top_p=request.top_p or 0.9
        )
        
        return OpenAIChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
