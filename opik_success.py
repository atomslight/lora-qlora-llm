from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import subprocess
import sys
from typing import List, Optional
import time
import uuid
from opik import track

# Initialize FastAPI app
app = FastAPI(title="Fine-tuned LLM API with Opik", version="1.0.0")

# Request/Response models for custom endpoint
class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.1
    max_tokens: int = 512
    top_p: float = 0.9

class ChatResponse(BaseModel):
    response: str
    model_name: str

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

# Global variables
model = None
tokenizer = None
MERGED_MODEL_PATH = "./merged_model"
SYSTEM_PROMPT = """You are a helpful, honest and harmless assistant designed to help engineers.
Think through each question logically and provide clear answers.
If you don't know something, say so honestly."""

def setup_opik():
    """Setup Opik configuration automatically"""
    try:
        # Install opik if not already installed
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opik"])
        
        # Configure Opik to use local instance
        import opik
        opik.configure(use_local=True)
        
        print("✅ Opik configured successfully for local use")
        return True
    except Exception as e:
        print(f"❌ Failed to setup Opik: {e}")
        print("Please run 'pip install opik && opik configure --use_local' manually")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize everything on startup"""
    global model, tokenizer
    
    print("🚀 Starting Fine-tuned LLM API with Opik integration...")
    
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
        print("\n📋 Available endpoints:")
        print("  • POST /chat - Custom chat endpoint")
        print("  • POST /v1/chat/completions - OpenAI-compatible endpoint")
        print("  • GET /health - Health check")
        print("  • GET /docs - API documentation")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None
        tokenizer = None

@track(project_name="fine-tuned-llm-api")
def generate_response(question: str, temperature: float = 0.1, max_tokens: int = 512, top_p: float = 0.9):
    """Generate response with Opik tracking"""
    if model is None or tokenizer is None:
        raise Exception("Model not loaded")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Extract new tokens only
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Custom chat endpoint with Opik tracking"""
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
        
        response = generate_response(
            question=request.message,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p
        )
        
        return ChatResponse(
            response=response,
            model_name="fine-tuned-model"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions", response_model=OpenAIChatResponse)
async def openai_chat_completions(request: OpenAIChatRequest):
    """OpenAI-compatible endpoint with Opik tracking"""
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
        
        # Extract user message
        user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        response = generate_response(
            question=user_message,
            temperature=request.temperature or 0.1,
            max_tokens=request.max_tokens or 512,
            top_p=request.top_p or 0.9
        )
        
        return OpenAIChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=response),
                    finish_reason="stop"
                )
            ]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "opik_configured": True
    }

@app.get("/")
async def root():
    """Root endpoint with usage instructions"""
    return {
        "message": "Fine-tuned LLM API with Opik Integration",
        "endpoints": {
            "chat": "POST /chat - Custom format",
            "openai": "POST /v1/chat/completions - OpenAI compatible",
            "health": "GET /health - Health check",
            "docs": "GET /docs - API documentation"
        },
        "opik_dashboard": "http://localhost:5173",
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🔧 Starting Fine-tuned LLM API with Opik integration...")
    print("📋 Make sure you have:")
    print("  1. Your model in ./merged_model directory")
    print("  2. Opik running locally (./opik.sh)")
    print("  3. Required packages installed")
    print("\n🚀 Starting server...")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
