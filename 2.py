import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

# Update this path to your MERGED model folder
MERGED_MODEL_PATH = "./merged_model" 

# Use 4-bit to keep it fast and lightweight
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print(f"Loading merged model from {MERGED_MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(
    MERGED_MODEL_PATH,
    trust_remote_code=True
)

# Set pad_token if not defined (common in Qwen)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

SYSTEM_PROMPT = """You are a helpful, honest and harmless assistant designed to help engineers.
Think through each question logically and provide clear answers.
If you don't know something, say so honestly."""

def test_model(question):
    """Test the merged model with a question."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    # Apply ChatML template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Extract only the assistant's new tokens
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

# Interactive mode
print("\n" + "="*70)
print("Merged Fine-tuned Model Ready! Type 'quit' to exit")
print("="*70 + "\n")

while True:
    question = input("You: ")
    if question.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if not question.strip():
        continue
        
    response = test_model(question)
    print(f"\nAssistant: {response}\n")
    print("-" * 30)
