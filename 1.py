import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

BASE_MODEL = "./local_models/Qwen3-4B"
ADAPTER_PATH = "./final_lora_adapter"

# Create offload directory
os.makedirs("offload", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    #fix_mistral_regex=True
)

# 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    max_memory={0: "7GB", "cpu": "16GB"}
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    model, 
    ADAPTER_PATH,
    offload_folder="offload",
    offload_state_dict=True
)

SYSTEM_PROMPT = """You are a helpful, honest and harmless assistant designed to help engineers.
Think through each question logically and provide clear answers.
If you don't know something, say so honestly.
"""

def test_model(question):
    """Test the fine-tuned model with a question."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the assistant's response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

# Interactive mode
print("\n" + "="*70)
print("Fine-tuned Model Ready! Type 'quit' to exit")
print("="*70 + "\n")

while True:
    question = input("You: ")
    if question.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    response = test_model(question)
    print(f"Assistant: {response}\n")
