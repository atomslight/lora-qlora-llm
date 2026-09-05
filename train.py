import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from colorama import Fore

# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset("data", split="train")

print(Fore.YELLOW + str(dataset[0]) + Fore.RESET)

# -----------------------------
# Chat formatting
# -----------------------------

SYSTEM_PROMPT = """You are a helpful, honest and harmless assistant designed to help engineers.
Think through each question logically and provide clear answers.
If you don't know something, say so honestly.
"""

def format_chat(batch):
    texts = []
    for q, a in zip(batch["question"], batch["answer"]):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

# -----------------------------
# Model paths
# -----------------------------

BASE_MODEL = "Qwen/Qwen3-4B"
LOCAL_PATH = "./local_models/Qwen3-4B"

if not os.path.exists(LOCAL_PATH):
    print("Downloading model first time...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(LOCAL_PATH)
    model.save_pretrained(LOCAL_PATH)
    del model
    torch.cuda.empty_cache()

# -----------------------------
# Load tokenizer
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_PATH,
    trust_remote_code=True,
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Reduce compute (important for 8GB GPU)
tokenizer.model_max_length = 1024

# -----------------------------
# Process dataset
# -----------------------------

train_dataset = dataset.map(
    format_chat,
    batched=True,
    remove_columns=dataset.column_names
)

# -----------------------------
# QLoRA 4-bit config
# -----------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# -----------------------------
# Load quantized model
# -----------------------------

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_PATH,
    quantization_config=bnb_config,
    device_map="cuda:0",
    trust_remote_code=True
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# -----------------------------
# LoRA config
# -----------------------------

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# -----------------------------
# Training config (RTX 4070 safe)
# -----------------------------

training_args = SFTConfig(
    output_dir="Qwen3-4B-QLoRA",
    num_train_epochs=10,

    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,

    logging_steps=10,
    save_steps=500,

    fp16=False,
    bf16=True,

    report_to="none"
)

# -----------------------------
# Trainer
# -----------------------------

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
    args=training_args,
)

# -----------------------------
# Train
# -----------------------------

print(Fore.CYAN + "Starting QLoRA fine-tuning..." + Fore.RESET)
trainer.train()

# -----------------------------
# Save
# -----------------------------

trainer.save_model("final_lora_adapter")
model.save_pretrained("complete_checkpoint")

print(Fore.GREEN + "Training finished successfully!" + Fore.RESET)
