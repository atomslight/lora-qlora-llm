import os
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from colorama import Fore
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import DataCollatorForLanguageModeling
# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset("data", split="train")

print(Fore.YELLOW + str(dataset[0]) + Fore.RESET)

# -----------------------------
# StratifiedShuffleSplit — BEFORE any processing (ML Rule §5a / §6)
# Bin answer length as complexity proxy (strongest predictor of response quality)
# Same concept as binning median_income before housing split
# -----------------------------

df = dataset.to_pandas()

df["answer_length_bin"] = np.digitize(
    df["answer"].str.len(),
    bins=np.percentile(df["answer"].str.len(), [20, 40, 60, 80])
)

print(Fore.CYAN + f"Total samples: {len(df)}" + Fore.RESET)
print(Fore.CYAN + f"Bin distribution:\n{df['answer_length_bin'].value_counts().sort_index()}" + Fore.RESET)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(df, df["answer_length_bin"]))

df_train = df.iloc[train_idx].drop(columns=["answer_length_bin"]).reset_index(drop=True)
df_val   = df.iloc[val_idx].drop(columns=["answer_length_bin"]).reset_index(drop=True)

train_dataset_raw = Dataset.from_pandas(df_train)
val_dataset_raw   = Dataset.from_pandas(df_val)

print(Fore.GREEN + f"Train size: {len(train_dataset_raw)} | Val size: {len(val_dataset_raw)}" + Fore.RESET)

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

# Load the tokenizer from a local directory.
# 'trust_remote_code=True' allows custom code within the tokenizer file to run.
# 'local_files_only=True' prevents the script from checking the internet (Hugging Face Hub).
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    fix_mistral_regex=True
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# LLMs are trained to "end" a sentence (EOS), but many don't have a default "padding" token.
# Padding is required to make all sentences in a batch the same length.
# If it's missing, we reuse the End-Of-Sentence (EOS) token as the Pad token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set the maximum sequence length the model will process (in tokens).
# 1024 is standard for a balance between performance and memory efficiency.
# Any text longer than this will be cut off (truncated).
tokenizer.model_max_length = 1024


# -----------------------------
# Chat formatting
# Tokenizer loaded first — then format applied to both splits consistently
# Critical ML Rule: no leakage, same transform on train and val
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
# Process both splits (ML Rule §5b: remove missing targets)
# -----------------------------

train_dataset = train_dataset_raw.map(
    format_chat,
    batched=True,
    remove_columns=train_dataset_raw.column_names
)

val_dataset = val_dataset_raw.map(
    format_chat,
    batched=True,
    remove_columns=val_dataset_raw.column_names
)

# Remove rows with missing/empty target — equivalent to dropna on target col
train_dataset = train_dataset.filter(lambda x: x["text"] is not None and len(x["text"]) > 0)
val_dataset   = val_dataset.filter(lambda x: x["text"] is not None and len(x["text"]) > 0)

print(Fore.GREEN + f"After cleaning — Train: {len(train_dataset)} | Val: {len(val_dataset)}" + Fore.RESET)

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
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# -----------------------------
# Training config (RTX 4070 safe)
# Added: eval_strategy, eval_steps, load_best_model_at_end
# Enables val loss tracking — prevents overfitting / saving wrong checkpoint
# -----------------------------

training_args = SFTConfig(
    output_dir="Qwen3-4B-QLoRA",
    num_train_epochs=10,

    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=12,
    save_steps=500,

    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    fp16=False,
    bf16=True,

    report_to="none"
)

# -----------------------------
# Trainer — eval_dataset added for val loss
# -----------------------------

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
    data_collator=data_collator,
    dataset_text_field="text",  # <--- ADD THIS
    args=training_args,
    packing=True, #Whether to group multiple sequences into fixed-length blocks to improve computational efficiency and reduce padding. 
    #activation_offloading=True #Only use when Cuda OOM helps save memory but is slow often
)

# -----------------------------
# Train
# -----------------------------

print(Fore.CYAN + "Starting QLoRA fine-tuning..." + Fore.RESET)
trainer.train()

# -----------------------------
# Final evaluation — print val loss for inspection
# -----------------------------

print(Fore.YELLOW + "\nRunning final evaluation on validation set..." + Fore.RESET)
eval_results = trainer.evaluate()
print(Fore.GREEN + f"Final Eval Loss : {eval_results['eval_loss']:.4f}" + Fore.RESET)
print(Fore.GREEN + f"Full eval results: {eval_results}" + Fore.RESET)

# -----------------------------
# Save
# -----------------------------

trainer.save_model("final_lora_adapter")
model.save_pretrained("complete_checkpoint")

print(Fore.GREEN + "Training finished successfully!" + Fore.RESET)
