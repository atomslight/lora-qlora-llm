#!/usr/bin/env python
# ----------------------------------------------------
# QLoRA fine-tune Qwen3-4B on a custom QA dataset
# ----------------------------------------------------
import os
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from sklearn.model_selection import StratifiedShuffleSplit

set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
LOCAL_ONLY = True  # flip to False to allow Hub download
BASE_MODEL = "Qwen/Qwen3-4B"
LOCAL_PATH = "./local_models/Qwen3-4B"

# ------------------------------------------------------------------
# 1. Dataset
# ------------------------------------------------------------------
# Replace with your actual loading logic
dataset = load_dataset("json", data_files={"train": "data/train.jsonl"})["train"]

# quick stratified split on answer length
df = dataset.to_pandas()
df["answer_length_bin"] = np.digitize(
    df["answer"].str.len(),
    bins=np.percentile(df["answer"].str.len(), [20, 40, 60, 80]),
)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(df, df["answer_length_bin"]))
train_dataset = Dataset.from_pandas(df.iloc[train_idx].drop(columns=["answer_length_bin"]), preserve_index=False)
val_dataset   = Dataset.from_pandas(df.iloc[val_idx].drop(columns=["answer_length_bin"]), preserve_index=False)

# ------------------------------------------------------------------
# 2. Tokeniser
# ------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_PATH if LOCAL_ONLY else BASE_MODEL,
    trust_remote_code=True,
    local_files_only=LOCAL_ONLY,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # CRITICAL for causal LM
tokenizer.model_max_length = 1024

# ------------------------------------------------------------------
# 3. Chat template
# ------------------------------------------------------------------
SYSTEM_PROMPT = "You are a helpful, honest and harmless assistant designed to help engineers."

def format_chat(batch):
    texts = []
    for q, a in zip(batch["question"], batch["answer"]):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    return {"text": texts}

train_dataset = train_dataset.map(format_chat, batched=True, remove_columns=train_dataset.column_names)
val_dataset   = val_dataset.map(format_chat, batched=True, remove_columns=val_dataset.column_names)

# drop empties
train_dataset = train_dataset.filter(lambda x: x["text"])
val_dataset   = val_dataset.filter(lambda x: x["text"])

# ------------------------------------------------------------------
# 4. Quantised base model
# ------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_PATH if LOCAL_ONLY else BASE_MODEL,
    quantization_config=bnb_config,
    device_map="cuda:0",
    trust_remote_code=True,
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# ------------------------------------------------------------------
# 5. LoRA
# ------------------------------------------------------------------
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# ------------------------------------------------------------------
# 6. Training arguments
# ------------------------------------------------------------------
training_args = SFTConfig(
    output_dir="Qwen3-4B-QLoRA",
    num_train_epochs=14,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=10,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    max_seq_length=512,  # packs short examples → big speed-up
    fp16=False,
    bf16=True,
    report_to="none",  # change to "wandb" if you like
    neftune_noise_alpha=5,  # free regularisation
)
# 3. Logging
report_to="wandb"  # or "tensorboard"
run_name="qwen3-4b-qlora-r32"
# ------------------------------------------------------------------
# 7. Trainer
# ------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
    dataset_text_field="text",
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ------------------------------------------------------------------
# 8. Go
# ------------------------------------------------------------------
trainer.train()
eval_results = trainer.evaluate()
print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
trainer.save_model("final_lora_adapter")  # saves adapter only
