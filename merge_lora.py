from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = r"E:\Ayyub_git\EndToEndLoRA\local_models\Qwen3-4B"

# load base + tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype="auto",
    device_map="cpu"
)

model = PeftModel.from_pretrained(base_model, "final_lora_adapter")

# merge
model = model.merge_and_unload()

# save BOTH
model.save_pretrained("merged_model_new")
tokenizer.save_pretrained("merged_model_new")   # ✅ CRITICAL
