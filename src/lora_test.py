from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL_NAME = "distilgpt2"

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("Original Parameters")

total = sum(p.numel() for p in model.parameters())
print(f"Total Parameters : {total:,}")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

print("\nAfter Applying LoRA\n")

model.print_trainable_parameters()