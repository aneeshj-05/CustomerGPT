from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "distilgpt2"
ADAPTER_PATH = "./adapters"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Model loaded successfully!")

prompt = """Customer: My package hasn't arrived.
Support:"""

inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,
)

print("\n==========================")
print("MODEL RESPONSE")
print("==========================\n")

print(tokenizer.decode(output[0], skip_special_tokens=True))