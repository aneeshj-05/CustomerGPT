from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "distilgpt2"
ADAPTER_PATH = "./adapters"

# -------------------------------------------------------
# Load Tokenizer
# -------------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
tokenizer.pad_token = tokenizer.eos_token

# -------------------------------------------------------
# Load Base Model
# -------------------------------------------------------

print("Loading Base Model...")

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# -------------------------------------------------------
# Load LoRA Model
# -------------------------------------------------------

print("Loading LoRA Model...")

lora_model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME),
    ADAPTER_PATH
)

prompt = """Customer: My package hasn't arrived.
Support:"""


def generate(model):
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


print("\n" + "=" * 70)
print("BASE MODEL")
print("=" * 70)

print(generate(base_model))

print("\n" + "=" * 70)
print("LORA MODEL")
print("=" * 70)

print(generate(lora_model))