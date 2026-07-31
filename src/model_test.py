from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

prompt = "Once upon a time"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
print(outputs)

print("="*60)

print(tokenizer.decode(outputs[0]))