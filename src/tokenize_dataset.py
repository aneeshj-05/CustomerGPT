from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train"
)


def format_example(example):
    text = (
        f"Customer: {example['instruction']}\n"
        f"Support: {example['response']}"
    )
    return {"text": text}


dataset = dataset.map(format_example)


def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )


tokenized_dataset = dataset.map(tokenize)

print(tokenized_dataset[0].keys())

print(tokenized_dataset[0]["input_ids"][:20])