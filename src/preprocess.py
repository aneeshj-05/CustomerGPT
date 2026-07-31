from datasets import load_dataset

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train"
)


def format_example(example):
    return {
        "text":
            f"Customer: {example['instruction']}\n"
            f"Support: {example['response']}"
    }


formatted_dataset = dataset.map(format_example)

print(formatted_dataset[0]["text"])