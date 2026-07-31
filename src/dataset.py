from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train"
)

print(dataset)
print()

print("Number of samples:", len(dataset))
print()

print(dataset[0])