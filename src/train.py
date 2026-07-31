from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "distilgpt2"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

print("Loading dataset...")

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split="train",
)

dataset = dataset.select(range(1000))


def format_example(example):
    text = (
        f"Customer: {example['instruction']}\n"
        f"Support: {example['response']}"
    )
    return {"text": text}


print("Formatting dataset...")

dataset = dataset.map(format_example)


def tokenize(example):
    encoding = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )

    encoding["labels"] = encoding["input_ids"].copy()

    return encoding


print("Tokenizing dataset...")

tokenized_dataset = dataset.map(tokenize)

tokenized_dataset = tokenized_dataset.remove_columns(
    ["flags", "instruction", "category", "intent", "response", "text"]
)

tokenized_dataset.set_format("torch")

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


print("Applying LoRA...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


training_args = TrainingArguments(
    output_dir="./outputs",
    overwrite_output_dir=True,

    num_train_epochs=1,

    per_device_train_batch_size=1,

    learning_rate=2e-4,

    logging_steps=10,

    save_strategy="epoch",

    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)


print("\nStarting Training...\n")

trainer.train()


print("\nSaving LoRA adapter...\n")

model.save_pretrained("./adapters")
tokenizer.save_pretrained("./adapters")

print("Training Completed Successfully!")