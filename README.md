# 🤖 CustomerGPT

A Parameter-Efficient Fine-Tuning (PEFT) project that adapts **DistilGPT-2** into a customer support assistant using **LoRA (Low-Rank Adaptation)**.

Instead of fine-tuning all **82 million** model parameters, this project trains only **0.18%** of the parameters by attaching lightweight LoRA adapters, making LLM fine-tuning feasible even on a CPU.

---

## 📌 Project Overview

Large Language Models are expensive to fine-tune because every parameter must be updated during training.

This project demonstrates how **LoRA (Low-Rank Adaptation)** can efficiently fine-tune a pretrained language model while keeping the original model weights frozen.

The model is trained on a customer support dataset from Hugging Face and learns to generate customer support style responses using only lightweight adapter weights.

---

## ✨ Features

- Fine-tuning using **LoRA (PEFT)**
- Customer support instruction dataset
- CPU-compatible training pipeline
- Adapter-based model saving
- Separate inference pipeline
- Base Model vs LoRA Model comparison
- Hugging Face Transformers integration

---

## 🏗 Project Structure

```
CustomerGPT/

├── adapters/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
│
├── outputs/
│
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── compare.py
│   ├── dataset.py
│   ├── preprocess.py
│   ├── tokenize_dataset.py
│   ├── model_test.py
│   └── lora_test.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙ Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- LoRA
- TRL
- Accelerate

---

## 📚 Dataset

Dataset:

**Bitext Customer Support LLM Chatbot Training Dataset**

https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

The dataset contains thousands of customer support instruction-response pairs covering topics such as:

- Order cancellation
- Refund requests
- Damaged products
- Delivery issues
- Account management

---

## 🧠 Model

Base Model

```
DistilGPT2
```

LoRA Configuration

| Parameter | Value |
|-----------|------:|
| Rank (r) | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target Module | c_attn |

Only **147,456** parameters are trained.

Total model parameters:

```
82,060,032
```

Trainable parameters:

```
147,456
```

Trainable Percentage

```
0.1797%
```

---

## 🚀 Training

Run

```bash
python src/train.py
```

The pipeline performs:

- Load Dataset
- Prompt Formatting
- Tokenization
- Load DistilGPT2
- Inject LoRA Adapters
- Fine-tuning
- Save Adapter

Training was performed on CPU.

---

## 🔍 Inference

Run

```bash
python src/inference.py
```

Example

Input

```
Customer:
My package hasn't arrived.

Support:
```

Output

```
I'm sorry for the inconvenience...

...
```

---

## 📊 Model Comparison

Run

```bash
python src/compare.py
```

This compares responses from:

- Base DistilGPT2
- LoRA Fine-tuned Model

allowing qualitative evaluation of the fine-tuning process.

---

## 📈 Training Results

Training Configuration

| Parameter | Value |
|-----------|------:|
| Epochs | 1 |
| Batch Size | 1 |
| Learning Rate | 2e-4 |
| Training Samples | 1000 |

Final Training Loss

```
1.78
```

Training Runtime

```
28 minutes
```

Hardware

```
CPU
```

---

## 💡 Key Learning Outcomes

This project demonstrates:

- Parameter Efficient Fine Tuning (PEFT)
- LoRA Adapter Training
- Hugging Face Transformers
- Dataset Preprocessing
- Tokenization Pipeline
- Model Saving & Loading
- LLM Inference
- Model Evaluation

---

## 🛠 Installation

Clone the repository

```bash
git clone https://github.com/aneeshj-05/CustomerGPT.git
```

Create virtual environment

```bash
python3 -m venv venv
```

Activate

Linux/macOS

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Future Improvements

- Train on the complete dataset
- Response-only loss masking
- Validation dataset
- Better prompt templates
- Larger instruction-tuned base models
- Quantized training
- Hugging Face Hub deployment

---

## Author

**Aneesh Jantikar**

GitHub

https://github.com/aneeshj-05

LinkedIn

https://linkedin.com/in/aneeshjantikar

---
