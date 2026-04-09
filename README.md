# 🚀 Fine-Tuning LLaMA 3.2 3B with Unsloth

This project demonstrates efficient fine-tuning of **LLaMA 3.2 3B Instruct** using **Unsloth**, **LoRA**, and **4-bit quantization** for memory-efficient training on limited hardware.

---

## 📌 Overview

This notebook implements a complete pipeline for:

* Loading a quantized LLaMA model
* Applying **LoRA (Low-Rank Adaptation)**
* Formatting conversational datasets
* Fine-tuning using supervised learning
* Running inference on the trained model

The goal is to achieve **fast, low-cost fine-tuning** without requiring high-end GPUs.

---

## 🧠 Model Details

* Base Model: `unsloth/Llama-3.2-3B-Instruct`
* Framework: Unsloth + Hugging Face Transformers
* Quantization: 4-bit (memory efficient)
* Fine-tuning Method: LoRA

---

## ⚙️ Tech Stack

* Python
* PyTorch
* Unsloth
* Transformers
* TRL (SFTTrainer)
* Datasets (Hugging Face)

---

## 🔧 Installation

```bash
pip install unsloth transformers trl datasets
```

---

## 🏗️ Training Pipeline

### 1. Load Model

* Loads quantized model using Unsloth
* Reduces VRAM usage significantly

### 2. Apply LoRA

* Targets key transformer layers:

  * Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  * MLP: `gate_proj`, `up_proj`, `down_proj`

### 3. Dataset Preparation

* Dataset: `mlabonne/FineTome-100k`
* Converted into **ShareGPT format**
* Standardized for chat-based training

### 4. Chat Formatting

* Applies LLaMA 3.1 chat template
* Ensures correct conversation structure

### 5. Tokenization

* Converts formatted conversations into model-readable tokens

### 6. Training

* Trainer: `SFTTrainer`
* Optimized with:

  * Gradient accumulation
  * Small batch sizes
  * Short training steps (demo setup)

---

## 🏋️ Training Configuration

* Batch Size: 2
* Gradient Accumulation: 4
* Max Steps: 60
* Sequence Length: 2048

---

## 💾 Model Saving

After training, the model is saved locally:

```bash
shanns_tuned_model/
```

---

## 🤖 Inference

Example:

```python
prompt = "What are the key principles of investment?"
```

The model:

1. Applies chat template
2. Tokenizes input
3. Generates response

---

## 🔑 Key Concepts Used

* **LoRA** → Efficient fine-tuning without updating full model
* **Quantization (4-bit)** → Reduced memory usage
* **Chat Templates** → Correct input formatting
* **ShareGPT Format** → Structured conversation dataset

---

## 🚀 Why This Project Matters

* Enables fine-tuning on **low-resource machines**
* Demonstrates modern LLM training techniques
* Useful for:

  * Chatbots
  * Domain-specific assistants
  * RAG pipelines






