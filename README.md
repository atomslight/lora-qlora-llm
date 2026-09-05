# 🛠️ Parameter Efficient Fine-Tuning (PEFT) with LoRA & QLoRA
> **Your end-to-end pipeline for generating synthetic data, ensuring quality, and fine-tuning local LLMs.**

## 📖 Overview
This project provides a comprehensive, end-to-end pipeline for fine-tuning Large Language Models (LLMs) on custom data. It solves the complex challenge of domain-specific model training by automating everything from parsing PDFs and generating synthetic Q&A datasets, to scoring the quality of that data, and finally performing parameter-efficient fine-tuning (QLoRA) on a `Qwen3-4B` model. It's designed to be run both locally and seamlessly scaled to cloud instances like RunPod.

## ✨ Features
- **Automated Synthetic Data Generation**: Ingests raw PDF documents and utilizes an LLM to generate high-quality question-and-answer pairs.
- **LLM-as-a-Judge Quality Filtering**: Automatically grades the accuracy and style of synthetic data, discarding low-quality pairs to ensure your fine-tuned model learns only from the best.
- **Memory-Efficient Fine-Tuning**: Uses QLoRA (4-bit quantization) and LoRA adapters, allowing you to train powerful models (like Qwen3-4B) on consumer-grade GPUs (e.g., RTX 4070 with 8GB+ VRAM).
- **RunPod Ready**: Includes configurations and instructions to transition seamlessly from local development to scalable cloud compute on RunPod.

## 🛠 Tech Stack
- **[Python (v3.12+)](https://www.python.org/)**: The core programming language powering the entire pipeline.
- **[uv](https://github.com/astral-sh/uv)**: An extremely fast Python package and project manager used for dependency management and running scripts.
- **[Docling](https://github.com/DS4SD/docling)**: Parses complex PDF documents and chunks them contextually for downstream processing.
- **[LiteLLM](https://github.com/BerriAI/litellm) & [Ollama](https://ollama.com/)**: Used in tandem to call local LLMs (`qwen3:4b`) for both synthetic data generation and quality evaluation.
- **[Hugging Face Ecosystem](https://huggingface.co/) (`transformers`, `datasets`, `peft`, `trl`)**: The backbone for loading the base model, configuring 4-bit quantization (BitsAndBytes), setting up the LoRA adapter, and executing the Supervised Fine-Tuning (SFT).
- **[PyTorch](https://pytorch.org/)**: The underlying deep learning framework handling model tensors and gradients.

## 📋 Prerequisites
Before you begin, ensure your system meets the following requirements:
- **Node.js**: (Optional for UI/Langflow deployments) v18+.
- **Python**: v3.12 or higher.
- **uv**: Installed on your machine. You can install it via pip (`pip install uv`) or by following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).
- **Git**: For cloning the repository.
- **Ollama**: Installed and running locally. You **must** pull the Qwen model beforehand.
  ```bash
  ollama pull qwen3:4b
  ```
- **Hardware**: A CUDA-compatible NVIDIA GPU with at least 8GB of VRAM (e.g., RTX 4070) is highly recommended for local training.

## 🚀 Local Development (Step-by-Step)

Follow these exact steps to set up and run the entire pipeline on your local machine.

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 2. Initialize and Install Dependencies
We use `uv` to manage the project environment cleanly.
```bash
# Initialize the uv project environment
uv init

# Sync and install all required dependencies from pyproject.toml
uv sync
```

### 3. Environment Variables
This project utilizes `.env` files for configuration if needed. If your project has an `.env.example` file (or requires API keys depending on the litellm backend you use):
```bash
cp .env.example .env
```
*(If you are exclusively using local models via Ollama, no additional API keys are required.)*

### 4. Generate Synthetic Data
Ensure you have a source document (e.g., `jain-2025-ijca-924786.pdf`) in your root directory. This script reads the PDF and generates raw question/answer pairs.
```bash
uv run syntheticdatageneration.py
```
*Note: This utilizes Ollama running locally. Ensure `ollama serve` is running in a separate terminal if it doesn't start automatically.*

### 5. Process and Format the Data
This step converts the generated raw data into a clean, unified format suitable for training.
```bash
uv run preprocessing.py
```

### 6. Filter for Data Quality
This script uses your local LLM to grade every generated Q&A pair. Only the highest-quality pairs (scoring 6 or higher in both accuracy and style) are kept for training.
```bash
uv run dataquality.py
```

### 7. Train the Model
Now, fine-tune the `Qwen3-4B` model using the filtered dataset!
```bash
uv run train.py
```
*Once completed, your new LoRA adapters will be saved in the `final_lora_adapter` directory.*

---

### ☁️ Cloud Deployment (RunPod)
If you prefer to train on the cloud (e.g., RunPod) for faster iteration or due to hardware constraints:

1. SSH into your RunPod instance.
2. Install `uv`:
   ```bash
   pip install uv
   ```
3. Initialize the project:
   ```bash
   uv init
   ```
4. Copy the specific RunPod toml file and rename it:
   ```bash
   cp pyproject_use_this_one_on_runpod.toml pyproject.toml
   uv sync
   ```
5. Upload your processed `data` folder (specifically `instructionquality.json`) from your local machine to the RunPod instance.
6. Run the training script:
   ```bash
   uv run train.py
   ```

## 🧠 How It Works (Architecture)
The magic of this project lies in its sequential pipeline, transforming unstructured data into a highly specialized language model:
1. **Ingestion & Chunking**: `syntheticdatageneration.py` uses Docling to read a PDF and break it down into manageable, contextual "chunks" of text.
2. **Generation**: We ask a base LLM (via LiteLLM and Ollama) to read these chunks and generate plausible questions and answers a user might ask about the text.
3. **Refinement**: `preprocessing.py` standardizes this data, while `dataquality.py` acts as an "LLM Judge." It critiques the generated Q&A pairs, throwing away hallucinations or poorly formatted answers.
4. **Quantization & LoRA**: During `train.py`, the massive base model is loaded in 4-bit precision (saving huge amounts of memory). Instead of retraining the whole model, we inject small "LoRA adapters" (extra layers) and train *only* those.
5. **Inference**: The final result is a lightweight adapter that, when combined with the base model, creates a highly specialized assistant without needing a supercomputer.

## 📁 Folder Structure
A simplified view of where the critical pieces live:

```text
├── README.md                              # This file
├── pyproject.toml                         # Project configuration and local dependencies
├── pyproject_use_this_one_on_runpod.toml  # Optimized dependencies for RunPod deployment
├── syntheticdatageneration.py             # 1️⃣ Reads PDFs & generates raw Q&A pairs
├── preprocessing.py                       # 2️⃣ Formats data for the next steps
├── dataquality.py                         # 3️⃣ Uses LLM-as-a-judge to filter high-quality data
├── train.py                               # 4️⃣ Main QLoRA training script
├── generated_prompt.py                    # Contains the prompt templates used for LLM generation
├── data/                                  # 📂 Output directory for processed datasets
│   ├── instruction.json                   # Preprocessed instructions
│   └── instructionquality.json            # Final, filtered high-quality dataset used for training
└── local_models/                          # 📂 (Generated during training) Base models downloaded from Hugging Face
```