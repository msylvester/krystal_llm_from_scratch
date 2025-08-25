# LLM from Scratch Project

This project is based on the "Build a Large Language Model from Scratch" book by Sebastian Raschka. The notebooks have been converted into a modular Python project that can be run with `python main.py`.

## Project Structure

```
├── main.py              # Main script to run demos
├── requirements.txt     # Python dependencies
├── tokenization.py      # Tokenization utilities (Chapter 2)
├── attention.py         # Attention mechanisms (Chapter 3)
├── gpt_model.py        # GPT model implementation (Chapter 4)
├── training_utils.py   # Training utilities (Chapter 5)
├── classification.py   # Classification utilities (Chapter 6)
├── README.md           # This file
└── ch*.ipynb          # Original Jupyter notebooks
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run all demos:
```bash
python main.py
```

### Run specific demos:
```bash
python main.py --demo tokenization    # Tokenization demo
python main.py --demo model          # Model creation demo
python main.py --demo generation     # Text generation demo
python main.py --demo training       # Training setup demo
python main.py --demo classification # Classification setup demo
```

## Features

### Tokenization (Chapter 2)
- Simple tokenizers with vocabulary handling
- GPT-2 tokenizer integration using tiktoken
- Data loading utilities for training

### Attention Mechanisms (Chapter 3)
- Self-attention implementations
- Causal (masked) attention for autoregressive models
- Multi-head attention

### GPT Model (Chapter 4)
- Complete GPT model implementation
- Layer normalization and feed-forward networks
- Transformer blocks with residual connections

### Training (Chapter 5)
- Training loop implementation
- Loss calculation utilities
- Text generation with temperature and top-k sampling
- Model evaluation functions

### Classification (Chapter 6)
- Spam classification dataset handling
- Model finetuning for classification tasks
- Classification-specific training loops

## Model Configurations

The project includes two pre-configured models:

- **gpt2-small**: 12 layers, 768 embedding dimensions (similar to GPT-2 124M)
- **gpt2-mini**: 6 layers, 384 embedding dimensions (smaller for quick testing)

## Educational Purpose

This project is designed for educational purposes to understand:
- How transformer-based language models work
- The training process for LLMs
- Different applications of LLMs (generation vs classification)
- Modern deep learning techniques used in NLP

## Original Source

Based on the book "Build a Large Language Model From Scratch" by Sebastian Raschka.
- Book: http://mng.bz/orYv
- Original code: https://github.com/rasbt/LLMs-from-scratch

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Other dependencies listed in requirements.txt

## Notes

- The models are simplified for educational purposes
- Training data is minimal (short story) for quick experimentation
- For production use, you would need larger datasets and more compute resources
- GPU is recommended but not required for the demos
