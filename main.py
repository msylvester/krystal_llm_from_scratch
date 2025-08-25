#!/usr/bin/env python3
"""
Main script for the LLM from Scratch project
Demonstrates key functionalities from the notebooks
"""

import argparse
import torch
import tiktoken
import urllib.request
import os
from pathlib import Path

from tokenization import create_dataloader_v1, text_to_token_ids, token_ids_to_text
from gpt_model import GPTModel, generate_text_simple, generate
from training_utils import train_model_simple, calc_loss_loader
from classification import SpamDataset, train_classifier_simple, classify_review


# Configuration for different GPT models
GPT_CONFIGS = {
    "gpt2-small": {
        "vocab_size": 50257,
        "context_length": 256,  # Reduced for educational purposes
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    },
    "gpt2-mini": {  # Smaller version for quick testing
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 384,
        "n_heads": 6,
        "n_layers": 6,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
}


def download_text_data():
    """Download the text data used for training"""
    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    
    if not os.path.exists(file_path):
        print("Downloading training data...")
        try:
            with urllib.request.urlopen(url) as response:
                text_data = response.read().decode('utf-8')
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_data)
            print(f"Downloaded {file_path}")
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    
    return text_data


def demo_tokenization():
    """Demonstrate tokenization functionality"""
    print("\n=== Tokenization Demo ===")
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    sample_text = "Hello, world! This is a sample text for tokenization."
    print(f"Original text: {sample_text}")
    
    # Tokenize
    token_ids = text_to_token_ids(sample_text, tokenizer)
    print(f"Token IDs: {token_ids}")
    
    # Detokenize
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(f"Decoded text: {decoded_text}")


def demo_model_creation():
    """Demonstrate model creation and basic forward pass"""
    print("\n=== Model Creation Demo ===")
    
    config = GPT_CONFIGS["gpt2-mini"]
    print(f"Creating GPT model with config: {config}")
    
    torch.manual_seed(123)
    model = GPTModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 2
    seq_length = 10
    dummy_input = torch.randint(0, config["vocab_size"], (batch_size, seq_length))
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")


def demo_text_generation():
    """Demonstrate text generation"""
    print("\n=== Text Generation Demo ===")
    
    config = GPT_CONFIGS["gpt2-mini"]
    model = GPTModel(config)
    model.eval()
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    start_text = "The future of AI is"
    print(f"Starting text: '{start_text}'")
    
    # Generate text (will be random since model is untrained)
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_text, tokenizer),
        max_new_tokens=10,
        context_size=config["context_length"]
    )
    
    generated_text = token_ids_to_text(token_ids, tokenizer)
    print(f"Generated text: '{generated_text}'")
    print("Note: Output is random since model is untrained")


def demo_training():
    """Demonstrate training setup (without actual training)"""
    print("\n=== Training Setup Demo ===")
    
    # Download data
    text_data = download_text_data()
    if text_data is None:
        print("Could not download training data, skipping training demo")
        return
    
    print(f"Training data length: {len(text_data)} characters")
    
    # Create data loaders
    train_ratio = 0.8
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    
    config = GPT_CONFIGS["gpt2-mini"]
    
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model
    model = GPTModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Calculate initial loss
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=2)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=2)
    
    print(f"Initial training loss: {train_loss:.3f}")
    print(f"Initial validation loss: {val_loss:.3f}")
    print("Training setup complete (actual training not performed for demo)")


def demo_classification_setup():
    """Demonstrate classification setup"""
    print("\n=== Classification Setup Demo ===")
    
    # For classification, we'd typically need labeled data
    # This is just showing how to modify the model for classification
    
    config = GPT_CONFIGS["gpt2-mini"]
    model = GPTModel(config)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace output head for binary classification
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=config["emb_dim"], out_features=num_classes)
    
    # Make last transformer block trainable
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    
    for param in model.final_norm.parameters():
        param.requires_grad = True
    
    print("Model modified for binary classification")
    print(f"Output head shape: {model.out_head.weight.shape}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params:.3f}")


def generate_response(input_text):
    """Generate a response for the given input text"""
    config = GPT_CONFIGS["gpt2-mini"]
    model = GPTModel(config)
    model.eval()
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Generate text
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer),
        max_new_tokens=20,
        context_size=config["context_length"]
    )
    
    generated_text = token_ids_to_text(token_ids, tokenizer)
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="LLM from Scratch Demo")
    parser.add_argument("input_text", nargs="?", help="Text to generate response for")
    parser.add_argument("--demo", type=str, choices=[
        "tokenization", "model", "generation", "training", "classification", "all"
    ], help="Which demo to run")
    
    args = parser.parse_args()
    
    # If input text is provided, generate response
    if args.input_text:
        response = generate_response(args.input_text)
        print(response)
        return
    
    # Default to running demos
    demo_to_run = args.demo or "all"
    
    print("🤖 LLM from Scratch Project Demo")
    print("Based on the 'Build a Large Language Model from Scratch' book")
    print("=" * 60)
    
    if demo_to_run in ["tokenization", "all"]:
        demo_tokenization()
    
    if demo_to_run in ["model", "all"]:
        demo_model_creation()
    
    if demo_to_run in ["generation", "all"]:
        demo_text_generation()
    
    if demo_to_run in ["training", "all"]:
        demo_training()
    
    if demo_to_run in ["classification", "all"]:
        demo_classification_setup()
    
    print("\n" + "=" * 60)
    print("Demo complete! 🎉")
    print("\nTo run specific demos:")
    print("  python main.py --demo tokenization")
    print("  python main.py --demo model")
    print("  python main.py --demo generation")
    print("  python main.py --demo training")
    print("  python main.py --demo classification")
    print("\nTo generate text:")
    print('  python main.py "your input text here"')


if __name__ == "__main__":
    main()