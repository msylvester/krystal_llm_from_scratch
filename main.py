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
        "context_length": 1024,  # Full context length for pretrained models
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.0,  # No dropout for inference
        "qkv_bias": True   # Required for pretrained weights
    },
    "gpt2-medium": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.0,
        "qkv_bias": True
    },
    "gpt2-large": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1280,
        "n_heads": 20,
        "n_layers": 36,
        "drop_rate": 0.0,
        "qkv_bias": True
    },
    "gpt2-xl": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1600,
        "n_heads": 25,
        "n_layers": 48,
        "drop_rate": 0.0,
        "qkv_bias": True
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

# Weight file mapping
WEIGHT_FILES = {
    "gpt2-small": "gpt2-small-124M.pth",
    "gpt2-medium": "gpt2-medium-355M.pth", 
    "gpt2-large": "gpt2-large-774M.pth",
    "gpt2-xl": "gpt2-xl-1558M.pth"
}


def download_weights(model_name):
    """Download pretrained weights for the specified model"""
    if model_name not in WEIGHT_FILES:
        print(f"No pretrained weights available for {model_name}")
        return None
        
    file_name = WEIGHT_FILES[model_name]
    url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"
    
    if not os.path.exists(file_name):
        print(f"Downloading pretrained weights: {file_name}")
        try:
            urllib.request.urlretrieve(url, file_name)
            print(f"Downloaded to {file_name}")
        except Exception as e:
            print(f"Error downloading weights: {e}")
            return None
    else:
        print(f"Using existing weights: {file_name}")
    
    return file_name


def load_pretrained_model(model_name="gpt2-small"):
    """Load a GPT model with pretrained weights"""
    if model_name not in GPT_CONFIGS:
        print(f"Unknown model: {model_name}")
        return None, None
        
    config = GPT_CONFIGS[model_name]
    
    # Download weights if needed
    weight_file = download_weights(model_name)
    if weight_file is None:
        print("Could not download weights, using untrained model")
        model = GPTModel(config)
        return model, config
    
    # Create model and load pretrained weights
    model = GPTModel(config)
    try:
        model.load_state_dict(torch.load(weight_file, weights_only=True))
        model.eval()
        print(f"Loaded pretrained {model_name} model")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Using untrained model")
    
    return model, config


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


def demo_text_generation(use_pretrained=False, model_name="gpt2-small"):
    """Demonstrate text generation"""
    print("\n=== Text Generation Demo ===")
    
    if use_pretrained:
        model, config = load_pretrained_model(model_name)
        if model is None:
            return
    else:
        config = GPT_CONFIGS["gpt2-mini"]
        model = GPTModel(config)
        model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    start_text = "Every effort moves"
    print(f"Starting text: '{start_text}'")
    print(f"Using model: {model_name if use_pretrained else 'gpt2-mini (untrained)'}")
    
    # Generate text
    torch.manual_seed(123)
    if use_pretrained and hasattr(model, 'generate'):
        # Use advanced generation if available
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(start_text, tokenizer).to(device),
            max_new_tokens=30,
            context_size=config["context_length"],
            top_k=1,
            temperature=1.0
        )
    else:
        # Use simple generation
        token_ids = generate_text_simple(
            model=model,
            idx=text_to_token_ids(start_text, tokenizer),
            max_new_tokens=30,
            context_size=config["context_length"]
        )
    
    generated_text = token_ids_to_text(token_ids, tokenizer)
    print(f"Generated text: '{generated_text}'")
    
    if not use_pretrained:
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


def generate_response(input_text, use_pretrained=True, model_name="gpt2-small", max_tokens=20):
    """Generate a response for the given input text"""
    if use_pretrained:
        model, config = load_pretrained_model(model_name)
        if model is None:
            print("Falling back to untrained model")
            config = GPT_CONFIGS["gpt2-mini"]
            model = GPTModel(config)
            model.eval()
            use_pretrained = False
    else:
        config = GPT_CONFIGS["gpt2-mini"]
        model = GPTModel(config)
        model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Generate text
    if use_pretrained:
        try:
            token_ids = generate(
                model=model,
                idx=text_to_token_ids(input_text, tokenizer).to(device),
                max_new_tokens=max_tokens,
                context_size=config["context_length"],
                top_k=50,
                temperature=0.8
            )
        except:
            # Fallback to simple generation
            token_ids = generate_text_simple(
                model=model,
                idx=text_to_token_ids(input_text, tokenizer),
                max_new_tokens=max_tokens,
                context_size=config["context_length"]
            )
    else:
        token_ids = generate_text_simple(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer),
            max_new_tokens=max_tokens,
            context_size=config["context_length"]
        )
    
    generated_text = token_ids_to_text(token_ids, tokenizer)
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="LLM from Scratch Demo")
    parser.add_argument("input_text", nargs="?", help="Text to generate response for")
    parser.add_argument("--demo", type=str, choices=[
        "tokenization", "model", "generation", "pretrained", "training", "classification", "all"
    ], help="Which demo to run")
    parser.add_argument("--model", type=str, default="gpt2-small", 
                       choices=["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                       help="Which pretrained model to use")
    parser.add_argument("--max-tokens", type=int, default=50, 
                       help="Maximum tokens to generate")
    parser.add_argument("--no-pretrained", action="store_true", 
                       help="Use untrained model instead of pretrained weights")
    
    args = parser.parse_args()
    
    # If input text is provided, generate response
    if args.input_text:
        response = generate_response(
            args.input_text, 
            use_pretrained=not args.no_pretrained,
            model_name=args.model,
            max_tokens=args.max_tokens
        )
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
        demo_text_generation(use_pretrained=False)
    
    if demo_to_run in ["pretrained", "all"]:
        demo_text_generation(use_pretrained=True, model_name=args.model)
    
    if demo_to_run in ["training", "all"]:
        demo_training()
    
    if demo_to_run in ["classification", "all"]:
        demo_classification_setup()
    
    print("\n" + "=" * 60)
    print("Demo complete! 🎉")
    print("\nTo run specific demos:")
    print("  python main.py --demo tokenization")
    print("  python main.py --demo model")
    print("  python main.py --demo generation      # Untrained model")
    print("  python main.py --demo pretrained     # Pretrained model")
    print("  python main.py --demo training")
    print("  python main.py --demo classification")
    print("\nTo generate text with pretrained model:")
    print('  python main.py "your input text here"')
    print('  python main.py "your input text here" --model gpt2-medium --max-tokens 100')
    print("\nTo generate text with untrained model:")
    print('  python main.py "your input text here" --no-pretrained')


if __name__ == "__main__":
    main()