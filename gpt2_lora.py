import argparse
import torch
import tiktoken

from gpt_download import download_and_load_gpt2
from utils import (
    load_weights_into_gpt,
    download_data,
    create_dataloaders,
    to_classifier,
)
from gpt2_layers import GPTModel
from train_lora import train_classifier


def initialize_gpt_model(model_variant: str) -> GPTModel:
    """Download GPT model variant (e.g. gpt2-small (124M)) from internet
    and load model weights into our own GPTModel."""

    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True  # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[model_variant])

    model_size = model_variant.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train a GPT-2 model with LoRA",
    )
    parser.add_argument("--model-variant", type=str, default="gpt2-small (124M)")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    torch.manual_seed(123)
    download_data(args.data_dir)
    train_loader, val_loader, test_loader = create_dataloaders(tiktoken.get_encoding("gpt2")
,args.data_dir)

    gpt_model = initialize_gpt_model(args.model_variant)
    gpt_model = to_classifier(gpt_model, dim=gpt_model.tok_emb.embedding_dim)

    train_classifier(gpt_model, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()
