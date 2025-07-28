from pathlib import Path
from torch.utils.data import DataLoader
import argparse
import time
import torch
import json
import os
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, snapshot_download

from utils import (
    load_weights_into_qwen,
    download_data,
    create_dataloaders,
    to_classifier,
)
from qwen_layers import Qwen3Model
from qwen_tokenizer import Qwen3Tokenizer
from train_lora import train


def initialize_qwen_tokenizer(model_variant: str, use_reasoning: bool) -> Qwen3Tokenizer:
    if use_reasoning:
        tokenizer_file_path = f"Qwen3-{model_variant}/tokenizer.json"
        repo_id = f"Qwen/Qwen3-{model_variant}"
    else:
        tokenizer_file_path = f"Qwen3-{model_variant}-Base/tokenizer.json"
        repo_id = f"Qwen/Qwen3-{model_variant}-Base"

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        repo_id=repo_id,
        add_generation_prompt=use_reasoning,
        add_thinking=use_reasoning
    )
    return tokenizer


def initialize_qwen_model(model_variant: str, use_reasoning: bool) -> Qwen3Model:
    if model_variant == "0.6B":
        qwen3_config = {
            "vocab_size": 151_936,  # Vocabulary size
            "context_length": 40_960,  # Context length that was used to train the model
            "emb_dim": 1024,  # Embedding dimension
            "n_heads": 16,  # Number of attention heads
            "n_layers": 28,  # Number of layers
            "hidden_dim": 3072,  # Size of the intermediate dimension in FeedForward
            "head_dim": 128,  # Size of the heads in GQA
            "qk_norm": True,  # Whether to normalize queries and values in GQA
            "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
            "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
            "dtype": torch.bfloat16,  # Lower-precision dtype to reduce memory usage
        }
    elif model_variant == "1.7B":
        qwen3_config = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 2048,  # 2x larger than above
            "n_heads": 16,
            "n_layers": 28,
            "hidden_dim": 6144,  # 2x larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        }
    elif model_variant == "4B":
        qwen3_config = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 2560,  # 25% larger than above
            "n_heads": 32,  # 2x larger than above
            "n_layers": 36,  # 29% larger than above
            "hidden_dim": 9728,  # ~3x larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        }
    elif model_variant == "8B":
        qwen3_config = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 4096,  # 60% larger than above
            "n_heads": 32,
            "n_layers": 36,  # 26% larger than above
            "hidden_dim": 12288,
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        }
    elif model_variant == "14B":
        qwen3_config = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 5120,  # 25% larger than above
            "n_heads": 40,  # 25% larger than above
            "n_layers": 40,  # 11% larger than above
            "hidden_dim": 17408,  # 42% larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        }
    elif model_variant == "32B":
        qwen3_config = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 5120,
            "n_heads": 64,  # 60% larger than above
            "n_layers": 64,  # 60% larger than above
            "hidden_dim": 25600,  # 47% larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        }
    else:
        raise ValueError(f"{model_variant} is not supported.")

    # Initialize model with random weights
    model = Qwen3Model(qwen3_config)

    # Download open source model weights from HuggingFace
    if use_reasoning:
        repo_id = f"Qwen/Qwen3-{model_variant}"
    else:
        repo_id = f"Qwen/Qwen3-{model_variant}-Base"

    local_dir = Path(repo_id).parts[-1]

    if model_variant == "0.6B":
        weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=local_dir,
        )
        weights_dict = load_file(weights_file)
    else:
        repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
        index_path = os.path.join(repo_dir, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)

        weights_dict = {}
        for filename in set(index["weight_map"].values()):
            shard_path = os.path.join(repo_dir, filename)
            shard = load_file(shard_path)
            weights_dict.update(shard)

    load_weights_into_qwen(model, qwen3_config, weights_dict)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train a Qwen model with LoRA",
    )
    parser.add_argument("--model-variant", type=str, default="0.6B")
    parser.add_argument("--use-reasoning", action="store_true")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    torch.manual_seed(123)
    download_data(args.data_dir)
    qwen_tokenizer = initialize_qwen_tokenizer(args.model_variant, args.use_reasoning)
    train_loader, val_loader, test_loader = create_dataloaders(qwen_tokenizer, args.data_dir)

    model = initialize_qwen_model(args.model_variant, args.use_reasoning)
    model = to_classifier(model, dim=model.tok_emb.embedding_dim)

    train(model, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()
