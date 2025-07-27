from pathlib import Path
from torch.utils.data import DataLoader
import argparse
import pandas as pd
import tiktoken
import time
import torch
import urllib

from gpt_download import download_and_load_gpt2
from utils import (
    download_and_unzip_spam_data,
    create_balanced_dataset,
    random_split,
    SpamDataset,
    GPTModel,
    load_weights_into_gpt,
    calc_accuracy_loader,
    train_classifier_simple,
    plot_values
)
from lora import LinearWithLoRA


def download_data(data_dir: str):
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    try:
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"Primary URL failed: {e}. Trying backup URL...")
        url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv(f"{data_dir}/train.csv", index=None)
    validation_df.to_csv(f"{data_dir}/validation.csv", index=None)
    test_df.to_csv(f"{data_dir}/test.csv", index=None)
    print(f"Train and Validation datasets saved to {data_dir}")


def create_dataloaders(data_dir: str) -> tuple:
    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = SpamDataset(f"{data_dir}/train.csv", max_length=None, tokenizer=tokenizer)
    val_dataset = SpamDataset(f"{data_dir}/validation.csv", max_length=train_dataset.max_length,
                              tokenizer=tokenizer)
    test_dataset = SpamDataset(f"{data_dir}/test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

    return train_loader, val_loader, test_loader


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


def gpt_to_classifier(model: GPTModel) -> GPTModel:
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=768, out_features=num_classes)
    return model


def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply the same function to child modules
            replace_linear_with_lora(module, rank, alpha)


def train(model: GPTModel, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
    # Freeze base model weights
    for param in model.parameters():
        param.requires_grad = False

    # Replace Linear layers with LinearWithLoRA layers
    replace_linear_with_lora(model, rank=16, alpha=16)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable LoRA parameters: {total_params:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model)

    # Start training
    print("Starting training...")
    model.train()

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # Plot training & validation losses
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses, label="loss")

    # Print model prediction accuracy after training
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Train a GPT-2 model with LoRA",
    )
    parser.add_argument("--model-variant", type=str, default="gpt2-small (124M)")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    torch.manual_seed(123)
    download_data(args.data_dir)
    train_loader, val_loader, test_loader = create_dataloaders(args.data_dir)

    gpt_model = initialize_gpt_model(args.model_variant)
    gpt_model = gpt_to_classifier(gpt_model)

    train(gpt_model, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()
