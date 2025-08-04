import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lora import replace_linear_with_lora
from utils import train_classifier_simple, plot_values, calc_accuracy_loader, train_model_simple


def train_classifier(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
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


def train_instruction(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_data: list[dict]):
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
    num_epochs = 2
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # Plot training & validation losses
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_values(epochs_tensor, tokens_seen, train_losses, val_losses, label="loss", x_axis="Tokens seen")

    # Save model
    finetuned_model_name = "qwen-sft.pt"
    torch.save(model.state_dict(), finetuned_model_name)
    print(f"Model saved as {finetuned_model_name}")
