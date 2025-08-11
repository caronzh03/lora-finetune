import json
import os
import urllib
from functools import partial

from torch.utils.data import Dataset, DataLoader
import torch


class InstructionDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer):
        """
        Data example:
        [{
            "instruction": "Evaluate the following phrase by transforming it into the spelling given.",
            "input": "freind --> friend",
            "output": "The spelling of the given phrase \"freind\" is incorrect, the correct spelling is \"friend\"."
        }]

        After applying chat template, it should become:
        <|im_start|>user
        Below is an instruction that describes a task. Write a response that appropriately completes the request.

        ### Instruction:
        Evaluate the following phrase by transforming it into the spelling given.

        ### Input:
        freind --> friend<|im_end|>
        <|im_start|>assistant
        The spelling of the given phrase \"freind\" is incorrect, the correct spelling is \"friend\".<|im_end|>
        """
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = entry['output']
            messages = [
                {"role": "user", "content": instruction_plus_input},
                {"role": "assistant", "content": response_text}
            ]
            full_text = tokenizer.format_qwen_chat(messages)
            self.encoded_texts.append(
                tokenizer.encode(full_text, use_chat_template=False)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def custom_collate_fn(
        batch,
        pad_token_id=151643,
        ignore_index=-100,
        allowed_max_length=None,
        device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def download_instruction_dataset(data_dir: str):
    file_path = f"{data_dir}/instruction-data.json"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)

    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)  # 10% for testing

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    print(50 * "-")
    return train_data, val_data, test_data


def create_instruction_dataloaders(tokenizer, train_data: list[dict], val_data: list[dict], test_data: list[dict],
                                   device: torch.device):
    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)
    num_workers = 0
    batch_size = 8

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
