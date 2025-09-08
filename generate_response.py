from tqdm import tqdm
import torch

from instruction_dataset import format_input
from redrafter.model import ReDrafter


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """Generate new tokens given a model and an initial sequence of token IDs."""

    is_redrafter = isinstance(model, ReDrafter)

    if is_redrafter:
        model = model.llm  # Use only the LLM part for generation
        # TODO: implement drafter-based generation

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def generate_response(model, tokenizer, test_data: list[dict], device="cuda"):
    print("Generating responses")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=40_960,
            eos_id=151645
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        print(generated_text)



# Some sample outputs

"""
<|im_start|>user
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Rewrite the given sentence to describe the same thing in a positive way.

### Input:
The food was not good.<|im_end|>
<|im_start|>assistant
The food could use some improvement.
"""

"""
<|im_start|>user
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Name three essential vitamins for human health.<|im_end|>
<|im_start|>assistant
1. Carbon
2. Iron
3. Vitamin C
"""
