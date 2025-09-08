# Qwen-from-scratch

This repository provides code for training, fine-tuning, and evaluating Qwen3 language models with LoRA and ReDrafter support. It includes utilities for loading open-source Qwen3 weights, applying LoRA adapters, training on classification and instruction datasets, and generating model responses.

## Features

- **Qwen3 Model Support:** Easily load and use various Qwen3 model variants (0.6B, 1.7B, 4B, 8B, 14B, 32B).
- **LoRA Finetuning:** Replace linear layers with LoRA adapters for efficient parameter updates.
- **ReDrafter Integration:** Add a drafter head for future token prediction.
- **Tokenizer Utilities:** Load and configure Qwen3 tokenizers for reasoning and base models.
- **Dataset Handling:** Download and preprocess spam classification and instruction-following datasets.
- **Training Scripts:** Train models for classification or instruction tasks with progress tracking and evaluation.
- **Response Generation:** Generate responses from trained models for evaluation.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/qwen-from-scratch.git
    cd qwen-from-scratch
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Train a Classifier with LoRA

```bash
python qwen_lora.py --classifier --model-variant 0.6B
```

### Instruction Fine-tuning with LoRA

```bash
python qwen_lora.py --model-variant 0.6B
```

### Instruction Fine-tuning with LoRA and ReDrafter Head (for inference acceleration)

```bash
python qwen_lora.py --redrafter --model-variant 0.6B
```


### Arguments

- `--model-variant`: Qwen3 model size (e.g., 0.6B, 1.7B, 4B, etc.)
- `--classifier`: Train a classifier head.
- `--redrafter`: Add and train a ReDrafter head.
- `--use-reasoning`: Use the reasoning variant of the tokenizer/model.
- `--data-dir`: Directory to store datasets.

## File Structure

- `qwen_lora.py`: Main training and evaluation script.
- `qwen_layers.py`: Qwen3 model architecture.
- `lora.py`: LoRA adapter implementation.
- `redrafter/model.py`: ReDrafter wrapper and drafter head.
- `redrafter/loss.py`: Drafter loss functions.
- `utils.py`: Utility functions for data loading, training, and evaluation.
- `instruction_dataset.py`: Instruction dataset download and preprocessing.
- `generate_response.py`: Model response generation.

## Citation

If you use this codebase, please cite the original Qwen3 and LoRA papers.

## References

- Sebastian Raschka. *Build A Large Language Model (From Scratch)*. Manning, 2024. ISBN: 978-1633437166.  
  [Book link](https://www.manning.com/books/build-a-large-language-model-from-scratch) | [GitHub](https://github.com/rasbt/LLMs-from-scratch)

- Aonan Zhang, Chong Wang, Yi Wang, Xuanyu Zhang, Yunfei Cheng.  
  "Recurrent Drafter for Fast Speculative Decoding in Large Language Models." arXiv:2403.09919, 2024.  
  [arXiv link](https://arxiv.org/abs/2403.09919) | [DOI](https://doi.org/10.48550/arXiv.2403.09919)

## License

See the `LICENSE` file for details.

---

**Note:** This repo is for research and educational purposes. For commercial use, check the respective model and