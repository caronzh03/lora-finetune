from pathlib import Path

from huggingface_hub import hf_hub_download

from tokenizers import Tokenizer


class Qwen3Tokenizer():
    def __init__(self, tokenizer_file_path="tokenizer.json", repo_id=None, add_generation_prompt=False,
                 add_thinking=False):
        self.tokenizer_file_path = tokenizer_file_path
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tokenizer_file_path_obj = Path(tokenizer_file_path)
        if not tokenizer_file_path_obj.is_file() and repo_id is not None:
            _ = hf_hub_download(
                repo_id=repo_id,
                filename=str(tokenizer_file_path_obj.name),
                local_dir=str(tokenizer_file_path_obj.parent.name)
            )
        self.tokenizer = Tokenizer.from_file(tokenizer_file_path)

    def encode(self, prompt, use_chat_template=True):
        if use_chat_template:
            messages = [
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.format_qwen_chat(
                messages,
                add_generation_prompt=self.add_generation_prompt,
                add_thinking=self.add_thinking
            )
        else:
            formatted_prompt = prompt
        return self.tokenizer.encode(formatted_prompt).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    @staticmethod
    def format_qwen_chat(messages, add_generation_prompt=False, add_thinking=False):
        prompt = ""
        for msg in messages:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        if add_generation_prompt:
            prompt += "<|im_start|>assistant"
            if not add_thinking:
                prompt += "<|think>\n\n<|/think>\n\n"
            else:
                prompt += "\n"
        return prompt
