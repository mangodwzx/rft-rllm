"""Raw prompt/answer data adapter for the unmodified official verl SFT loss.

No ChatTemplateParser is involved. Prompt and completion are tokenized as
separate sequences, just as autoregressive sampling starts from fixed prompt
IDs. Prompt tokens are masked; only the completion and its EOS are supervised.
"""
from __future__ import annotations

from pathlib import Path

from .prompts import INSTRUCTION


def tokenize_raw_messages(messages, tokenizer, max_length: int) -> tuple[list[int], list[int]]:
    messages = list(messages)
    if len(messages) != 2 or [m.get("role") for m in messages] != ["user", "assistant"]:
        raise ValueError("Raw SFT requires exactly one raw user prompt and one assistant completion")
    prompt, response = (m.get("content") for m in messages)
    if not isinstance(prompt, str) or not prompt.startswith(INSTRUCTION + "\n\n") or not prompt.endswith("\nSolution:"):
        raise ValueError("SFT prompt does not match the shared raw math protocol")
    if not isinstance(response, str) or not response:
        raise ValueError("Raw SFT requires a nonempty completion")
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    response_ids = list(tokenizer.encode(response, add_special_tokens=False))
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Raw SFT requires an EOS token")
    if not response_ids or response_ids[-1] != eos:
        response_ids.append(eos)
    ids = prompt_ids + response_ids
    if len(ids) > max_length:
        raise ValueError(f"Raw SFT sample has {len(ids)} tokens, exceeding max_length={max_length}; refusing to truncate")
    return ids, [0] * len(prompt_ids) + [1] * len(response_ids)


class RawCompletionSFTDataset:
    """The ``data.custom_cls`` protocol accepted by verl's SFTTrainer."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        import pandas as pd

        files = [parquet_files] if isinstance(parquet_files, (str, Path)) else list(parquet_files)
        self.messages = [messages for path in files for messages in pd.read_parquet(path)["messages"].tolist()]
        if max_samples > 0:
            self.messages = self.messages[:max_samples]
        if not self.messages:
            raise ValueError("Raw SFT dataset is empty")
        if config.pad_mode != "no_padding":
            raise ValueError("RawCompletionSFTDataset requires pad_mode=no_padding")
        self.tokenizer = tokenizer
        self.max_length = int(config.max_length)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, item):
        import torch

        ids, mask = tokenize_raw_messages(self.messages[item], self.tokenizer, self.max_length)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "position_ids": torch.arange(len(ids), dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.long),
        }
