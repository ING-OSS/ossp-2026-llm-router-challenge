"""Task-independent whole-prompt WordPiece packing."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def episode_text(episode: object) -> str:
    """Render every prompt-visible field without guessing a task boundary."""

    if episode.prompt is not None:
        return episode.prompt
    return "\n".join(
        f"[{message.role}] {message.content}" for message in episode.messages
    )


class WordPieceTokenizer:
    def __init__(self, vocab_path: Path) -> None:
        from tokenizers import BertWordPieceTokenizer

        self.backend = BertWordPieceTokenizer(str(vocab_path), lowercase=True)
        self.cls_token_id = self.backend.token_to_id("[CLS]")
        self.sep_token_id = self.backend.token_to_id("[SEP]")
        self.pad_token_id = self.backend.token_to_id("[PAD]")

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self.backend.encode(text, add_special_tokens=add_special_tokens).ids


def _head_tail(token_ids: Sequence[int], limit: int) -> list[int]:
    if limit <= 0:
        return []
    if len(token_ids) <= limit:
        return list(token_ids)
    head = (limit + 1) // 2
    tail = limit - head
    return [*token_ids[:head], *token_ids[-tail:]] if tail else list(token_ids[:head])


def tokenize_inputs(
    inputs: object,
    tokenizer: WordPieceTokenizer,
    *,
    max_length: int = 96,
) -> dict[str, np.ndarray]:
    """Keep whole-prompt boundaries without semantic or task-format heuristics."""

    rows: list[list[int]] = []
    for episode in inputs.episodes:
        token_ids = tokenizer.encode(episode_text(episode))
        selected = _head_tail(token_ids, max_length - 2)
        rows.append(
            [tokenizer.cls_token_id, *selected, tokenizer.sep_token_id]
        )

    input_ids = np.full(
        (len(rows), max_length), tokenizer.pad_token_id, dtype=np.int64
    )
    attention_mask = np.zeros_like(input_ids)
    token_type_ids = np.zeros_like(input_ids)
    for index, row in enumerate(rows):
        input_ids[index, : len(row)] = row
        attention_mask[index, : len(row)] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def prepare_inputs(
    inputs: object,
    vocab_path: Path,
    preprocess: Mapping[str, object],
) -> dict[str, np.ndarray]:
    tokenizer = WordPieceTokenizer(vocab_path)
    return tokenize_inputs(
        inputs,
        tokenizer,
        max_length=int(preprocess["max_length"]),
    )
