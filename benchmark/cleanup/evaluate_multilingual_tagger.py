#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = [
#   "numpy>=2.3,<3",
#   "onnxruntime>=1.27,<2",
#   "openpyxl>=3.1,<4",
#   "transformers>=5.4,<6",
# ]
# ///
"""Evaluate an exported multilingual tagger on DISCO's held-out test split."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from train_multilingual_tagger import (
    DELETE,
    IGNORE,
    Example,
    load_disco,
    metrics_for_threshold,
    split_examples,
)
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def aligned_batch(
    tokenizer: Any,
    examples: list[Example],
    *,
    max_length: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    encoded = tokenizer(
        [list(example.words) for example in examples],
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    aligned_labels: list[list[int]] = []
    for batch_index, example in enumerate(examples):
        word_ids = encoded.word_ids(batch_index=batch_index)
        previous_word_id: int | None = None
        labels: list[int] = []
        for word_id in word_ids:
            if word_id is None or word_id == previous_word_id:
                labels.append(IGNORE)
            else:
                labels.append(example.labels[word_id])
            previous_word_id = word_id
        aligned_labels.append(labels)
    inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }
    return inputs, np.asarray(aligned_labels, dtype=np.int64)


def evaluate_language(
    session: ort.InferenceSession,
    tokenizer: Any,
    examples: list[Example],
    *,
    max_length: int,
    batch_size: int,
) -> tuple[list[float], list[int]]:
    probabilities: list[float] = []
    labels: list[int] = []
    for start in range(0, len(examples), batch_size):
        inputs, batch_labels = aligned_batch(
            tokenizer,
            examples[start : start + batch_size],
            max_length=max_length,
        )
        (logits,) = session.run(None, inputs)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        softmax = np.exp(shifted)
        softmax /= np.sum(softmax, axis=-1, keepdims=True)
        valid = batch_labels != IGNORE
        probabilities.extend(softmax[..., DELETE][valid].tolist())
        labels.extend(batch_labels[valid].tolist())
    return probabilities, labels


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.model_root / "metadata.json").read_text())
    threshold = float(metadata["delete_threshold"])
    max_length = int(metadata["max_length"])
    seed = int(metadata["training"]["seed"])
    examples, data_stats = load_disco(args.data_dir)
    _, _, testing = split_examples(examples, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, use_fast=True)
    session = ort.InferenceSession(
        str(args.model_root / "model.int8.onnx"),
        providers=["CPUExecutionProvider"],
    )

    by_language: dict[str, Any] = {}
    all_probabilities: list[float] = []
    all_labels: list[int] = []
    for language in sorted({example.language for example in testing}):
        language_examples = [example for example in testing if example.language == language]
        probabilities, labels = evaluate_language(
            session,
            tokenizer,
            language_examples,
            max_length=max_length,
            batch_size=args.batch_size,
        )
        all_probabilities.extend(probabilities)
        all_labels.extend(labels)
        by_language[language] = {
            "examples": len(language_examples),
            **asdict(metrics_for_threshold(probabilities, labels, threshold)),
        }

    report = {
        "model_root": str(args.model_root),
        "artifact": "model.int8.onnx",
        "threshold": threshold,
        "data": data_stats,
        "aggregate": asdict(metrics_for_threshold(all_probabilities, all_labels, threshold)),
        "by_language": by_language,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
