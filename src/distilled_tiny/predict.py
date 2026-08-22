"""Predict Tiny quality and adjacent gains for challenge inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from distilled_tiny.config import MODEL_DIR, TOKENIZER_DIR
from distilled_tiny.tokenization import prepare_inputs
from distilled_tiny.runtime import run_batches, session, sha256
from ossp_router.protocol import load_input


def predict_quality(
    input_path: Path,
    model_path: Path = MODEL_DIR / "student.onnx",
    preprocess_path: Path = MODEL_DIR / "preprocess.json",
    vocab_path: Path = TOKENIZER_DIR / "vocab.txt",
    *,
    threads: int | None = None,
    batch_size: int | None = None,
) -> tuple[object, np.ndarray, dict[str, object]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"input not found: {input_path}")
    inputs = load_input(input_path)
    quality, preprocess = predict_inputs(
        inputs,
        model_path,
        preprocess_path,
        vocab_path,
        threads=threads,
        batch_size=batch_size,
    )
    return inputs, quality, preprocess


def predict_inputs(
    inputs: object,
    model_path: Path = MODEL_DIR / "student.onnx",
    preprocess_path: Path = MODEL_DIR / "preprocess.json",
    vocab_path: Path = TOKENIZER_DIR / "vocab.txt",
    *,
    threads: int | None = None,
    batch_size: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Predict q0/q1/q2 from an already parsed official InputBatch."""

    for label, path in {
        "ONNX model": model_path,
        "preprocess manifest": preprocess_path,
        "BERT vocabulary": vocab_path,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
    if preprocess.get("artifact_type") != "ossp-distilled-tiny-preprocess-v1":
        raise ValueError("unsupported preprocess manifest")
    actual_hash = sha256(model_path)
    expected_hash = preprocess.get("models", {}).get(model_path.name)
    if expected_hash is None:
        expected_hash = preprocess.get("model_sha256")
    if expected_hash != actual_hash:
        raise ValueError(f"model SHA-256 mismatch: {actual_hash} != {expected_hash}")
    feeds = prepare_inputs(inputs, vocab_path, preprocess)
    inference = session(model_path, threads or int(preprocess["threads"]))
    quality = run_batches(
        inference, feeds, batch_size or int(preprocess["batch_size"])
    ).astype(np.float32)
    if quality.shape != (len(inputs.episodes), 3):
        raise ValueError(f"unexpected ONNX output shape: {quality.shape}")
    return quality, preprocess


def run(args: argparse.Namespace) -> dict[str, object]:
    inputs, quality, preprocess = predict_quality(
        args.input,
        args.model,
        args.preprocess,
        args.vocab,
        threads=args.threads,
        batch_size=args.batch_size,
    )
    rows = [
        {
            "episode_id": episode.episode_id,
            "q0": float(values[0]),
            "q1": float(values[1]),
            "q2": float(values[2]),
            "g01": float(values[1] - values[0]),
            "g12": float(values[2] - values[1]),
        }
        for episode, values in zip(inputs.episodes, quality, strict=True)
    ]
    report = {
        "artifact_type": "ossp-distilled-tiny-quality-predictions-v1",
        "note": "Quality predictions, not a challenge submission.",
        "model_order": preprocess["model_order"],
        "model_sha256": sha256(args.model),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "student.onnx")
    parser.add_argument(
        "--preprocess", type=Path, default=MODEL_DIR / "preprocess.json"
    )
    parser.add_argument("--vocab", type=Path, default=TOKENIZER_DIR / "vocab.txt")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run(args)
    print(json.dumps({"output": str(args.output), "rows": len(report["rows"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
