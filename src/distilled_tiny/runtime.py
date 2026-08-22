"""Small ONNX-only runtime helpers; no Torch dependency."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def run_batches(
    inference: ort.InferenceSession,
    feeds: dict[str, np.ndarray],
    batch_size: int,
) -> np.ndarray:
    values = []
    for start in range(0, len(feeds["input_ids"]), batch_size):
        selected = slice(start, start + batch_size)
        values.append(
            inference.run(
                ["quality"], {name: value[selected] for name, value in feeds.items()}
            )[0]
        )
    return np.concatenate(values)
