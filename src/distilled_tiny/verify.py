"""Verify packaged checkpoint, preprocessing, and ONNX numerical parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from distilled_tiny.config import MODEL_DIR, ROOT, TRAINING_DIR
from distilled_tiny.export import load_student
from distilled_tiny.tokenization import prepare_inputs
from distilled_tiny.model import OnnxStudent
from distilled_tiny.predict import predict_quality
from distilled_tiny.runtime import sha256
from ossp_router.protocol import load_input


def run(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = Path(__file__).with_name("MANIFEST.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package_dir = Path(__file__).resolve().parent
    checked = 0
    for relative, expected in manifest["sha256"].items():
        path = package_dir / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"artifact digest mismatch: {relative}")
        checked += 1

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    student = load_student(checkpoint, args.model_dir)
    runtime = OnnxStudent(student, float(checkpoint.get("temperature", 1.0))).eval()
    preprocess = json.loads(args.preprocess.read_text(encoding="utf-8"))
    inputs = load_input(args.inputs)
    feeds = prepare_inputs(inputs, args.model_dir / "vocab.txt", preprocess)
    with torch.no_grad():
        raw = runtime(
            *[
                torch.from_numpy(feeds[name])
                for name in ("input_ids", "attention_mask", "token_type_ids")
            ]
        ).numpy()
    _inputs, onnx, _preprocess = predict_quality(
        args.inputs,
        args.model,
        args.preprocess,
        args.model_dir / "vocab.txt",
        threads=args.threads,
        batch_size=args.batch_size,
    )
    maximum_error = float(np.max(np.abs(raw - onnx)))
    argmax_changes = int(np.sum(raw.argmax(axis=1) != onnx.argmax(axis=1)))
    if maximum_error > 1e-5 or argmax_changes:
        raise ValueError("Float ONNX does not match the checkpoint")
    _inputs, train_dev_onnx, _preprocess = predict_quality(
        args.inputs,
        args.train_dev_model,
        args.preprocess,
        args.model_dir / "vocab.txt",
        threads=args.threads,
        batch_size=args.batch_size,
    )
    if train_dev_onnx.shape != onnx.shape or not np.isfinite(train_dev_onnx).all():
        raise ValueError("Train+Dev ONNX output is invalid")
    teacher = np.load(args.teacher_targets, allow_pickle=False)
    expected_teacher_shapes = {
        "train_quality": (1760, 3),
        "train_variance": (1760, 3),
        "train_representation": (1760, 256),
        "dev_quality": (880, 3),
        "dev_variance": (880, 3),
        "dev_representation": (880, 256),
    }
    for name, expected_shape in expected_teacher_shapes.items():
        if teacher[name].shape != expected_shape:
            raise ValueError(f"unexpected {name} shape")
    return {
        "status": "passed",
        "manifest_artifacts": checked,
        "rows": len(onnx),
        "float_max_absolute_error": maximum_error,
        "float_argmax_changes": argmax_changes,
        "train_dev_rows": len(train_dev_onnx),
        "train_dev_output_range": [
            float(train_dev_onnx.min()),
            float(train_dev_onnx.max()),
        ],
        "teacher_target_shapes": {name: list(teacher[name].shape) for name in teacher.files},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "student.pt")
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "student.onnx")
    parser.add_argument(
        "--train-dev-model",
        type=Path,
        default=MODEL_DIR / "student.train-dev.onnx",
    )
    parser.add_argument("--preprocess", type=Path, default=MODEL_DIR / "preprocess.json")
    parser.add_argument(
        "--teacher-targets",
        type=Path,
        default=TRAINING_DIR / "full-teacher-targets.npz",
    )
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/bert-tiny")
    parser.add_argument(
        "--inputs", type=Path, default=ROOT / "data/materialized/dev/inputs.json"
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(_parser().parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
