"""Export a Tiny checkpoint to validated Float and optional INT8 ONNX."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from distilled_tiny.config import (
    MODEL_DIR,
    REPRODUCTION_DIR,
    ROOT,
    load_method,
)
from distilled_tiny.tokenization import prepare_inputs
from distilled_tiny.model import OnnxStudent, TinyResponseStudent
from distilled_tiny.runtime import run_batches, session, sha256
from ossp_router.protocol import MODEL_IDS, load_input


def load_student(checkpoint: dict[str, object], model_dir: Path) -> TinyResponseStudent:
    config = checkpoint["model_config"]
    model = TinyResponseStudent(
        model_dir,
        hidden_size=int(config["hidden_size"]),
        fusion_size=int(config["fusion_size"]),
        trainable_layers=int(config["trainable_layers"]),
        dropout=float(config["dropout"]),
        initial_quality=config["initial_quality"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def _preprocess(checkpoint: dict[str, object], model_hashes: dict[str, str]) -> dict[str, object]:
    tokenization = checkpoint["tokenization"]
    method = load_method()
    return {
        "artifact_type": "ossp-distilled-tiny-preprocess-v1",
        "model_order": list(MODEL_IDS),
        "models": model_hashes,
        "max_length": int(tokenization["max_length"]),
        "context_selection": str(tokenization["context_selection"]),
        "threads": int(method["runtime"]["threads"]),
        "batch_size": int(method["runtime"]["batch_size"]),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    for label, path in {
        "checkpoint": args.checkpoint,
        "Tiny config": args.model_dir / "config.json",
        "Tiny vocabulary": args.model_dir / "vocab.txt",
        "validation inputs": args.inputs,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("candidate_name") != "generic_head_tail_gain_rank_relation":
        raise ValueError("unsupported checkpoint candidate")
    student = load_student(checkpoint, args.model_dir)
    runtime = OnnxStudent(student, float(checkpoint.get("temperature", 1.0))).eval()
    sequence = int(checkpoint["tokenization"]["max_length"])
    sample = (
        torch.ones((2, sequence), dtype=torch.long),
        torch.ones((2, sequence), dtype=torch.long),
        torch.zeros((2, sequence), dtype=torch.long),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.model_name).name != args.model_name or not args.model_name.endswith(
        ".onnx"
    ):
        raise ValueError("model name must be a plain .onnx file name")
    float_path = args.output_dir / args.model_name
    int8_path = args.output_dir / f"{Path(args.model_name).stem}.int8.onnx"
    with torch.no_grad():
        torch.onnx.export(
            runtime,
            sample,
            float_path,
            input_names=("input_ids", "attention_mask", "token_type_ids"),
            output_names=("quality",),
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                "quality": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    model_hashes = {float_path.name: sha256(float_path)}
    if not args.float_only:
        quantize_dynamic(
            float_path,
            int8_path,
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=("MatMul", "Gemm"),
        )
        model_hashes[int8_path.name] = sha256(int8_path)
    preprocess = _preprocess(checkpoint, model_hashes)
    preprocess_path = args.output_dir / "preprocess.json"
    preprocess_path.write_text(
        json.dumps(preprocess, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    inputs = load_input(args.inputs)
    feeds = prepare_inputs(inputs, args.model_dir / "vocab.txt", preprocess)
    with torch.no_grad():
        expected = runtime(
            *[
                torch.from_numpy(feeds[name])
                for name in ("input_ids", "attention_mask", "token_type_ids")
            ]
        ).numpy()
    report: dict[str, object] = {
        "rows": len(expected),
        "float_size_bytes": float_path.stat().st_size,
        "float_sha256": model_hashes[float_path.name],
        "model_name": float_path.name,
        "training_scope": checkpoint.get("training_scope", "train"),
    }
    for name, path in (("float", float_path), ("int8", int8_path)):
        if not path.is_file() or (name == "int8" and args.float_only):
            continue
        inference = session(path, args.threads)
        started = time.perf_counter()
        actual = run_batches(inference, feeds, args.batch_size)
        report[f"{name}_seconds"] = time.perf_counter() - started
        report[f"{name}_max_absolute_error"] = float(np.max(np.abs(actual - expected)))
        report[f"{name}_argmax_changes"] = int(
            np.sum(actual.argmax(axis=1) != expected.argmax(axis=1))
        )
    report_path = args.output_dir / "export-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "student.pt")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/bert-tiny")
    parser.add_argument(
        "--inputs", type=Path, default=ROOT / "data/materialized/dev/inputs.json"
    )
    parser.add_argument("--output-dir", type=Path, default=REPRODUCTION_DIR / "model")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--model-name", default="student.onnx")
    parser.add_argument("--float-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(_parser().parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
