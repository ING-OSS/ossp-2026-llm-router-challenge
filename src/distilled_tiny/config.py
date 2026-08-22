"""Shared paths and manifest loading for the Tiny module."""

from __future__ import annotations

import json
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]
METHOD_PATH = Path(__file__).with_name("method.v1.json")
ARTIFACT_DIR = PACKAGE_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "model"
TOKENIZER_DIR = ARTIFACT_DIR / "tokenizer"
TRAINING_DIR = ARTIFACT_DIR / "training"
REPRODUCTION_DIR = ROOT / "build/distilled-tiny-reproduction"


def load_method(path: Path = METHOD_PATH) -> dict[str, object]:
    method = json.loads(path.read_text(encoding="utf-8"))
    if method.get("artifact_type") != "ossp-distilled-tiny-method-v1":
        raise ValueError(f"unsupported method manifest: {path}")
    student = method.get("student", {})
    if student.get("candidate") != "generic_head_tail_gain_rank_relation":
        raise ValueError(
            "the frozen method requires generic_head_tail_gain_rank_relation"
        )
    return method
