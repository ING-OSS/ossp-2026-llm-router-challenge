"""Train the fixed distilled BERT-Tiny quality model."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from distilled_tiny.config import (
    REPRODUCTION_DIR,
    ROOT,
    TRAINING_DIR,
    load_method,
)
from distilled_tiny.tokenization import WordPieceTokenizer, tokenize_inputs
from distilled_tiny.model import TinyResponseStudent, distillation_loss, optimizer_groups
from ossp_router.protocol import MODEL_IDS, load_input, load_outcomes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-inputs",
        type=Path,
        default=ROOT / "data/materialized/train/inputs.json",
    )
    parser.add_argument(
        "--train-outcomes", type=Path, default=ROOT / "data/train/outcomes.json"
    )
    parser.add_argument(
        "--dev-inputs",
        type=Path,
        default=ROOT / "data/materialized/dev/inputs.json",
    )
    parser.add_argument(
        "--dev-outcomes", type=Path, default=ROOT / "data/dev/outcomes.json"
    )
    parser.add_argument(
        "--include-dev",
        action="store_true",
        help="Fit the final artifact on Train+Dev instead of Train only.",
    )
    parser.add_argument(
        "--teacher-targets",
        type=Path,
        default=TRAINING_DIR / "full-teacher-targets.npz",
    )
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/bert-tiny")
    parser.add_argument("--output-dir", type=Path, default=REPRODUCTION_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--check-only", action="store_true")
    return parser


def _targets(inputs: object, outcomes: object) -> tuple[np.ndarray, np.ndarray]:
    by_key = {
        (outcome.episode_id, outcome.model_id): outcome for outcome in outcomes.outcomes
    }
    quality = np.empty((len(inputs.episodes), len(MODEL_IDS)), dtype=np.float32)
    generations = np.empty(len(inputs.episodes), dtype=np.float32)
    for row, episode in enumerate(inputs.episodes):
        selected = [by_key[(episode.episode_id, model_id)] for model_id in MODEL_IDS]
        counts = {outcome.num_generations for outcome in selected}
        if len(counts) != 1:
            raise ValueError(f"generation counts differ for {episode.episode_id}")
        generations[row] = counts.pop()
        quality[row] = [float(outcome.score) for outcome in selected]
    return quality, generations


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate(args: argparse.Namespace) -> dict[str, object]:
    required = {
        "train inputs": args.train_inputs,
        "train outcomes": args.train_outcomes,
        "teacher targets": args.teacher_targets,
        "Tiny config": args.model_dir / "config.json",
        "Tiny vocabulary": args.model_dir / "vocab.txt",
    }
    if args.include_dev:
        required.update(
            {
                "dev inputs": args.dev_inputs,
                "dev outcomes": args.dev_outcomes,
            }
        )
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required files: " + ", ".join(missing))
    if not any(
        (args.model_dir / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin")
    ):
        raise FileNotFoundError(f"Tiny weights not found in {args.model_dir}")
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("this training path requires a torch build with torch.optim.Muon")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    method = load_method()
    cache = np.load(args.teacher_targets, allow_pickle=False)
    shapes = {name: list(cache[name].shape) for name in cache.files}
    expected = {"train_quality", "train_variance", "train_representation"}
    if args.include_dev:
        expected.update({"dev_quality", "dev_variance", "dev_representation"})
    if not expected.issubset(cache.files):
        raise ValueError(f"teacher target keys must include {sorted(expected)}")
    return {
        "status": "passed",
        "method": method["method_id"],
        "device": str(device),
        "teacher_shapes": shapes,
        "output_dir": str(args.output_dir),
        "training_scope": "train+dev" if args.include_dev else "train",
    }


@torch.no_grad()
def _predict(
    model: TinyResponseStudent,
    feeds: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    rows = len(feeds["input_ids"])
    values = []
    for start in range(0, rows, batch_size):
        selected = slice(start, start + batch_size)
        batch = [
            torch.as_tensor(feeds[name][selected], device=device)
            for name in ("input_ids", "attention_mask", "token_type_ids")
        ]
        logits, _representation = model(*batch)
        values.append(logits.sigmoid().float().cpu().numpy())
    return np.concatenate(values)


def run(args: argparse.Namespace) -> dict[str, object]:
    preflight = _validate(args)
    started = time.perf_counter()
    method = load_method()
    student = method["student"]
    seed = int(student["seed"]) + 100_000
    epochs = int(student["max_epochs"])
    batch_size = int(student["batch_size"])
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    _seed_everything(seed)

    input_parts = [load_input(args.train_inputs)]
    observed_parts = [_targets(input_parts[0], load_outcomes(args.train_outcomes))]
    teacher = np.load(args.teacher_targets, allow_pickle=False)
    teacher_prefixes = ["train"]
    if args.include_dev:
        input_parts.append(load_input(args.dev_inputs))
        observed_parts.append(_targets(input_parts[1], load_outcomes(args.dev_outcomes)))
        teacher_prefixes.append("dev")
    observed_quality = np.concatenate([part[0] for part in observed_parts])
    generations = np.concatenate([part[1] for part in observed_parts])
    teacher_quality = np.concatenate(
        [teacher[f"{prefix}_quality"].astype(np.float32) for prefix in teacher_prefixes]
    )
    teacher_variance = np.concatenate(
        [teacher[f"{prefix}_variance"].astype(np.float32) for prefix in teacher_prefixes]
    )
    teacher_representation = np.concatenate(
        [
            teacher[f"{prefix}_representation"].astype(np.float32)
            for prefix in teacher_prefixes
        ]
    )
    if teacher_quality.shape != observed_quality.shape:
        raise ValueError("teacher targets do not align with training rows")
    if teacher_variance.shape != observed_quality.shape:
        raise ValueError("teacher variance does not align with training rows")
    if len(teacher_representation) != len(observed_quality):
        raise ValueError("teacher representations do not align with training rows")

    tokenizer = WordPieceTokenizer(args.model_dir / "vocab.txt")
    token_parts = [
        tokenize_inputs(
            inputs,
            tokenizer,
            max_length=int(student["max_length"]),
        )
        for inputs in input_parts
    ]
    feeds = {
        name: np.concatenate([part[name] for part in token_parts])
        for name in ("input_ids", "attention_mask", "token_type_ids")
    }

    trials = generations[:, None]
    initial_quality = (observed_quality * trials).sum(axis=0) / trials.sum(axis=0)
    model = TinyResponseStudent(
        args.model_dir,
        hidden_size=int(student["hidden_size"]),
        fusion_size=int(student["fusion_size"]),
        trainable_layers=int(student["trainable_layers"]),
        dropout=float(student["dropout"]),
        initial_quality=initial_quality,
    ).to(device)
    muon_parameters, adam_parameters = optimizer_groups(model)
    muon = torch.optim.Muon(
        muon_parameters,
        lr=float(student["muon_lr"]),
        weight_decay=float(student["weight_decay"]),
        momentum=0.95,
        ns_steps=5,
        adjust_lr_fn="match_rms_adamw",
    )
    adam = torch.optim.AdamW(
        adam_parameters,
        lr=float(student["adam_lr"]),
        betas=(0.9, 0.95),
        weight_decay=float(student["weight_decay"]),
    )
    steps_per_epoch = math.ceil(len(observed_quality) / batch_size)
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, round(total_steps * float(student["warmup_fraction"])))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    schedulers = (
        torch.optim.lr_scheduler.LambdaLR(muon, schedule),
        torch.optim.lr_scheduler.LambdaLR(adam, schedule),
    )
    tensors = {
        name: torch.as_tensor(value)
        for name, value in {
            **feeds,
            "observed": observed_quality,
            "trials": generations,
            "teacher_quality": teacher_quality,
            "teacher_variance": teacher_variance,
            "teacher_representation": teacher_representation,
        }.items()
    }
    generator = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        model.set_training_mode()
        totals: dict[str, list[float]] = {}
        permutation = generator.permutation(len(observed_quality))
        for start in range(0, len(permutation), batch_size):
            indexes = permutation[start : start + batch_size]
            muon.zero_grad(set_to_none=True)
            adam.zero_grad(set_to_none=True)
            batch = [
                tensors[name][indexes].to(device)
                for name in ("input_ids", "attention_mask", "token_type_ids")
            ]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits, representation = model(*batch)
                loss, components = distillation_loss(
                    logits,
                    representation,
                    tensors["observed"][indexes].to(device),
                    tensors["trials"][indexes].to(device),
                    tensors["teacher_quality"][indexes].to(device),
                    tensors["teacher_variance"][indexes].to(device),
                    tensors["teacher_representation"][indexes].to(device),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(student["max_grad_norm"]))
            muon.step()
            adam.step()
            for scheduler in schedulers:
                scheduler.step()
            for name, value in components.items():
                totals.setdefault(name, []).append(float(value))
        history.append(
            {"epoch": epoch + 1, **{name: sum(v) / len(v) for name, v in totals.items()}}
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "student.pt"
    checkpoint = {
        "artifact_type": "ossp-tiny-response-student-v1",
        "candidate_name": str(student["candidate"]),
        "candidate": method["objective"],
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "model_config": {
            "hidden_size": int(student["hidden_size"]),
            "fusion_size": int(student["fusion_size"]),
            "trainable_layers": int(student["trainable_layers"]),
            "dropout": float(student["dropout"]),
            "initial_quality": initial_quality.tolist(),
        },
        "tokenization": {
            "max_length": int(student["max_length"]),
            "context_selection": str(student["context_selection"]),
            "tokenizer_prefix": str(student["tokenizer_prefix"]),
        },
        "temperature": 1.0,
        "training_scope": "train+dev" if args.include_dev else "train",
        "training_rows": len(observed_quality),
    }
    torch.save(checkpoint, checkpoint_path)
    predictions = _predict(model, feeds, device, int(student["eval_batch_size"]))
    np.savez_compressed(args.output_dir / "train-predictions.npz", quality=predictions)
    report = {
        **preflight,
        "status": "trained",
        "rows": len(observed_quality),
        "training_scope": "train+dev" if args.include_dev else "train",
        "epochs": epochs,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "parameters": sum(value.numel() for value in model.parameters()),
        "train_quality_mae": float(np.abs(predictions - observed_quality).mean()),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "training-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.check_only:
        print(json.dumps(_validate(args), ensure_ascii=False, indent=2))
        return 0
    report = run(args)
    print(json.dumps({key: report[key] for key in ("status", "checkpoint", "elapsed_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
