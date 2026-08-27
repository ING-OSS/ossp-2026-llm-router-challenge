# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Release-safe entry point for the exact V32.1c/V24 routing chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from . import orchestrator as _orchestrator


_ORIGINAL_RUN_STAGE = _orchestrator._v321_run_stage
_DISABLED_TIER_FALLBACKS: dict[str, str] = {}


def _last_valid_submission(outputs: Mapping[str, Any]) -> str | None:
    files = outputs.get("files", {})
    if not isinstance(files, Mapping):
        return None
    for value in reversed(tuple(files.values())):
        if isinstance(value, str) and Path(value).is_file():
            return value
    return None


def _run_stage_with_safe_fallback(
    stage: Mapping[str, Any],
    bundle_root: Path,
    input_path: Path,
    outputs: Dict[str, Dict[str, str]],
    work_root: Path,
    tier: str,
) -> Dict[str, Dict[str, str]]:
    """Keep the last valid policy when a later micro-allocation is inapplicable."""

    reference = stage.get("reference_files", {}).get(tier)
    disabled_fallback = _DISABLED_TIER_FALLBACKS.get(tier)
    if disabled_fallback is not None and isinstance(reference, str):
        recovered = {
            "dirs": dict(outputs.get("dirs", {})),
            "files": dict(outputs.get("files", {})),
        }
        recovered["files"][reference] = disabled_fallback
        return recovered

    try:
        return _ORIGINAL_RUN_STAGE(
            stage, bundle_root, input_path, outputs, work_root, tier
        )
    except RuntimeError as exc:
        fallback = _last_valid_submission(outputs)
        if fallback is None or not isinstance(reference, str):
            raise

        recovered = {
            "dirs": dict(outputs.get("dirs", {})),
            "files": dict(outputs.get("files", {})),
        }
        recovered["files"][reference] = fallback
        _DISABLED_TIER_FALLBACKS[tier] = fallback
        print(
            f"경고: {stage.get('name', 'unknown')} 단계를 적용할 수 없어 "
            "마지막 유효 라우팅 결과를 사용합니다: "
            f"{str(exc).splitlines()[0]}",
            file=_orchestrator._v321_sys.stderr,
        )
        return recovered


_orchestrator._v321_run_stage = _run_stage_with_safe_fallback


def main(argv: list[str] | None = None) -> int:
    """Run the final router through the release-safe stage adapter."""

    _DISABLED_TIER_FALLBACKS.clear()
    return _orchestrator.final_router_main(argv)
