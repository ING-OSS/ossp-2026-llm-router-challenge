# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Create the final SKT technical-submission JSON from a pushed image digest."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "submission-ossp-skt.json"
REPOSITORY_URL = (
    "https://github.com/ING-OSS/ossp-2026-llm-router-challenge"
)
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
IMAGE_DIGEST = re.compile(
    r"^[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+"
    r"@sha256:[0-9a-f]{64}$"
)


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _require_clean_code_tree() -> None:
    dirty = []
    for line in _git("status", "--porcelain", "--untracked-files=all").splitlines():
        path = line[3:] if len(line) > 3 else line
        if path != OUTPUT.name:
            dirty.append(line)
    if dirty:
        raise SystemExit(
            "오류: 이미지와 코드 커밋의 대응을 고정할 수 있도록 먼저 모든 "
            "소스 변경을 커밋하세요:\n" + "\n".join(dirty)
        )


def _write_atomic(path: pathlib.Path, payload: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="공개 ARM64 이미지 digest로 최종 기술 제출 JSON을 만듭니다."
    )
    parser.add_argument(
        "--image-digest",
        required=True,
        help="ghcr.io/...@sha256:<64자리> 형태의 공개 이미지",
    )
    args = parser.parse_args()

    _require_clean_code_tree()
    commit_sha = _git("rev-parse", "HEAD")
    if COMMIT_SHA.fullmatch(commit_sha) is None:
        raise SystemExit(f"오류: 올바르지 않은 코드 커밋 SHA: {commit_sha}")
    if IMAGE_DIGEST.fullmatch(args.image_digest) is None:
        raise SystemExit(
            "오류: 이미지에는 태그가 아니라 "
            "registry/repository@sha256:<64자리 소문자> 전체 digest가 필요합니다."
        )

    payload = {
        "schema_version": 1,
        "challenge_id": "ossp-2026-llm-router-challenge",
        "repository_url": REPOSITORY_URL,
        "commit_sha": commit_sha,
        "image_digest": args.image_digest,
        "primary_license": "Apache-2.0",
    }
    _write_atomic(OUTPUT, payload)
    print(f"작성 완료: {OUTPUT}")
    print(f"이미지 빌드 코드 커밋: {commit_sha}")
    print("다음 명령: python3 tools/validate_technical_submission.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
