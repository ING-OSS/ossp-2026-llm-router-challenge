<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# Third-party data notices

This notice applies to the adapted public prompts in
`data/train/inputs-base.json` and `data/dev/inputs-base.json`. Those files are
collections. Each source-derived part retains the license below; the project
Apache-2.0 license does not relicense third-party material. Exact revisions,
artifact hashes, and license-evidence hashes are in
`data/sources/source-pins.v1.json`.

## Belebele Korean

Source: Meta's Belebele `kor_Hang` configuration. Licensed under
[CC BY-SA 4.0](LICENSES/CC-BY-SA-4.0.txt). The released adaptation selects a
subset, formats passage, question, and choices as a prompt, omits answer
labels, and assigns opaque episode IDs. No endorsement is implied.

Attribution: Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe,
Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke
Zettlemoyer, and Madian Khabsa, *The Belebele Benchmark: a Parallel Reading
Comprehension Dataset in 122 Language Variants*, ACL 2024.

## CRUXEval

Copyright (c) 2023 Meta. Licensed under the [MIT License](LICENSES/MIT.txt).
The adaptation selects public examples, applies the direct input-prediction or
output-prediction prompt, omits reference inputs or outputs, and assigns opaque
episode IDs.

## GSM8K

Copyright (c) 2021 OpenAI. Licensed under the
[MIT License](LICENSES/MIT.txt). The adaptation selects public test questions,
omits solutions and answers, and assigns opaque episode IDs.

## BABILong 4K/16K components

The bAbI tasks component is copyright (c) 2015-present Facebook, Inc. and is
licensed under [BSD-3-Clause](LICENSES/BSD-3-Clause.txt). BABILong code and the
PG-19 component are licensed under [Apache-2.0](LICENSES/Apache-2.0.txt).
The adaptation uses only approved 4K and 16K configurations, adds a zero-shot task
instruction, omits targets, and assigns opaque episode IDs. Neither Facebook
nor any contributor endorses this project.

## Apache-2.0 sources

The following adapted public prompts are licensed under
[Apache-2.0](LICENSES/Apache-2.0.txt): DeepMind Mathematics, HRMCR, RuleTaker,
and TruthfulQA. Each adaptation selects an approved subset, formats only the
question-side prompt, omits gold answers and solutions, and assigns opaque
episode IDs. DeepMind Mathematics prompts are independently reproduced from
the pinned upstream generator and verified against the reference hashes in the
source record.

## Source-fetch-only material

AIME problem text is not included in this repository or release archive.
`data/train/aime-selection.json` and `data/dev/aime-selection.json` contain
only public source keys and expected prompt hashes. Users fetch the pinned
public sources and materialize those prompts locally.

## Submitted router runtime

### AI model

해당 없음 — 실행 이미지에 AI 모델을 탑재하지 않음.

사전학습 언어모델, 토크나이저 또는 임베딩 가중치를 포함하지 않는다.
실행 이미지에 포함되는 학습 산출물은 공개 Train 자료로 이 저장소의 코드가
학습한 통계적 라우터뿐이다.

### Participant-trained router artifact

| 항목 | 내용 |
| --- | --- |
| 파일 | `artifacts/router.joblib` |
| SHA-256 | `1316754ed7b0aec1b2de90964f1dd712e3d312be8ebc155054c60f41438093b5` |
| 크기 | 53,745,354 B |
| 용도 | 프롬프트별 모델 품질·비용 예측 및 예산 인식 라우팅 |
| 생성 명령 | `python3 train_router.py --repo-root . --out artifacts/router.joblib` |
| 학습 자료 | 공개 Train 1,760문항 |
| 외부 가중치 | 없음 |

### Runtime Python dependencies

`container/requirements.txt`에 버전을 고정하고 이미지 빌드 단계에서 설치한다.
실행 중 다운로드하지 않는다.

| 패키지 | 버전 | 라이선스 | 업스트림 |
| --- | --- | --- | --- |
| numpy | 2.4.4 | BSD-3-Clause | https://github.com/numpy/numpy |
| scipy | 1.17.1 | BSD-3-Clause | https://github.com/scipy/scipy |
| scikit-learn | 1.8.0 | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn |
| joblib | 1.5.3 | BSD-3-Clause | https://github.com/joblib/joblib |
| threadpoolctl | 3.6.0 | BSD-3-Clause | https://github.com/joblib/threadpoolctl |

### Runtime base image

| 항목 | 내용 |
| --- | --- |
| 이미지 | `python:3.11.15-slim-bookworm` |
| SHA-256 index digest | `sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3` |
| 제출 플랫폼 | `linux/arm64` |

기반 운영체제와 Python 런타임 및 직접 포함하는 Python 패키지의 라이선스는
각 업스트림의 조건을 유지하며, 이 저장소의 Apache-2.0 라이선스로
재라이선스하지 않는다.
