<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# SKT 지정과제 최종 제출 절차

이 문서는 `release-safe` 브랜치를 처음 받는 상태부터 최종 프로젝트 URL을
만드는 과정만 설명합니다. 원본 SKT 저장소로 Pull Request를 만들 필요는
없습니다.

## 1. 서버에서 최종 코드 받기

```console
cd /home/korean/seokbeom/llm_routing/ossp-2026-llm-router-challenge
git fetch origin --prune
git switch release-safe
git pull --ff-only origin release-safe
git status --short
git rev-parse HEAD
```

`git status --short`에 아무 내용도 없어야 합니다. 이때 출력되는 HEAD가 이미지
빌드 코드 커밋입니다.

## 2. 기본 검증

```console
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'

for tier in fast balanced premium; do
  PYTHONPATH=src python3 container/entrypoint.py \
    --input data/toy/inputs.json \
    --tier "$tier" \
    --output "/tmp/ossp-${tier}.json"
done
```

Docker를 사용할 수 있으면 공개 Train/Dev materialization 후 공식 제한 검사를
실행합니다.

```console
SOURCE_MANIFEST_SHA256="$(PYTHONPATH=src python3 tools/benchmark_runtime.py \
  --print-source-manifest-sha256)"

docker build --pull --platform linux/arm64 \
  --build-arg "SOURCE_MANIFEST_SHA256=${SOURCE_MANIFEST_SHA256}" \
  --file container/Dockerfile \
  --tag ossp-router:check .

PYTHONPATH=src python3 tools/check_runtime.py \
  --image ossp-router:check \
  --report build/runtime-check-report.json
```

## 3. GitHub Actions에서 최종 ARM64 이미지 만들기

1. GitHub 저장소의 **Actions** 탭으로 이동합니다.
2. **Build final ARM64 image**를 선택합니다.
3. **Run workflow**에서 `release-safe` 브랜치를 선택하고 실행합니다.
4. 완료된 실행의 Summary에서 `Image:` 전체 값을 복사합니다.
5. 저장소의 **Packages**에서 새 컨테이너 패키지를 **Public**으로 변경합니다.

이미지는 다음 형식이어야 합니다.

```text
ghcr.io/ing-oss/ossp-2026-llm-router-challenge@sha256:<64자리>
```

태그가 붙은 주소가 아니라 `@sha256:` 전체 digest를 사용합니다.

## 4. 기술 제출 JSON 만들기

Actions가 빌드한 코드와 로컬 HEAD가 같은지 먼저 확인합니다. Actions Summary의
`Code commit`과 아래 값이 같아야 합니다.

```console
git rev-parse HEAD
```

복사한 전체 이미지 digest로 파일을 생성합니다.

```console
python3 tools/prepare_final_submission.py \
  --image-digest 'ghcr.io/ing-oss/ossp-2026-llm-router-challenge@sha256:<64자리>'

python3 tools/validate_technical_submission.py
```

## 5. JSON만 별도 커밋

```console
git status --short
git add submission-ossp-skt.json
git commit -m "Add SKT technical submission metadata"
git push origin release-safe
git rev-parse HEAD
```

마지막 `git rev-parse HEAD`가 결과보고서에 적을 제출 스냅샷 SHA입니다.

```text
https://github.com/ING-OSS/ossp-2026-llm-router-challenge/tree/<제출 스냅샷 SHA>
```

`submission-ossp-skt.json` 안의 `commit_sha`는 앞쪽 이미지 빌드 코드 커밋이고,
결과보고서 URL은 JSON까지 포함한 뒤쪽 커밋을 가리킵니다. 대회 사이트에는
JSON을 별도로 올리지 않고, 결과보고서 원본과 PDF를 업로드합니다.
