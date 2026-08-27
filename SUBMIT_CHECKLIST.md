# 제출 체크리스트

현재 dev **0.689943** (공식 self-check), near_budget 0개, 여유 9.5 / 13.7 / 18.4%.

## 파일 배치

| 파일 | 위치 |
| --- | --- |
| `features.py`, `core.py`, `router_run.py`, `train_router.py` | 저장소 루트 |
| `Dockerfile.router`, `router-run`, `requirements.txt` | `container/` |
| `router.joblib` | `artifacts/` (train_router.py 로 생성) |
| `submission-ossp-skt.json` | 저장소 루트 |
| `THIRD_PARTY_NOTICES_ADD.md` 내용 | 기존 `THIRD_PARTY_NOTICES.md` 에 병합 |

`container/router-run` 은 실행 권한 필요: `chmod +x container/router-run`

## 1. 기반 이미지 다이제스트 고정

```bash
docker buildx imagetools inspect python:3.11.15-slim-bookworm
```

출력의 인덱스 다이제스트로 `Dockerfile.router` 의 `REPLACE_WITH_INDEX_DIGEST`
두 곳을 교체한다. 규정상 태그만 있는 참조는 받지 않는다.

## 2. artifact 생성

```bash
python3 train_router.py --repo-root . --out artifacts/router.joblib
```

약 130초. 결과 58.4 MB. SHA-256 을 THIRD_PARTY_NOTICES.md 에 기록한다.

## 3. arm64 빌드

```bash
docker buildx build --platform linux/arm64 \
  --file container/Dockerfile.router \
  --tag ossp-router:check --load .
```

## 4. 공식 자원 한도 검사  ← 가장 중요

```bash
PYTHONPATH=src python3 tools/check_runtime.py \
  --image ossp-router:check \
  --report build/runtime-check-report.json
```

x86 비격리 환경 실측값(참고): 880문항 6.0초, 1,760문항 9.1초, RSS 621 MB,
최대 스레드 2개. **arm64 격리에서 재확인 필요.**

| 한도 | 값 | 예상 사용률 |
| --- | --- | --- |
| 등급별 실행 시간 | 90초 | 10~20% |
| 메모리 | 2 GiB | 30% |
| 프로세스·스레드 | 32개 | 6% |
| 압축 계층 합계 | 1 GiB | 미측정 (sklearn 포함, 여유 예상) |
| 루트 파일시스템 | 2 GiB | 미측정 |

## 5. push 및 다이제스트 확보

```bash
docker buildx build --platform linux/arm64 --push \
  --file container/Dockerfile.router \
  --tag ghcr.io/<계정>/ossp-router:submission .
docker buildx imagetools inspect ghcr.io/<계정>/ossp-router:submission
```

## 6. 커밋 순서 (규정상 두 개로 분리)

```bash
git add -A && git commit -m "라우터 구현"      # 코드 커밋
git rev-parse HEAD                              # -> commit_sha
# 이 커밋에서 이미지 빌드 & push -> image_digest
# submission-ossp-skt.json 의 세 필드 교체 후
git add submission-ossp-skt.json && git commit -m "기술 제출 정보"
python3 tools/validate_technical_submission.py
git rev-parse HEAD                              # -> 결과보고서 URL 용
```

결과보고서 `프로젝트 등록 URL`:
`https://github.com/<계정>/<저장소>/tree/<두 번째 커밋 SHA>`

## 7. 결과보고서

- 5페이지 이내, 첫 쪽 안내 문구 삭제
- 파일명 `2026 오픈소스 개발자대회 결과보고서_접수번호(팀명)`
- 원본(한글/Word) 1개 + PDF 1개
- **AI 모델 항목**: `해당 없음 — 실행 이미지에 AI 모델을 탑재하지 않음`
- AI 코딩 도구 사용 범위 기재
- SBOM: `container/requirements.txt` 5개 패키지 + 기반 이미지

## 마감

2026-08-27 18:00 KST, https://osscontest.kr/
