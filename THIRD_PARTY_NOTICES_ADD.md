# 제출 이미지 구성 요소 고지 (기존 THIRD_PARTY_NOTICES.md 에 추가)

`docs/SUBMISSION.md`가 요구하는 항목을 기재한다.

## AI 모델

**해당 없음 — 실행 이미지에 AI 모델을 탑재하지 않음.**

사전학습 언어모델, 토크나이저, 임베딩 가중치를 포함하지 않는다. 이미지에 들어가는
학습 산출물은 공개 Train 자료로 이 저장소의 코드가 적합시킨 통계 모델뿐이다.

## 학습한 분류기 (자체 산출물)

| 항목 | 내용 |
| --- | --- |
| 파일 | `artifacts/router.joblib` |
| SHA-256 | `237579e0183febd1900168fd0469c3c057ddd2d7e9489b01a7e9aed98ad29f4c` |
| 크기 | 58,364,928 B |
| 용도 | 프롬프트별 모델 품질·비용 예측 |
| 생성 방법 | `python3 train_router.py --repo-root . --out artifacts/router.joblib` |
| 학습 자료 | 공개 Train 1,760문항 (`data/materialized/train/inputs.json`, `data/train/outcomes.json`) |
| 라이선스 | Apache-2.0 (이 저장소와 동일) |
| 외부 가중치 | 없음 |

포함 구성 요소:

- `ng` 분류기 — HistGradientBoostingClassifier
- `in_tok` 회귀 3개 — Ridge (문자 클래스 8개 입력)
- `out_tok` 회귀 6개 — HistGradientBoostingRegressor 3 + ExtraTreesRegressor 3
- 비용 보정 — TfidfVectorizer(char_wb 3-5gram) + MiniBatchKMeans(k=200) 군집별 smearing, 전역 κ
- 품질 예측 — TfidfVectorizer(word 1-2gram, char_wb 2-5gram) + LogisticRegression 3,
  ExtraTreesRegressor 3

재현: 같은 커밋에서 위 명령을 실행하면 동일한 산출물이 나온다. 해시 함수는
FNV-1a 64비트를 직접 구현해 `PYTHONHASHSEED`에 의존하지 않는다.

## 런타임 의존성

`container/requirements.txt`로 고정하며 빌드 단계에서만 설치한다.
실행 중 다운로드는 없다.

| 패키지 | 버전 | 라이선스 | 업스트림 |
| --- | --- | --- | --- |
| numpy | 2.4.4 | BSD-3-Clause | https://github.com/numpy/numpy |
| scipy | 1.17.1 | BSD-3-Clause | https://github.com/scipy/scipy |
| scikit-learn | 1.8.0 | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn |
| joblib | 1.5.3 | BSD-3-Clause | https://github.com/joblib/joblib |
| threadpoolctl | 3.6.0 | BSD-3-Clause | https://github.com/joblib/threadpoolctl |

전부 BSD-3-Clause로 `docs/CHALLENGE_RULES.md`의 허용 목록에 있다. 배포 시 각
패키지의 저작권 고지와 라이선스 본문을 이미지 안에 보존한다(`/usr/local/lib/
python3.11/site-packages/*.dist-info/`).

## 기반 이미지

| 항목 | 내용 |
| --- | --- |
| 이미지 | `python:3.11.15-slim-bookworm` |
| 다이제스트 | (빌드 시 고정 — `docker buildx imagetools inspect` 결과 기재) |
| 플랫폼 | `linux/arm64` |
| 라이선스 | Python-2.0 (Python), 각 Debian 패키지 라이선스 |

기준 예시의 `python:3.11.15-alpine3.23` 대신 debian slim을 쓴다. alpine은 musl
libc라 numpy·scipy·scikit-learn의 공식 휠이 배포되지 않아 소스 빌드가 필요하고,
빌드 도구를 이미지에 넣거나 다단계 빌드로도 재현성이 떨어진다.
`docs/RUNTIME.md`는 기준 Dockerfile 사용을 강제하지 않는다.

이 저장소의 Apache-2.0 라이선스는 기반 이미지 안의 Python, Debian 패키지를
재라이선스하지 않는다.
