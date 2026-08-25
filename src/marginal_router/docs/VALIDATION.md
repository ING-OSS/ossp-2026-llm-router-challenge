# Validation

## Dev exact reproduction

최종 runtime chain을 독립적으로 재실행했을 때 기존 primary submission과 episode-level decision이 모두 일치했다.

```text
fast     exact=True diff=0
balanced exact=True diff=0
premium  exact=True diff=0
```

## Source integration exact check

`container/entrypoint.py`가 최종 runtime을 호출하도록 적용한 뒤에도 동일했다.

```text
fast     source_exact=True diff=0
balanced source_exact=True diff=0
premium  source_exact=True diff=0
```

## Python 3.11 authoritative unit tests

```text
Ran 261 tests
OK (skipped=19)
OFFICIAL_PY311_UNIT_TESTS_OK=True
```

## Docker image checks

검증 중 다음 항목은 PASS했다.

```text
image_linux_arm64=PASS
image_no_volume=PASS

fast_exact_expected=PASS
balanced_exact_expected=PASS
premium_exact_expected=PASS

fast_deterministic=PASS
balanced_deterministic=PASS
premium_deterministic=PASS

fast_runtime_90s=PASS
balanced_runtime_90s=PASS
premium_runtime_90s=PASS

all ID/order invariants=PASS
```

당시 `official_runtime_check`와 source audit는 별도 packaging/source-policy 문제 때문에 최종 gate에서 추가 정리가 필요했다. 이는 모델 decision correctness와는 별개다.

## Runtime latency observed on host dry-run

```text
Fast      ~24 s
Balanced  ~32 s
Premium   ~16 s
```

최종 제출 판단은 challenge의 제한된 Docker 환경에서 다시 검증해야 한다.
