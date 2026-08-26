# Marginal Router yongsoon

`marginal_router`는 SKT Efficient LLM Routing Challenge에서 개발한 **prompt-only, budget-aware LLM router 연구 라인**을 팀 내부에서 공유하기 위한 문서화 패키지다.

> 이 디렉터리는 최종 제출 runtime 자체가 아니다. 실제 제출용 엔트리포인트와 통합 코드는 `src/ossp_router/`와 `container/entrypoint.py`에 있다. 여기서는 방법론, 실험 발전 과정, 선택 기준, 재현 가능한 구성요소를 읽기 쉬운 형태로 정리한다.

## 핵심 아이디어

라우팅 문제를 단순한 3-class 분류로 보지 않고, 각 prompt에 대해 **상위 모델로 올릴 때 얻는 품질 증가와 추가 비용을 비교하는 marginal allocation 문제**로 다뤘다.

모델 계층은 다음과 같다.

```text
ax31-light  ->  ax31  ->  axk1-think
```

각 upgrade에 대해 다음 신호를 사용한다.

- 예상 품질 증가 확률 / 크기
- 예상 비용 증가
- tail-risk
- 현재 tier budget headroom
- donor swap으로 회수 가능한 비용
- upgrade 이후 전체 score 변화

따라서 Fast tier에서는 `ax31 -> axk1-think` promotion을 무조건 늘리지 않고, 필요한 경우 일부 `ax31 -> ax31-light` donor를 함께 선택해 budget을 맞춘다.

## 전체 흐름

```text
Prompt
  |
  v
Prompt-only features
  |
  +--> quality / gain predictors
  +--> cost predictors
  +--> tail-risk predictors
  |
  v
Marginal utility estimation
  |
  v
Tier-specific allocation
  |
  +--> promotion candidates
  +--> donor candidates
  +--> budget / tail constraints
  |
  v
Model ID
```

## 주요 실험 계보

| Version | 핵심 변화 | 의미 |
| --- | --- | --- |
| V5 | Marginal-value base | quality/cost 기반 기본 allocator 확립 |
| V9 | Conditional swap | prompt 조건부 upgrade/downgrade 신호 강화 |
| V10 | Economic cost | 비용 예측을 별도 모델링 |
| V11 | Hard-tail gate | Balanced의 tail-risk 방어 |
| V12 | Fast AX31 swap | Fast의 AX31/Light 경계 정교화 |
| V16 | Fast K1 headroom | Fast에 처음 K1 promotion을 실제 도입 |
| V17 | Margin recovery | K1을 유지하면서 donor로 budget margin 회복 |
| V18 | Marginal K1 | 추가 K1 후보 탐색으로 score champion 갱신 |
| V20 | Donor frontier | score를 유지하면서 donor 조합 탐색 |
| V22 | 7th K1 on champion | 기존 champion 위에서 K1 추가 |
| V23 | Consensus recovery | 여러 ranking 신호 합의 기반 Fast 개선 |
| V24 | Consensus frontier | 최종 primary 후보; score/margin 균형 |
| V26 | Microfund | K1 promotion을 donor로 micro-funding |
| V28 | 9th K1 | raw score 추가 개선 |
| V29 | Margin recovery | raw champion의 budget margin 회복 |
| V30 | Final stress | raw score와 robustness를 함께 비교 |

자세한 발전 과정은 [`docs/EXPERIMENT_HISTORY.md`](docs/EXPERIMENT_HISTORY.md)에 정리했다.

## 최종 후보

### Primary: V24

```text
final score : 0.706335227273
Fast quality: 0.669602272727
Fast cost   : 1.242828927920
Fast margin : 0.007171072080
Fast K1     : 7
```

V24는 raw score 최고점은 아니지만, V30 stress test에서 score 손실 대비 budget margin과 failure-risk의 균형이 가장 좋았다.

### Aggressive: V29

```text
final score : 0.707244318182
Fast quality: 0.671875000000
Fast cost   : 1.246792056836
Fast margin : 0.003207943164
Fast K1     : 9
```

V29는 public/dev 기준 raw score champion이다. 단, Fast budget margin이 V24보다 작아 hidden distribution shift에서 budget violation risk가 더 크다.

### Defensive: V20

```text
final score : 0.705426136364
Fast quality: 0.667329545455
Fast cost   : 1.238180334357
Fast margin : 0.011819665643
Fast K1     : 6
```

점수는 낮지만 margin이 가장 넓은 방어형 후보다.

## 최종 선택 논리

V30 stress test에서 다음 세 축을 비교했다.

1. raw score
2. Fast budget margin
3. IID / mixture perturbation failure risk

결론은 다음과 같다.

```text
Raw-score leader        : V29
Margin/risk pick <=0.001: V24
Defensive fallback      : V20
```

최종 제출 primary는 V24를 기준으로 검증했다.

## 현재 runtime 통합 상태

최종 exact-chain runtime은 다음 계보를 재실행한다.

```text
Fast:
V5 competition
 -> V16 k1_8
 -> V17 add4
 -> V18 balanced-r3-d6
 -> V19 ratio-o1-d6
 -> V20 window-o6-d4
 -> V22 balanced-r4-d4
 -> V23 consensus-o7-d4
 -> V24 prefix-total7

Balanced:
V5 competition -> V11 gate20_cost

Premium:
V5 competition
```

이 chain은 Dev에서 3-tier 모두 기존 최종 submission과 exact match를 확인했다.

```text
fast     exact=True diff=0
balanced exact=True diff=0
premium  exact=True diff=0
```

## 이 디렉터리와 제출 runtime의 경계

```text
src/marginal_router/
    연구 방법론, 실험 역사, 선택 근거, 팀 공유 문서

src/ossp_router/
    공식 challenge protocol/runtime 및 최종 통합 코드

container/entrypoint.py
    실제 컨테이너 entrypoint
```

`marginal_router`의 목적은 최종 제출 코드를 복제하는 것이 아니라, **왜 그런 router가 만들어졌는지 팀원이 이해하고 재현할 수 있게 하는 것**이다.

## 문서

- [`docs/METHOD.md`](docs/METHOD.md): 알고리즘과 allocator 구조
- [`docs/EXPERIMENT_HISTORY.md`](docs/EXPERIMENT_HISTORY.md): V5~V30 실험 발전 과정
- [`docs/VALIDATION.md`](docs/VALIDATION.md): 최종 후보 검증 및 runtime 검증 결과
- [`method.v1.json`](method.v1.json): 기계 판독 가능한 method summary

---

## --- To 석범 ---

# Marginal Router — V24

`marginal_router`는 SKT Efficient LLM Routing Challenge에서 개발한 **prompt-only, budget-aware LLM router 연구 라인**이다.

최종 선택 모델은 **V24**이며, 단순한 prompt 난이도 3-class 분류가 아니라 각 모델 upgrade의 **marginal quality gain, 예상 비용, tail risk**를 추정하고 tier별 budget 안에서 전체 allocation을 최적화한다.

> 이 디렉터리는 방법론과 실험 내용을 팀 내부에 공유하기 위한 문서화 영역이다.
> 실제 최종 runtime은 `src/ossp_router/orchestrator.py`와 `container/entrypoint.py`에 통합되어 있다.

---

## 1. 최종 코드

최종 검증된 runtime은 다음 Git branch/commit에 있다.

```text
branch:
submission-v24-final

commit:
f67644eb6d6e2d84243c87cb85cd7ecd06f58d9f

commit message:
Finalize v24 submission runtime
```

실제 최종 runtime 변경 파일:

```text
src/ossp_router/orchestrator.py
```

최종 제출 담당자는 위 commit을 기준으로 Docker image와 technical submission metadata를 생성하면 된다.

---

## 2. 핵심 아이디어

전체 방법론은 다음과 같이 요약할 수 있다.

> **Budget-Constrained Marginal Utility Routing with Pairwise Conditional Swaps and Donor-Based Cost Recovery**

사용 가능한 모델 계층:

```text
ax31-light
    ↓
ax31
    ↓
axk1-think
```

라우터는 prompt를 직접 `easy / medium / hard`로 분류하지 않는다.

대신 다음과 같은 **upgrade의 가치**를 추정한다.

```text
ax31-light → ax31
ax31       → axk1-think
```

각 upgrade에 대해 다음 신호를 사용한다.

* 예상 품질 증가 확률
* 예상 품질 증가 크기
* 예상 incremental cost
* cost-tail probability
* tier budget headroom
* donor downgrade로 회수할 수 있는 비용

즉 핵심 질문은 다음이다.

> “이 prompt에 더 비싼 모델을 쓰는 것이 추가 비용만큼 가치가 있는가?”

---

## 3. 전체 구조

```text
Prompt
  │
  ▼
Prompt-only feature extraction
  │
  ├── Quality / gain predictors
  ├── Cost predictors
  └── Tail-risk predictors
  │
  ▼
Marginal utility estimation
  │
  ▼
Tier-specific allocation
  │
  ├── Promotion candidates
  ├── Donor candidates
  ├── Pairwise conditional swaps
  └── Budget / risk constraints
  │
  ▼
Final model ID
```

---

## 4. Promotion / Donor 구조

Fast tier에서는 특히 **promotion + donor** 방식이 중요하다.

고가치 prompt를:

```text
ax31 → axk1-think
```

으로 올리면 품질 향상 가능성이 있지만 비용이 증가한다.

이 비용을 확보하기 위해 상대적으로 `ax31`의 marginal value가 낮은 prompt를:

```text
ax31 → ax31-light
```

로 내린다.

이 downgrade prompt를 **donor**라고 한다.

결과적으로:

```text
고가치 prompt  → 더 비싼 모델
저가치 prompt  → 더 싼 모델
전체 allocation → tier budget 유지
```

를 동시에 최적화한다.

---

## 5. Pairwise Conditional Swap

초기 allocation 이후에도 K1 slot이 가장 가치 있는 prompt에 배치되었는지 다시 확인한다.

예를 들어:

```text
Prompt A → axk1-think
Prompt B → ax31
```

인 상태에서 B의 K1 marginal gain이 더 크다고 예측되면:

```text
Prompt A → ax31
Prompt B → axk1-think
```

으로 교환할 수 있다.

Swap 판단에는 다음 정보를 함께 사용한다.

* conditional K1 gain probability
* regression-based gain
* predicted incremental cost
* receiver tail-risk
* donor 대비 relative tail-risk

---

## 6. 주요 실험 계보

| Version | 핵심 변화               | 역할                             |
| ------- | ------------------- | ------------------------------ |
| V5      | Marginal Value      | quality/cost 기반 초기 allocator   |
| V9      | Conditional Swap    | pairwise conditional routing   |
| V10     | Economic Cost       | 비용 및 cost-tail 예측              |
| V11     | Hard-tail Gate      | Balanced tier risk-aware swap  |
| V12     | AX31 Swap           | Fast AX31/Light 경계 개선          |
| V16     | K1 Headroom         | Fast에 K1 promotion 도입          |
| V17     | Margin Recovery     | donor를 이용한 비용 회수               |
| V18     | Marginal K1         | 추가 K1 후보 탐색                    |
| V19     | Ratio Donor         | quality-loss / saving 기반 donor |
| V20     | Donor Frontier      | 추가 donor frontier 탐색           |
| V22     | 7th K1              | champion 위에 추가 K1 promotion    |
| V23     | Consensus Recovery  | 여러 ranking 신호 결합               |
| V24     | Consensus Frontier  | 최종 primary candidate           |
| V28     | 9th K1              | raw score 추가 개선                |
| V29     | Aggressive Champion | Dev raw score 최고               |
| V30     | Final Stress        | score / budget robustness 비교   |

---

## 7. 최종 V24 runtime chain

### Fast

```text
V5 competition
 → V16 k1_8
 → V17 add4
 → V18 balanced-r3-d6
 → V19 ratio-o1-d6
 → V20 window-o6-d4
 → V22 balanced-r4-d4
 → V23 consensus-o7-d4
 → V24 prefix-total7
```

Fast에서는 K1 promotion과 donor allocation을 가장 적극적으로 수행한다.

### Balanced

```text
V5 competition
 → V11 gate20_cost
```

V11에서는 K1 slot을 재배치할 때 다음 조건을 hard gate로 사용한다.

* minimum expected quality gain
* maximum receiver tail probability
* donor 대비 relative tail-risk
* predicted cost non-increase

### Premium

```text
V5 competition
```

---

## 8. V24를 선택한 이유

Dev raw score 최고 후보는 V29였다.

```text
V29 final score : 0.707244318182
V24 final score : 0.706335227273
```

하지만 Fast budget margin은:

```text
V29 : 0.003207943164
V24 : 0.007171072080
```

으로 V24가 더 넓었다.

V30 stress test에서도 V24가 V29보다 budget failure risk 측면에서 더 안정적이었다.

최종 판단:

```text
Raw-score leader        : V29
Primary score/risk pick : V24
Defensive fallback      : V20
```

따라서 최종 primary candidate는 **V24**로 선정했다.

---

## 9. 최종 Dev 성능

V24:

```text
Final score  : 0.706335227273

Fast quality : 0.669602272727
Fast cost    : 1.242828927920
Fast margin  : 0.007171072080
```

Fast model allocation:

```text
ax31-light : 607
ax31       : 266
axk1-think :   7
```

비교 후보:

```text
V20 defensive
Final score = 0.705426136364

V29 aggressive
Final score = 0.707244318182
```

---

## 10. 최종 runtime 일반화 수정

초기 exact-chain은 Dev 880문항 기준으로 개발되었기 때문에 일부 단계에 다음과 같은 regression guard가 있었다.

```text
V19 expected 6 K1 in base
```

공식 runtime checker는 공개 Train+Dev 전체 **2,640문항**을 실행하므로 K1 개수가 정상적으로 달라지고 해당 assertion이 실패했다.

최종 runtime에서는:

* routing algorithm은 그대로 유지
* Dev dataset에만 종속된 fixed K1-count assertion만 제거

했다.

수정 후 Dev 결과:

```text
fast     diff=0
balanced diff=0
premium  diff=0
```

즉 기존 V24와 정확히 동일하다.

---

## 11. V11 runtime 최적화

Balanced tier의 초기 runtime은 2,640문항에서 90초 제한에 근접했다.

분리 측정:

```text
V5  : 41.24 sec
V11 : 49.24 sec
```

프로파일 결과 V11에서 동일한 prompt의 `build_meta_features()`를 반복 계산하고 있었다.

기존:

```text
2,640 episodes × 2 fallback × 2 repeated extraction
= 10,560 build_meta_features calls
```

최종 구현에서는 prompt meta-feature가 fallback-independent라는 점을 이용해:

```text
episode당 1회 계산
→ 두 gain/cost head에서 공유
```

하도록 cache했다.

결과:

```text
V11 runtime
49.24 sec → 약 13.5 sec
```

최적화 전후 결과 비교:

```text
V11_SUBMISSION_EXACT=True
V11_REPORT_EXACT=True
```

즉 runtime 최적화로 인해 routing decision은 전혀 바뀌지 않았다.

---

## 12. 공식 공개 Train+Dev runtime 검증

공식 runtime checker 입력:

```text
2,640 episodes
11,780,297 bytes
```

최종 결과:

```text
fast:     PASS  64.192 sec / 90 sec
balanced: PASS  58.774 sec / 90 sec
premium:  PASS  44.100 sec / 90 sec
```

세 tier 모두 공식 90초 제한을 통과했다.

---

## 13. 최종 V31.1 validation

최종 runtime에 대해 다음 검증을 통과했다.

```text
REPO_HYGIENE_OK=True
EXPECTED_SUBMISSIONS_OK=True
SOURCE_AUDIT_OK=True
OFFICIAL_PY311_UNIT_TESTS_OK=True
```

Docker / runtime:

```text
image_linux_arm64=PASS
image_no_volume=PASS
official_runtime_check=PASS
```

Exact output:

```text
fast_exact_expected=PASS
balanced_exact_expected=PASS
premium_exact_expected=PASS
```

Determinism:

```text
fast_deterministic=PASS
balanced_deterministic=PASS
premium_deterministic=PASS
```

Runtime:

```text
fast_runtime_90s=PASS
balanced_runtime_90s=PASS
premium_runtime_90s=PASS
```

Metamorphic test:

```text
id_only_*       = PASS
order_only_*    = PASS
id_and_order_*  = PASS
```

최종 image validation:

```text
IMAGE_VALIDATION_OK=True
```

코드/runtime 관점에서는 최종 제출 가능한 상태까지 검증했다.

---

## 14. 최종 제출 담당자가 사용할 정보

```text
Repository:
ING-OSS/ossp-2026-llm-router-challenge

Branch:
submission-v24-final

Validated code commit:
f67644eb6d6e2d84243c87cb85cd7ecd06f58d9f
```

최종 제출 담당자는 위 commit을 기준으로:

1. `linux/arm64` Docker image build
2. 공개 registry push
3. immutable image digest 확보
4. `submission-ossp-skt.json` 작성
5. technical submission validation
6. 최종 제출 commit 생성

을 진행하면 된다.

특별한 문제가 없다면 검증이 끝난 `orchestrator.py`의 routing logic을 추가 수정하지 않는 것을 권장한다.

---

## 15. 디렉터리 역할

```text
src/marginal_router/
    방법론, 실험 과정, 모델 선택 근거 및 팀 공유 문서

src/ossp_router/
    challenge protocol/runtime 및 최종 통합 router

container/entrypoint.py
    공식 컨테이너 entrypoint
```

---

## 16. 한 문장 요약

> **V24 Marginal Router는 prompt별 모델 upgrade의 예상 marginal quality gain과 비용·tail risk를 추정하고, promotion과 donor downgrade를 조합하여 tier budget 안에서 전체 품질을 최대화하는 cost-aware routing system이다.**


