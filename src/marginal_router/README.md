# Marginal Router

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
