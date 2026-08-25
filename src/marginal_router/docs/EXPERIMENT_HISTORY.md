# Experiment History

이 문서는 score champion이 어떻게 발전했는지와 각 실험의 목적을 요약한다.

## V5 — Marginal value base

기본 quality/cost 예측을 결합한 competition profile을 확립했다.

```text
final = 0.703607954545
Fast cost = 1.179872168161
```

이 단계에서는 Fast K1이 0이었다. 이후 실험은 남는 Fast headroom을 활용해 K1을 안전하게 추가하는 데 집중했다.

## V15 — Donor-funded K1 attempt

AX31 -> K1 promotion을 AX31 -> Light donor로 funding하려 했지만 조건이 너무 보수적이어서 K1이 추가되지 않았다.

## V16 — Headroom-funded K1

기존 Fast budget headroom을 직접 활용하면서 첫 K1 promotion에 성공했다.

```text
k1_8 profile
K1 = 5
final = 0.704744318182
```

## V17 — Margin recovery

V16 K1을 유지하면서 quality-neutral donor를 추가해 budget margin을 회복했다.

`add4`가 score를 유지하면서 비용을 낮췄다.

## V18 — Additional marginal K1

추가 K1 후보를 탐색해 새로운 score champion을 만들었다.

```text
final = 0.705198863636
```

## V19 — Zero-loss donor recovery

V18 score를 그대로 유지하면서 Fast cost를 낮추는 donor 조합을 탐색했다.

## V20 — Donor frontier

score와 budget margin의 Pareto frontier를 탐색했다.

```text
window-o6-d4
final = 0.705653409091
```

동시에 `prefix-total18`은 score를 조금 희생하고 margin을 크게 늘린 방어형 후보가 됐다.

## V21 / V22 — 7th K1

V21에서는 새로운 champion을 만들지 못했지만, V22에서 기존 champion 위에 K1을 추가해:

```text
final = 0.705880681818
```

까지 상승했다.

## V23 — Consensus recovery

여러 ranking signal의 consensus로 donor/promotion을 정리해:

```text
final = 0.706335227273
```

을 달성했다.

## V24 — Primary frontier

V23 score를 그대로 유지하면서 Fast cost를 줄였다.

```text
prefix-total7
final = 0.706335227273
Fast cost = 1.242828927920
Fast margin = 0.007171072080
```

이 profile이 최종 primary 후보가 됐다.

## V25 — 8th K1 search

추가 K1 후보를 대규모 탐색했지만 score 개선은 없었다.

## V26 — Microfund

K1 추가를 소규모 donor bundle로 funding하는 방식으로 score를 크게 개선했다.

```text
final = 0.707017045455
```

## V27 — Margin recovery

V26 score를 유지하면서 donor를 추가해 margin을 회복했다.

## V28 — 9th K1

추가 K1 하나가 실제 score 개선으로 이어졌다.

```text
final = 0.707244318182
```

## V29 — Champion margin recovery

V28 raw score를 유지하면서 Fast cost를 줄였다.

```text
consensus-o18-d4
final = 0.707244318182
Fast cost = 1.246792056836
Fast margin = 0.003207943164
```

이 profile이 public/dev raw-score champion이다.

## V30 — Final selection stress

최종 후보들을 score, margin, IID perturbation, mixture perturbation 관점에서 비교했다.

| Candidate | Final | Fast cost | Margin | K1 |
| --- | ---: | ---: | ---: | ---: |
| V29 aggressive | 0.707244318182 | 1.246792057 | 0.003207943 | 9 |
| V27 | 0.707017045455 | 1.246188374 | 0.003811626 | 8 |
| V24 primary | 0.706335227273 | 1.242828928 | 0.007171072 | 7 |
| V20 defensive | 0.705426136364 | 1.238180334 | 0.011819666 | 6 |
| V5 reference | 0.703607954545 | 1.179872168 | 0.070127832 | 0 |

최종 판단:

```text
raw score pick      = V29
margin/risk pick    = V24
primary submission  = V24
```
