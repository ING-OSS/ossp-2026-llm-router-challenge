# Method

## 1. 문제 정의

각 prompt마다 `ax31-light`, `ax31`, `axk1-think` 중 하나를 선택해야 한다. 각 tier에는 비용 제약이 있으므로 품질만 최대화할 수 없다.

이 연구 라인은 다음 최적화 관점에서 출발한다.

```text
maximize expected quality
subject to tier budget
```

핵심은 절대 품질보다 **현재 선택에서 한 단계 upgrade했을 때의 marginal gain**을 예측하는 것이다.

## 2. Marginal signals

각 prompt에 대해 개념적으로 다음을 추정한다.

```text
G01 = gain(ax31-light -> ax31)
G12 = gain(ax31 -> axk1-think)
C01 = incremental cost(ax31-light -> ax31)
C12 = incremental cost(ax31 -> axk1-think)
```

그리고 단순 ratio 하나만 사용하는 대신 다음을 함께 고려한다.

- gain probability
- gain magnitude
- predicted token/cost inflation
- cost-tail probability
- regression probability
- donor loss
- remaining tier headroom

## 3. Promotion / donor allocation

Fast에서 K1 promotion이 추가될수록 품질은 증가할 수 있지만 budget margin은 빠르게 줄어든다.

따라서 두 종류의 candidate를 동시에 다룬다.

### Promotion

```text
ax31 -> axk1-think
```

높은 expected gain을 가지는 prompt를 선택한다.

### Donor

```text
ax31 -> ax31-light
```

품질 손실 가능성이 작으면서 비용 절감 효과가 큰 prompt를 선택한다.

promotion으로 발생한 추가 비용을 donor가 일부 상쇄한다.

## 4. Tier별 전략

### Fast

가장 빡빡한 budget 때문에 marginal allocation과 donor funding이 중요하다. V16 이후 실험의 대부분은 Fast의 K1 수와 margin을 조절하는 방향으로 진행됐다.

### Balanced

V11의 hard-tail gate를 적용해 고비용 tail을 제한했다.

### Premium

V5 competition policy가 이미 충분한 score/budget trade-off를 보여 추가 변경을 최소화했다.

## 5. 왜 단순 classifier가 아닌가

최종 선택은 class probability만으로 결정되지 않는다.

동일한 `axk1-think` 선호 prompt라도:

- 예상 gain이 작거나
- 비용 tail이 크거나
- Fast headroom이 부족하면

promotion을 하지 않는다.

반대로 classification confidence가 중간이어도 gain/cost ratio가 좋고 budget이 남아 있으면 promotion할 수 있다.

즉 decision은 **prompt-level prediction + batch-level allocation**의 결합이다.

## 6. Robustness 관점

V29는 public/dev raw score가 가장 높지만 Fast margin이 약 0.0032다. V24는 score가 약 0.00091 낮지만 margin은 약 0.00717이다.

V30에서 IID 및 mixture perturbation을 통해 이 차이를 비교했고, 최종 primary는 V24로 선택했다.

## 7. Runtime 구현

최종 runtime은 최종 Dev label을 새 classifier로 학습하는 방식이 아니라, 원래 실험 chain의 prompt-only predictor와 allocator를 그대로 실행하도록 구성했다.

따라서 runtime 재현성 검증은 **기존 submission과 episode-level model ID exact match**로 수행했다.
