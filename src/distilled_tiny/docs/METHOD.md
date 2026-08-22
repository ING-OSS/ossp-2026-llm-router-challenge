# Distilled Tiny 방법론

## 1. 목적

Distilled Tiny는 prompt만 보고 세 후보 모델의 예상 정답 확률을 동시에 추정한다.

$$
\begin{aligned}
q_0 &= P(\text{correct}\mid x,\ \text{ax31-light}), \\
q_1 &= P(\text{correct}\mid x,\ \text{ax31}), \\
q_2 &= P(\text{correct}\mid x,\ \text{axk1-think}), \\
g_{01} &= q_1-q_0, \\
g_{12} &= q_2-q_1.
\end{aligned}
$$

`q0/q1/q2`는 모델별 품질 추정치이고, `g01/g12`는 한 단계 비싼 모델로 올렸을 때의 예상 이득이다. 세 값을 하나의 encoder와 하나의 joint head에서 함께 예측하므로, 특정 task 이름을 알려 주거나 규칙 기반 expert를 선택할 필요가 없다.

라우팅에 직접 필요한 값은 `g01/g12`지만, gain만 학습하면 세 모델의 공통 확률 수준이 정해지지 않는다. 따라서 관측된 `q0/q1/q2`를 함께 학습해 확률의 기준점을 고정하고, 예산별 승격 판단이 흔들리지 않도록 했다.

이 모듈은 **품질 모델**이다. 최종 라우터는 이 출력과 별도로 얻은 호출 비용을 결합해 예산 안에서 모델을 선택해야 한다. 품질 예측과 비용 배분을 분리하면 비용표나 예산 조건이 달라져도 Tiny를 다시 학습하지 않고 allocator만 바꿀 수 있다.

## 2. 입력과 모델 구조

### 2.1 Whole-prompt head/tail packing

과제 입력에 `prompt` 필드가 있으면 그 값을 사용한다. 대신 `messages` 배열이 들어 있으면 각 message의 role과 content를 순서대로 연결한다. 이렇게 얻은 prompt 전체를 하나의 문자열로 만든 뒤 BERT WordPiece tokenizer로 변환한다.

최대 길이는 96 token이다. `[CLS]`와 `[SEP]`를 제외한 94개 자리에 다음 규칙을 적용한다.

- 94 token 이하: 전체를 보존한다.
- 94 token 초과: 앞 47개와 뒤 47개를 보존한다.

이 규칙은 prompt 안에 특정 구분 문자열이 있다고 가정하지 않는다. 앞부분의 지시·배경과 뒷부분의 실제 질문·출력 조건을 함께 남기면서, 같은 입력에는 항상 같은 결과를 내는 전처리를 모든 task에 적용한다.

ONNX 입력은 아래 세 tensor뿐이다.

| 입력 | shape | 의미 |
| --- | --- | --- |
| `input_ids` | `[batch, 96]` | WordPiece token ID |
| `attention_mask` | `[batch, 96]` | 실제 token과 padding을 구분 |
| `token_type_ids` | `[batch, 96]` | 모두 0인 단일 sequence 표시 |

별도의 task label이나 수작업 numeric feature는 사용하지 않는다.

### 2.2 Encoder와 pooling

BERT-Tiny의 두 Transformer layer는 모두 학습하되 embedding table은 고정한다. Encoder가 만든 contextual representation에서 다음 네 view를 구한다.

1. `[CLS]` representation
2. Padding을 제외한 전체 sequence의 mean pooling
3. 선택된 content 앞 절반의 mean pooling
4. 선택된 content 뒤 절반의 mean pooling

네 view를 이어 붙인 뒤 128차원 semantic projection과 192차원 residual MLP를 통과시킨다. 마지막 linear head가 세 logit을 만들고 sigmoid를 적용해 `q0/q1/q2`를 출력한다. 모델 간 우열을 강제로 고정하지 않으므로 prompt에 따라 `q1 < q0` 또는 `q2 < q1`도 표현할 수 있다.

## 3. Qwen teacher를 이용한 distillation

Tiny는 **distilled student**다. 두 response probe는 고정된 Qwen prompt embedding을 입력으로 학습된다. 두 probe가 예측한 품질의 평균과 각 probe의 representation을 teacher target으로 사용하고, 품질 예측의 분산으로 teacher confidence를 계산한다.

```text
prompt
  ├─ observed model outcomes ───────────────┐
  ├─ frozen Qwen teacher quality / gain ────┼─> Tiny joint training
  └─ frozen Qwen teacher representation ────┘
                                                │
                                                └─> q0, q1, q2
```

Teacher target은 `artifacts/training/full-teacher-targets.npz`에 고정되어 있다. Train과 Dev의 quality·uncertainty·representation을 모두 포함하므로, Qwen을 다시 실행하지 않고도 Train-only와 Train+Dev student를 학습할 수 있다. 추론 시에는 teacher가 전혀 필요하지 않다.

## 4. 학습 objective

최종 objective는 실제 관측 정답, upgrade 이득, prompt 사이의 upgrade 순위, teacher embedding의 관계 구조를 함께 학습한다.

### 4.1 관측 outcome loss

prompt $i$, 모델 $j$에서 $n_i$번 생성 중 $s_{ij}$번 맞았다면 관측 정답률은 $y_{ij}=s_{ij}/n_i$이다. student logit을 $z_{ij}$라고 할 때 다음 binary cross-entropy를 사용한다.

$$
\mathcal{L}_{\text{observed}}
=
\frac{\sum_i n_i\left[\frac{1}{3}\sum_{j=0}^{2}
\operatorname{BCEWithLogits}(z_{ij},y_{ij})\right]}
{\sum_i n_i}.
$$

이는 binomial negative log-likelihood에서 학습에 영향을 주지 않는 상수항을 제외한 형태다. 정답률을 계산한 뒤에도 `num_generations`를 버리지 않고 row별 가중치로 다시 반영한다. 따라서 `num_generations`가 큰 prompt의 정답률에 더 큰 신뢰도를 부여한다.

### 4.2 Adjacent gain loss

라우팅에 직접 필요한 값은 비싼 모델이 주는 추가 이득이다. student와 teacher의 인접 upgrade 이득을 각각

$$
\hat{\mathbf g}_i=(\hat q_{i1}-\hat q_{i0},\ \hat q_{i2}-\hat q_{i1}),
\qquad
\mathbf g_i^{T}=(q_{i1}^{T}-q_{i0}^{T},\ q_{i2}^{T}-q_{i1}^{T})
$$

로 만들고 Huber loss로 맞춘다. Teacher ensemble의 평균 분산을 $v_i$라 할 때 confidence는

$$
c_i=\frac{1}{1+100v_i}
$$

이며, gain loss는 $c_i$로 가중한다. Teacher ensemble 구성원 사이의 예측 차이가 큰 prompt가 distillation 학습을 과도하게 좌우하지 않도록 하기 위해서다.

### 4.3 Gain ranking loss

한 batch 안의 prompt 쌍에 대해 어느 쪽의 upgrade 이득이 더 큰지도 학습한다. Teacher gain 차이를 soft target으로 바꾸고 student gain 차이에 binary cross-entropy를 적용한다.

$$
t_{ik}=\sigma\!\left(\frac{g_{ik}^{T}-g_{rk}^{T}}{0.025}\right),
\qquad
\mathcal{L}_{\text{rank}}
=
\operatorname{BCEWithLogits}\!\left(
\frac{\hat g_{ik}-\hat g_{rk}}{0.025},t_{ik}
\right).
$$

여기서 $r$은 batch 안에서 짝지은 다른 prompt다. Teacher gain 차이가 거의 없는 쌍에는 낮은 가중치를 주며, epoch마다 row 순서를 섞어 다양한 쌍을 비교한다. 이 항은 절대 오차보다 “제한된 예산을 어느 prompt에 먼저 써야 하는가”를 직접 겨냥한다.

### 4.4 Relation loss

Student의 128차원 representation과 teacher의 256차원 representation을 각각 정규화한 뒤, batch 안에서 서로 다른 모든 prompt 쌍의 cosine similarity matrix를 만든다. 두 행렬 사이에 Huber loss를 적용한다.

$$
\mathcal{L}_{\text{relation}}
=
\operatorname{Huber}\!\left(
\hat R_{ir},R^T_{ir};\ \delta=0.05
\right),\qquad i\ne r.
$$

Teacher embedding의 좌표값 자체를 복제하지 않고 prompt 사이의 상대적 구조를 보존하는 것이 목적이다. 따라서 student와 teacher의 embedding 차원이 달라도 적용할 수 있다.

### 4.5 전체 objective와 optimizer

$$
\mathcal{L}
=
\mathcal{L}_{\text{observed}}
+\mathcal{L}_{\text{gain}}
+0.1\mathcal{L}_{\text{rank}}
+0.1\mathcal{L}_{\text{relation}}.
$$

Quality head를 제외한 학습 가능한 2차원 weight matrix는 Muon(`lr=2e-4`)으로 최적화한다. Quality head·bias·normalization parameter에는 AdamW(`lr=5e-4`)를 사용한다. Weight decay는 `0.01`, warmup은 전체 step의 `10%`, gradient norm 한도는 `1.0`이다. 고정 재현 설정은 seed `2026`, batch size `128`, 9 epochs다.

## 5. 검증 결과

### 5.1 과제 최종 채점식

아래 점수는 Tiny를 별도의 prompt-cost predictor와 예산 allocator까지 포함한 전체 routing pipeline에 연결한 뒤, 과제의 최종 채점식으로 계산했다.

| 검증 조건 | 최종 점수 | 이 수치가 답하는 질문 |
| --- | ---: | --- |
| Content-stratified 5-fold OOF, 3 seeds | **0.671510 ± 0.001114** | 학습 때 등장한 task 안에서 보지 않은 prompt로 일반화하는가? |
| Leave-One-Task-Out OOF, seed 2026 | **0.663977** | 학습에서 빠진 task에도 일반화하는가? |

`OOF`는 각 prompt가 학습 fold에 포함되지 않은 모델로 해당 prompt를 예측했다는 뜻이다. `Leave-One-Task-Out`은 task 하나 전체를 학습에서 제외한 뒤 해당 task를 평가하는 더 강한 분포 이동 검사다. 두 검증 모두 allocator까지 포함한 최종 점수를 사용한다.
Content-stratified 결과의 `±` 값은 세 seed에서 얻은 점수의 평균과 표준편차다. Leave-One-Task-Out은 seed 2026으로 한 번만 실행했으므로 분산을 제시하지 않는다.

전체 Train으로 학습한 `student.onnx`의 공식 계산 결과는 다음과 같다. Public Dev에서 121개 후보로 구성한 safety grid를 탐색해 예산을 보정했다. 따라서 이 값은 Public Dev에 맞춰 allocator를 보정한 진단 결과이며, hidden test 성능의 unbiased 추정치로 해석해서는 안 된다.

| Tier | 품질 점수 | 실제 예산 비율 | 한도 | 통과 |
| --- | ---: | ---: | ---: | :---: |
| Fast | 0.667045 | 1.244109 | 1.25 | O |
| Balanced | 0.694602 | 1.982301 | 2.00 | O |
| Premium | 0.733523 | 3.999305 | 4.00 | O |

세 tier를 공식 방식으로 합산한 최종 점수는 **0.695255681818**이며, 모든 예산 제약을 통과했다.

### 5.2 Hash-regex와 역할 차이

공개 hash-regex baseline은 정규식 기반 구조 feature와 signed feature hashing으로 만든 word unigram·bigram을 linear quality/cost head에 입력한다. 따라서 반복되는 형식과 어휘를 이용해, 어떤 모델이 잘 맞는지에 대한 task 계열별 평균 경향을 포착하는 데 유리하다. Distilled Tiny는 이러한 수작업 구조 feature 없이 contextual WordPiece representation으로 `q0/q1/q2`를 예측한다. 두 방법은 같은 prompt를 입력으로 받지만 서로 다른 inductive bias를 가진다.

Public Dev 880개에서 각 방법의 품질 신호를 예산 allocator에 연결하고 공식 채점한 결과는 다음과 같다. `L/A/T`는 각각 `ax31-light / ax31 / axk1-think`를 선택한 횟수다.

| Tier | Hash-regex 점수 | Tiny 점수 | Tiny−Hash | Hash L/A/T | Tiny L/A/T |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fast | 0.663068 | **0.667045** | +0.003977 | 543 / 335 / 2 | 422 / 458 / 0 |
| Balanced | 0.693750 | **0.694602** | +0.000852 | 372 / 446 / 62 | 136 / 710 / 34 |
| Premium | **0.740057** | 0.733523 | −0.006534 | 279 / 409 / 192 | 99 / 631 / 150 |
| 공식 가중 최종 점수 | **0.695369** | 0.695256 | −0.000114 | — | — |

이 결과에서 직접 말할 수 있는 것은 다음과 같다.

- Tiny는 Fast와 Balanced에서 더 높은 품질 점수를 냈다. 특히 hash-regex보다 `ax31`을 훨씬 많이 선택해 light→ax31 upgrade를 더 자주 수행했다.
- Hash-regex는 Premium에서 0.006534 높았다. Think 모델에 예산을 배정할 여지가 큰 Premium tier에서는 현재 Tiny보다 구조·어휘 기반 신호가 더 효과적이었다.
- 최종 점수는 거의 같지만 선택 분포는 크게 다르다. 두 모델이 사실상 같은 신호를 내는 것은 아니므로 결합을 검토할 가치가 있다.
- 이 표만으로 ensemble의 성능 향상을 보장할 수는 없다. 두 방법을 Public Dev에서 각각 safety calibration한 결과이므로, Dev에서 고른 고정 혼합 비율의 성능을 hidden test 성능으로 해석해서는 안 된다.

통합할 때는 각 라우터가 내린 최종 decision에 사후 투표를 적용하기보다 **품질 예측 단계**에서 두 신호를 결합하는 편이 적절하다. Hash-regex의 세 quality와 Tiny의 `q0/q1/q2`, 그리고 필요하다면 양쪽의 adjacent gain을 하나의 task-independent combiner에 입력하는 방식을 권장한다. Combiner의 weight는 Train OOF prediction만으로 정해야 한다. 결합 결과가 OOF와 Leave-One-Task-Out의 최종 점수를 모두 개선하는지 확인한 뒤, 공통 비용 allocator로 세 tier의 decision을 만든다. 현재 package에는 검증된 combiner가 포함되어 있지 않으므로 Tiny 출력은 독립적인 품질 신호로 제공한다.

### 5.3 Train-only와 Train+Dev artifact

Package에는 같은 architecture와 objective로 학습한 두 Float ONNX가 들어 있다.

| 파일 | 학습 범위 | rows | 사용 시점 |
| --- | --- | ---: | --- |
| `student.onnx` | Train | 1,760 | 방법 비교와 Public Dev 추론 검증 |
| `student.train-dev.onnx` | Train+Dev | 2,640 | 방법 선택을 마친 뒤 hidden test 추론 |

Train+Dev 모델에도 Train-only 모델과 같은 seed, 9 epochs, Muon+AdamW 설정을 적용했다. Dev teacher target은 같은 Qwen ensemble을 전체 prompt에 적용해 만들었다. 이 ensemble로 Train teacher target을 다시 생성했을 때 quality·uncertainty·representation이 기존 값과 완전히 일치했다.

`student.train-dev.onnx`는 Dev outcome을 학습에 사용했으므로 5.1과 5.2의 Public Dev 점수 계산에는 사용하지 않았다. 이 모델은 Dev와 겹치지 않는 hidden test에서만 유효하게 평가할 수 있다. 따라서 검증용 기본 모델은 `student.onnx`로 유지하며, 최종 통합 단계에서만 Train+Dev 파일을 명시적으로 선택한다.

### 5.4 배포 검증

| 항목 | 결과 |
| --- | ---: |
| Train-only Float ONNX 크기 | 18.07 MB |
| Train-only CPU 추론 시간 | Dev 880 prompts / 0.51 s |
| Train+Dev Float ONNX 크기 | 18.07 MB |
| Train+Dev CPU 추론 시간 | Dev 880 prompts / 0.54 s |
| 측정 조건 | ONNX Runtime, CPU 2 threads, batch 128 |
| Train-only PyTorch↔ONNX 최대 절대 오차 | $2.09\times10^{-7}$ |
| Train+Dev PyTorch↔ONNX 최대 절대 오차 | $2.38\times10^{-7}$ |
| PyTorch↔ONNX argmax 변화 | 두 모델 모두 0 / 880 |

속도는 현재 환경에서 측정한 참고값이므로 제출 장비에서 다시 확인해야 한다. 수치 동등성 검증 결과, package에 포함된 Float ONNX는 학습 checkpoint의 출력을 사실상 그대로 보존했다.

## 6. 통합 방법과 한계

통합 코드는 Tiny가 반환한 `q0/q1/q2`에서 `g01/g12`를 계산하고, 별도의 prompt별 모델 비용과 함께 예산 allocator에 전달하면 된다. Tiny 안에는 비용표나 tier threshold가 들어 있지 않다.

Hash-regex와 함께 사용할 때도 비용 예측과 budget constraint는 하나의 allocator가 담당해야 한다. Tiny와 hash-regex가 별도로 만든 tier decision을 나중에 섞으면 전체 예산을 다시 보장하기 어렵기 때문이다.

현재 모델의 주요 한계는 다음과 같다.

- 최대 96 WordPiece token만 사용하므로 긴 prompt의 중간 정보는 보지 못한다.
- 영어 중심 BERT-Tiny vocabulary를 사용하므로 다국어 tokenization 효율이 낮을 수 있다.
- Leave-One-Task-Out 점수가 content-stratified OOF보다 낮아, 학습에 없던 task로의 일반화가 여전히 핵심 병목이다.
- Public Dev 점수는 allocator calibration에 사용한 split의 결과이며 hidden test 성능을 보장하지 않는다.
- 포함된 teacher target으로 student는 재학습할 수 있지만, teacher 자체를 새 데이터에 생성하는 pipeline은 이 모듈의 범위 밖이다.

설정의 단일 기준은 `method.v1.json`, artifact 무결성의 기준은 `MANIFEST.json`이다. `python -m distilled_tiny.verify`로 checkpoint, ONNX, tokenizer, teacher target과 수치 동등성을 함께 검사할 수 있다.
