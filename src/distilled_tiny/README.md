# Distilled Tiny

Qwen embedding teacher의 신호를 BERT-Tiny에 증류한 **순수 의미 기반 품질 예측 모듈**이다. 이 디렉터리에는 추론 코드, Float ONNX, tokenizer 어휘 파일, 재학습 코드와 teacher target이 모두 들어 있다. 실행 환경 구성과 최종 라우터 통합은 상위 프로젝트에서 담당한다.

모델은 prompt마다 다음 세 모델의 예상 정답률을 동시에 출력한다.

$$
\begin{aligned}
q_0 &= q(\text{ax31-light}), \\
q_1 &= q(\text{ax31}), \\
q_2 &= q(\text{axk1-think}), \\
g_{01} &= q_1-q_0, \\
g_{12} &= q_2-q_1.
\end{aligned}
$$

출력은 최종 제출 파일이 아니다. 통합 담당자는 `q0/q1/q2` 또는 `g01/g12`를 별도의 비용 예측값 및 tier 예산 allocator와 결합해야 한다.

## 모듈 경계

- 추론 입력은 `input_ids`, `attention_mask`, `token_type_ids`뿐이다.
- 세 입력 tensor 외에는 prompt-derived numeric input이나 task label을 요구하지 않는다.
- Tiny는 **distilled student**다. 학습할 때 Qwen teacher의 품질·upgrade·표현 관계를 배우지만, 추론할 때 teacher는 필요 없다.
- 과제의 공식 입출력 형식은 상위 저장소에 공개된 `ossp_router.protocol`을 사용한다.

## 구성

```text
src/distilled_tiny/
├─ artifacts/
│  ├─ model/                 Train/Train+Dev Float ONNX, checkpoint, preprocess.json
│  ├─ tokenizer/             WordPiece vocabulary
│  └─ training/              고정 Qwen teacher target
├─ docs/METHOD.md            현재 설계, objective, hash-regex 역할 비교와 한계
├─ config.py                 경로와 method manifest 검증
├─ tokenization.py           task-independent whole-prompt head/tail packing
├─ model.py                  Tiny architecture와 distillation objective
├─ train.py                  GPU 재학습
├─ export.py                 checkpoint → ONNX
├─ predict.py                CPU 품질 추론
├─ runtime.py                Torch가 필요 없는 ONNX 실행부
├─ verify.py                 artifact 및 수치 동등성 검증
└─ method.v1.json            고정 학습·추론 설정
```

## 추론

Python 3.11 환경에 `requirements-runtime.txt`의 의존성을 설치한 뒤 실행한다.

```powershell
python -m distilled_tiny.predict `
  --input data\materialized\dev\inputs.json `
  --output build\tiny-quality.json
```

상위 코드가 이미 공식 `InputBatch`를 읽었다면 파일을 다시 만들 필요가 없다.

```python
from distilled_tiny.predict import predict_inputs

quality, preprocess = predict_inputs(inputs)
g01 = quality[:, 1] - quality[:, 0]
g12 = quality[:, 2] - quality[:, 1]
```

기본 제공 artifact는 검증된 Float ONNX다. CPU 2-thread 설정에서 Dev 880개를 약 `0.51초`에 추론했다. PyTorch 출력과의 최대 절대 오차는 `2.09e-7`이었고, argmax가 달라진 prompt는 없었다. `export.py`로 INT8 모델도 만들 수 있지만, 실제 배포 형식은 통합 환경에서 측정한 benchmark 결과에 따라 선택해야 한다.

두 Float ONNX의 용도는 다르다.

| 파일 | 학습 데이터 | 용도 |
| --- | --- | --- |
| `student.onnx` | Train 1,760개 | OOF·Public Dev 평가로 확정한 Train-only 기준 모델 |
| `student.train-dev.onnx` | Train+Dev 2,640개 | 방법 선택이 끝난 뒤 hidden test에 사용할 final-fit 모델 |

기본값은 문서의 성능표와 대응하는 `student.onnx`다. Train+Dev 모델을 사용하려면 `--model` 경로만 바꾸면 된다.

```powershell
python -m distilled_tiny.predict `
  --input path\to\hidden-inputs.json `
  --model src\distilled_tiny\artifacts\model\student.train-dev.onnx `
  --output build\tiny-quality.json
```

`student.train-dev.onnx`는 Public Dev outcome을 학습에 사용했으므로 Public Dev 점수를 검증 성능으로 보고하지 않는다.

## 재학습과 ONNX export

학습은 GPU를 사용한다. Quality head를 제외한 학습 가능한 2차원 weight matrix에는 Muon을, quality head·bias·normalization parameter에는 AdamW를 적용한다.

```powershell
python -m distilled_tiny.train --check-only
python -m distilled_tiny.train
python -m distilled_tiny.export `
  --checkpoint build\distilled-tiny-reproduction\model\student.pt `
  --float-only
```

Train+Dev final-fit artifact는 다음처럼 재현한다.

```powershell
python -m distilled_tiny.train `
  --include-dev `
  --output-dir build\distilled-tiny-train-dev-reproduction
python -m distilled_tiny.export `
  --checkpoint build\distilled-tiny-train-dev-reproduction\model\student.pt `
  --output-dir build\distilled-tiny-train-dev-reproduction\model `
  --model-name student.train-dev.onnx `
  --float-only
```

Train-only 재학습에는 과제 Train 입력·outcome과 상위 저장소의 BERT-Tiny 초기 가중치가 필요하다. `--include-dev`를 사용하면 Dev 입력·outcome도 추가로 읽는다. 각 파일의 경로는 명령행 옵션으로 바꿀 수 있다. 결과는 기본적으로 `build/distilled-tiny-reproduction/`에 저장되므로 package에 포함된 artifact를 덮어쓰지 않는다.

## 검증

```powershell
python -m distilled_tiny.verify
```

검증 명령은 package manifest의 SHA-256, checkpoint의 strict loading, PyTorch↔Float ONNX 수치 동등성, 출력 shape와 teacher target shape를 확인한다. 방법론과 수치의 정의는 [METHOD.md](docs/METHOD.md)에 정리했다.
