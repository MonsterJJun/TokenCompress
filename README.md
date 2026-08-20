# TokenCompress
AE(Reconstruct trigger token), BE(Behavior Equivalent Token) or something tokens. 

Reconstruction(recon) + Knowledge Distillation(KD) = Total loss를
결합해서, 압축 토큰이 원본 프롬프트를 재현하면서도 실제 태스크 성능을 유지하도록 학습시킴.

## 객체 구성

| 객체 | 역할 |
|---------------------------|---------------------------------------------------------------|
| `CustomTrainingArgs`      | Hyper Parameter                                               |
| `TokenSetup`              | Add special token → Model Resize → Backbone freeze 준비 과정   |
| `PromptBuilder`           | recon/KD/ check input format                                  |
| `EmbeddingManager`        | initialize special token·gradient 격리·복원·분석               |
| `CustomTokenTrainer`      | Train Loop                                                    |
| `TrainerDebugger`         | Debugging Input/Output, compute loss 과정                     |
| `Evaluator`               | Metric to Reconstruct, Task Response(Distillation)            |
| `MetricTracker`           | loss(store history), select best checkpoint, store loss graph |
| `CheckpointManager`       | model/history 저장·정리·로드                                   |
| `save_experiment_report`  | setup·loss·metric 결과를 실험마다 폴더 하나에 report로 저장     |

`TokenSetup → CustomTokenTrainer(PromptBuilder/EmbeddingManager 자동 초기화) → Evaluator → save_experiment_report` 순서로 사용.

---

## 1. TokenSetup

모델에 special token을 추가, 임베딩 행렬 resize, backbone freeze 준비 과정 담당.

### 생성자

```python
TokenSetup(model, tokenizer, new_tokens: list, mean_resizing: bool = False,
           freeze_backbone: bool = True, verbose: bool = True)
```

- `new_tokens`: 추가할 토큰 문자열 리스트. 예: `["<|BE|>", "<|AE|>"]`. 이미 등록된
  special token과 섞여 들어와도 자동으로 중복 제거.
- `mean_resizing`: `model.resize_token_embeddings(..., mean_resizing=...)`에 그대로 전달.
  - `True`(HF default): 기존 임베딩들의 평균+공분산 기반 다변량 정규분포에서 샘플링(Hewitt 방식).
    새 토큰들이 전부 평균값의 복제본이 되어 같은 값을 가지는 벡터가 생성됨.
  - `False`: 모델의 표준 초기화 함수(정규분포)로 각 토큰을 독립적으로 랜덤 샘플링
- `freeze_backbone`: `True`면 전체 파라미터를 얼리고 입력 임베딩 레이어만
  `requires_grad=True`로 켬. `CustomTokenTrainer`(정확히는 내부의 `EmbeddingManager`)가
  이 작업을 다시 하므로, 바로 이어서 트레이너를 만들 거라면 `False`로 둬도 무방
- `verbose`: 각 단계 결과를 콘솔에 출력할지 여부.

### 생성 시 실행되는 5단계

1. 추가 전 vocab 크기 / 임베딩 shape / 기존 special token 목록 기록
2. `tokenizer.add_special_tokens(...)` 호출
3. `model.resize_token_embeddings(...)` 호출
4. 요청한 각 토큰의 id와 초기 벡터 norm 출력
5. backbone freeze

### Method

- `get_id(token: str) -> int`: 토큰 문자열로 id 조회. `new_tokens`에 없던,
  이미 등록된 토큰도 조회 가능.
- `get_ids(tokens: list) -> list`: 여러 개를 한 번에 조회.
- `freeze_backbone()`: 생성자에서 안 했다면 나중에 수동으로 호출 가능.
- `summary()`: 추가된 토큰 정보, vocab 크기 변화, resize 방식 등을 dict로 반환.

### Example

```python
from TokenCompress import TokenSetup

setup = TokenSetup(
    model, tokenizer,
    new_tokens=["<|BE|>", "<|AE|>"],
    mean_resizing=False,
    freeze_backbone=True,
)

BE_token_id = setup.get_id("<|BE|>")
AE_token_id = setup.get_id("<|AE|>")
```

---

## 2. PromptBuilder

recon 학습, KD 학습, teacher 데이터셋 생성, 실제 생성 평가까지

### 생성자

```python
PromptBuilder(tokenizer, fix_prompt, target_token_ids, ae_token_id,
              user_template=None, device="cuda", end_token="<|im_end|>",
              prompt_mode="chat", enable_thinking=False, raw_separator="\n\n")
```

- `fix_prompt`: 압축하려는 target system prompt.
- `target_token_ids`: 압축된 내용이 들어갈 special Tokens.
- `prompt_mode`:
  - `"chat"`: use `tokenizer.apply_chat_template()`
  - `"raw"`: chat template 없이 raw text
- `enable_thinking`:
  - `True`/`False`: `apply_chat_template(enable_thinking=...)`
  - `None`: Tokenizer/Template 자체의 기본 동작.
  - `raw` 모드에서는 `None`/`False`를 동일하게 취급해 think prefill(`\n<think>\n\n</think>\n\n`)을
    붙임(실험적으로 raw 모드엔 필요하다고 확인됨). `True`일 때만 생략.

### Method

| 메서드 | 용도 |
|---|---|
| `fill_template(query)` | `user_template`의 `{user_text}` 자리에 query 채워넣기 |
| `build_recon()` | recon 학습용 `input_ids`/`labels`/`prefix_len` 딕셔너리 반환 |
| `build_kd(query)` | KD 학습용 teacher/student prefix(텍스트+토큰id) 딕셔너리 반환 |
| `build_inference_prefix(query, use_student=True)` | 평가/생성 시 쓸 prefix 문자열 (student 또는 teacher) |
| `build_generate_input()` | recon 검증용 `[target_token_ids..., ae_token_id]` 시작 입력 |
| `build_teacher_generate_prefix(query)` | teacher 데이터셋 생성 시 쓸 prefix |
| `summary()` | 현재 설정을 dict로 반환 (디버깅/리포트용) |

### 사용 예시

```python
builder = PromptBuilder(
    tokenizer, fix_prompt=prompt_636,
    target_token_ids=[BE_token_id], ae_token_id=AE_token_id,
    prompt_mode="chat", enable_thinking=False,
)

r = builder.build_recon()          # {"input_ids":..., "labels":..., "prefix_len":...}
kd = builder.build_kd(query)       # {"teacher_prefix":..., "student_prefix":..., ...}
```

`CustomTokenTrainer`를 쓰면 내부에서 자동으로 생성되므로(`trainer.builder`로 접근),
보통은 직접 생성할 일이 적고 `Evaluator`나 `TrainerDebugger`가 `trainer.builder`를
공유해서 쓰는 방식으로 사용함.

---

## 3. CustomTokenTrainer

`PromptBuilder` / `EmbeddingManager` / `CheckpointManager` / `MetricTracker` 사용한 Train Loop

### 생성자 핵심 인자

```python
CustomTokenTrainer(
    model, tokenizer, target_prompt, target_token_ids, ae_token_id,
    args=None, user_template=None, init_mode="random",
    recon_reduction="mean", kd_tau_squared=True,
    prompt_mode="chat", enable_thinking=False,
    sync_output_embedding=False, clamp_norm_multiplier=None,
)
```

- `init_mode`: `"random"`(순수 랜덤, 기본) | `"mean"`(Hewitt류 평균+노이즈) | `"none"`(resize 결과 그대로)
- `recon_reduction`: `"mean"`(기본, 논문 실측값과 일치) | `"sum"`(논문 수식)
- `kd_tau_squared`: KD loss에 `τ²`를 곱할지(`True`, Hinton 2015 관례, 기본값)
- `clamp_norm_multiplier`: `None`(기본, 제한 없음) 또는 숫자. 압축 토큰 임베딩의 norm이
  `vocab 평균 × 이 값`을 넘으면 강제로 줄임(norm 폭주로 인한 자기예측 문제 대응용).
  단, recon이 필요로 하는 표현력 자체를 제한할 수 있어 신중히 사용해야 함
  (대안으로 `Evaluator`의 생성 단계 토큰 억제 기능 참고).

### Train Step

```
1. compute_recon_loss()
   - PromptBuilder.build_recon()으로 [target..., AE, fix_prompt, end] 구성
   - teacher forcing으로 forward, reduction="mean"이면 HF의 out.loss 그대로 사용,
     "sum"일 때만 직접 F.cross_entropy 계산

2. compute_kd_loss(query, answer_ids)
   - teacher: [fix_prompt 기반 prefix] + answer_ids → forward(no_grad)
   - student: [압축 토큰 기반 prefix] + answer_ids → forward(grad 흐름)
   - 두 분포의 KL(teacher‖student) 계산

3. total = (1 - λ) * recon + λ * kd   (λ = args.lambda_weight)

4. backward() → grad_norm clip → optimizer.step() → scheduler.step()

5. EmbeddingManager.restore_frozen_rows()
   - target 토큰 외 vocab 전체를 원본 값으로 복원 (weight_decay 등의 부작용 차단)
   - (옵션) sync_output_embedding, clamp_norm_multiplier 적용
```

### Train Loop

```python
trainer.train(
    teacher_dataset, eval_dataset=None,
    live_plot_every=0, live_plot_unit="step",
    live_plot_smooth=20, live_plot_log=False, live_plot_both_scales=False,
    early_stopping_patience=None,
)
```

```
for epoch in range(num_train_epochs):
    for batch in teacher_dataset:
        train_step(batch)                     # 위 5단계
        (선택) live_plot_every 스텝마다 그래프 갱신+저장
        (선택) save_steps마다 중간 체크포인트 저장

    if eval_dataset:
        evaluate() → eval_total/eval_recon/eval_kd 계산
        best_eval_loss 갱신 시 → checkpoint-best 저장
        best_eval_kd 갱신 시   → checkpoint-best_kd 저장 (recon 개선에 KD 악화가
                                  가려지는 걸 막기 위해 별도 추적)
    (선택) live_plot_unit="epoch"면 이 시점에 그래프 갱신
    (선택) early_stopping_patience번 연속 미개선 시 학습 조기 종료

checkpoint-final 저장
```

- `live_plot_every=0`, `early_stopping_patience=None`이 기본값이며, 이 경우 위 옵션
  기능들은 전혀 개입하지 않고 기존 학습 흐름과 동일하게 동작함.
- 가벼운 실험용으로 `train_steps(teacher_dataset, n_steps, ...)`도 제공 — epoch/eval/save
  없이 지정한 스텝 수만 빠르게 실행.

---

## 4. TrainerDebugger

 `trainer.debugger`에 붙였을 때만 내부 계산 과정을 출력하는 관찰 도구. `with`를 사용.

### 기본 사용법

```python
from TokenCompress import TrainerDebugger

with TrainerDebugger(trainer, keys={"recon"}) as dbg:
    trainer.compute_recon_loss()
# with 블록을 벗어나면 자동으로 trainer.debugger = None (detach)
```

- `keys`: `True`(전부) | `False`(끔) | `{"init","grad","recon","kd","step","sched","prompt"}`의 부분집합
- `max_rows`: 위치별 표를 몇 줄까지 보여줄지 (`None`이면 전체)
- `collect`: `True`면 출력 여부와 무관하게 계산된 텐서를 `dbg.last_recon`/`dbg.last_kd`에
  저장해둠 — 나중에 그래프를 그리거나 직접 값을 뜯어볼 때 사용. 기본값 `False`(끔).

### 상황별 사용 패턴

```python
# 학습 전 정적 점검 (초기화/gradient/optimizer 설정)
with TrainerDebugger(trainer, keys={"init"}) as dbg:
    dbg.inspect_setup()

# 프롬프트 조립 결과 검증 (chat/raw, thinking 옵션이 실제로 어떻게 반영됐는지)
with TrainerDebugger(trainer, keys={"prompt"}) as dbg:
    dbg.inspect_prompts(query, show_ids=True, run_generate=True)

# recon loss 계산 과정 — 위치별 CE, 확률, argmax까지 전부 표로 확인
with TrainerDebugger(trainer, keys={"recon"}, max_rows=None) as dbg:
    trainer.compute_recon_loss()

# gradient가 target 토큰 행에만 흐르는지, 값이 실제로 얼마나 움직였는지
with TrainerDebugger(trainer, keys={"step", "grad"}) as dbg:
    trainer.train_step(query, answer_ids)

# 텍스트 출력 없이 조용히 데이터만 수집 (여러 샘플 반복할 때)
with TrainerDebugger(trainer, keys=False, collect=True) as dbg:
    for i in range(20):
        trainer.compute_kd_loss(train_ds[i]["query"], train_ds[i]["answer_ids"])
        k = dbg.last_kd   # 매 반복마다 최신 값으로 갱신됨
```

### 관찰 가능한 항목(`keys`)

| key | 확인 내용 |
|---|---|
| `init` | 임베딩 초기화 방식, gradient 설정, optimizer 하이퍼파라미터, tied 여부, clamp 상한선 |
| `prompt` | 템플릿 치환, recon/KD 입력 조립 결과, 각종 assert 검증 |
| `recon` | shift/flatten/CE 계산 전 과정, 위치별 CE·확률·argmax |
| `kd` | teacher/student 분포 비교(top3), 위치별 KL 기여도 |
| `step` | loss 결합식, backward 후 grad norm, 파라미터 업데이트 크기, 복원 여부 |
| `grad` | gradient가 target 토큰 행에만 흐르는지 assert 검증 |
| `sched` | 스케줄러/스텝 계산 정보 |

---

## 5. save_experiment_report

Setup·loss·metric result를 실험 이름별 폴더 하나에 result 저장.

### 사용 예시

```python
from save_experiment_report import save_experiment_report

save_experiment_report(
    output_dir=trainer.args.output_dir,
    name="prompt_636",
    trainer=trainer,
    recon_result=recon_result,        # Evaluator.reconstruct() 반환값
    task_eval_result=task_result,     # test_student_refuses / evaluate_against_gold 반환값
    loss_plot_prefix=loss_prefix,     # 미리 저장해둔 loss 그래프 경로(같은 폴더로 자동 이동)
)
```

### 저장되는 폴더 구조

```
{output_dir}/{name}/
├── config.json           # 초기화·optimizer·PromptBuilder 설정 + args 전체(기본값 포함)
├── recon_eval.json        # 재구성 평가 원본 결과
├── task_eval.json         # 태스크 평가 원본 결과 (응답 분포, 불일치 케이스 포함)
├── metrics_summary.json   # 최종 loss 요약 (best_eval_loss, best_eval_kd 등)
├── summary.txt            # 위 내용을 사람이 읽기 좋게 합친 텍스트 리포트
├── loss.png
├── loss_log.png
├── loss_eval.png
└── loss_eval_log.png
```

### `summary.txt` 구성 — `===` 구분선으로 섹션화됨

```
[실험: {name}]

[설정 — 임베딩/옵티마이저]
  init_mode, 각 target 토큰 norm, lr(설정값 vs 학습 종료 시점 값 구분 표시),
  betas, weight_decay, recon_reduction, kd_tau_squared,
  sync_output_embedding, clamp_norm_multiplier

[설정 — PromptBuilder]
  prompt_mode, enable_thinking, fix_prompt 토큰 길이 등

[설정 — CustomTrainingArgs 전체 (기본값 포함)]
  args 객체의 모든 필드를 자동 나열 (vars(args) 기반, 손으로 고르지 않음)

[최종 loss]
  recon/kd의 처음→마지막→최솟값, best_eval_loss/best_eval_kd와 각각의 달성 epoch

[Reconstruction 평가]
  Exact Match, 토큰 일치 길이/비율, ROUGE-L F1, 정상 종결 여부

[태스크 평가]
  exact_accuracy 등 지표, 응답 분포 상위 10개, 불일치 케이스 최대 10개
```

`lr(설정값)`과 `lr(학습 종료 시점)`을 분리 표시하는 이유: cosine 등 스케줄러가
학습 종료 시점엔 lr을 거의 0까지 감쇠시키므로, 두 값이 다른 것이 정상임을
명시하기 위함.
