"""
CustomTrainingArgs — 학습 하이퍼파라미터를 담는 dataclass.
CustomTokenTrainer/BatchCustomTokenTrainer, TrainerDebugger, CheckpointManager 등
코드 전체에서 실제로 참조하는 필드를 빠짐없이 담음 (save_epochs, save_total_limit 등
나중에 추가된 것들 포함).
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CustomTrainingArgs:
    # --- 출력 / 저장 / 로깅 ---
    output_dir: str = "./output"
    logging_steps: int = 10
    save_steps: int = 500                  # N step마다 checkpoint-{step} 저장 (0이면 비활성)
    save_epochs: int = 0                   # N epoch마다 checkpoint-epoch{N} 저장 (0이면 비활성)
    save_total_limit: Optional[int] = 3    # 최대 보관 체크포인트 개수 (None이면 무제한)
    eval_steps: int = 0                    # N step마다 평가 (0이면 epoch마다 평가)
    seed: int = 42

    # --- Resume ---
    resume_from_checkpoint: Optional[str] = None

    # --- 학습 스케줄 ---
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1   # KD 특성상 1 권장 (배치 버전 쓸 때만 의미 있음)
    gradient_accumulation_steps: int = 1
    max_steps: int = -1                    # -1이면 epoch 기반

    # --- Optimizer ---
    learning_rate: float = 1e-4
    weight_decay: float = 0.0              # target 행만 학습하려면 0이 안전 (restore로 이미 방어되긴 함)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # --- LR 스케줄러 ---
    lr_scheduler_type: str = "constant"    # "constant" | "linear" | "cosine" 등
    warmup_steps: int = 0

    # --- 정밀도 / 디바이스 ---
    device: str = "cuda"
    bf16: bool = False
    fp16: bool = False

    # --- BE Token 전용 ---
    lambda_weight: float = 0.9             # KD 비중
    temperature: float = 2.0               # KD temperature
    max_teacher_new_tokens: int = 1024     # teacher 응답 생성 시 최대 길이