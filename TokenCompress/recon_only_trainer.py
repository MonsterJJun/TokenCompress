"""
ReconOnlyTrainer — CustomTokenTrainer 상속. KD를 완전히 배제하고 recon loss만으로 학습.

KD를 계산하지 않는 이유: recon은 query와 무관한(fix_prompt 고정) 재구성 태스크라
teacher_dataset(query/answer_ids) 자체가 필요 없음. lambda_weight=0으로 KD를 곱해서
버리는 방식은 매 스텝 KD forward(teacher+student 2회)를 낭비하므로, 이 클래스는
KD 관련 호출 자체를 아예 하지 않도록 train_step/evaluate/train을 오버라이드함.

compute_recon_loss, compute_kd_loss(안 씀), _restore_frozen_rows 등 나머지 메서드는
부모 것을 그대로 상속 — recon 계산 로직은 단 한 줄도 바뀌지 않음.
"""
from tqdm.auto import tqdm
import torch

from .trainer import CustomTokenTrainer


class ReconOnlyTrainer(CustomTokenTrainer):

    # -----------------------------------------------------------------
    # 한 step — KD 관련 코드가 아예 없음 (부모 train_step과 달리 compute_kd_loss
    # 호출 자체가 없어서, query/answer_ids 인자도 필요 없음)
    # -----------------------------------------------------------------
    def train_step(self):
        self.model.train()

        recon = self.compute_recon_loss()   # 부모 것 그대로 상속
        total = recon                        # KD 항 없음 — (1-lam)*recon + lam*kd 식 자체가 불필요

        if self.debugger:
            self.debugger.on_loss_combine(recon, 0.0, total, 0.0)

        self.optimizer.zero_grad()
        total.backward()

        if self.debugger:
            self.debugger.on_backward()

        if self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_([self.embed.weight], self.args.max_grad_norm)

        if self.debugger:
            self.debugger.before_optimizer_step()

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.debugger:
            self.debugger.after_optimizer_step()

        self.emb.restore_frozen_rows()   # 부모 것 그대로 상속

        if self.debugger:
            self.debugger.after_restore()

        return {"total": total.item(), "recon": recon.item(), "kd": 0.0}

    # -----------------------------------------------------------------
    # Evaluation — recon만 반복 평가 (eval_dataset 자체가 필요 없음,
    # recon은 고정 시퀀스라 "평가"도 매번 같은 값이 나오는 게 정상이지만
    # 인터페이스 일관성을 위해 남겨둠 — 원하면 n_repeat으로 여러 번 평균 가능)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, n_repeat=1):
        self.model.eval()
        total = 0.0
        for _ in range(n_repeat):
            total += self.compute_recon_loss().item()
        avg = total / n_repeat
        return {"eval_total": avg, "eval_recon": avg, "eval_kd": 0.0}

    # -----------------------------------------------------------------
    # 학습 루프 — teacher_dataset 인자 자체가 없음. n_steps만 받음.
    # -----------------------------------------------------------------
    def train(self, n_steps, log_every=10, eval_every=None, save_epochs=0):
        """
        n_steps: 총 optimizer.step() 횟수.
        eval_every: N스텝마다 evaluate() 호출 (None이면 평가 안 함).
        save_epochs: 0이면 저장 안 함, N이면 N스텝마다 저장(epoch 개념이 없으므로
                     여기서는 그대로 step 간격으로 사용).
        """
        from transformers import get_scheduler
        self.scheduler = get_scheduler(
            name=self.args.lr_scheduler_type, optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_steps, num_training_steps=n_steps,
        )

        if self.debugger:
            self.debugger.on_train_start(n_steps, n_steps)

        print(f"[Recon-only 설정] lr={self.args.learning_rate:.2e}, "
              f"betas=({self.args.adam_beta1}, {self.args.adam_beta2}), "
              f"wd={self.args.weight_decay}, total_steps={n_steps}, "
              f"warmup={self.args.warmup_steps}")

        pbar = tqdm(total=n_steps, desc="Recon-only Training")
        for step in range(1, n_steps + 1):
            logs = self.train_step()

            lr_now = self.scheduler.get_last_lr()[0]
            self.metrics.log_step(step, logs["total"], logs["recon"], logs["kd"], lr_now)

            pbar.set_postfix(recon=f"{logs['recon']:.6f}", lr=f"{lr_now:.2e}")
            pbar.update(1)

            if log_every and step % log_every == 0:
                print(f"[step {step}/{n_steps}] recon={logs['recon']:.6f} lr={lr_now:.3e}")

            if eval_every and step % eval_every == 0:
                ev = self.evaluate()
                print(f"  [eval @ step {step}] recon={ev['eval_recon']:.6f}")

            if save_epochs and step % save_epochs == 0:
                self.save(step)

        pbar.close()
        self.save("final")
        print("Recon-only 학습 완료")
        return self.emb.get_vectors()