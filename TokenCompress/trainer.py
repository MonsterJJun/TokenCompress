"""
CustomTokenTrainer — PromptBuilder / EmbeddingManager / CheckpointManager /
MetricTracker를 조립한 학습기. 기존 단일 파일 버전과 동일하게 동작.

디버그가 필요하면 TrainerDebugger를 attach해서 사용:
    with TrainerDebugger(trainer, keys={"recon"}):
        trainer.train_step(query, answer_ids)
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Optional

from .prompt_builder import PromptBuilder
from .embedding_manager import EmbeddingManager
from .checkpoint_manager import CheckpointManager
from .metric_tracker import MetricTracker


class CustomTokenTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        target_prompt: str,
        target_token_ids: list,
        ae_token_id: int,
        args=None,
        user_template: Optional[str] = None,
        init_mode: str = "random",       # "random" | "mean" | "none"
        recon_reduction: str = "mean",   # "mean"(논문 실측값과 일치) | "sum"(수식 그대로)
        kd_tau_squared: bool = True,     # True: τ² 곱함(논문 실측값과 일치)
        prompt_mode: str = "chat",       # "chat" | "raw"
        enable_thinking=False,           # True | False | None
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.device = args.device
        self.recon_reduction = recon_reduction
        self.kd_tau_squared = kd_tau_squared
        assert recon_reduction in ("sum", "mean"), \
            f"recon_reduction must be 'sum' or 'mean', got {recon_reduction}"

        torch.manual_seed(args.seed)

        if isinstance(target_token_ids, int):
            target_token_ids = [target_token_ids]

        # ── 조립: 입력 구성 ──
        self.builder = PromptBuilder(
            tokenizer=tokenizer,
            fix_prompt=target_prompt,
            target_token_ids=target_token_ids,
            ae_token_id=ae_token_id,
            user_template=user_template,
            device=self.device,
            prompt_mode=prompt_mode,
            enable_thinking=enable_thinking,
        )

        # ── 조립: 임베딩 초기화 / grad 격리 / 복원 ──
        self.emb = EmbeddingManager(
            model=model,
            target_token_ids=target_token_ids,
            n_extra_special=1,          # AE 1개
            init_mode=init_mode,
            token_names=self.builder.target_token_strs,   # 실제 토큰 문자열을 이름표로 사용
        )

        # ── 조립: 저장 관리 ──
        self.ckpt = CheckpointManager(
            output_dir=args.output_dir,
            save_total_limit=getattr(args, "save_total_limit", None),
        )

        # ── 조립: 지표 추적 ──
        self.metrics = MetricTracker()

        # 편의 참조 (기존 코드 호환)
        self.target_token_ids = self.builder.target_token_ids
        self.n_target = self.builder.n_target
        self.target_token_strs = self.builder.target_token_strs
        self.ae_token_id = ae_token_id
        self.end_id = self.builder.end_id
        self.end_str = self.builder.end_token
        self.target_prompt_text = target_prompt
        self.prompt_ids = self.builder.prompt_ids
        self.prompt_ids_tensor = self.builder.prompt_ids_tensor
        self.embed = self.emb.embed
        self.init_mode = self.emb.init_mode_desc
        self._frozen_embed = self.emb._frozen

        # optimizer
        self.optimizer = torch.optim.AdamW(
            [self.embed.weight],
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
        self.scheduler = None
        self.debugger = None

    # 기존 인터페이스 호환용 (디버거가 호출)
    def build_recon_inputs(self):
        r = self.builder.build_recon()
        return r["input_ids"], r["labels"], r["prefix_len"]

    def build_kd_inputs(self, query, answer_ids=None):
        return self.builder.build_kd(query)

    # -----------------------------------------------------------------
    # Stage 1: Reconstruction
    # -----------------------------------------------------------------
    def compute_recon_loss(self):
        """
        논문 수식(원표기): L_recon = -Σ_j log P(s_j | [target..],[AE],s_<j)  — sum.
        논문 실측 loss 크기와 대조 검증한 결과 mean이 일치해서 기본값을 "mean"으로 둠.
        HF의 out.loss(labels 넘겼을 때 자동 계산)에 의존하지 않고 직접 계산 —
        reduction 전환을 명시적으로 제어하기 위함.
        """
        r = self.builder.build_recon()
        input_ids, labels = r["input_ids"], r["labels"]

        out = self.model(input_ids=input_ids)   # labels 안 넘김 — reduction 직접 제어
        logits = out.logits.float()

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction=self.recon_reduction,
        )

        if self.debugger:
            self.debugger.on_recon(input_ids, labels, out, r["prefix_len"], loss)
        return loss

    # -----------------------------------------------------------------
    # Stage 2: KD
    # -----------------------------------------------------------------
    def compute_kd_loss(self, query: str, answer_ids: torch.Tensor):
        """
        논문 수식(3) 원표기: L_KD = (1/T') Σ KL(σ(z^T/τ) ‖ σ(z^S/τ))  — τ² 없음.
        논문 실측 loss 크기와 대조 검증한 결과 τ²를 곱한 쪽이 일치해서
        기본값을 kd_tau_squared=True로 둠(Hinton 2015 관례).
        """
        if not torch.is_tensor(answer_ids):
            answer_ids = torch.tensor(answer_ids, dtype=torch.long)
        answer_ids = answer_ids.to(self.device)
        T_prime = answer_ids.shape[0]
        if T_prime == 0:
            return None

        built = self.builder.build_kd(query)
        t_ids, s_ids = built["teacher_prefix_ids"], built["student_prefix_ids"]

        teacher_full = torch.cat([
            torch.tensor(t_ids, device=self.device), answer_ids
        ]).unsqueeze(0)
        with torch.no_grad():
            teacher_out = self.model(input_ids=teacher_full)
        start_t = len(t_ids) - 1
        teacher_logits = teacher_out.logits[0, start_t:start_t + T_prime, :]

        student_full = torch.cat([
            torch.tensor(s_ids, device=self.device), answer_ids
        ]).unsqueeze(0)
        student_out = self.model(input_ids=student_full)
        start_s = len(s_ids) - 1
        student_logits = student_out.logits[0, start_s:start_s + T_prime, :]

        tau = self.args.temperature
        teacher_prob = F.softmax(teacher_logits / tau, dim=-1)
        student_logprob = F.log_softmax(student_logits / tau, dim=-1)

        kd = F.kl_div(student_logprob, teacher_prob, reduction="batchmean")
        if self.kd_tau_squared:
            kd = kd * (tau ** 2)

        if self.debugger:
            self.debugger.on_kd(query, answer_ids, built, teacher_logits, student_logits,
                                teacher_prob, student_logprob, kd, T_prime, start_t, start_s)
        return kd

    def _restore_frozen_rows(self):
        self.emb.restore_frozen_rows()

    # -----------------------------------------------------------------
    # 한 step
    # -----------------------------------------------------------------
    def train_step(self, query: str, answer_ids: torch.Tensor):
        self.model.train()

        recon = self.compute_recon_loss()
        kd = self.compute_kd_loss(query, answer_ids)

        lam = self.args.lambda_weight
        if kd is None:
            total = (1 - lam) * recon
            kd_val = 0.0
        else:
            total = (1 - lam) * recon + lam * kd
            kd_val = kd.item()

        if self.debugger:
            self.debugger.on_loss_combine(recon, kd_val, total, lam)

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

        self.emb.restore_frozen_rows()

        if self.debugger:
            self.debugger.after_restore()

        return {"total": total.item(), "recon": recon.item(), "kd": kd_val}

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, eval_dataset):
        self.model.eval()
        loader = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                            collate_fn=lambda x: x[0])
        total_loss, total_recon, total_kd, n = 0.0, 0.0, 0.0, 0
        lam = self.args.lambda_weight

        for batch in loader:
            recon = self.compute_recon_loss()
            kd = self.compute_kd_loss(batch["query"], batch["answer_ids"])
            if kd is None:
                total = (1 - lam) * recon; kd_val = 0.0
            else:
                total = (1 - lam) * recon + lam * kd; kd_val = kd.item()
            total_loss += total.item(); total_recon += recon.item()
            total_kd += kd_val; n += 1

        return {"eval_total": total_loss / n, "eval_recon": total_recon / n,
                "eval_kd": total_kd / n}

    # -----------------------------------------------------------------
    # 학습 루프
    # -----------------------------------------------------------------
    def train(self, teacher_dataset, eval_dataset=None):
        loader = DataLoader(teacher_dataset, batch_size=1, shuffle=True,
                            collate_fn=lambda x: x[0])

        total_steps = (self.args.max_steps if self.args.max_steps > 0
                       else len(loader) * self.args.num_train_epochs)

        from transformers import get_scheduler
        self.scheduler = get_scheduler(
            name=self.args.lr_scheduler_type, optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_steps, num_training_steps=total_steps,
        )

        if self.debugger:
            self.debugger.on_train_start(len(loader), total_steps)

        self._print_config(len(loader), total_steps, accum_steps=1, batch_size=1)

        save_steps = getattr(self.args, "save_steps", 0)
        save_epochs = getattr(self.args, "save_epochs", 0)

        step = 0
        pbar = tqdm(total=total_steps, desc="Token Training")

        for epoch in range(self.args.num_train_epochs):
            for batch in loader:
                logs = self.train_step(batch["query"], batch["answer_ids"])
                step += 1

                lr_now = self.scheduler.get_last_lr()[0] if self.scheduler else 0.0
                self.metrics.log_step(step, logs["total"], logs["recon"], logs["kd"], lr_now)

                pbar.set_postfix(epoch=epoch + 1, total=f"{logs['total']:.4f}",
                                 recon=f"{logs['recon']:.4f}", kd=f"{logs['kd']:.4f}",
                                 lr=f"{lr_now:.2e}")
                pbar.update(1)

                if step % self.args.logging_steps == 0:
                    print(f"[epoch {epoch+1} step {step}] total={logs['total']:.4f} "
                          f"recon={logs['recon']:.4f} kd={logs['kd']:.4f} lr={lr_now:.3e}")

                if save_steps > 0 and step % save_steps == 0:
                    self.save(step)
                if 0 < self.args.max_steps <= step:
                    break

            if eval_dataset is not None:
                self._run_eval_and_save(eval_dataset, epoch)

            if save_epochs > 0 and (epoch + 1) % save_epochs == 0:
                self.save(f"epoch{epoch+1}")

            if 0 < self.args.max_steps <= step:
                break

        pbar.close()
        self.save("final")
        print("학습 완료")
        return self.emb.get_vectors()

    # -----------------------------------------------------------------
    # 가벼운 스텝 단위 학습 (디버깅/실험용)
    # -----------------------------------------------------------------
    def train_steps(self, teacher_dataset, n_steps, shuffle=True, log_every=10):
        """epoch/eval/save 없이 n_steps만 실행. history엔 기록되므로 plot 가능."""
        loader = DataLoader(teacher_dataset, batch_size=1, shuffle=shuffle,
                            collate_fn=lambda x: x[0])

        def _cycle(dl):
            while True:
                for b in dl:
                    yield b

        it = _cycle(loader)
        logs_list = []

        for i in range(n_steps):
            batch = next(it)
            logs = self.train_step(batch["query"], batch["answer_ids"])
            logs_list.append(logs)

            lr_now = self.scheduler.get_last_lr()[0] if self.scheduler else self.args.learning_rate
            step_num = len(self.metrics.history["step"]) + 1
            self.metrics.log_step(step_num, logs["total"], logs["recon"], logs["kd"], lr_now)

            if log_every and (i + 1) % log_every == 0:
                print(f"[step {i+1}/{n_steps}] total={logs['total']:.4f} "
                      f"recon={logs['recon']:.4f} kd={logs['kd']:.4f} lr={lr_now:.3e}")

        print(f"train_steps 완료 ({n_steps} steps)")
        return logs_list

    # -----------------------------------------------------------------
    # 공통 헬퍼 (배치 버전도 재사용)
    # -----------------------------------------------------------------
    def _print_config(self, n_batches, total_steps, accum_steps, batch_size):
        b = self.builder
        print(f"[스텝 계산] batches/epoch={n_batches}, accum={accum_steps}, "
              f"total_steps={total_steps}, warmup={self.args.warmup_steps} "
              f"({100*self.args.warmup_steps/max(total_steps,1):.1f}%)")
        print(f"[설정] lr={self.args.learning_rate:.2e}, "
              f"betas=({self.args.adam_beta1}, {self.args.adam_beta2}), "
              f"wd={self.args.weight_decay}, lambda={self.args.lambda_weight}, "
              f"tau={self.args.temperature}")
        print(f"       effective_batch={batch_size * accum_steps}, "
              f"recon_reduction={self.recon_reduction}, kd_tau_squared={self.kd_tau_squared}")
        print(f"       prompt_mode={b.prompt_mode}, enable_thinking={b.enable_thinking}, "
              f"init={self.init_mode}")

    def _run_eval_and_save(self, eval_dataset, epoch):
        ev = self.evaluate(eval_dataset)
        improved = self.metrics.log_eval(epoch + 1, ev["eval_total"],
                                         ev["eval_recon"], ev["eval_kd"])
        print(f"[epoch {epoch+1} EVAL] eval_total={ev['eval_total']:.4f} "
              f"eval_recon={ev['eval_recon']:.4f} eval_kd={ev['eval_kd']:.4f}")
        if improved:
            print(f"[Best 갱신 - total] {self.metrics.best_eval_loss:.6f}")
            self.save("best")
        if self.metrics.kd_improved(epoch + 1):
            print(f"[Best 갱신 - KD] {self.metrics.best_eval_kd:.6f}")
            self.save("best_kd")
        return ev

    # -----------------------------------------------------------------
    # 저장 (CheckpointManager에 위임)
    # -----------------------------------------------------------------
    def save(self, tag):
        path = self.ckpt.save(self.model, self.tokenizer, tag)
        self.ckpt.save_history(self.metrics.history, self.metrics.eval_history, tag)
        return path

    # 편의 속성 (기존 코드 호환)
    @property
    def history(self):
        return self.metrics.history

    @property
    def eval_history(self):
        return self.metrics.eval_history

    @property
    def best_eval_loss(self):
        return self.metrics.best_eval_loss