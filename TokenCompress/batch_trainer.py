"""
BatchCustomTokenTrainer — CustomTokenTrainer 상속.

args.per_device_train_batch_size=1, gradient_accumulation_steps=1로 두면
CustomTokenTrainer 학습과 수학적으로 동일한 결과가 나옴 (검증용).
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from .trainer import CustomTokenTrainer


class BatchCustomTokenTrainer(CustomTokenTrainer):

    # -----------------------------------------------------------------
    # KD (배치) — 단일 샘플 compute_kd_loss와 수식·슬라이싱 동일
    #   단일: teacher_full = prefix + answer, start = len(prefix)-1
    #         logits[start : start+T']
    #   배치: left-padding이라 "끝 기준"으로 세면 동일 위치 → logits[-T-1:-1]
    # -----------------------------------------------------------------
    def compute_kd_loss_batch(self, queries: list, answer_ids_list: list):
        B = len(queries)
        tokenizer = self.tokenizer
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
                 else tokenizer.eos_token_id

        def to_list(ans):
            return ans.tolist() if torch.is_tensor(ans) else list(ans)

        teacher_seqs, student_seqs, ans_lens = [], [], []
        builts = []
        for q, ans in zip(queries, answer_ids_list):
            ans_list = to_list(ans)
            built = self.builder.build_kd(q)      # 부모와 동일한 PromptBuilder 사용
            builts.append(built)
            teacher_seqs.append(built["teacher_prefix_ids"] + ans_list)
            student_seqs.append(built["student_prefix_ids"] + ans_list)
            ans_lens.append(len(ans_list))

        def pad_left(seqs, pad_val):
            max_len = max(len(s) for s in seqs)
            ii = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
            am = torch.zeros((len(seqs), max_len), dtype=torch.long)
            for i, s in enumerate(seqs):
                L = len(s)
                ii[i, -L:] = torch.tensor(s)
                am[i, -L:] = 1
            return ii, am

        t_ids, t_mask = pad_left(teacher_seqs, pad_id)
        s_ids, s_mask = pad_left(student_seqs, pad_id)
        t_ids, t_mask = t_ids.to(self.device), t_mask.to(self.device)
        s_ids, s_mask = s_ids.to(self.device), s_mask.to(self.device)

        with torch.no_grad():
            t_out = self.model(input_ids=t_ids, attention_mask=t_mask, use_cache=False)
        s_out = self.model(input_ids=s_ids, attention_mask=s_mask, use_cache=False)

        tau = self.args.temperature
        total_kl = 0.0
        for i in range(B):
            T = ans_lens[i]
            t_logits = t_out.logits[i, -T-1:-1, :]
            s_logits = s_out.logits[i, -T-1:-1, :]
            teacher_prob = F.softmax(t_logits / tau, dim=-1)
            student_logprob = F.log_softmax(s_logits / tau, dim=-1)
            kl = F.kl_div(student_logprob, teacher_prob, reduction="batchmean")
            if self.kd_tau_squared:               # 부모와 동일한 옵션 존중
                kl = kl * (tau ** 2)
            total_kl = total_kl + kl

            # 디버거는 배치 첫 샘플만 상세 출력 (로그 폭탄 방지)
            if self.debugger and i == 0:
                ans_t = torch.tensor(to_list(answer_ids_list[i]), device=self.device)
                self.debugger.on_kd(
                    queries[i], ans_t, builts[i], t_logits, s_logits,
                    teacher_prob, student_logprob, kl, T,
                    len(builts[i]["teacher_prefix_ids"]) - 1,
                    len(builts[i]["student_prefix_ids"]) - 1,
                )

        return total_kl / B

    @staticmethod
    def _collate_batch(batch_list):
        return {
            "query": [b["query"] for b in batch_list],
            "answer_ids": [b["answer_ids"] for b in batch_list],
        }

    # -----------------------------------------------------------------
    # 한 step (배치)
    #   loss 결합식, clip, optimizer.step, scheduler.step, restore 순서 모두
    #   부모 train_step과 동일. 디버거 콜백도 같은 지점에서 호출.
    # -----------------------------------------------------------------
    def train_step_batch(self, queries, answer_ids_list,
                          accum_steps=1, is_accum_boundary=True):
        self.model.train()

        recon = self.compute_recon_loss()          # 부모 것 그대로 (고정 시퀀스 1회)
        kd = self.compute_kd_loss_batch(queries, answer_ids_list)

        lam = self.args.lambda_weight
        total = (1 - lam) * recon + lam * kd
        kd_val = kd.item()

        if self.debugger:
            self.debugger.on_loss_combine(recon, kd_val, total, lam)

        (total / accum_steps).backward()

        if self.debugger:
            self.debugger.on_backward()

        if is_accum_boundary:
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_([self.embed.weight],
                                                self.args.max_grad_norm)

            if self.debugger:
                self.debugger.before_optimizer_step()

            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()

            if self.debugger:
                self.debugger.after_optimizer_step()

            self.emb.restore_frozen_rows()         # 부모와 동일

            if self.debugger:
                self.debugger.after_restore()

        return {"total": total.item(), "recon": recon.item(), "kd": kd_val}

    # -----------------------------------------------------------------
    # Evaluation (배치)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, eval_dataset):
        self.model.eval()
        loader = DataLoader(eval_dataset,
                            batch_size=self.args.per_device_train_batch_size,
                            shuffle=False, collate_fn=self._collate_batch)
        total_loss, total_recon, total_kd, n = 0.0, 0.0, 0.0, 0
        lam = self.args.lambda_weight

        for batch in loader:
            recon = self.compute_recon_loss()
            kd = self.compute_kd_loss_batch(batch["query"], batch["answer_ids"])
            total = (1 - lam) * recon + lam * kd
            total_loss += total.item(); total_recon += recon.item()
            total_kd += kd.item(); n += 1

        return {"eval_total": total_loss / n, "eval_recon": total_recon / n,
                "eval_kd": total_kd / n}

    # -----------------------------------------------------------------
    # 학습 루프 (배치)
    #   scheduler.step()은 optimizer.step()이 일어난 시점(boundary)에만 →
    # -----------------------------------------------------------------
    def train(self, teacher_dataset, eval_dataset=None):
        loader = DataLoader(teacher_dataset,
                            batch_size=self.args.per_device_train_batch_size,
                            shuffle=True, collate_fn=self._collate_batch)

        accum_steps = max(1, getattr(self.args, "gradient_accumulation_steps", 1))
        micro_per_epoch = len(loader)
        opt_steps_per_epoch = math.ceil(micro_per_epoch / accum_steps)
        total_steps = (self.args.max_steps if self.args.max_steps > 0
                       else opt_steps_per_epoch * self.args.num_train_epochs)

        from transformers import get_scheduler
        self.scheduler = get_scheduler(
            name=self.args.lr_scheduler_type, optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_steps, num_training_steps=total_steps,
        )

        if self.debugger:
            self.debugger.on_train_start(len(loader), total_steps)

        self._print_config(micro_per_epoch, total_steps, accum_steps,
                           self.args.per_device_train_batch_size)
        print(f"       opt_steps/epoch={opt_steps_per_epoch}")

        save_steps = getattr(self.args, "save_steps", 0)
        save_epochs = getattr(self.args, "save_epochs", 0)

        step = 0
        self.optimizer.zero_grad()
        pbar = tqdm(total=total_steps, desc="Batch Token Training")

        stop = False
        for epoch in range(self.args.num_train_epochs):
            n_batches = len(loader)
            for micro_idx, batch in enumerate(loader):
                is_boundary = ((micro_idx + 1) % accum_steps == 0) or \
                              (micro_idx + 1 == n_batches)

                logs = self.train_step_batch(batch["query"], batch["answer_ids"],
                                              accum_steps=accum_steps,
                                              is_accum_boundary=is_boundary)
                if not is_boundary:
                    continue

                step += 1
                lr_now = self.scheduler.get_last_lr()[0] if self.scheduler else 0.0
                self.metrics.log_step(step, logs["total"], logs["recon"],
                                      logs["kd"], lr_now)

                pbar.set_postfix(epoch=epoch+1, total=f"{logs['total']:.4f}",
                                 recon=f"{logs['recon']:.4f}", kd=f"{logs['kd']:.4f}",
                                 lr=f"{lr_now:.2e}")
                pbar.update(1)

                if step % self.args.logging_steps == 0:
                    print(f"[epoch {epoch+1} step {step}] total={logs['total']:.4f} "
                          f"recon={logs['recon']:.4f} kd={logs['kd']:.4f} lr={lr_now:.3e}")

                if save_steps > 0 and step % save_steps == 0:
                    self.save(step)
                if 0 < self.args.max_steps <= step:
                    stop = True
                    break

            if eval_dataset is not None:
                self._run_eval_and_save(eval_dataset, epoch)

            if save_epochs > 0 and (epoch + 1) % save_epochs == 0:
                self.save(f"epoch{epoch+1}")

            if stop:
                break

        pbar.close()
        self.save("final")
        print("학습 완료")
        return self.emb.get_vectors()

    # -----------------------------------------------------------------
    # 가벼운 스텝 단위 학습 (배치)
    # -----------------------------------------------------------------
    def train_steps(self, teacher_dataset, n_steps, shuffle=True, log_every=10):
        loader = DataLoader(teacher_dataset,
                            batch_size=self.args.per_device_train_batch_size,
                            shuffle=shuffle, collate_fn=self._collate_batch)

        def _cycle(dl):
            while True:
                for b in dl:
                    yield b

        it = _cycle(loader)
        logs_list = []

        for i in range(n_steps):
            batch = next(it)
            logs = self.train_step_batch(batch["query"], batch["answer_ids"],
                                          accum_steps=1, is_accum_boundary=True)
            logs_list.append(logs)

            lr_now = self.scheduler.get_last_lr()[0] if self.scheduler \
                     else self.args.learning_rate
            step_num = len(self.metrics.history["step"]) + 1
            self.metrics.log_step(step_num, logs["total"], logs["recon"],
                                  logs["kd"], lr_now)

            if log_every and (i + 1) % log_every == 0:
                print(f"[step {i+1}/{n_steps}] total={logs['total']:.4f} "
                      f"recon={logs['recon']:.4f} kd={logs['kd']:.4f} lr={lr_now:.3e}")

        print(f"train_steps 완료 ({n_steps} steps)")
        return logs_list