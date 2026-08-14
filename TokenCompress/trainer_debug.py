"""
TrainerDebugger — CustomTokenTrainer에 붙여서 내부 동작을 관찰하는 debugger
"""
import torch
import torch.nn.functional as F


class TrainerDebugger:
    ALL_KEYS = {"init", "grad", "recon", "kd", "step", "sched", "prompt"}

    def __init__(self, trainer, keys=True, max_rows=20, collect=False):
        self.t = trainer
        self.keys = keys
        self.max_rows = max_rows
        self.collect = collect
        self._before_snapshot = None
        self.last_recon = None
        self.last_kd = None

    # ── 활성화 제어 ────────────────────────────────────────────────
    def on(self, key):
        if self.keys is True:
            return True
        if self.keys is False or self.keys is None:
            return False
        return key in self.keys

    def attach(self):
        self.t.debugger = self
        return self

    def detach(self):
        self.t.debugger = None
        return self

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.detach()
        return False

    def _header(self, key, title):
        if self.on(key):
            print("\n" + "=" * 78, flush=True)
            print(f"[{key}] {title}", flush=True)
            print("=" * 78, flush=True)

    @staticmethod
    def _token_names(t):
        """t.target_token_strs가 없는 구버전 trainer 대비 fallback."""
        names = getattr(t, "target_token_strs", None)
        if names:
            return names
        return [f"token{i+1}" for i in range(len(t.target_token_ids))]

    # ── 정적 점검 (학습 전 1회) ────────────────────────────────────
    def inspect_setup(self):
        """초기화 / gradient 설정 / optimizer 설정 점검."""
        t = self.t
        self._header("init", "임베딩 초기화 / gradient / optimizer 점검")

        n_special = 1 + t.n_target
        orig_vocab = t.embed.weight.shape[0] - n_special
        trainable = [n for n, p in t.model.named_parameters() if p.requires_grad]
        names = self._token_names(t)

        print(f"  초기화 방식        : {t.init_mode}")
        print(f"  vocab 크기         : {t.embed.weight.shape[0]} "
              f"(신규 {n_special}개 제외 시 {orig_vocab})")
        print(f"  hidden_dim         : {t.embed.weight.shape[1]}")
        print(f"  requires_grad=True : {trainable}")
        print(f"  target_token_ids   : {t.target_token_ids} {names}")
        print(f"  ae_token_id        : {t.ae_token_id} ({t.tokenizer.decode([t.ae_token_id])!r})")
        print(f"  end_id             : {t.end_id} ({t.end_str!r})")

        ref_std = t.embed.weight[:orig_vocab].std().item()
        for i, tid in enumerate(t.target_token_ids):
            v = t.embed.weight[tid]
            print(f"  {names[i]}(id={tid}) norm={v.float().norm().item():.4f} "
                  f"std={v.float().std().item():.6f} (기존 vocab std={ref_std:.6f})")

        g = t.optimizer.param_groups[0]
        print(f"  optimizer: AdamW lr={g['lr']} betas={g['betas']} eps={g['eps']} "
              f"weight_decay={g['weight_decay']}")
        print(f"  관리 파라미터 수: {len(g['params'])} (shape={tuple(g['params'][0].shape)})")

        recon_reduction = getattr(t, "recon_reduction", "(속성 없음 — 구버전)")
        kd_tau_squared = getattr(t, "kd_tau_squared", "(속성 없음 — 구버전)")
        print(f"  recon_reduction={recon_reduction}, kd_tau_squared={kd_tau_squared}")

        if hasattr(t, "builder"):
            print()
            print(f"  [PromptBuilder 설정]")
            for k, v in t.builder.summary().items():
                print(f"    {k}: {v}")
        else:
            print("\n  ⚠️ trainer.builder가 없음 — 구버전 CustomTokenTrainer로 보임 "
                  "(prompt_mode/enable_thinking 관련 기능 사용 불가)")
        return self

    # ── 프롬프트 입출력 검증 ──────────────────────────────────────
    def inspect_prompts(self, query, show_ids=False, run_generate=False, max_new_tokens=32):
        """
        현재 prompt_mode/enable_thinking 설정으로 실제 어떤 입력이 만들어지는지,
        (선택) 그 입력에 모델이 뭐라고 응답하는지까지 확인.
        trainer.builder(PromptBuilder)가 있어야 사용 가능.
        """
        t = self.t
        if not hasattr(t, "builder"):
            raise AttributeError(
                "trainer.builder가 없음 — 이 기능은 PromptBuilder를 쓰는 최신 "
                "CustomTokenTrainer에서만 동작함. TokenCompress/trainer.py와 "
                "prompt_builder.py가 최신 버전인지 확인할 것."
            )
        b = t.builder
        tok = t.tokenizer
        names = self._token_names(t)

        self._header("prompt", "프롬프트 입출력 검증")
        print("[설정]")
        for k, v in b.summary().items():
            print(f"  {k}: {v}")
        print()

        # 1) 템플릿 치환
        filled = b.fill_template(query)
        print("[1] fill_template()")
        print(f"  입력 query : {query[:]!r}")
        print(f"  치환 결과  : {filled[:]!r}")
        if b.user_template is not None:
            assert query in filled, "⚠️ query가 템플릿에 안 들어감!"
            assert "{user_text}" not in filled, "⚠️ 치환 안 된 자리표시자 남음!"
            print("  ✅ 치환 정상")
        print()

        # 2) recon 입력
        r = b.build_recon()
        ii, lb, pl = r["input_ids"], r["labels"], r["prefix_len"]
        token_desc = "+".join(names)   # "S x2" 대신 실제 토큰명들을 그대로 나열
        print("[2] build_recon()")
        print(f"  input_ids shape={tuple(ii.shape)}, prefix_len={pl}")
        print(f"  구조: [{token_desc}] + [AE] + [fix_prompt x{len(b.prompt_ids)}] + [end]")
        assert ii.shape == lb.shape, "⚠️ input_ids/labels 길이 불일치!"
        assert (lb[0, :pl] == -100).all(), "⚠️ prefix 마스킹 안 됨!"
        assert (lb[0, pl:] != -100).all(), "⚠️ prompt 구간이 마스킹됨!"
        assert ii[0, -1].item() == b.end_id, "⚠️ 종결 토큰 없음!"
        print(f"  마스킹 {int((lb[0]==-100).sum())}개 / CE 대상 {int((lb[0]!=-100).sum())}개")
        print("  ✅ recon 입력 정상")
        if show_ids:
            print(f"  앞부분 ids: {ii[0, :pl+5].tolist()}")
        print()

        # 3) KD 입력
        kd = b.build_kd(query)
        print(f"[3] build_kd()  — mode={b.prompt_mode}, thinking={b.enable_thinking}")
        print(f"  teacher prefix 토큰수 : {len(kd['teacher_prefix_ids'])}")
        print(f"  student prefix 토큰수 : {len(kd['student_prefix_ids'])}")
        print(f"  압축 비율             : "
              f"{len(kd['teacher_prefix_ids'])/max(len(kd['student_prefix_ids']),1):.1f}x")
        print()
        print("  --- teacher prefix (전문) ---")
        print(f"  {kd['teacher_prefix']!r}")
        print()
        print(f"  --- student prefix (전문, system=[{token_desc}]) ---")
        print(f"  {kd['student_prefix']!r}")
        print()

        assert b.category_prefix_str in kd["student_prefix"], "⚠️ target 토큰이 student에 없음!"
        assert b.fix_prompt[:40] in kd["teacher_prefix"], "⚠️ teacher에 fix_prompt 없음!"
        reenc = tok.encode(b.category_prefix_str, add_special_tokens=False)
        assert reenc == b.target_token_ids, f"⚠️ target 토큰이 쪼개짐! {reenc}"

        if b.enable_thinking is False:
            has_think = "<think>" in kd["student_prefix"]
            print(f"  enable_thinking=False → 빈 think 블록 포함 여부: {has_think}")
            if b.prompt_mode == "raw":
                assert has_think, "⚠️ raw 모드인데 think prefill이 안 붙음!"
        print("  ✅ KD 입력 정상")
        print()

        # 4) (선택) 실제 생성 결과
        if run_generate:
            print("[4] 실제 생성 결과")
            for label, prefix in [("teacher", kd["teacher_prefix"]),
                                   ("student", kd["student_prefix"])]:
                inputs = tok(prefix, add_special_tokens=False,
                             return_tensors="pt").to(t.model.device)
                with torch.no_grad():
                    out = t.model.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                        eos_token_id=[tok.eos_token_id, b.end_id],
                        pad_token_id=tok.pad_token_id,
                    )
                gen = out[0, inputs["input_ids"].shape[1]:]
                print(f"  [{label}] {tok.decode(gen, skip_special_tokens=True).strip()!r}")
                print(f"           raw: {tok.decode(gen, skip_special_tokens=False)!r}")
            print()

        return {"filled": filled, "recon": r, "kd": kd}

    # ── 콜백: trainer가 호출 ───────────────────────────────────────
    def on_train_start(self, n_batches, total_steps):
        if not self.on("sched"):
            return
        t = self.t
        self._header("sched", "스케줄러 / 스텝 계산")
        n_epochs = getattr(t.args, "num_train_epochs", "N/A")
        print(f"  len(loader)={n_batches}, num_train_epochs={n_epochs}")
        print(f"  total_steps = {total_steps}  (= train_step 호출 = scheduler.step 호출)")
        print(f"  scheduler={t.args.lr_scheduler_type}, warmup={t.args.warmup_steps} "
              f"({100*t.args.warmup_steps/max(total_steps,1):.1f}%)")
        if t.scheduler is not None:
            print(f"  초기 lr={t.scheduler.get_last_lr()[0]:.3e}")

    def on_recon(self, input_ids, labels, out, prefix_len, loss_value=None):
        if not self.on("recon") and not self.collect:
            return
        t = self.t
        tok = t.tokenizer
        L = input_ids.shape[1]
        logits = out.logits

        padded = F.pad(labels, (0, 1), value=-100)
        shift_labels = padded[..., 1:].contiguous()
        vocab_size = logits.shape[-1]
        flat_logits = logits.float().view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        n_valid = (flat_labels != -100).sum().item()
        n_ignored = (flat_labels == -100).sum().item()

        loss_final = loss_value.item() if torch.is_tensor(loss_value) else \
                     (loss_value if loss_value is not None else
                      (out.loss.item() if getattr(out, "loss", None) is not None else None))

        positions, tokens_in, tokens_target, p_target_list, ce_list, argmax_list = \
            [], [], [], [], [], []
        for i in range(L):
            lbl = shift_labels[0, i].item()
            if lbl == -100:
                continue
            probs = F.softmax(flat_logits[i], dim=-1)
            p_t = probs[lbl].item()
            positions.append(i)
            tokens_in.append(tok.decode([input_ids[0, i].item()]))
            tokens_target.append(tok.decode([lbl]))
            p_target_list.append(p_t)
            ce_list.append(-torch.log(torch.tensor(p_t)).item())
            argmax_list.append(tok.decode([probs.argmax().item()]))

        if self.collect:
            self.last_recon = {
                "input_ids": input_ids, "labels": labels, "logits": logits,
                "flat_logits": flat_logits, "flat_labels": flat_labels,
                "shift_labels": shift_labels, "prefix_len": prefix_len,
                "positions": positions, "tokens_in": tokens_in,
                "tokens_target": tokens_target, "p_target": p_target_list,
                "ce": ce_list, "argmax": argmax_list,
                "loss": loss_final,
                "loss_sum": sum(ce_list),
                "loss_mean": sum(ce_list) / len(ce_list) if ce_list else None,
            }

        if not self.on("recon"):
            return

        names = self._token_names(t)
        token_desc = "+".join(names)
        recon_reduction = getattr(t, "recon_reduction", "mean")

        self._header("recon", "Recon Loss 계산 과정")
        print(f"[입력]")
        print(f"  input_ids shape : {tuple(input_ids.shape)}  "
              f"([{token_desc}] + AE 1개 + prompt {len(t.prompt_ids)}개 + end 1개)")
        print(f"  labels shape    : {tuple(labels.shape)}")
        print(f"  prefix_len(마스킹 구간) = {prefix_len}")
        print(f"[출력]")
        print(f"  logits shape : {tuple(logits.shape)}")
        print()
        print(f"[loss 계산 — reduction={recon_reduction}]")
        print(f"  1) labels 오른쪽 -100 pad : {labels.shape[1]} -> {padded.shape[1]}")
        print(f"  2) [...,1:] shift         : {shift_labels.shape[1]} "
              f"(logits {logits.shape[1]}와 일치)")
        print(f"  3) flatten                : {tuple(flat_logits.shape)}")
        print(f"  4) CE 대상                : 무시 {n_ignored} / 반영 {n_valid} "
              f"(= prompt {len(t.prompt_ids)} + end 1)")
        print()

        print(f"[위치별 대응 — logits[i] ↔ shift_labels[i]]")
        n_rows = len(positions) if self.max_rows is None else min(self.max_rows, len(positions))
        for k in range(n_rows):
            i = positions[k]
            print(f"  [{i:>4}] 입력={tokens_in[k]!r:<18} -> 정답={tokens_target[k]!r}  "
                  f"P(정답)={p_target_list[k]:.6f}  CE={ce_list[k]:.4f}  "
                  f"argmax={argmax_list[k]!r}")
        if n_rows < len(positions):
            print(f"  ... (총 {len(positions)}개 중 {n_rows}개 표시, max_rows로 조절)")
        print()

        manual_sum = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="sum")
        manual_mean = manual_sum / max(n_valid, 1)
        print(f"[검증] trainer 반환 loss={loss_final:.6f}")
        print(f"       수동재현 sum={manual_sum.item():.6f}  mean={manual_mean.item():.6f}")

    def on_kd(self, query, answer_ids, built, teacher_logits, student_logits,
              teacher_prob, student_logprob, kd, T_prime, start_t, start_s):
        if not self.on("kd") and not self.collect:
            return
        t = self.t
        tok = t.tokenizer
        tau = t.args.temperature
        tp_ids, sp_ids = built["teacher_prefix_ids"], built["student_prefix_ids"]
        names = self._token_names(t)
        token_desc = "+".join(names)

        pos_data = []
        for i in range(T_prime):
            ans_id = answer_ids[i].item()
            t_prob = teacher_prob[i]
            s_prob = student_logprob[i].exp()
            kl_pos = (t_prob * (torch.log(t_prob + 1e-12) - student_logprob[i])).sum().item()
            pos_data.append({
                "target_token": tok.decode([ans_id]), "target_id": ans_id,
                "teacher_p_target": t_prob[ans_id].item(),
                "student_p_target": s_prob[ans_id].item(),
                "kl": kl_pos,
                "teacher_top3": torch.topk(t_prob, 3),
                "student_top3": torch.topk(s_prob, 3),
            })

        if self.collect:
            self.last_kd = {
                "query": query, "answer_ids": answer_ids,
                "teacher_prefix": built["teacher_prefix"],
                "student_prefix": built["student_prefix"],
                "teacher_prefix_ids": tp_ids, "student_prefix_ids": sp_ids,
                "teacher_logits": teacher_logits, "student_logits": student_logits,
                "teacher_prob": teacher_prob, "student_logprob": student_logprob,
                "T_prime": T_prime, "positions": pos_data, "kd_value": kd.item(),
            }

        if not self.on("kd"):
            return

        prompt_mode = getattr(t.builder, "prompt_mode", "(정보 없음)") if hasattr(t, "builder") else "(builder 없음)"
        enable_thinking = getattr(t.builder, "enable_thinking", "(정보 없음)") if hasattr(t, "builder") else "(builder 없음)"
        kd_tau_squared = getattr(t, "kd_tau_squared", "(속성 없음)")

        self._header("kd", "KD Loss 계산 과정")
        print(f"[입력]")
        print(f"  mode={prompt_mode}, thinking={enable_thinking}")
        print(f"  query      : {query[:120]!r}{'...' if len(query) > 120 else ''}")
        print(f"  answer_ids : {answer_ids.tolist()} -> {tok.decode(answer_ids)!r}")
        print(f"  T_prime    : {T_prime}")
        print()
        print(f"  [Teacher] prefix 길이={len(tp_ids)}, slice 시작={start_t}")
        print(f"    prefix: {built['teacher_prefix'][:]!r}")
        print()
        print(f"  [Student] prefix 길이={len(sp_ids)}, slice 시작={start_s}  "
              f"(system=[{token_desc}])")
        print(f"    prefix: {built['student_prefix'][:]!r}")
        print()
        print(f"  ※ 압축 비율: {len(tp_ids)}/{len(sp_ids)} = "
              f"{len(tp_ids)/max(len(sp_ids),1):.1f}x")
        print()
        print(f"[출력]")
        print(f"  teacher_logits : {tuple(teacher_logits.shape)}")
        print(f"  student_logits : {tuple(student_logits.shape)}")
        print()

        print(f"[분포 비교 — temperature={tau}, tau_squared={kd_tau_squared}]")
        for i, pd in enumerate(pos_data):
            t_top, s_top = pd["teacher_top3"], pd["student_top3"]
            print(f"  위치{i} (정답={pd['target_token']!r}, id={pd['target_id']})")
            print("    teacher top3: " + ", ".join(
                f"{tok.decode([j])!r}={p:.4f}"
                for p, j in zip(t_top.values.tolist(), t_top.indices.tolist())))
            print("    student top3: " + ", ".join(
                f"{tok.decode([j])!r}={p:.4f}"
                for p, j in zip(s_top.values.tolist(), s_top.indices.tolist())))
            print(f"    정답 확률 — teacher={pd['teacher_p_target']:.6f}  "
                  f"student={pd['student_p_target']:.6f}")
            print(f"    이 위치 KL(teacher||student) = {pd['kl']:.6f}")
        print()
        print(f"[loss] kd = {kd.item():.8f}")

    def on_loss_combine(self, recon, kd_val, total, lam):
        if not self.on("step"):
            return
        self._header("step", "train_step — loss 결합 및 파라미터 업데이트")
        print(f"  total = (1-{lam})×recon + {lam}×kd")
        print(f"        = {1-lam:.2f}×{recon.item():.6f} + {lam:.2f}×{kd_val:.6f} "
              f"= {total.item():.6f}")

    def on_backward(self):
        if not self.on("step") and not self.on("grad"):
            return
        t = self.t
        names = self._token_names(t)
        g = t.embed.weight.grad
        print(f"  backward 후 grad norm(전체) : {g.norm().item():.6f}")
        for i, tid in enumerate(t.target_token_ids):
            print(f"    {names[i]}(id={tid}) grad norm   : {g[tid].norm().item():.6f}")
        nonzero = (g.abs().sum(dim=1) > 0).nonzero(as_tuple=True)[0].tolist()
        print(f"  grad!=0 인 행 : {nonzero[:10]}{' ...' if len(nonzero) > 10 else ''}  "
              f"(target={t.target_token_ids})")
        assert set(nonzero) <= set(t.target_token_ids), "⚠️ target 외 행에 grad가 흐름!"

    def before_optimizer_step(self):
        if not self.on("step"):
            return
        t = self.t
        self._before_snapshot = {tid: t.embed.weight[tid].detach().clone()
                                 for tid in t.target_token_ids}

    def after_optimizer_step(self):
        if not self.on("step") or self._before_snapshot is None:
            return
        t = self.t
        names = self._token_names(t)
        for i, tid in enumerate(t.target_token_ids):
            delta = (t.embed.weight[tid] - self._before_snapshot[tid]).float().norm().item()
            print(f"    {names[i]}(id={tid}) 업데이트 크기 : {delta:.8f}")
        if t.scheduler is not None:
            print(f"  현재 lr : {t.scheduler.get_last_lr()[0]:.3e}")

    def after_restore(self):
        if not self.on("step"):
            return
        t = self.t
        diff = (t.embed.weight - t._frozen_embed).abs().sum(dim=1)
        changed = (diff > 0).nonzero(as_tuple=True)[0].tolist()
        print(f"  복원 후 원본과 다른 행 : {changed}  (target만 남아야 정상)")