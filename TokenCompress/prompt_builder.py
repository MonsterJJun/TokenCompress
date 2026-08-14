"""
PromptBuilder — 모든 입력 시퀀스 구성을 한곳에서 담당.
학습(recon/KD), teacher 데이터셋 생성, 검증(generate)이 전부 이 클래스를 공유해야
"학습 프롬프트와 생성 프롬프트가 어긋나는" 문제가 원천 차단됨.

prompt_mode:
  "chat" — tokenizer.apply_chat_template 사용 (system/user 역할 분리)
  "raw"  — chat template 없이 순수 텍스트 결합

enable_thinking (3단계 값):
  True  — apply_chat_template(enable_thinking=True) 명시 전달
  False — apply_chat_template(enable_thinking=False) 명시 전달 (빈 <think></think> 삽입)
  None  — apply_chat_template 호출 시 이 인자를 아예 넘기지 않음
          -> 템플릿/토크나이저 자체의 기본 동작을 그대로 따름.
  raw 모드에서는 None/False를 동일 취급해 think prefill을 붙임(실험으로 필요성 검증됨).
  True일 때만 prefill을 생략함.
"""
import torch


class PromptBuilder:
    THINK_PREFILL = "\n<think>\n\n</think>\n\n"

    def __init__(self, tokenizer, fix_prompt, target_token_ids, ae_token_id,
                 user_template=None, device="cuda", end_token="<|im_end|>",
                 prompt_mode="chat", enable_thinking=False, raw_separator="\n\n"):
        assert prompt_mode in ("chat", "raw"), \
            f"prompt_mode must be 'chat' or 'raw', got {prompt_mode}"
        assert enable_thinking in (True, False, None), \
            f"enable_thinking must be True, False, or None, got {enable_thinking}"

        self.tokenizer = tokenizer
        self.fix_prompt = fix_prompt
        self.user_template = user_template
        self.device = device
        self.prompt_mode = prompt_mode
        self.enable_thinking = enable_thinking
        self.raw_separator = raw_separator

        self.target_token_ids = list(target_token_ids)
        self.n_target = len(self.target_token_ids)
        # 실제 토큰 문자열 (예: '<|BE|>', '<|CATEGORY1|>') — S1/S2 같은 고정 이름 대신
        # 이 값이 EmbeddingManager/디버그 출력 등에서 표시용 이름(token_names)으로 쓰임
        self.target_token_strs = [tokenizer.decode([t]) for t in self.target_token_ids]
        self.category_prefix_str = "".join(self.target_token_strs)
        self.ae_token_id = ae_token_id

        self.end_token = end_token
        self.end_id = tokenizer.convert_tokens_to_ids(end_token)

        self.prompt_ids = tokenizer.encode(fix_prompt, add_special_tokens=False)
        self.prompt_ids_tensor = torch.tensor(self.prompt_ids, device=device)

    # ── 템플릿 치환 ────────────────────────────────────────────────
    def fill_template(self, query):
        """user_template의 {user_text} 치환. 이중 중괄호({{user_text}})도 안전 처리."""
        if self.user_template is None:
            return query
        safe = self.user_template.replace("{{", "{").replace("}}", "}")
        return safe.format(user_text=query)

    # ── 프롬프트 조립 (mode에 따라 분기) ───────────────────────────
    def _build_prefix(self, system_content, user_content):
        """prompt_mode에 따라 prefix 텍스트를 만들어 반환."""
        if self.prompt_mode == "chat":
            kwargs = dict(tokenize=False, add_generation_prompt=True)
            if self.enable_thinking is not None:
                kwargs["enable_thinking"] = self.enable_thinking
            return self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_content},
                 {"role": "user", "content": user_content}],
                **kwargs,
            )

        # raw 모드 — apply_chat_template을 안 쓰므로 "인자 생략" 개념이 없음.
        text = f"{system_content}{self.raw_separator}{user_content}"
        if self.enable_thinking is not True:
            text = text.rstrip() + self.THINK_PREFILL
        return text

    # ── Recon 입력: [target..., AE, fix_prompt, end] (mode 무관) ──
    def build_recon(self):
        target_t = torch.tensor(self.target_token_ids, device=self.device)
        ae_t = torch.tensor([self.ae_token_id], device=self.device)
        end_t = torch.tensor([self.end_id], device=self.device)

        input_ids = torch.cat([target_t, ae_t, self.prompt_ids_tensor, end_t]).unsqueeze(0)
        prefix_len = self.n_target + 1
        labels = torch.cat([
            torch.full((prefix_len,), -100, device=self.device, dtype=torch.long),
            self.prompt_ids_tensor, end_t,
        ]).unsqueeze(0)
        return {"input_ids": input_ids, "labels": labels, "prefix_len": prefix_len}

    # ── KD 입력: teacher(system=fix_prompt) vs student(system=target토큰) ──
    def build_kd(self, query):
        user_content = self.fill_template(query)

        teacher_prefix = self._build_prefix(self.fix_prompt, user_content)
        student_prefix = self._build_prefix(self.category_prefix_str, user_content)

        t_ids = self.tokenizer(teacher_prefix, add_special_tokens=False)["input_ids"]
        s_ids = self.tokenizer(student_prefix, add_special_tokens=False)["input_ids"]

        return {
            "user_content": user_content,
            "teacher_prefix": teacher_prefix, "teacher_prefix_ids": t_ids,
            "student_prefix": student_prefix, "student_prefix_ids": s_ids,
        }

    # ── 추론용 prefix (Evaluator가 사용, 학습과 동일 형식 보장) ────
    def build_inference_prefix(self, query, use_student=True):
        built = self.build_kd(query)
        return built["student_prefix"] if use_student else built["teacher_prefix"]

    # ── 검증용: recon 생성 시작 입력 [target..., AE] ──────────────
    def build_generate_input(self):
        return torch.tensor([self.target_token_ids + [self.ae_token_id]], device=self.device)

    # ── teacher 데이터셋 생성용 ────────────────────────────────────
    def build_teacher_generate_prefix(self, query):
        return self.build_kd(query)["teacher_prefix"]

    def summary(self):
        thinking_desc = {True: "True(명시)", False: "False(명시)",
                          None: "None(인자 생략, 템플릿 기본값 따름)"}[self.enable_thinking]
        return {
            "prompt_mode": self.prompt_mode,
            "enable_thinking": thinking_desc,
            "fix_prompt_tokens": len(self.prompt_ids),
            "n_target": self.n_target,
            "target_token_strs": self.target_token_strs,
            "ae_token": self.tokenizer.decode([self.ae_token_id]),
            "end_token": self.end_token,
            "has_user_template": self.user_template is not None,
            "raw_separator": repr(self.raw_separator) if self.prompt_mode == "raw" else None,
        }