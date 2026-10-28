"""
Evaluator — 학습 loss와 별개로 '실제 생성 품질'을 재는 도구.
PromptBuilder를 공유하므로 prompt_mode/enable_thinking이 학습과 항상 일치함.
"""
import torch
from tqdm.auto import tqdm
from collections import Counter


class Evaluator:
    def __init__(self, model, tokenizer, builder):
        self.model = model
        self.tokenizer = tokenizer
        self.builder = builder

    # ── ROUGE-L ───────────────────────────────────────────────────
    @staticmethod
    def _lcs_len(a, b):
        n, m = len(a), len(b)
        if n == 0 or m == 0:
            return 0
        dp = [0] * (m + 1)
        for i in range(1, n + 1):
            prev, ai = 0, a[i-1]
            for j in range(1, m + 1):
                tmp = dp[j]
                dp[j] = prev + 1 if ai == b[j-1] else max(dp[j], dp[j-1])
                prev = tmp
        return dp[m]

    def rouge_l_f1(self, ref_ids, hyp_ids):
        """
        토큰 ID 단위 ROUGE-L F1 (직접 구현, LCS 기반).
        토크나이저의 서브워드 경계 그대로 비교하므로, "토큰 단위로 정확히 재현했는가"를
        측정하는 이 프로젝트의 목적에 더 정확히 맞음.
        공식 rouge_score 라이브러리와 동일한 값을 내는지 검증 완료(표준 ROUGE-L F1, β=1).
        """
        if not ref_ids or not hyp_ids:
            return 0.0
        lcs = self._lcs_len(ref_ids, hyp_ids)
        if lcs == 0:
            return 0.0
        p, r = lcs / len(hyp_ids), lcs / len(ref_ids)
        return 2 * p * r / (p + r)

    def rouge_l_text(self, ref_text, hyp_text, use_stemmer=False):
        """
        텍스트(단어) 단위 ROUGE-L — 공식 rouge_score 라이브러리 사용.
        다른 논문/보고서의 단어-단위 ROUGE-L과 직접 비교하고 싶을 때 사용.
        (rouge_l_f1과는 채점 단위가 달라 값이 서로 다를 수 있음 — 토큰 단위 재현
         정확도를 보려면 rouge_l_f1을, 일반적인 텍스트 유사도 보고용이면 이걸 사용)
        """
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            raise ImportError(
                "rouge_l_text()는 'rouge_score' 패키지가 필요함. "
                "pip install rouge-score 로 설치 후 사용."
            )
        if not hasattr(self, "_rouge_scorer"):
            self._rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)
        scores = self._rouge_scorer.score(ref_text, hyp_text)["rougeL"]
        return {"precision": scores.precision, "recall": scores.recall, "fmeasure": scores.fmeasure}

    # ── Recon 재구성 평가 ─────────────────────────────────────────
    @torch.no_grad()
    def reconstruct(self, max_new_tokens=2048):
        """[S1..Sn, AE]만 주고 fix_prompt를 재현하는지 확인."""
        self.model.eval()
        b = self.builder
        input_ids = b.build_generate_input()

        out = self.model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=[self.tokenizer.eos_token_id, b.end_id],
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_ids = out[0, input_ids.shape[1]:].tolist()
        clean_ids = [t for t in gen_ids if t not in (self.tokenizer.eos_token_id, b.end_id)]
        gen_text = self.tokenizer.decode(clean_ids, skip_special_tokens=True).strip()

        match_len = 0
        for x, y in zip(clean_ids, b.prompt_ids):
            if x != y:
                break
            match_len += 1

        return {
            "gen_text": gen_text,
            "gen_raw": self.tokenizer.decode(gen_ids, skip_special_tokens=False),
            "exact_match": gen_text == b.fix_prompt.strip(),
            "ended_clean": bool(gen_ids) and gen_ids[-1] in (self.tokenizer.eos_token_id, b.end_id),
            "hit_max": len(gen_ids) >= max_new_tokens,
            "match_len": match_len,
            "target_len": len(b.prompt_ids),
            "match_ratio": match_len / max(len(b.prompt_ids), 1),
            "rouge_l": self.rouge_l_f1(b.prompt_ids, clean_ids),
        }

    def report_reconstruction(self, max_new_tokens=2048, show_divergence=True):
        r = self.reconstruct(max_new_tokens)
        b = self.builder
        print("=" * 70)
        print("[Reconstruction 평가]")
        print("=" * 70)
        print(f"Exact Match    : {r['exact_match']}")
        print(f"토큰 일치 길이 : {r['match_len']} / {r['target_len']} ({100*r['match_ratio']:.1f}%)")
        print(f"ROUGE-L F1     : {100*r['rouge_l']:.2f}%")
        print(f"정상 종결(EOS) : {r['ended_clean']}   hit_max: {r['hit_max']}")

        if show_divergence and not r["exact_match"]:
            m = r["match_len"]
            lo, hi = max(0, m-5), m+5
            print(f"\n[{m}번째 토큰부터 어긋남]")
            print("  정답:", [self.tokenizer.decode([t]) for t in b.prompt_ids[lo:hi]])
            gen_ids = self.tokenizer.encode(r["gen_text"], add_special_tokens=False)
            print("  생성:", [self.tokenizer.decode([t]) for t in gen_ids[lo:hi]])
        print("=" * 70)
        return r

    # ── 단일 쿼리 생성 (학습과 동일한 prompt_mode 사용) ───────────
    @torch.no_grad()
    def generate_response(self, query, max_new_tokens=64, use_student=True,
                           return_prefix=False):
        self.model.eval()
        b = self.builder
        prefix = b.build_inference_prefix(query, use_student=use_student)

        inputs = self.tokenizer(prefix, add_special_tokens=False,
                                return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=[self.tokenizer.eos_token_id, b.end_id],
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=False)

        result = {
            "text": text, "raw": raw, "gen_ids": gen_ids.tolist(),
            "ended_clean": len(gen_ids) > 0 and
                           gen_ids[-1].item() in (self.tokenizer.eos_token_id, b.end_id),
        }
        if return_prefix:
            result["prefix"] = prefix
        return result

    def classify(self, query, max_new_tokens=64, use_student=True):
        """텍스트만 반환하는 간편 버전."""
        return self.generate_response(query, max_new_tokens, use_student)["text"]

    # ── 압축 토큰 + 쿼리 → 출력, gold와 종합 비교 (진행바 포함) ───
    @torch.no_grad()
    def evaluate_against_gold(self, dataset, n=None, query_key="query",
                               gold_key="gold_response", max_new_tokens=64,
                               compare_gold=True, compare_teacher=True,
                               verbose=True, show_wrong=10):
        """
        student(압축 토큰) 출력을, 원하면 gold와, 원하면 teacher와 비교.
        compare_gold / compare_teacher를 독립적으로 켜고 끌 수 있음:
          - compare_gold=True, compare_teacher=False  -> gold 비교만
          - compare_gold=False, compare_teacher=True   -> teacher 비교만
          - 둘 다 True                                  -> 둘 다 (기본값)
          - 둘 다 False                                  -> response_distribution만
        gold_key가 데이터에 없으면 compare_gold=True여도 자동으로 gold 비교는 생략됨.
        n: 평가할 샘플 개수. None이면 dataset 전체.
        """
        b = self.builder
        n = len(dataset) if n is None else min(n, len(dataset))

        rows = []
        s_exact_gold = s_contains_gold = 0
        t_exact_gold = t_contains_gold = 0
        s_match_t = 0
        s_ended_clean = 0
        has_gold = None
        response_counter = Counter()
        wrong_cases = []

        pbar = tqdm(range(n), desc="평가", dynamic_ncols=True)
        for i in pbar:
            sample = dataset[i]
            query = sample[query_key]
            gold = None
            if compare_gold:
                gold = str(sample[gold_key]).strip() if gold_key in sample else None
            if has_gold is None:
                has_gold = gold is not None

            s_res = self.generate_response(query, max_new_tokens, use_student=True)
            s_text = s_res["text"]
            response_counter[s_text] += 1
            s_ended_clean += s_res["ended_clean"]

            t_text = None
            if compare_teacher:
                t_text = self.generate_response(query, max_new_tokens, use_student=False)["text"]
                if s_text == t_text:
                    s_match_t += 1

            row = {"query": query, "student": s_text, "teacher": t_text, "gold": gold,
                   "student_ended_clean": s_res["ended_clean"]}

            if gold is not None:
                s_exact = (s_text == gold)
                s_contains = gold.rstrip(".").lower() in s_text.lower()
                s_exact_gold += s_exact
                s_contains_gold += s_contains
                row.update({"student_exact": s_exact, "student_contains": s_contains})
                if t_text is not None:
                    t_exact = (t_text == gold)
                    t_exact_gold += t_exact
                    t_contains_gold += (gold.rstrip(".").lower() in t_text.lower())
                    row["teacher_exact"] = t_exact
                if not s_exact:
                    wrong_cases.append(row)

            rows.append(row)

            # 진행바 옆에 실시간 지표 표시
            postfix = {"s==t": f"{100*s_match_t/(i+1):.0f}%"} if compare_teacher else {}
            if has_gold:
                postfix["s==gold"] = f"{100*s_exact_gold/(i+1):.0f}%"
            pbar.set_postfix(**postfix)
        pbar.close()

        metrics = {
            "n": n,
            "prompt_mode": b.prompt_mode,
            "enable_thinking": b.enable_thinking,
            "student_ended_clean_rate": s_ended_clean / n,
        }
        if has_gold:
            metrics.update({
                "student_exact_gold": s_exact_gold / n,
                "student_contains_gold": s_contains_gold / n,
            })
            if compare_teacher:
                metrics.update({
                    "teacher_exact_gold": t_exact_gold / n,
                    "teacher_contains_gold": t_contains_gold / n,
                })
        if compare_teacher:
            metrics["student_vs_teacher"] = s_match_t / n

        if verbose:
            print("=" * 72)
            print(f"[압축 토큰 성능 평가]  n={n}  "
                  f"(mode={b.prompt_mode}, thinking={b.enable_thinking})")
            print("=" * 72)
            print(f"student 정상 종결(EOS)     : {100*metrics['student_ended_clean_rate']:.1f}%")
            if compare_teacher:
                print(f"student == teacher         : {100*metrics['student_vs_teacher']:.1f}%"
                      f"   <- 압축 충실도")
            if has_gold:
                print(f"student == gold (정확일치) : {100*metrics['student_exact_gold']:.1f}%")
                print(f"student ⊇ gold  (포함)     : {100*metrics['student_contains_gold']:.1f}%")
                if compare_teacher:
                    print(f"teacher == gold (정확일치) : {100*metrics['teacher_exact_gold']:.1f}%"
                          f"   <- 상한선")
            print()
            print("[student 실제 응답 분포 (상위 10)]")
            for resp, cnt in response_counter.most_common(10):
                mark = ""
                if has_gold and rows[0]["gold"] is not None:
                    mark = "  ✓" if resp == rows[0]["gold"] else ""
                print(f"  {resp!r:45s} : {cnt:>4}개{mark}")

            if has_gold and wrong_cases and show_wrong:
                print()
                print(f"[불일치 케이스 (총 {len(wrong_cases)}개 중 "
                      f"{min(show_wrong, len(wrong_cases))}개)]")
                for item in wrong_cases[:show_wrong]:
                    print(f"  query   : {item['query']}")
                    print(f"  student : {item['student']!r}")
                    print(f"  gold    : {item['gold']!r}")
                    if item.get("teacher") is not None:
                        print(f"  teacher : {item['teacher']!r}")
                    print()
            print("=" * 72)

        return {"metrics": metrics, "rows": rows,
                "response_distribution": dict(response_counter),
                "wrong_cases": wrong_cases}

    # ── 페르소나 유지 평가 — exact match 대신 행동 규칙 체크 + 유사도 ──
    @torch.no_grad()
    def evaluate_persona_consistency(
        self, dataset, n=None, query_key="query", max_new_tokens=128,
        persona_identity_keywords=None,      # 응답에 있어야 할(주제 유도) 키워드
        forbidden_self_disclosure=None,      # 있으면 안 되는 정체성 노출 문구
        forbidden_other_names=None,          # 있으면 안 되는 다른 AI 이름
        verbose=True,
    ):
        """
        persona처럼 "정답이 하나로 정해지지 않는" 태스크용 평가.
        exact match 대신 아래 세 가지를 확인:
          1. 정체성 노출 안 했는가 (forbidden_self_disclosure)
          2. 다른 이름을 자칭 안 했는가 (forbidden_other_names)
          3. 주제 이탈 질문에 리다이렉트했는가 (persona_identity_keywords 중 하나라도 포함)
        + teacher와의 ROUGE-L 유사도(연속 점수, exact match보다 훨씬 공정한 비교)
        """
        if forbidden_self_disclosure is None:
            forbidden_self_disclosure = [
                "as an ai", "i am an ai", "i'm an ai", "language model",
                "i don't have a physical", "i am a program", "i'm a program",
                "artificial intelligence", "as a large language",
            ]
        if forbidden_other_names is None:
            forbidden_other_names = [
                "chatgpt", "gpt-4", "gpt-3", "claude", "gemini", "llama",
                "bard", "copilot", "i am gpt", "i'm gpt",
            ]
        if persona_identity_keywords is None:
            persona_identity_keywords = ["nimbus", "storage", "account", "billing", "cloud"]

        b = self.builder
        n = len(dataset) if n is None else min(n, len(dataset))

        n_no_disclosure = 0
        n_no_other_name = 0
        n_redirected = 0
        rouge_scores = []
        rows = []

        pbar = tqdm(range(n), desc="persona 일관성 평가", dynamic_ncols=True)
        for i in pbar:
            sample = dataset[i]
            query = sample[query_key]

            s_res = self.generate_response(query, max_new_tokens, use_student=True)
            t_res = self.generate_response(query, max_new_tokens, use_student=False)
            s_text_lower = s_res["text"].lower()

            no_disclosure = not any(p in s_text_lower for p in forbidden_self_disclosure)
            no_other_name = not any(p in s_text_lower for p in forbidden_other_names)
            redirected = any(k in s_text_lower for k in persona_identity_keywords)

            n_no_disclosure += no_disclosure
            n_no_other_name += no_other_name
            n_redirected += redirected

            s_ids = self.tokenizer.encode(s_res["text"], add_special_tokens=False)
            t_ids = self.tokenizer.encode(t_res["text"], add_special_tokens=False)
            rouge = self.rouge_l_f1(t_ids, s_ids)
            rouge_scores.append(rouge)

            rows.append({
                "query": query, "student": s_res["text"], "teacher": t_res["text"],
                "no_disclosure": no_disclosure, "no_other_name": no_other_name,
                "redirected": redirected, "rouge_l": rouge,
            })

            pbar.set_postfix(정체성비노출=f"{100*n_no_disclosure/(i+1):.0f}%",
                             리다이렉트=f"{100*n_redirected/(i+1):.0f}%")
        pbar.close()

        metrics = {
            "n": n,
            "no_self_disclosure_rate": n_no_disclosure / n,
            "no_other_name_rate": n_no_other_name / n,
            "redirect_rate": n_redirected / n,
            "avg_rouge_l_vs_teacher": sum(rouge_scores) / n,
        }

        if verbose:
            print("=" * 72)
            print(f"[페르소나 일관성 평가]  n={n}")
            print("=" * 72)
            print(f"정체성 비노출 비율     : {100*metrics['no_self_disclosure_rate']:.1f}%  "
                  f"(AI/언어모델이라고 밝히지 않은 비율)")
            print(f"다른 이름 미자칭 비율  : {100*metrics['no_other_name_rate']:.1f}%")
            print(f"주제 리다이렉트 비율   : {100*metrics['redirect_rate']:.1f}%  "
                  f"(관련 키워드로 유도한 비율)")
            print(f"teacher 대비 평균 유사도(ROUGE-L): {100*metrics['avg_rouge_l_vs_teacher']:.1f}%")
            print()
            broken = [r for r in rows if not (r["no_disclosure"] and r["no_other_name"])]
            if broken:
                print(f"[페르소나 붕괴 케이스 (최대 5개, 총 {len(broken)}개)]")
                for item in broken[:5]:
                    print(f"  query  : {item['query']}")
                    print(f"  student: {item['student']}")
                    print()
            print("=" * 72)

        return {"metrics": metrics, "rows": rows}

    # ── GSM8K 전용 평가 — 형식 준수(Step.../Answer:)와 최종 숫자 정답을 분리해서 채점 ──
    @staticmethod
    def _extract_final_number(text):
        """텍스트에서 'Answer:' 뒤에 오는 첫 숫자를 뽑음. 콤마/소수점/음수 처리."""
        import re
        match = re.search(r"answer\s*:\s*(-?[\d,]+\.?\d*)", text, flags=re.IGNORECASE)
        if not match:
            return None
        num_str = match.group(1).replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            return None

    @staticmethod
    def _extract_gsm8k_gold_number(gold_answer_text):
        """GSM8K 원본 answer 필드(...#### 42 형식)에서 정답 숫자만 추출."""
        if "####" in gold_answer_text:
            tail = gold_answer_text.split("####")[-1].strip()
        else:
            tail = gold_answer_text.strip()
        tail = tail.replace(",", "")
        try:
            return float(tail)
        except ValueError:
            return None

    @torch.no_grad()
    def evaluate_gsm8k(self, dataset, n=None, query_key="question", gold_key="answer",
                       teacher_key="response", max_new_tokens=256, verbose=True, show_wrong=10):
        """
        gold_key는 GSM8K 원본 answer 필드(전체 풀이 + '#### 42' 형식) 또는
        미리 뽑아둔 순수 숫자(정답)여도 됨 — 둘 다 자동 처리.
        teacher_key: 데이터셋에 이미 저장된 teacher(원본 모델) 응답 필드명.
                     teacher를 다시 생성하지 않고 이 필드를 그대로 재사용 — student만 새로 생성.
        format_compliance: "Step"과 "Answer:"가 응답에 포함됐는가 (형식 준수)
        exact_answer: 최종 숫자가 gold와 정확히 일치하는가 (내용 정답)
        """
        n = len(dataset) if n is None else min(n, len(dataset))

        n_format_ok = 0
        n_correct = 0
        n_both = 0
        n_matches_teacher = 0   # student 숫자 == teacher 숫자 (gold와 무관, distillation 충실도)
        rouge_scores = []
        rows = []

        pbar = tqdm(range(n), desc="GSM8K 평가", dynamic_ncols=True)
        for i in pbar:
            sample = dataset[i]
            query = sample[query_key]
            gold_raw = str(sample[gold_key])
            t_text = sample[teacher_key]   # 이미 데이터셋에 저장된 teacher 응답 재사용

            s_res = self.generate_response(query, max_new_tokens, use_student=True)
            s_text = s_res["text"]

            has_step = "step" in s_text.lower()
            has_answer_tag = "answer" in s_text.lower() and ":" in s_text
            format_ok = has_step and has_answer_tag

            pred_num = self._extract_final_number(s_text)
            gold_num = self._extract_gsm8k_gold_number(gold_raw)
            correct = (pred_num is not None and gold_num is not None
                       and abs(pred_num - gold_num) < 1e-6)

            teacher_num = self._extract_final_number(t_text)
            matches_teacher = (pred_num is not None and teacher_num is not None
                              and abs(pred_num - teacher_num) < 1e-6)

            t_ids = self.tokenizer.encode(t_text, add_special_tokens=False)
            s_ids = self.tokenizer.encode(s_text, add_special_tokens=False)
            rouge = self.rouge_l_f1(t_ids, s_ids)
            rouge_scores.append(rouge)

            n_format_ok += format_ok
            n_correct += correct
            n_both += (format_ok and correct)
            n_matches_teacher += matches_teacher

            rows.append({
                "query": query, "student": s_text, "teacher": t_text,
                "pred_num": pred_num, "gold_num": gold_num, "teacher_num": teacher_num,
                "format_ok": format_ok, "correct": correct,
                "matches_teacher": matches_teacher, "rouge_l_vs_teacher": rouge,
            })

            pbar.set_postfix(형식준수=f"{100*n_format_ok/(i+1):.0f}%",
                             정답=f"{100*n_correct/(i+1):.0f}%",
                             teacher일치=f"{100*n_matches_teacher/(i+1):.0f}%")
        pbar.close()

        metrics = {
            "n": n,
            "format_compliance_rate": n_format_ok / n,
            "exact_answer_rate": n_correct / n,
            "student_vs_teacher_rate": n_matches_teacher / n,
            "both_format_and_correct_rate": n_both / n,
            "avg_rouge_l_vs_teacher": sum(rouge_scores) / n,
        }

        if verbose:
            print("=" * 72)
            print(f"[GSM8K 평가]  n={n}")
            print("=" * 72)
            print(f"형식 준수율(Step.../Answer:) : {100*metrics['format_compliance_rate']:.1f}%")
            print(f"정답률(최종 숫자 일치)        : {100*metrics['exact_answer_rate']:.1f}%")
            print(f"형식+정답 둘 다 만족           : {100*metrics['both_format_and_correct_rate']:.1f}%")
            print(f"teacher 대비 평균 유사도(ROUGE-L): {100*metrics['avg_rouge_l_vs_teacher']:.1f}%  "
                  f"(참고용 — 정답 여부와 별개로 풀이 스타일이 얼마나 비슷한지)")

            wrong = [r for r in rows if not r["correct"]]
            if wrong and show_wrong:
                print(f"\n[오답 케이스 (최대 {show_wrong}개, 총 {len(wrong)}개)]")
                for item in wrong[:show_wrong]:
                    print(f"  query : {item['query']}")
                    print(f"  예측  : {item['pred_num']}   정답: {item['gold_num']}")
                    print(f"  형식준수: {item['format_ok']}")
                    print()
            print("=" * 72)

        return {"metrics": metrics, "rows": rows}

    # ── Classification 전용 — sklearn 표준 지표 (Accuracy, F1, Precision, Recall) ──
    @torch.no_grad()
    def evaluate_classification(self, dataset, n=None, query_key="query", gold_key="gold_label",
                                teacher_key="response", max_new_tokens=16, verbose=True,
                                compare_teacher=True):
        from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                     recall_score, classification_report)

        n = len(dataset) if n is None else min(n, len(dataset))
        y_true, y_pred, y_teacher, rows = [], [], [], []

        pbar = tqdm(range(n), desc="Classification 평가", dynamic_ncols=True)
        for i in pbar:
            sample = dataset[i]
            query = sample[query_key]
            gold = str(sample[gold_key]).strip().lower()

            s_res = self.generate_response(query, max_new_tokens, use_student=True)
            pred = s_res["text"].strip().lower()

            row = {"query": query, "student": pred, "gold": gold}
            if compare_teacher:
                teacher_pred = str(sample[teacher_key]).strip().lower()   # 데이터셋 필드 재사용
                row["teacher"] = teacher_pred
                y_teacher.append(teacher_pred)

            y_true.append(gold)
            y_pred.append(pred)
            rows.append(row)
            postfix = {"acc": f"{100*accuracy_score(y_true, y_pred):.0f}%"}
            if compare_teacher:
                postfix["s==t"] = f"{100*accuracy_score(y_teacher, y_pred):.0f}%"
            pbar.set_postfix(**postfix)
        pbar.close()

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "n": n,
        }
        if compare_teacher:
            # student가 gold와 무관하게 teacher를 얼마나 그대로 흉내내는지 (distillation 충실도)
            metrics["student_vs_teacher_accuracy"] = accuracy_score(y_teacher, y_pred)

        if verbose:
            print("=" * 72)
            print(f"[Classification 평가]  n={n}")
            print("=" * 72)
            print(f"Accuracy       : {100*metrics['accuracy']:.1f}%")
            print(f"F1 (macro)     : {100*metrics['f1_macro']:.1f}%")
            print(f"Precision(macro): {100*metrics['precision_macro']:.1f}%")
            print(f"Recall (macro) : {100*metrics['recall_macro']:.1f}%")
            if "student_vs_teacher_accuracy" in metrics:
                print(f"Student==Teacher: {100*metrics['student_vs_teacher_accuracy']:.1f}%  "
                      f"(gold와 무관하게 teacher를 얼마나 그대로 따르는지)")
            print()
            print(classification_report(y_true, y_pred, zero_division=0))
            print("=" * 72)

        return {"metrics": metrics, "rows": rows}

    # ── QA 전용 — SQuAD 공식 EM/F1 (토큰 겹침 기반, 공식 예시로 검증된 구현) ──
    @staticmethod
    def _squad_normalize(s):
        import re, string
        s = s.lower()
        s = "".join(ch for ch in s if ch not in string.punctuation)
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        return " ".join(s.split())

    @classmethod
    def _squad_f1(cls, prediction, ground_truth):
        from collections import Counter
        pred_tokens = cls._squad_normalize(prediction).split()
        gt_tokens = cls._squad_normalize(ground_truth).split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return float(pred_tokens == gt_tokens)
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        return 2 * precision * recall / (precision + recall)

    @classmethod
    def _squad_em(cls, prediction, ground_truth):
        return float(cls._squad_normalize(prediction) == cls._squad_normalize(ground_truth))

    @torch.no_grad()
    def evaluate_qa_squad(self, dataset, n=None, query_key="query", gold_key="gold_label",
                          teacher_key="response", max_new_tokens=64, verbose=True,
                          show_wrong=10, compare_teacher=True):
        """
        SQuAD 공식 평가 방식(EM/F1). gold_key 값이 리스트(복수 gold 후보)면 최댓값을,
        단일 문자열이면 그대로 사용.
        compare_teacher=True면 데이터셋에 저장된 teacher 응답(teacher_key)과 비교해서,
        student가 gold와 무관하게 teacher를 얼마나 그대로 따르는지(distillation 충실도)도
        EM/F1로 함께 측정. teacher를 다시 생성하지 않음 — student만 새로 생성.
        """
        n = len(dataset) if n is None else min(n, len(dataset))
        em_scores, f1_scores, rows = [], [], []
        teacher_em_scores, teacher_f1_scores = [], []

        pbar = tqdm(range(n), desc="QA(SQuAD) 평가", dynamic_ncols=True)
        for i in pbar:
            sample = dataset[i]
            query = sample[query_key]
            gold = sample[gold_key]
            golds = gold if isinstance(gold, list) else [gold]

            s_res = self.generate_response(query, max_new_tokens, use_student=True)
            pred = s_res["text"]

            em = max(self._squad_em(pred, g) for g in golds)
            f1 = max(self._squad_f1(pred, g) for g in golds)
            em_scores.append(em)
            f1_scores.append(f1)
            row = {"query": query, "student": pred, "gold": gold, "em": em, "f1": f1}

            if compare_teacher:
                teacher_pred = sample[teacher_key]   # 데이터셋에 이미 있는 teacher 응답 재사용
                t_em = self._squad_em(pred, teacher_pred)
                t_f1 = self._squad_f1(pred, teacher_pred)
                teacher_em_scores.append(t_em)
                teacher_f1_scores.append(t_f1)
                row.update({"teacher": teacher_pred, "teacher_em": t_em, "teacher_f1": t_f1})

            rows.append(row)

            postfix = {"EM": f"{100*sum(em_scores)/(i+1):.0f}%", "F1": f"{100*sum(f1_scores)/(i+1):.0f}%"}
            if compare_teacher:
                postfix["s==t_EM"] = f"{100*sum(teacher_em_scores)/(i+1):.0f}%"
            pbar.set_postfix(**postfix)
        pbar.close()

        metrics = {
            "exact_match": 100 * sum(em_scores) / n,
            "f1": 100 * sum(f1_scores) / n,
            "n": n,
        }
        if compare_teacher:
            metrics["teacher_exact_match"] = 100 * sum(teacher_em_scores) / n
            metrics["teacher_f1"] = 100 * sum(teacher_f1_scores) / n

        if verbose:
            print("=" * 72)
            print(f"[QA 평가 — SQuAD 공식 EM/F1]  n={n}")
            print("=" * 72)
            print(f"Exact Match (vs gold)    : {metrics['exact_match']:.1f}")
            print(f"F1          (vs gold)    : {metrics['f1']:.1f}")
            if compare_teacher:
                print(f"Exact Match (vs teacher) : {metrics['teacher_exact_match']:.1f}  "
                      f"(gold와 무관하게 teacher를 얼마나 그대로 따르는지)")
                print(f"F1          (vs teacher) : {metrics['teacher_f1']:.1f}")

            wrong = [r for r in rows if r["em"] < 1.0]
            if wrong and show_wrong:
                print(f"\n[EM 기준 불일치 케이스 (최대 {show_wrong}개, 총 {len(wrong)}개)]")
                for item in wrong[:show_wrong]:
                    print(f"  query: {item['query']}")
                    print(f"  예측 : {item['student']!r}   F1={item['f1']:.2f}")
                    print(f"  정답 : {item['gold']!r}")
                    print()
            print("=" * 72)

        return {"metrics": metrics, "rows": rows}

    # ── Summarize 전용 — ROUGE (student vs teacher, 정답이 하나로 안 정해지는 태스크) ──
    @torch.no_grad()
    def evaluate_summarize(self, dataset, n=None, query_key="query", gold_key=None,
                           teacher_key="response", max_new_tokens=96, verbose=True):
        """
        gold_key: cnn_dailymail의 highlights, Dolly summarization의 response처럼
                  실제 참고 요약이 있으면 그 필드명을 지정. 없으면 None(vs teacher만 비교)
                  — CNN/DailyMail 표준 벤치마크도 모델 요약을 ROUGE로 highlights와 비교하는
                  방식이라 형식(문장 수)이 달라도 표준적인 비교.
        teacher_key: 데이터셋에 이미 저장된 teacher 응답 필드명. teacher를 다시 생성하지
                     않고 이 필드를 그대로 재사용 — student만 새로 생성.
        """
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

        n = len(dataset) if n is None else min(n, len(dataset))
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        gold_scores = {"rouge1": [], "rouge2": [], "rougeL": []} if gold_key else None
        rows = []

        pbar = tqdm(range(n), desc="Summarize 평가", dynamic_ncols=True)
        for i in pbar:
            sample = dataset[i]
            query = sample[query_key]
            t_text = sample[teacher_key]   # 이미 데이터셋에 저장된 teacher 응답 재사용

            s_res = self.generate_response(query, max_new_tokens, use_student=True)

            result = scorer.score(t_text, s_res["text"])
            for k in scores:
                scores[k].append(result[k].fmeasure)

            row = {"query": query, "student": s_res["text"], "teacher": t_text,
                  "rougeL": result["rougeL"].fmeasure}

            if gold_key:
                gold = sample[gold_key]
                gold_result = scorer.score(gold, s_res["text"])
                for k in gold_scores:
                    gold_scores[k].append(gold_result[k].fmeasure)
                row["gold"] = gold
                row["rougeL_vs_gold"] = gold_result["rougeL"].fmeasure

            rows.append(row)
            postfix = {"rougeL(t)": f"{100*sum(scores['rougeL'])/(i+1):.0f}%"}
            if gold_key:
                postfix["rougeL(g)"] = f"{100*sum(gold_scores['rougeL'])/(i+1):.0f}%"
            pbar.set_postfix(**postfix)
        pbar.close()

        metrics = {
            "rouge1": 100 * sum(scores["rouge1"]) / n,
            "rouge2": 100 * sum(scores["rouge2"]) / n,
            "rougeL": 100 * sum(scores["rougeL"]) / n,
            "n": n,
        }
        if gold_key:
            metrics["rouge1_vs_gold"] = 100 * sum(gold_scores["rouge1"]) / n
            metrics["rouge2_vs_gold"] = 100 * sum(gold_scores["rouge2"]) / n
            metrics["rougeL_vs_gold"] = 100 * sum(gold_scores["rougeL"]) / n

        if verbose:
            print("=" * 72)
            print(f"[Summarize 평가]  n={n}")
            print("=" * 72)
            print(f"[vs teacher] ROUGE-1: {metrics['rouge1']:.1f}  ROUGE-2: {metrics['rouge2']:.1f}  "
                  f"ROUGE-L: {metrics['rougeL']:.1f}  (distillation 충실도)")
            if gold_key:
                print(f"[vs gold]    ROUGE-1: {metrics['rouge1_vs_gold']:.1f}  "
                      f"ROUGE-2: {metrics['rouge2_vs_gold']:.1f}  "
                      f"ROUGE-L: {metrics['rougeL_vs_gold']:.1f}  (실제 정답 대비 품질)")
            print("=" * 72)

        return {"metrics": metrics, "rows": rows}