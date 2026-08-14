"""
Evaluator — 학습 loss와 별개로 '실제 생성 품질'을 재는 도구.
PromptBuilder를 공유하므로 prompt_mode/enable_thinking이 학습과 항상 일치함.
"""
import torch
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
        if not ref_ids or not hyp_ids:
            return 0.0
        lcs = self._lcs_len(ref_ids, hyp_ids)
        if lcs == 0:
            return 0.0
        p, r = lcs / len(hyp_ids), lcs / len(ref_ids)
        return 2 * p * r / (p + r)

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

    # ── 압축 토큰 + 쿼리 → 출력, gold와 종합 비교 ─────────────────
    @torch.no_grad()
    def evaluate_against_gold(self, dataset, n=None, query_key="query",
                               gold_key="gold_response", max_new_tokens=64,
                               compare_teacher=True, verbose=True, show_wrong=10):
        """
        student(압축 토큰) 출력을 gold response와 비교하고, 원하면 teacher와도 비교.
        gold_key가 데이터에 없으면 teacher 비교만 수행.
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

        for i in range(n):
            sample = dataset[i]
            query = sample[query_key]
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
                    print(f"  query   : {item['query'][:60]}...")
                    print(f"  student : {item['student']!r}")
                    print(f"  gold    : {item['gold']!r}")
                    if item.get("teacher") is not None:
                        print(f"  teacher : {item['teacher']!r}")
                    print()
            print("=" * 72)

        return {"metrics": metrics, "rows": rows,
                "response_distribution": dict(response_counter),
                "wrong_cases": wrong_cases}