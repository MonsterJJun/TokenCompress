"""
EmbeddingManager — 학습 대상 임베딩 행의 초기화 / gradient 격리 / 복원 / 분석.
"""
import torch
import torch.nn.functional as F


class EmbeddingManager:
    def __init__(self, model, target_token_ids, n_extra_special=1, init_mode="random",
                 token_names=None):
        """
        n_extra_special: target 외에 새로 추가된 special 토큰 수 (AE 1개 -> 1)
        init_mode: "random" | "mean" | "none"
        token_names: 각 target_token_id에 대응하는 표시용 토큰 문자열 리스트
                     (예: ["<|BE|>"] 또는 ["<|CATEGORY1|>", "<|CATEGORY2|>"]).
                     None이면 "S1","S2",... 로 자동 생성.
        """
        self.model = model
        self.target_token_ids = list(target_token_ids)
        self.n_target = len(self.target_token_ids)
        self.n_special = self.n_target + n_extra_special

        if token_names is not None:
            assert len(token_names) == self.n_target, \
                f"token_names 길이({len(token_names)})가 target_token_ids 개수({self.n_target})와 다름"
            self.token_names = list(token_names)
        else:
            self.token_names = [f"S{i+1}" for i in range(self.n_target)]

        self.embed = model.get_input_embeddings()
        self.orig_vocab = self.embed.weight.shape[0] - self.n_special
        self.hidden_dim = self.embed.weight.shape[1]

        self.freeze_backbone()
        self.register_grad_hook()
        self.init_mode_desc = self.initialize(init_mode)
        self._frozen = self.embed.weight.detach().clone()
        self._initial_vectors = self.snapshot()

    def freeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = False
        self.embed.weight.requires_grad = True

    def register_grad_hook(self):
        ids = torch.tensor(self.target_token_ids, device=self.embed.weight.device)

        def hook(grad):
            mask = torch.zeros_like(grad)
            mask[ids] = 1.0
            return grad * mask

        self._hook_handle = self.embed.weight.register_hook(hook)
        return self._hook_handle

    def initialize(self, mode="random"):
        with torch.no_grad():
            if mode == "none":
                return "none (resize 결과 그대로 사용)"
            if mode == "random":
                std = self.embed.weight[:self.orig_vocab].std()
                for tid in self.target_token_ids:
                    self.embed.weight[tid] = torch.randn(
                        self.hidden_dim, device=self.embed.weight.device,
                        dtype=self.embed.weight.dtype,
                    ) * std
                return f"random (std={std.item():.6f})"
            if mode == "mean":
                avg = self.embed.weight[:self.orig_vocab].mean(dim=0)
                for tid in self.target_token_ids:
                    self.embed.weight[tid] = avg + torch.randn_like(avg) * 0.01
                return "mean-based (avg + noise*0.01)"
        raise ValueError(f"unknown init_mode: {mode}")

    def restore_frozen_rows(self):
        """optimizer.step() 후 target 외 행을 원본으로 되돌림."""
        with torch.no_grad():
            saved = {t: self.embed.weight[t].clone() for t in self.target_token_ids}
            self.embed.weight.copy_(self._frozen)
            for t, row in saved.items():
                self.embed.weight[t] = row

    def get_vectors(self):
        return {name: self.embed.weight[t].detach().clone()
                for name, t in zip(self.token_names, self.target_token_ids)}

    def check_isolation(self):
        """grad가 target 행에만 흐르는지 검사. (backward 직후 호출)"""
        g = self.embed.weight.grad
        if g is None:
            return {"ok": None, "msg": "grad 없음 (backward 전)"}
        nonzero = (g.abs().sum(dim=1) > 0).nonzero(as_tuple=True)[0].tolist()
        ok = set(nonzero) <= set(self.target_token_ids)
        return {"ok": ok, "nonzero_rows": nonzero, "target": self.target_token_ids,
                "grad_norms": {name: g[t].norm().item()
                              for name, t in zip(self.token_names, self.target_token_ids)}}

    def check_frozen_intact(self):
        """target 외 행이 원본과 동일한지 검사."""
        diff = (self.embed.weight - self._frozen).abs().sum(dim=1)
        changed = (diff > 0).nonzero(as_tuple=True)[0].tolist()
        return {"ok": set(changed) <= set(self.target_token_ids), "changed_rows": changed}

    # ── 코사인 유사도 분석 ─────────────────────────────────────────
    def snapshot(self):
        """현재 target 벡터들을 저장 (나중에 compare_snapshots에 사용)."""
        return {name: self.embed.weight[t].detach().clone()
                for name, t in zip(self.token_names, self.target_token_ids)}

    @staticmethod
    def cosine(vec_a, vec_b):
        # bf16/fp16이면 자기 자신과의 코사인도 1.0이 안 나올 수 있어 float32로 업캐스트
        return F.cosine_similarity(
            vec_a.float().unsqueeze(0), vec_b.float().unsqueeze(0)
        ).item()

    def compare_snapshots(self, snap_before, snap_after=None):
        if snap_after is None:
            snap_after = self.get_vectors()
        result = {}
        for name in snap_before:
            a, b = snap_before[name], snap_after[name]
            result[name] = {
                "cosine": self.cosine(a, b),
                "l2_dist": (a - b).float().norm().item(),
                "norm_before": a.float().norm().item(),
                "norm_after": b.float().norm().item(),
            }
        return result

    def compare_to_initial(self):
        """학습 시작 직후(초기화 완료 시점) 대비 현재 상태."""
        return self.compare_snapshots(self._initial_vectors)

    def report_drift(self, snap_before=None, label_before="초기화 직후"):
        """고정폭 정렬 표로 출력. 각 벡터를 자기 자신의 이전 시점과 비교."""
        cmp = self.compare_to_initial() if snap_before is None \
              else self.compare_snapshots(snap_before)
        names = list(cmp.keys())
        col_w = max(max(len(n) for n in names), 8) + 2

        def cell(x):
            return f"{x:^{col_w}}" if isinstance(x, str) else f"{x:^{col_w}.4f}"

        def row(row_label, key):
            return "|" + cell(row_label) + "|" + "|".join(cell(cmp[n][key]) for n in names) + "|"

        print(f"[벡터 이동 — {label_before} 대비 현재]")
        print("|" + cell("") + "|" + "|".join(cell(n) for n in names) + "|")
        print("|" + "-"*col_w + "|" + "|".join("-"*col_w for _ in names) + "|")
        print(row("cosine", "cosine"))
        print(row("L2거리", "l2_dist"))
        print(row("norm(전)", "norm_before"))
        print(row("norm(후)", "norm_after"))
        return cmp

    def pairwise_cosine_matrix(self, vectors=None):
        """token_name_i 대 token_name_j 서로 간의 코사인 유사도 (대각선은 1.0)."""
        if vectors is None:
            vectors = self.get_vectors()
        names = list(vectors.keys())
        return {a: {b: self.cosine(vectors[a], vectors[b]) for b in names} for a in names}

    def report_pairwise_similarity(self, vectors=None,
                                    title="벡터 간 코사인 유사도"):
        """가로/세로 헤더가 동일한 정사각 표로 출력."""
        mat = self.pairwise_cosine_matrix(vectors)
        names = list(mat.keys())
        col_w = max(max(len(n) for n in names), 6) + 2

        def cell(x):
            return f"{x:^{col_w}}" if isinstance(x, str) else f"{x:^{col_w}.4f}"

        print(f"[{title}]")
        print("|" + cell("") + "|" + "|".join(cell(n) for n in names) + "|")
        print("|" + "-"*col_w + "|" + "|".join("-"*col_w for _ in names) + "|")
        for a in names:
            print("|" + cell(a) + "|" + "|".join(cell(mat[a][b]) for b in names) + "|")

        if len(names) > 1:
            off_diag = [(mat[a][b], a, b) for a in names for b in names if a != b]
            max_sim, a, b = max(off_diag)
            print(f"\n  가장 유사한 쌍: {a} vs {b} = {max_sim:.4f}"
                  + ("  ⚠️ 매우 유사 — 중복 학습 가능성" if max_sim > 0.9 else ""))
        return mat

    def top_similar_vocab(self, tokenizer, k, exclude_ids=None):
        """각 target 토큰이 기존 vocab의 어떤 단어와 가장 가까운지 (행렬곱 1회로 전체 계산)."""
        exclude_ids = set(exclude_ids or []) | set(self.target_token_ids)
        with torch.no_grad():
            vocab_norm = F.normalize(self.embed.weight[:self.orig_vocab].float(), dim=-1)
            results = {}
            for name, tid in zip(self.token_names, self.target_token_ids):
                v_norm = F.normalize(self.embed.weight[tid].float().unsqueeze(0), dim=-1)
                sims = (vocab_norm @ v_norm.T).squeeze(-1)
                for eid in exclude_ids:
                    if eid < self.orig_vocab:
                        sims[eid] = -float("inf")
                topk = torch.topk(sims, k)
                results[name] = [
                    (tokenizer.decode([idx.item()]), val.item())
                    for val, idx in zip(topk.values, topk.indices)
                ]
        return results

    def report_top_similar_vocab(self, tokenizer, k, exclude_ids=None):
        results = self.top_similar_vocab(tokenizer, k, exclude_ids)
        for name, items in results.items():
            print(f"\n[{name} — 기존 vocab과 가장 유사한 토큰 Top{k}]")
            print("| 순위 |    토큰    | 유사도  |")
            print("|------|------------|---------|")
            for rank, (tok_str, sim) in enumerate(items, 1):
                print(f"| {rank:>4} | {tok_str!r:>10} | {sim:.4f}  |")
        return results