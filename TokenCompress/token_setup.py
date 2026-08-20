"""
TokenSetup — 모델에 새 special token을 추가하고, resize하고, backbone freeze
  1. 현재 vocab/special token 상태 확인
  2. tokenizer.add_special_tokens(...)
  3. model.resize_token_embeddings(..., mean_resizing=...)
  4. 각 토큰의 id/벡터값 출력
  5. 전체 파라미터 freeze + 임베딩 레이어만 requires_grad=True
"""
import torch


class TokenSetup:
    def __init__(self, model, tokenizer, new_tokens: list, mean_resizing: bool = False,
                 freeze_backbone: bool = True, verbose: bool = True):
        """
        new_tokens: 추가할 special token 문자열 리스트. 이미 등록된 토큰이 섞여 있어도
                    자동으로 중복 제거됨(순서는 유지).
        mean_resizing: resize_token_embeddings에 그대로 전달.
                       True(HF 기본값) -> Hewitt 방식(평균+공분산 샘플링, 드물게 공분산이
                       비정상이면 전부 동일한 값으로 복제되는 문제 있음, 이전에 확인함).
                       False -> 모델 표준 초기화(정규분포)로 각각 독립적인 랜덤값.
        freeze_backbone: True면 전체 파라미터를 얼리고 임베딩 레이어만 학습 가능하게 켬.
                         CustomTokenTrainer의 EmbeddingManager가 이미 이 작업을 하므로,
                         Trainer를 바로 이어서 쓸 거라면 False로 두고 EmbeddingManager에
                         맡겨도 됨(중복 실행 자체는 무해하지만).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.new_tokens = list(new_tokens)
        self.mean_resizing = mean_resizing
        self.verbose = verbose

        self._log_before_state()
        self.num_added_tokens, self.total_special_tokens = self._add_tokens()
        self._resize()
        self.token_ids = self._collect_token_ids()
        self._log_vectors()

        if freeze_backbone:
            self.freeze_backbone()

    # ── 1. 추가 전 상태 기록 ──
    def _log_before_state(self):
        self.vocab_size_before = self.tokenizer.vocab_size
        self.len_tokenizer_before = len(self.tokenizer)
        self.embed_shape_before = tuple(self.model.get_input_embeddings().weight.shape)
        self.special_tokens_before = list(self.tokenizer.all_special_tokens)

        if self.verbose:
            print("=" * 70)
            print("[추가 전 상태]")
            print("=" * 70)
            print(f"  tokenizer.vocab_size : {self.vocab_size_before}")
            print(f"  len(tokenizer)       : {self.len_tokenizer_before}")
            print(f"  embedding weight shape: {self.embed_shape_before}")
            print(f"  기존 special tokens ({len(self.special_tokens_before)}개): "
                  f"{self.special_tokens_before}")

    # ── 2. 토큰 추가 ──
    def _add_tokens(self):
        current = self.tokenizer.all_special_tokens
        # dict.fromkeys로 순서를 유지하면서 중복 제거
        total = list(dict.fromkeys(current + self.new_tokens))
        num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": total})

        if self.verbose:
            print()
            print("=" * 70)
            print("[토큰 추가]")
            print("=" * 70)
            print(f"  요청한 new_tokens     : {self.new_tokens}")
            print(f"  추가된 토큰 개수      : {num_added}")
            print(f"  업데이트된 vocab 크기 : {len(self.tokenizer)}")
            print(f"  최종 special tokens 순서: {total}")
        return num_added, total

    # ── 3. resize ──
    def _resize(self):
        self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=self.mean_resizing)

        if self.verbose:
            print()
            print("=" * 70)
            print("[Resize]")
            print("=" * 70)
            print(f"  mean_resizing={self.mean_resizing}")
            print(f"  tokenizer.vocab_size(변화없음) : {self.tokenizer.vocab_size}")
            print(f"  len(tokenizer)                 : {len(self.tokenizer)}")
            print(f"  embedding weight shape(갱신후) : "
                  f"{tuple(self.model.get_input_embeddings().weight.shape)}")

    # ── 4. 요청한 new_tokens 각각의 id 수집 ──
    def _collect_token_ids(self):
        ids = {}
        for token in self.new_tokens:
            ids[token] = self.tokenizer.convert_tokens_to_ids(token)
        return ids

    def _log_vectors(self):
        if not self.verbose:
            return
        print()
        print("=" * 70)
        print("[추가된 토큰의 id / 초기 벡터]")
        print("=" * 70)
        embed = self.model.get_input_embeddings()
        for token in self.total_special_tokens:
            tid = self.tokenizer.convert_tokens_to_ids(token)
            vector = embed.weight[tid].detach().cpu()
            marker = " ← added Token" if token in self.new_tokens else ""
            print(f"  {token!r:>15} | id={tid:>7} | norm={vector.float().norm().item():.4f}{marker}")

    # ── 5. backbone freeze ──
    def freeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = False
        embed = self.model.get_input_embeddings()
        embed.weight.requires_grad = True

        if self.verbose:
            trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
            print()
            print("=" * 70)
            print("[Backbone Freeze]")
            print("=" * 70)
            print(f"  requires_grad=True인 파라미터: {trainable}")
        return self

    # ── 편의 조회 ──
    def get_id(self, token: str) -> int:
        """new_tokens에 없던 토큰이라도 조회 가능(등록만 돼 있으면)."""
        return self.tokenizer.convert_tokens_to_ids(token)

    def get_ids(self, tokens: list) -> list:
        return [self.get_id(t) for t in tokens]

    def summary(self):
        return {
            "new_tokens": self.new_tokens,
            "num_added_tokens": self.num_added_tokens,
            "token_ids": self.token_ids,
            "vocab_size_before": self.vocab_size_before,
            "vocab_size_after": len(self.tokenizer),
            "embed_shape_before": self.embed_shape_before,
            "embed_shape_after": tuple(self.model.get_input_embeddings().weight.shape),
            "mean_resizing": self.mean_resizing,
        }