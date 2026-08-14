"""
CheckpointManager — 저장/정리/복원 담당.
save_steps, save_epochs, save_total_limit을 독립적으로 처리.
"""
import os
import json
import shutil
import torch


class CheckpointManager:
    def __init__(self, output_dir, save_total_limit=None):
        self.output_dir = output_dir
        self.save_total_limit = save_total_limit
        self._saved = []      # best/final 제외, 저장 순서 기록
        os.makedirs(output_dir, exist_ok=True)

    def path_for(self, tag):
        return os.path.join(self.output_dir, f"checkpoint-{tag}")

    def save(self, model, tokenizer, tag, extra_state=None):
        path = self.path_for(tag)
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

        if extra_state is not None:
            torch.save(extra_state, os.path.join(path, "training_state.pt"))

        print(f"  → 저장 완료: {path}")

        if str(tag) not in ("best", "final"):
            if path in self._saved:
                self._saved.remove(path)
            self._saved.append(path)
        self.cleanup()
        return path

    def save_vectors(self, vectors, tag):
        """학습된 S 벡터만 가볍게 저장 (모델 전체 저장 없이)."""
        path = self.path_for(tag)
        os.makedirs(path, exist_ok=True)
        torch.save({k: v.cpu() for k, v in vectors.items()},
                   os.path.join(path, "learned_vectors.pt"))
        return path

    def save_history(self, history, eval_history, tag):
        path = self.path_for(tag)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "history.json"), "w") as f:
            json.dump({"history": history, "eval_history": eval_history}, f)

    def cleanup(self):
        """save_total_limit개만 남기고 오래된 것부터 삭제. best/final은 제외."""
        limit = self.save_total_limit
        if not limit or limit <= 0:
            return
        while len(self._saved) > limit:
            old = self._saved.pop(0)
            shutil.rmtree(old, ignore_errors=True)
            print(f"  → 오래된 체크포인트 삭제: {old}")

    def load_vectors(self, tag):
        p = os.path.join(self.path_for(tag), "learned_vectors.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return torch.load(p)

    def list_checkpoints(self):
        if not os.path.isdir(self.output_dir):
            return []
        return sorted(d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-"))