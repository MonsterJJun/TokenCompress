"""
MetricTracker — 학습 지표 기록 / best 추적 / 그래프.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

_VALID_EXTS = {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".webp", ".tif", ".tiff", ".eps"}


class MetricTracker:
    def __init__(self):
        self.history = {"step": [], "total": [], "recon": [], "kd": [], "lr": []}
        self.eval_history = {"epoch": [], "eval_total": [], "eval_recon": [], "eval_kd": []}
        self.best_eval_loss = float("inf")
        self.best_epoch = None
        # KD 기준 best도 별도 추적 (recon 개선에 가려 KD 악화를 놓치지 않도록)
        self.best_eval_kd = float("inf")
        self.best_kd_epoch = None
        # early stopping용 — best_eval_loss 기준 연속 미개선 횟수
        self.epochs_without_improvement = 0

    def log_step(self, step, total, recon, kd, lr=None):
        self.history["step"].append(step)
        self.history["total"].append(total)
        self.history["recon"].append(recon)
        self.history["kd"].append(kd)
        self.history["lr"].append(lr if lr is not None else 0.0)

    def log_eval(self, epoch, eval_total, eval_recon, eval_kd):
        self.eval_history["epoch"].append(epoch)
        self.eval_history["eval_total"].append(eval_total)
        self.eval_history["eval_recon"].append(eval_recon)
        self.eval_history["eval_kd"].append(eval_kd)

        improved = eval_total < self.best_eval_loss
        if improved:
            self.best_eval_loss = eval_total
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if eval_kd < self.best_eval_kd:
            self.best_eval_kd = eval_kd
            self.best_kd_epoch = epoch

        return improved

    def kd_improved(self, epoch):
        """직전 log_eval에서 KD 기준 best가 갱신됐는지."""
        return self.best_kd_epoch == epoch

    def should_stop(self, patience):
        """연속 미개선 횟수가 patience 이상이면 True (early stopping 조건 충족)."""
        return self.epochs_without_improvement >= patience

    def summary(self):
        h = self.history
        if not h["step"]:
            return "기록 없음"
        return {
            "steps": len(h["step"]),
            "recon": {"first": h["recon"][0], "last": h["recon"][-1],
                      "min": min(h["recon"]), "delta": h["recon"][-1] - h["recon"][0]},
            "kd": {"first": h["kd"][0], "last": h["kd"][-1],
                   "min": min(h["kd"]), "delta": h["kd"][-1] - h["kd"][0]},
            "best_eval_loss": self.best_eval_loss,
            "best_epoch": self.best_epoch,
            "best_eval_kd": self.best_eval_kd,
            "best_kd_epoch": self.best_kd_epoch,
        }

    @staticmethod
    def _sanitize_log(values, name=""):
        arr = np.asarray(values, dtype=float)
        n_bad = int((arr <= 0).sum())
        if n_bad:
            print(f"[log_scale] {name}: 0 이하 {n_bad}개 -> 1e-8로 치환")
            arr = np.where(arr <= 0, 1e-8, arr)
        return arr

    @staticmethod
    def _save_path(save_path, suffix):
        base, ext = os.path.splitext(save_path)
        if ext.lower() not in _VALID_EXTS:
            base, ext = save_path, ".png"
        return f"{base}{suffix}{ext}"

    def plot(self, save_path=None, smooth=1, log_scale=False):
        def ma(x, w):
            return list(x) if w <= 1 else list(np.convolve(x, np.ones(w)/w, mode="valid"))

        steps = self.history["step"]
        recon, kd, total = (ma(self.history[k], smooth) for k in ("recon", "kd", "total"))
        steps_s = steps[smooth-1:] if smooth > 1 else steps

        if log_scale:
            recon = self._sanitize_log(recon, "recon")
            kd = self._sanitize_log(kd, "kd")
            total = self._sanitize_log(total, "total")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, data, title, color in zip(
            axes, [recon, kd, total], ["Recon", "KD", "Total"], ["tab:blue", "tab:orange", "tab:green"]
        ):
            ax.plot(steps_s, data, color=color)
            ax.set_title(f"{title} Loss" + (" (log)" if log_scale else ""))
            ax.set_xlabel("step"); ax.grid(alpha=0.3)
            if log_scale:
                ax.set_yscale("log")
        axes[0].set_ylabel("loss")
        plt.tight_layout()
        if save_path:
            p = self._save_path(save_path, "_log" if log_scale else "")
            plt.savefig(p, dpi=150); print(f"저장됨: {p}")
        plt.show()

        if self.eval_history["epoch"]:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            for k, lbl in [("eval_recon", "recon"), ("eval_kd", "kd"), ("eval_total", "total")]:
                vals = self._sanitize_log(self.eval_history[k], k) if log_scale else self.eval_history[k]
                ax2.plot(self.eval_history["epoch"], vals, marker="o", label=lbl)
            ax2.set_xlabel("epoch"); ax2.set_ylabel("loss")
            ax2.set_title("Eval Loss" + (" (log)" if log_scale else ""))
            if log_scale:
                ax2.set_yscale("log")
            ax2.legend(); ax2.grid(alpha=0.3)
            plt.tight_layout()
            if save_path:
                # 첫 그래프와 파일명이 겹치지 않도록 "_eval" 접미사 추가
                p2 = self._save_path(save_path, ("_eval_log" if log_scale else "_eval"))
                plt.savefig(p2, dpi=150); print(f"저장됨: {p2}")
            plt.show()

    def live_update(self, output_dir, tag=None, smooth=20, log_scale=False, both_scales=False):
        """
        학습 도중 N스텝/N에폭마다 호출해서 지금까지의 loss 그래프를 그림.
        기존 셀 출력은 지우지 않고(clear_output 없음) 그 자리에 새 그래프를 계속 이어서 출력.
        args.output_dir 아래에 파일로도 저장해서, 학습 도중 언제든 디스크에서 진행상황을 확인 가능.

        tag: 파일명에 붙일 구분자(보통 step 번호나 "epoch3" 같은 값). None이면 저장 안 하고 화면 출력만.
        both_scales: True면 log_scale 인자와 무관하게 선형/로그 그래프를 둘 다 그림
                     (파일도 live_loss_{tag}.png / live_loss_{tag}_log.png 두 개로 저장).
        """
        if not self.history["step"]:
            return
        save_path = None
        if tag is not None:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"live_loss_{tag}.png")

        if both_scales:
            self.plot(save_path=save_path, smooth=smooth, log_scale=False)
            self.plot(save_path=save_path, smooth=smooth, log_scale=True)
        else:
            self.plot(save_path=save_path, smooth=smooth, log_scale=log_scale)