"""
run_lr_search — 하나의 프롬프트(태스크)를 고정해두고, learning rate만 바꿔가며
순차적으로 학습+평가하는 그리드/그리디 서치.
run_generalization_experiment.py와 거의 동일한 구조를 재사용하되,
순회 축이 "태스크"가 아니라 "lr"이라는 점만 다름.
"""
import gc
import os
import copy
import torch
from tqdm.auto import tqdm

from .trainer import CustomTokenTrainer
from .trainer_debug import TrainerDebugger
from .evaluator import Evaluator
from .save_experiment_report import save_experiment_report


def run_lr_search(
    fix_prompt, train_ds, validation_ds, test_data,
    lr_list,   # 예: [1e-3, 3e-3, 5e-3, 7e-3, 1e-2]
    model, tokenizer, base_args,
    target_token_ids, ae_token_id,
    task_name="lr_search",       # 평가 방식 선택용: gsm8k/classification/qa/summarize 중 하나,
                                   # 또는 그 외 모든 태스크(예: "I cannot find..."류 거절 태스크)는 "generic"
    gold_key="gold_label",
    user_template=None,
    init_mode="random",
    recon_reduction="mean",
    kd_tau_squared=True,
    prompt_mode="chat",
    enable_thinking=False,
    max_new_tokens_recon=2048,
    max_new_tokens_task=256,
    live_plot_every=1, live_plot_unit="epoch",
    live_plot_smooth=1, live_plot_both_scales=True,
    n_gold_eval=100,
    train_mode="epoch",          # "epoch" -> trainer.train() 사용, "step" -> trainer.train_by_steps() 사용
    total_steps=None,             # train_mode="step"일 때 필수 (총 스텝 수)
    eval_every_steps=500,         # train_mode="step"일 때 사용 (평가 주기)
    early_stopping_patience=None, # 둘 다에서 사용 가능
):
    """
    lr_list에 있는 각 lr마다:
      1) CustomTokenTrainer를 새로 만들어(임베딩 새로 초기화) 해당 lr로 학습
         (train_mode="epoch"면 base_args.num_train_epochs만큼, "step"이면 total_steps만큼)
      2) recon 평가 + task 평가
      3) {base_args.output_dir}/lr_{lr}/ 에 결과 저장 (run_generalization_experiment와 동일한 방식)
    끝나면 lr별 비교표 출력.
    """
    if train_mode == "step" and total_steps is None:
        raise ValueError('train_mode="step"이면 total_steps를 반드시 지정해야 함')
    if train_mode not in ("epoch", "step"):
        raise ValueError(f'train_mode는 "epoch" 또는 "step"만 가능 (받은 값: {train_mode!r})')
    results = {}
    outer_pbar = tqdm(lr_list, desc="LR Search", total=len(lr_list), dynamic_ncols=True)

    for lr in outer_pbar:
        lr_tag = f"lr_{lr:.0e}"   # 예: lr_1e-03
        outer_pbar.set_postfix(현재=lr_tag)
        outer_pbar.write("\n" + "#" * 90)
        outer_pbar.write(f"# LR 실험: {lr_tag} (lr={lr})")
        outer_pbar.write("#" * 90)

        # ── 이 lr 전용 args (얕은 output_dir — checkpoint 저장용) ──
        exp_args = copy.deepcopy(base_args)
        exp_args.learning_rate = lr
        exp_args.output_dir = os.path.join(base_args.output_dir, lr_tag)
        os.makedirs(exp_args.output_dir, exist_ok=True)

        # ── 트레이너 생성 (매 lr마다 target 토큰 임베딩 새로 초기화) ──
        trainer = CustomTokenTrainer(
            model, tokenizer,
            target_prompt=fix_prompt,
            target_token_ids=target_token_ids,
            ae_token_id=ae_token_id,
            args=exp_args,
            user_template=user_template,
            init_mode=init_mode,
            recon_reduction=recon_reduction,
            kd_tau_squared=kd_tau_squared,
            prompt_mode=prompt_mode,
            enable_thinking=enable_thinking,
        )

        # ── 학습 전 점검 — 초기화 상태 + 실제 input/output이 의도대로 조립됐는지 확인 ──
        sample_query = train_ds[0]["query"] if hasattr(train_ds, "__getitem__") else None
        with TrainerDebugger(trainer, keys={"init"}) as dbg:
            dbg.inspect_setup()
        with TrainerDebugger(trainer, keys={"prompt"}) as dbg:
            # recon용 input(=target_prompt 재구성 시퀀스)과
            # KD용 input(=query+teacher forcing answer_ids) 조립 결과를 실제로 디코딩해서 출력.
            # run_generate=True로 짧게 실제 생성도 한 번 돌려서, output 쪽도 붕 뜬 상태가 아닌지 확인.
            dbg.inspect_prompts(str(sample_query)[:] if sample_query else "", show_ids=True,
                                run_generate=False, max_new_tokens=32)

        # ── 학습 (train_mode에 따라 epoch 단위 / step 단위 선택) ──
        outer_pbar.write(f"  [{lr_tag}] 학습 시작 (lr={lr}, mode={train_mode})...")
        if train_mode == "epoch":
            train_result = trainer.train(
                train_ds, validation_ds,
                live_plot_every=live_plot_every, live_plot_unit=live_plot_unit,
                live_plot_smooth=live_plot_smooth, live_plot_both_scales=live_plot_both_scales,
                early_stopping_patience=early_stopping_patience,
            )
        else:  # "step"
            train_result = trainer.train_by_steps(
                train_ds, total_steps=total_steps, eval_dataset=validation_ds,
                eval_every_steps=eval_every_steps,
                live_plot_every=live_plot_every, live_plot_unit="step",
                live_plot_smooth=live_plot_smooth, live_plot_both_scales=live_plot_both_scales,
                early_stopping_patience=early_stopping_patience,
            )
        vectors = train_result["selected"]

        # ── 벡터 저장 (report_dir = {output}/{lr_tag}/{lr_tag}/, run_generalization과 동일한 이중 구조) ──
        report_dir = os.path.join(exp_args.output_dir, lr_tag)
        os.makedirs(report_dir, exist_ok=True)

        final_path = os.path.join(report_dir, "compressed_vectors_final.pt")
        torch.save({k: v.cpu() for k, v in train_result["final"].items()}, final_path)

        best_path = os.path.join(report_dir, "compressed_vectors_best.pt")
        if trainer.best_vectors is not None:
            torch.save({k: v.cpu() for k, v in trainer.best_vectors.items()}, best_path)
        else:
            torch.save({k: v.cpu() for k, v in train_result["final"].items()}, best_path)

        # ── 재구성 평가 ──
        evaluator = Evaluator(model, tokenizer, trainer.builder)
        recon_result = evaluator.report_reconstruction(max_new_tokens=max_new_tokens_recon)

        # ── 태스크 평가 (task_name에 맞는 메서드로) ──
        if task_name == "gsm8k":
            task_result_raw = evaluator.evaluate_gsm8k(
                test_data, n=n_gold_eval, query_key="query", gold_key=gold_key,
                teacher_key="response", max_new_tokens=max_new_tokens_task,
            )
            wrong_pred_gold = lambda r: not r["correct"]
            wrong_pred_teacher = lambda r: not r["matches_teacher"]
        elif task_name == "classification":
            task_result_raw = evaluator.evaluate_classification(
                test_data, n=n_gold_eval, query_key="query", gold_key=gold_key,
                teacher_key="response", max_new_tokens=max_new_tokens_task,
            )
            wrong_pred_gold = lambda r: r["student"] != r["gold"]
            wrong_pred_teacher = lambda r: r["student"] != r.get("teacher")
        elif task_name == "qa":
            task_result_raw = evaluator.evaluate_qa_squad(
                test_data, n=n_gold_eval, query_key="query", gold_key=gold_key,
                teacher_key="response", max_new_tokens=max_new_tokens_task,
            )
            wrong_pred_gold = lambda r: r["em"] < 1.0
            wrong_pred_teacher = lambda r: r.get("teacher_em", 0) < 1.0
        elif task_name == "summarize":
            task_result_raw = evaluator.evaluate_summarize(
                test_data, n=n_gold_eval, query_key="query", gold_key=gold_key,
                teacher_key="response", max_new_tokens=max_new_tokens_task,
            )
            wrong_pred_teacher = lambda r: r["rougeL"] < 0.3
            wrong_pred_gold = (lambda r: r.get("rougeL_vs_gold", 1.0) < 0.3) if gold_key else None
        elif task_name == "generic":
            # gsm8k/classification/qa/summarize 4개 특화 메서드에 안 맞는 모든 태스크용
            # (예: "I cannot find..."류 거절 태스크). 범용 evaluate_against_gold 사용 —
            # exact/contains 매치를 gold 기준으로 계산하고, compare_teacher=True로
            # student-teacher 비교도 같이 함.
            task_result_raw = evaluator.evaluate_against_gold(
                test_data, n=n_gold_eval, gold_key=gold_key,
                max_new_tokens=max_new_tokens_task, compare_teacher=True,
            )
            wrong_pred_gold = lambda r: not r.get("student_exact", r["student"] == r["gold"])
            wrong_pred_teacher = lambda r: r["student"] != r.get("teacher")
        else:
            raise ValueError(f"알 수 없는 task_name: {task_name} "
                             f"(gsm8k/classification/qa/summarize/generic 중 하나)")

        task_result_raw["wrong_cases_vs_gold"] = (
            [r for r in task_result_raw["rows"] if wrong_pred_gold(r)] if wrong_pred_gold else []
        )
        task_result_raw["wrong_cases_vs_teacher"] = (
            [r for r in task_result_raw["rows"] if wrong_pred_teacher(r)] if wrong_pred_teacher else []
        )

        # ── loss 그래프 저장 ──
        loss_prefix = f"{exp_args.output_dir}/{lr_tag}_loss"
        trainer.metrics.plot(save_path=loss_prefix + ".png", smooth=20)
        trainer.metrics.plot(save_path=loss_prefix + ".png", smooth=20, log_scale=True)

        # ── 종합 리포트 저장 (run_generalization_experiment와 동일한 이중 폴더 구조) ──
        save_experiment_report(
            output_dir=exp_args.output_dir,
            name=lr_tag,
            trainer=trainer,
            recon_result=recon_result,
            task_eval_result=task_result_raw,
            loss_plot_prefix=loss_prefix,
            live_log_source_dir=os.path.join(exp_args.output_dir, "live_log"),
        )

        results[lr_tag] = {
            "lr": lr,
            "final_metrics": trainer.metrics.summary(),
            "history": dict(trainer.metrics.history),           # step별 loss 원본 (plotting용)
            "eval_history": dict(trainer.metrics.eval_history),   # epoch별 eval loss 원본
            "recon": recon_result,
            "task_metrics": task_result_raw["metrics"],
            "vectors": {k: v.cpu() for k, v in vectors.items()},
            "vectors_path": final_path,
            "output_dir": report_dir,
        }

        del trainer, evaluator
        gc.collect()
        torch.cuda.empty_cache()

    _print_lr_comparison(results)
    return results


def _print_lr_comparison(results):
    print("\n" + "#" * 90)
    print("# LR 서치 비교")
    print("#" * 90)
    tags = list(results.keys())
    col_w = max(max(len(t) for t in tags), 12) + 2

    def cell(x):
        if isinstance(x, float):
            return f"{x:^{col_w}.4f}"
        return f"{x:^{col_w}}"

    rows = [
        ("lr", lambda r: r["lr"]),
        ("recon exact_match", lambda r: r["recon"]["exact_match"]),
        ("recon match_ratio", lambda r: f"{100*r['recon']['match_ratio']:.1f}%"),
        ("recon ROUGE-L", lambda r: f"{100*r['recon']['rouge_l']:.1f}%"),
        ("best_eval_loss(total)", lambda r: r["final_metrics"].get("best_eval_loss", "-")),
        ("best_eval_kd", lambda r: r["final_metrics"].get("best_eval_kd", "-")),
    ]
    print("|" + cell("지표") + "|" + "|".join(cell(t) for t in tags) + "|")
    print("|" + "-"*col_w + "|" + "|".join("-"*col_w for _ in tags) + "|")
    for row_name, fn in rows:
        print("|" + cell(row_name) + "|" + "|".join(cell(fn(results[t])) for t in tags) + "|")

    # recon 기준 최적 lr 하이라이트
    best_tag = max(tags, key=lambda t: results[t]["recon"]["match_ratio"])
    print(f"\n★ recon match_ratio 기준 최적: {best_tag} (lr={results[best_tag]['lr']})")


# =====================================================================
# lr별 학습 곡선을 한 그래프에 겹쳐서 비교
# =====================================================================
def plot_lr_comparison(results, metric="recon", smooth=1, log_scale=True,
                       figsize=(10, 6), save_path=None):
    """
    results: run_lr_search()가 반환한 딕셔너리.
    metric: "recon" | "kd" | "total" 중 하나 — 어떤 loss를 겹쳐볼지.
    smooth: 이동평균 윈도우 크기 (노이즈 완화용).
    log_scale: y축을 로그 스케일로.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=figsize)

    for lr_tag, r in results.items():
        steps = r["history"]["step"]
        values = r["history"][metric]
        if len(values) == 0:
            continue

        if smooth > 1 and len(values) >= smooth:
            values_arr = np.array(values)
            kernel = np.ones(smooth) / smooth
            smoothed = np.convolve(values_arr, kernel, mode="valid")
            plot_steps = steps[smooth - 1:]
        else:
            smoothed = values
            plot_steps = steps

        ax.plot(plot_steps, smoothed, label=f"{lr_tag} (lr={r['lr']:.0e})", alpha=0.8)

    ax.set_xlabel("step")
    ax.set_ylabel(f"{metric} loss" + (" (log scale)" if log_scale else ""))
    ax.set_title(f"LR별 {metric} loss 비교")
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장: {save_path}")
    plt.show()
    return fig


def plot_lr_eval_comparison(results, metric="eval_total", figsize=(10, 6), save_path=None):
    """
    lr별 "eval 시점"의 loss(매 epoch 평가값)를 겹쳐 비교 — train loss보다 노이즈가 적어
    수렴 양상을 더 깔끔하게 볼 수 있음.
    metric: "eval_total" | "eval_recon" | "eval_kd"
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for lr_tag, r in results.items():
        epochs = r["eval_history"]["epoch"]
        values = r["eval_history"][metric]
        if len(values) == 0:
            continue
        ax.plot(epochs, values, marker="o", label=f"{lr_tag} (lr={r['lr']:.0e})", alpha=0.8)

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"LR별 {metric} 비교 (평가 시점)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장: {save_path}")
    plt.show()
    return fig


def plot_lr_final_summary(results, figsize=(14, 6), save_path=None):
    """
    lr(x축)별 최종 성능을 recon과 KD 두 서브플롯으로 나란히 비교.
      - 왼쪽: recon match_ratio(%) — 초록=완전일치, 빨강=불일치
      - 오른쪽: best_eval_kd(낮을수록 좋음) — distillation이 얼마나 잘 됐는지
    recon만 보고 최적 lr을 고르면, "재구성은 완벽한데 KD는 전혀 안 된" lr을
    잘못 고를 위험이 있어 KD도 함께 봐야 함.
    """
    import matplotlib.pyplot as plt

    tags = list(results.keys())
    lrs = [r["lr"] for r in results.values()]
    match_ratios = [100 * r["recon"]["match_ratio"] for r in results.values()]
    exact_matches = [r["recon"]["exact_match"] for r in results.values()]
    best_kds = [r["final_metrics"].get("best_eval_kd") for r in results.values()]
    lr_labels = [f"{lr:.0e}" for lr in lrs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ── 왼쪽: recon match_ratio ──
    colors = ["#2ecc71" if em else "#e74c3c" for em in exact_matches]
    ax1.bar(lr_labels, match_ratios, color=colors)
    ax1.set_xlabel("learning rate")
    ax1.set_ylabel("recon match_ratio (%)")
    ax1.set_title("Recon 성능 (초록=완전일치, 빨강=불일치)")
    ax1.set_ylim(0, 105)
    ax1.grid(True, axis="y", alpha=0.3)

    # ── 오른쪽: best_eval_kd (낮을수록 좋음 -> 가장 낮은 막대를 강조) ──
    valid_kds = [k for k in best_kds if k is not None]
    kd_colors = ["#3498db"] * len(best_kds)
    if valid_kds:
        min_kd = min(valid_kds)
        kd_colors = ["#2ecc71" if k == min_kd else "#3498db" for k in best_kds]
    ax2.bar(lr_labels, [k if k is not None else 0 for k in best_kds], color=kd_colors)
    ax2.set_xlabel("learning rate")
    ax2.set_ylabel("best_eval_kd (낮을수록 좋음)")
    ax2.set_title("KD 성능 (초록=최저값=distillation 최적)")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("LR별 최종 recon vs KD 성능 비교")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장: {save_path}")
    plt.show()

    # ── recon과 KD 둘 다 좋은 lr을 놓치지 않도록, 텍스트로도 요약 ──
    print("\n[LR별 최종 요약]")
    for tag, lr, mr, em, kd in zip(tags, lrs, match_ratios, exact_matches, best_kds):
        kd_str = f"{kd:.4f}" if kd is not None else "N/A"
        print(f"  {tag:>10} (lr={lr:.0e}): recon={mr:.1f}%({'✓' if em else '✗'})  best_kd={kd_str}")

    return fig