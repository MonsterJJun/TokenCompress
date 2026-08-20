"""
save_experiment_report — 학습 하나가 끝날 때마다 설정/loss/평가 결과를 전부
한 폴더에 묶어서 저장. loss 그래프(png)와 나란히 두면, 나중에 실험별로
결과를 다시 열어볼 때 흩어지지 않고 한눈에 확인 가능.

저장물:
  {output_dir}/{name}/config.json          — 초기화/하이퍼파라미터 설정
  {output_dir}/{name}/recon_eval.json      — 재구성 평가 (Evaluator.reconstruct 결과)
  {output_dir}/{name}/task_eval.json       — 거절태스크/gold 비교 등 커스텀 평가 결과
  {output_dir}/{name}/summary.txt          — 사람이 읽기 좋은 종합 리포트 (콘솔과 동일 형식)
  {output_dir}/{name}/loss.png, loss_log.png, loss_eval.png, loss_eval_log.png
                                             — metrics.plot()이 저장한 그래프 (자동 이동)
"""
import os
import json
import shutil


def _to_jsonable(obj):
    """텐서/기타 non-JSON 타입을 안전하게 문자열/숫자로 변환."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "item"):   # torch.Tensor(스칼라)
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if hasattr(obj, "tolist"):  # torch.Tensor(비스칼라)
        return obj.tolist()
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def build_config_dict(trainer):
    """trainer/builder/args로부터 실험 설정 딕셔너리 구성 (inspect_setup 출력과 대응)."""
    t = trainer
    b = t.builder
    n_special = 1 + t.n_target
    orig_vocab = t.embed.weight.shape[0] - n_special
    ref_std = t.embed.weight[:orig_vocab].std().item()

    token_info = []
    for i, tid in enumerate(t.target_token_ids):
        v = t.embed.weight[tid]
        token_info.append({
            "name": t.target_token_strs[i], "id": tid,
            "norm": v.float().norm().item(), "std": v.float().std().item(),
        })

    g = t.optimizer.param_groups[0]

    # args 객체가 가진 모든 속성을 기본값까지 포함해 자동 수집 (dataclass든 일반 객체든)
    all_args = {}
    for k, v in vars(t.args).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            all_args[k] = v
        else:
            all_args[k] = str(v)

    return {
        "init_mode": t.init_mode,
        "vocab_size": t.embed.weight.shape[0],
        "hidden_dim": t.embed.weight.shape[1],
        "target_tokens": token_info,
        "ref_vocab_std": ref_std,
        "ae_token_id": t.ae_token_id,
        "end_id": t.end_id,
        "optimizer": {
            "lr_initial_설정값": t.args.learning_rate,          # args에 준 원래 값
            "lr_current_학습종료시점": g["lr"],                  # 스케줄러가 감쇠시킨 현재 값
            "betas": g["betas"], "eps": g["eps"],
            "weight_decay": g["weight_decay"],
        },
        "recon_reduction": getattr(t, "recon_reduction", None),
        "kd_tau_squared": getattr(t, "kd_tau_squared", None),
        "sync_output_embedding": getattr(t, "sync_output_embedding", None),
        "clamp_norm_multiplier": getattr(t, "clamp_norm_multiplier", None),
        "prompt_builder": b.summary(),
        "args_전체(기본값포함)": all_args,
    }


def _format_summary_text(name, config, recon_result, task_eval_result, metrics_summary):
    SEP = "=" * 70
    lines = []
    lines.append(SEP)
    lines.append(f"[실험: {name}]")
    lines.append(SEP)

    # ── 설정 ──
    lines.append(f"\n{SEP}")
    lines.append("[설정 — 임베딩/옵티마이저]")
    lines.append(SEP)
    lines.append(f"  init_mode        : {config['init_mode']}")
    for tok in config["target_tokens"]:
        lines.append(f"  {tok['name']}(id={tok['id']}) norm={tok['norm']:.4f} std={tok['std']:.6f}")
    opt = config["optimizer"]
    lines.append(f"  lr(설정값, 초기)      : {opt['lr_initial_설정값']}")
    lines.append(f"  lr(학습 종료 시점)    : {opt['lr_current_학습종료시점']:.6e}  "
                 f"※ 스케줄러가 감쇠시킨 최종 값 — 설정값과 다른 게 정상")
    lines.append(f"  betas={opt['betas']}  eps={opt['eps']}  weight_decay={opt['weight_decay']}")
    lines.append(f"  recon_reduction  : {config['recon_reduction']}, "
                  f"kd_tau_squared={config['kd_tau_squared']}")
    lines.append(f"  sync_output_embedding : {config['sync_output_embedding']}")
    lines.append(f"  clamp_norm_multiplier : {config['clamp_norm_multiplier']}")

    # ── PromptBuilder 설정 ──
    lines.append(f"\n{SEP}")
    lines.append("[설정 — PromptBuilder]")
    lines.append(SEP)
    for k, v in config["prompt_builder"].items():
        lines.append(f"  {k:<20}: {v}")

    # ── args 전체(기본값 포함) ──
    lines.append(f"\n{SEP}")
    lines.append("[설정 — CustomTrainingArgs 전체 (기본값 포함)]")
    lines.append(SEP)
    for k, v in config["args_전체(기본값포함)"].items():
        lines.append(f"  {k:<30}: {v}")

    # ── 최종 loss ──
    if metrics_summary and metrics_summary != "기록 없음":
        lines.append(f"\n{SEP}")
        lines.append("[최종 loss]")
        lines.append(SEP)
        lines.append(f"  recon: {metrics_summary['recon']['first']:.4f} -> "
                      f"{metrics_summary['recon']['last']:.4f} (min={metrics_summary['recon']['min']:.4f})")
        lines.append(f"  kd   : {metrics_summary['kd']['first']:.4f} -> "
                      f"{metrics_summary['kd']['last']:.4f} (min={metrics_summary['kd']['min']:.4f})")
        lines.append(f"  best_eval_loss(total) = {metrics_summary['best_eval_loss']:.6f} "
                      f"@ epoch {metrics_summary['best_epoch']}")
        lines.append(f"  best_eval_kd          = {metrics_summary['best_eval_kd']:.6f} "
                      f"@ epoch {metrics_summary['best_kd_epoch']}")

    # ── Reconstruction 평가 ──
    if recon_result:
        lines.append(f"\n{SEP}")
        lines.append("[Reconstruction 평가]")
        lines.append(SEP)
        lines.append(f"  Exact Match    : {recon_result['exact_match']}")
        lines.append(f"  토큰 일치 길이 : {recon_result['match_len']} / {recon_result['target_len']} "
                      f"({100*recon_result['match_ratio']:.1f}%)")
        lines.append(f"  ROUGE-L F1     : {100*recon_result['rouge_l']:.2f}%")
        lines.append(f"  정상 종결(EOS) : {recon_result['ended_clean']}   "
                      f"hit_max: {recon_result['hit_max']}")

    # ── 태스크 평가 ──
    if task_eval_result:
        lines.append(f"\n{SEP}")
        lines.append("[태스크 평가]")
        lines.append(SEP)
        for key in ("exact_accuracy", "contains_accuracy", "accuracy", "format_compliance",
                    "student_vs_teacher", "student_exact_gold", "teacher_exact_gold"):
            if key in task_eval_result:
                val = task_eval_result[key]
                lines.append(f"  {key:<22}: {100*val:.1f}%" if isinstance(val, float) else
                             f"  {key:<22}: {val}")

        dist = task_eval_result.get("response_distribution")
        if dist:
            lines.append("\n  [응답 분포 (상위 10)]")
            for resp, cnt in sorted(dist.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"    {resp!r:45s} : {cnt}개")

        wrong = task_eval_result.get("wrong_cases")
        if wrong:
            lines.append(f"\n  [불일치 케이스 (총 {len(wrong)}개 중 최대 10개)]")
            for item in wrong[:10]:
                q = item.get("query", "")[:60]
                pred = item.get("student") or item.get("pred") or ""
                lines.append(f"    query: {q}...")
                lines.append(f"    pred : {pred!r}")

    lines.append(f"\n{SEP}")
    return "\n".join(lines)


def save_experiment_report(output_dir, name, trainer, recon_result=None,
                            task_eval_result=None, move_loss_plots=True,
                            loss_plot_prefix=None):
    """
    output_dir/name/ 폴더에 config.json, recon_eval.json, task_eval.json,
    summary.txt를 저장. loss_plot_prefix로 지정한 경로에 이미 저장된
    {prefix}.png/_log.png/_eval.png/_eval_log.png가 있으면 같은 폴더로 옮겨줌.
    """
    exp_dir = os.path.join(output_dir, name)
    os.makedirs(exp_dir, exist_ok=True)

    config = build_config_dict(trainer)
    metrics_summary = trainer.metrics.summary()

    # ── JSON 저장 ──
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(config), f, ensure_ascii=False, indent=2)

    if recon_result is not None:
        with open(os.path.join(exp_dir, "recon_eval.json"), "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(recon_result), f, ensure_ascii=False, indent=2)

    if task_eval_result is not None:
        with open(os.path.join(exp_dir, "task_eval.json"), "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(task_eval_result), f, ensure_ascii=False, indent=2)

    if metrics_summary and metrics_summary != "기록 없음":
        with open(os.path.join(exp_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(metrics_summary), f, ensure_ascii=False, indent=2)

    # ── 사람이 읽기 좋은 텍스트 리포트 ──
    summary_text = _format_summary_text(name, config, recon_result, task_eval_result, metrics_summary)
    with open(os.path.join(exp_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(summary_text)

    # ── loss 그래프 png들을 같은 폴더로 이동 ──
    if move_loss_plots and loss_plot_prefix:
        for suffix in ("", "_log", "_eval", "_eval_log"):
            src = f"{loss_plot_prefix}{suffix}.png"
            if os.path.exists(src):
                dst = os.path.join(exp_dir, f"loss{suffix}.png")
                shutil.move(src, dst)

    print(f"\n리포트 저장 완료: {exp_dir}/")
    return exp_dir