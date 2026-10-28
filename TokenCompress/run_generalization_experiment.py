"""
run_generalization_experiment — 미리 만들어둔 (프롬프트, 데이터셋) 조합들을 순회하며
"학습 전 점검 -> 학습 -> 재구성/태스크 평가"를 순차 실행.

teacher 데이터셋은 셀에서 미리 만들어서 넘겨준다는 전제이므로,
이 함수 내부에서는 데이터셋을 새로 생성하지 않음.
"""
import gc
import os
import copy
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .trainer import CustomTokenTrainer
from .trainer_debug import TrainerDebugger
from .evaluator import Evaluator
from .save_experiment_report import save_experiment_report


class TeacherDataset(Dataset):
    """
    미리 만들어둔 teacher 데이터셋(hf_dataset 또는 list[dict])을 CustomTokenTrainer가
    기대하는 {"query":..., "answer_ids":...} 형태로 감싸는 래퍼.
    response_ids/answer_ids 둘 중 어떤 필드명으로 저장돼 있든 자동으로 처리.
    """
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        ids_key = "response_ids" if "response_ids" in item else "answer_ids"
        return {
            "query": item["query"],
            "answer_ids": torch.tensor(item[ids_key], dtype=torch.long),
        }


def run_generalization_experiment(
    configs: dict,
    # configs 구조: {
    #   "gsm8k": {
    #       "prompt": fix_prompt,
    #       "train": TeacherDataset 인스턴스,
    #       "validation": TeacherDataset 인스턴스,
    #       "test": [{"query":..., "gold_label" 또는 "gold_response":...}, ...] (원본 리스트),
    #       "gold_key": "gold_label" 또는 "gold_response" 또는 None,
    #   },
    #   ...
    # }
    model, tokenizer, args,
    target_token_ids, ae_token_id,
    user_template=None,
    init_mode="random",
    recon_reduction="mean",
    kd_tau_squared=True,
    prompt_mode="chat",
    enable_thinking=None,
    max_new_tokens_recon=2048,
    live_plot_every=1, live_plot_unit="epoch",
    live_plot_smooth=1, live_plot_both_scales=True,
    max_new_tokens_task=256,
    n_gold_eval=100,
    debug_query=None,
    delete=True,
    # ── 학습 방식 선택 ──
    train_mode="epoch",           # "epoch" -> trainer.train() 사용, "step" -> trainer.train_by_steps() 사용
    total_steps=None,              # train_mode="step"일 때 필수 (총 스텝 수)
    eval_every_steps=500,          # train_mode="step"일 때 사용 (평가 주기)
    early_stopping_patience=None,  # epoch/step 둘 다 사용 가능. None이면 조기 종료 안 함
):
    """
    각 실험마다 CustomTokenTrainer를 새로 생성해 target 토큰 임베딩을 초기화하므로,
    같은 model 객체를 재사용해도 이전 실험 결과가 섞이지 않음.

    train_mode="epoch": trainer.train()을 사용, args.num_train_epochs만큼 학습.
                         live_plot_unit은 인자로 준 값 그대로 사용됨.
    train_mode="step" : trainer.train_by_steps()를 사용, total_steps만큼 학습.
                         eval_every_steps마다 평가 + best 갱신 체크.
                         live_plot_unit은 내부적으로 항상 "step"으로 강제됨
                         (epoch 개념이 없으므로).
    early_stopping_patience: 두 모드 모두에서 사용 가능. eval_dataset(=cfg["validation"])이
                              있어야 실제로 작동함(validation을 안 주는 config는 자동으로 무시됨
                              -> 끝까지 학습).
    """
    if train_mode not in ("epoch", "step"):
        raise ValueError(f'train_mode는 "epoch" 또는 "step"만 가능 (받은 값: {train_mode!r})')
    if train_mode == "step" and total_steps is None:
        raise ValueError('train_mode="step"이면 total_steps를 반드시 지정해야 함')

    results = {}
    outer_pbar = tqdm(configs.items(), desc="일반화 실험", total=len(configs), dynamic_ncols=True)

    for name, cfg in outer_pbar:
        outer_pbar.set_postfix(현재=name)
        outer_pbar.write("\n" + "#" * 90)
        outer_pbar.write(f"# 실험: {name}")
        outer_pbar.write("#" * 90)

        fix_prompt = cfg["prompt"]
        n_tokens = len(tokenizer.encode(fix_prompt, add_special_tokens=False))
        outer_pbar.write(f"  fix_prompt 토큰 길이: {n_tokens}")
        outer_pbar.write(f"  train={len(cfg['train'])}, validation={len(cfg['validation'])}, "
                         f"test={len(cfg['test'])}")

        # ── 실험별 독립 output_dir (전체 실험이 args.output_dir을 공유하면
        #    checkpoint-best 등이 서로 덮어써버리므로, 실험 이름별 하위 폴더로 분리) ──
        exp_args = copy.deepcopy(args)
        exp_args.output_dir = os.path.join(args.output_dir, name)
        os.makedirs(exp_args.output_dir, exist_ok=True)

        # ── 트레이너 생성 (매 실험마다 target 토큰 임베딩 새로 초기화) ──
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

        # ── 학습 전 점검 ──
        sample_query = debug_query or cfg["train"][0]["query"]
        with TrainerDebugger(trainer, keys={"init"}) as dbg:
            dbg.inspect_setup()
        with TrainerDebugger(trainer, keys={"prompt"}) as dbg:
            dbg.inspect_prompts(str(sample_query)[:], show_ids=False, run_generate=False)

        # ── 학습 (train_mode에 따라 epoch 단위 / step 단위 선택) ──
        outer_pbar.write(f"  [{name}] 학습 시작 (mode={train_mode})...")
        if train_mode == "epoch":
            train_result = trainer.train(
                cfg["train"], cfg["validation"],
                live_plot_every=live_plot_every, live_plot_unit=live_plot_unit,
                live_plot_smooth=live_plot_smooth, live_plot_both_scales=live_plot_both_scales,
                early_stopping_patience=early_stopping_patience,
            )
        else:  # "step"
            train_result = trainer.train_by_steps(
                cfg["train"], total_steps=total_steps, eval_dataset=cfg["validation"],
                eval_every_steps=eval_every_steps,
                live_plot_every=live_plot_every, live_plot_unit="step",
                live_plot_smooth=live_plot_smooth, live_plot_both_scales=live_plot_both_scales,
                early_stopping_patience=early_stopping_patience,
            )
        vectors = train_result["selected"]   # 이후 평가는 select_best_model 기준으로 선택된 시점

        # ── final/best 벡터를 항상 둘 다 저장 ({vector_name}_final.pt / {vector_name}_best.pt).
        #    select_best_model 설정과 무관하게 저장 — 전체 모델(수 GB) 대신 벡터(몇 KB)만 남김,
        #    backbone은 항상 원본과 동일하므로 이것만으로 완전히 재현 가능 ──
        vector_name = "compressed_vectors"

        final_path = os.path.join(exp_args.output_dir, f"{vector_name}_final.pt")
        torch.save({k: v.cpu() for k, v in train_result["final"].items()}, final_path)
        outer_pbar.write(f"  압축 벡터(final) 저장: {final_path}")

        best_path = os.path.join(exp_args.output_dir, f"{vector_name}_best.pt")
        if trainer.best_vectors is not None:
            torch.save({k: v.cpu() for k, v in trainer.best_vectors.items()}, best_path)
            outer_pbar.write(f"  압축 벡터(best) 저장: {best_path}")
        else:
            # eval_dataset을 안 줬거나 한 번도 개선이 없었던 경우 -> best 스냅샷 자체가 없음.
            # 이 경우엔 final과 동일한 값으로 대체 저장(파일은 항상 둘 다 존재하게 유지).
            torch.save({k: v.cpu() for k, v in train_result["final"].items()}, best_path)
            outer_pbar.write(f"  압축 벡터(best) 저장: {best_path}  (best 스냅샷 없어 final로 대체)")

        vectors_path = final_path if train_result["selected_tag"] == "final" else best_path

        # ── 재구성 평가 ──
        evaluator = Evaluator(model, tokenizer, trainer.builder)
        recon_result = evaluator.report_reconstruction(max_new_tokens=max_new_tokens_recon)

        # ── 태스크 평가 (태스크마다 표준에 맞는 Evaluator 메서드로 분기) ──
        gold_key = cfg.get("gold_key")

        if name == "gsm8k":
            task_result_raw = evaluator.evaluate_gsm8k(
                cfg["test"], n=n_gold_eval, query_key="query", gold_key=gold_key,
                max_new_tokens=max_new_tokens_task,
            )
            wrong_pred_gold = lambda r: not r["correct"]
            wrong_pred_teacher = lambda r: not r["matches_teacher"]
        elif name == "classification":
            task_result_raw = evaluator.evaluate_classification(
                cfg["test"], n=n_gold_eval, query_key="query", gold_key=gold_key,
                max_new_tokens=max_new_tokens_task,
            )
            wrong_pred_gold = lambda r: r["student"] != r["gold"]
            wrong_pred_teacher = lambda r: r["student"] != r.get("teacher")
        elif name == "qa":
            task_result_raw = evaluator.evaluate_qa_squad(
                cfg["test"], n=n_gold_eval, query_key="query", gold_key=gold_key,
                max_new_tokens=max_new_tokens_task,
            )
            wrong_pred_gold = lambda r: r["em"] < 1.0
            wrong_pred_teacher = lambda r: r.get("teacher_em", 0) < 1.0
        elif name == "summarize":
            task_result_raw = evaluator.evaluate_summarize(
                cfg["test"], n=n_gold_eval, query_key="query", gold_key=gold_key,
                max_new_tokens=max_new_tokens_task,
            )
            wrong_pred_teacher = lambda r: r["rougeL"] < 0.3          # student vs teacher 유사도 낮음
            wrong_pred_gold = (lambda r: r.get("rougeL_vs_gold", 1.0) < 0.3) if gold_key else None
        else:
            raise ValueError(f"알 수 없는 태스크 이름: {name}")

        # ── wrong_cases를 gold 기준/teacher 기준 각각 별도로 계산해서 저장 (rows는 항상 전체 보존) ──
        task_result_raw["wrong_cases_vs_gold"] = (
            [r for r in task_result_raw["rows"] if wrong_pred_gold(r)] if wrong_pred_gold else []
        )
        task_result_raw["wrong_cases_vs_teacher"] = (
            [r for r in task_result_raw["rows"] if wrong_pred_teacher(r)] if wrong_pred_teacher else []
        )

        # ── loss 그래프 저장 ──
        loss_prefix = f"{exp_args.output_dir}/{name}_loss"
        trainer.metrics.plot(save_path=loss_prefix + ".png", smooth=1)
        trainer.metrics.plot(save_path=loss_prefix + ".png", smooth=1, log_scale=True)

        # ── 종합 리포트 저장 — evaluator가 반환한 {"metrics":..., "rows":...}에
        #    wrong_cases_vs_gold/wrong_cases_vs_teacher까지 이미 채워 넣은 상태로 그대로 넘김 ──
        save_experiment_report(
            output_dir=args.output_dir,   # exp_args.output_dir(이미 .../name)이 아니라 원래 루트를 넘김
                                            # -> save_experiment_report 내부에서 .../name 하나만 생성됨
                                            #    (vectors_path와 정확히 같은 폴더로 합쳐짐)
            name=name,
            trainer=trainer,
            recon_result=recon_result,
            task_eval_result=task_result_raw,
            loss_plot_prefix=loss_prefix,
            live_log_source_dir=os.path.join(exp_args.output_dir, "live_log"),
        )

        results[name] = {
            "prompt_tokens": n_tokens,
            "final_metrics": trainer.metrics.summary(),
            "recon": recon_result,
            "task_metrics": task_result_raw["metrics"],
            "vectors": {k: v.cpu() for k, v in vectors.items()},   # 메모리에도 바로 접근 가능
            "vectors_path": vectors_path,                           # 디스크 저장 위치
            "output_dir": exp_args.output_dir,
        }

        if delete:
            del trainer, evaluator
            gc.collect()
            torch.cuda.empty_cache()
        else:
            _print_comparison(results)
            return results[name], trainer

    _print_comparison(results)
    return results


def _print_comparison(results):
    print("\n" + "#" * 90)
    print("# 일반화 실험 비교")
    print("#" * 90)
    names = list(results.keys())
    col_w = max(max(len(n) for n in names), 12) + 2

    def cell(x):
        if isinstance(x, float):
            return f"{x:^{col_w}.4f}"
        return f"{x:^{col_w}}"

    rows = [
        ("prompt 토큰수", lambda r: r["prompt_tokens"]),
        ("recon exact_match", lambda r: r["recon"]["exact_match"]),
        ("recon match_ratio", lambda r: f"{100*r['recon']['match_ratio']:.1f}%"),
        ("recon ROUGE-L", lambda r: f"{100*r['recon']['rouge_l']:.1f}%"),
    ]
    print("|" + cell("지표") + "|" + "|".join(cell(n) for n in names) + "|")
    print("|" + "-"*col_w + "|" + "|".join("-"*col_w for _ in names) + "|")
    for row_name, fn in rows:
        print("|" + cell(row_name) + "|" + "|".join(cell(fn(results[n])) for n in names) + "|")

    # 태스크마다 metrics 키가 서로 다르므로(accuracy vs exact_match vs rougeL 등),
    # 공통 표 대신 태스크별로 자기 지표를 그대로 나열
    print()
    for name in names:
        print(f"[{name}] task_metrics:")
        for k, v in results[name]["task_metrics"].items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")