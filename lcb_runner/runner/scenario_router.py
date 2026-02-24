from typing import Union

from lcb_runner.utils.scenarios import Scenario
from lcb_runner.lm_styles import LanguageModel
from lcb_runner.evaluation import (
    codegen_metrics,
    test_output_metrics,
    code_execution_metrics,
)

from lcb_runner.prompts import (
    format_prompt_generation,
    format_prompt_test_output,
    format_prompt_execution,
    format_prompt_execution_cot,
    format_prompt_self_repair,
)
from lcb_runner.utils.extraction_utils import (
    extract_code,
    extract_test_output_code,
    extract_execution_code,
)

from lcb_runner.benchmarks import (
    CodeGenerationProblem,
    TestOutputPredictionProblem,
    CodeExecutionProblem,
    load_code_generation_dataset,
    load_code_generation_dataset_not_fast,
    load_test_prediction_dataset,
    load_code_execution_dataset,
)

# BenchMarkType = list[CodeGenerationProblem | TestOutputPredictionProblem]
BenchMarkType = list[
    Union[CodeGenerationProblem, CodeExecutionProblem, TestOutputPredictionProblem]
]


def build_prompt_benchmark(
    args,
) -> tuple[
    list[CodeExecutionProblem]
    | list[CodeGenerationProblem]
    | list[TestOutputPredictionProblem],
    callable,
]:
    scenario: Scenario = args.scenario

    if scenario == Scenario.codegeneration:
        not_fast: bool = args.not_fast
        if not_fast:
            benchmark = load_code_generation_dataset_not_fast(args.release_version)
        else:
            benchmark = load_code_generation_dataset(
                args.release_version,
                start_date=args.start_date,
                end_date=args.end_date
            )
        benchmark = sorted(benchmark, key=lambda x: x.question_id)
        format_prompt = format_prompt_generation
    elif scenario == Scenario.testoutputprediction:
        benchmark = load_test_prediction_dataset(args.release_version)
        benchmark = sorted(benchmark, key=lambda x: (x.question_id, x.test_id))
        format_prompt = format_prompt_test_output
    elif scenario == Scenario.selfrepair:
        benchmark = load_code_generation_dataset(args.release_version)
        benchmark = sorted(benchmark, key=lambda x: x.question_id)
        format_prompt = format_prompt_self_repair
    elif scenario == Scenario.codeexecution:
        cot_code_execution: bool = args.cot_code_execution
        benchmark = load_code_execution_dataset(args.release_version)
        benchmark = sorted(benchmark, key=lambda x: int(x.id.split("_")[1]))
        if cot_code_execution:
            format_prompt = format_prompt_execution_cot
        else:
            format_prompt = format_prompt_execution
    else:
        raise ValueError(f"Scenario {scenario} not implemented")
    return benchmark, format_prompt


def combine_results(
    scenario: Scenario,
    results: list[list[str]],
    model: LanguageModel,
    cot_code_execution: bool = False,
):
    if scenario == Scenario.codegeneration:
        combined_results = [
            (
                outputs_list,
                [extract_code(output, model.model_style) for output in outputs_list],
            )
            for outputs_list in results
        ]
    elif scenario == Scenario.testoutputprediction:
        combined_results = [
            (
                outputs_list,
                [
                    extract_test_output_code(output, model.model_style)
                    for output in outputs_list
                ],
            )
            for outputs_list in results
        ]
    elif scenario == Scenario.selfrepair:
        combined_results = [
            (
                [
                    output[0] if type(output) is list else output
                    for output in outputs_list
                ],
                [
                    (
                        extract_code(output[0], model.model_style)
                        if type(output) is list
                        else extract_code(output, model.model_style)
                    )
                    for output in outputs_list
                ],
            )
            for outputs_list in results
        ]
    elif scenario == Scenario.codeexecution:
        combined_results = [
            (
                outputs_list,
                [
                    extract_execution_code(
                        output, model.model_style, cot=cot_code_execution
                    )
                    for output in outputs_list
                ],
            )
            for outputs_list in results
        ]
    else:
        raise ValueError(f"Scenario {scenario} not implemented")

    return combined_results


def sort_and_extract_save_results(scenario: Scenario, save_results: list[dict]):
    if scenario == Scenario.codegeneration:
        save_results = sorted(save_results, key=lambda x: x["question_id"])
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["code_list"])
            for save_result_instance in save_results
        ]

    elif scenario == Scenario.testoutputprediction:
        save_results = sorted(
            save_results, key=lambda x: (x["question_id"], x["test_id"])
        )
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["pred_list"])
            for save_result_instance in save_results
        ]
    elif scenario == Scenario.selfrepair:
        save_results = sorted(save_results, key=lambda x: x["question_id"])
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["code_list"])
            for save_result_instance in save_results
        ]
    elif scenario == Scenario.codeexecution:
        save_results = sorted(save_results, key=lambda x: int(x["id"].split("_")[1]))
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["pred_list"])
            for save_result_instance in save_results
        ]

    else:
        raise ValueError(f"Scenario {scenario} not implemented")

    return save_results, combined_results


def get_metrics(
    scenario: Scenario,
    args,
    benchmark: list[
        CodeGenerationProblem | CodeExecutionProblem | TestOutputPredictionProblem
    ],
    combined_results,
):
    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
    generations = [extracted for _, extracted in combined_results]

    if scenario == Scenario.codegeneration or scenario == Scenario.selfrepair:
        metrics = codegen_metrics(
            eval_samples,
            generations,
            num_process_evaluate=args.num_process_evaluate,
            timeout=args.timeout,
        )

    elif args.scenario == Scenario.testoutputprediction:
        metrics = test_output_metrics(
            eval_samples,
            generations,
            k_list=[1, 5],
        )

    elif args.scenario == Scenario.codeexecution:
        metrics = code_execution_metrics(
            eval_samples,
            generations,
        )

    else:
        raise ValueError(f"Scenario {scenario} not implemented")

    _print_metrics(scenario, metrics, benchmark)

    return metrics


def _print_metrics(scenario: Scenario, metrics, benchmark):
    import json as _json

    summary = metrics[0]
    # detail keys are integers when coming directly from codegen_metrics, but
    # strings when loaded back from a saved JSON file — normalise to strings.
    per_problem = {
        str(k): v
        for k, v in metrics[0].get("detail", {}).get("pass@1", {}).items()
    }
    metadatas = metrics[2] if len(metrics) > 2 else None

    # ── per-problem table ────────────────────────────────────────────────────
    col_id    = 14
    col_title = 32
    col_diff  = 8
    col_res   = 6
    sep = "─" * (col_id + col_title + col_diff + col_res + 3 * 3 + 2)

    header = (
        f"{'ID':<{col_id}}   "
        f"{'Title':<{col_title}}   "
        f"{'Diff':<{col_diff}}   "
        f"{'Pass':<{col_res}}"
    )
    print()
    print(sep)
    print(header)
    print(sep)

    by_diff: dict[str, list[float]] = {}

    for idx, instance in enumerate(benchmark):
        key      = str(idx)
        passed   = per_problem.get(key, None)
        mark     = "✓" if passed == 1.0 else ("✗" if passed == 0.0 else "?")

        qid   = getattr(instance, "question_id",    "?")[:col_id]
        title = getattr(instance, "question_title", "?")[:col_title]
        diff  = str(getattr(instance, "difficulty",  "?"))

        # strip "Difficulty." prefix if present
        if "." in diff:
            diff = diff.split(".", 1)[1]
        diff_label = diff.lower()
        diff = diff[:col_diff]

        # accumulate difficulty breakdown
        if passed is not None:
            by_diff.setdefault(diff_label, []).append(passed)

        # pull the first metadata entry for this problem (error msg or time)
        note = ""
        if metadatas and idx < len(metadatas):
            raw = metadatas[idx]
            if raw:
                try:
                    m = _json.loads(raw[0])
                    if "error_message" in m:
                        note = m["error_message"][:100]
                    elif "execution time" in m:
                        note = f"exec {m['execution time']:.4f}s"
                except Exception:
                    note = str(raw[0])[:80]

        row = (
            f"{qid:<{col_id}}   "
            f"{title:<{col_title}}   "
            f"{diff:<{col_diff}}   "
            f"{mark:<{col_res}}"
        )
        print(row)
        if note:
            print(f"  {'':>{col_id}}  ↳ {note}")

    print(sep)

    # ── summary ──────────────────────────────────────────────────────────────
    total  = len(per_problem)
    solved = sum(1 for v in per_problem.values() if v == 1.0)
    pass_at_1 = summary.get("pass@1", solved / total if total else 0.0)
    print(f"\n  pass@1 : {pass_at_1:.1%}  ({solved}/{total} solved)")

    # difficulty breakdown (easy → medium → hard, skip tiers with no data)
    _DIFF_ORDER = ["easy", "medium", "hard"]
    tiers = [d for d in _DIFF_ORDER if d in by_diff] + \
            [d for d in sorted(by_diff) if d not in _DIFF_ORDER]
    if tiers:
        print()
        col_d = max(len(d) for d in tiers)
        for d in tiers:
            scores = by_diff[d]
            n = len(scores)
            s = sum(1 for x in scores if x == 1.0)
            pct = s / n
            bar_filled = round(pct * 20)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            print(f"  {d.capitalize():<{col_d}}  {bar}  {pct:5.1%}  ({s}/{n})")
    print()
