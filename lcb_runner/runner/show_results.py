"""
show_results.py — display a saved LiveCodeBench run without re-executing.

Usage:
    python -m lcb_runner.runner.show_results output/MyModel/Scenario.codegeneration_1_0.0
    python -m lcb_runner.runner.show_results output/MyModel/Scenario.codegeneration_1_0.0_eval_all.json
"""
import argparse
import json
import os
import sys


def _error_code_label(code: int) -> str:
    labels = {
        -1: "compile error",
        -2: "runtime error",
        -3: "timeout",
        -4: "syntax error",
        -5: "wrong answer",
    }
    return labels.get(code, f"error {code}")


def show(eval_all_path: str, eval_path: str | None = None, show_code: bool = False):
    with open(eval_all_path) as f:
        problems = json.load(f)

    # summary metrics (may come from _eval.json or inline in eval_all)
    summary: dict = {}
    if eval_path and os.path.exists(eval_path):
        with open(eval_path) as f:
            raw = json.load(f)
        # eval.json is [summary_dict, per_problem_dict, metadatas_list]
        if isinstance(raw, list) and raw:
            summary = raw[0]
        elif isinstance(raw, dict):
            summary = raw

    pass_at_1_overall = summary.get("pass@1", None)

    # ── per-problem table ────────────────────────────────────────────────────
    col_id    = 14
    col_title = 36
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

    total = len(problems)
    solved = 0
    by_diff: dict[str, list[float]] = {}

    for p in problems:
        qid    = str(p.get("question_id",    "?"))[:col_id]
        title  = str(p.get("question_title", "?"))[:col_title]
        diff   = str(p.get("difficulty",     "?")).lower()
        passed = p.get("pass@1", None)

        mark = "✓" if passed == 1.0 else ("✗" if passed == 0.0 else "?")
        if passed == 1.0:
            solved += 1
        if passed is not None:
            by_diff.setdefault(diff, []).append(passed)

        print(
            f"{qid:<{col_id}}   "
            f"{title:<{col_title}}   "
            f"{diff.upper()[:col_diff]:<{col_diff}}   "
            f"{mark:<{col_res}}"
        )

        # metadata: error or timing per sample
        for meta_raw in p.get("metadata", []):
            try:
                m = json.loads(meta_raw)
            except Exception:
                m = {}
            if "error_message" in m:
                ec = m.get("error_code", "")
                label = _error_code_label(ec) if isinstance(ec, int) else ""
                note = m["error_message"]
                print(f"  {'':>{col_id}}  ↳ [{label}] {note}")
            elif "execution time" in m:
                print(f"  {'':>{col_id}}  ↳ exec {m['execution time']:.4f}s")

        # optionally show extracted code
        if show_code:
            for i, code in enumerate(p.get("code_list", [])):
                print(f"  {'':>{col_id}}  ── sample {i} code ──")
                for line in (code or "(empty)").splitlines():
                    print(f"  {'':>{col_id}}  | {line}")

    print(sep)

    # ── summary ──────────────────────────────────────────────────────────────
    score = pass_at_1_overall if pass_at_1_overall is not None else (solved / total if total else 0.0)
    print(f"\n  pass@1 : {score:.1%}  ({solved}/{total} solved)")

    # difficulty breakdown (easy → medium → hard)
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


def main():
    parser = argparse.ArgumentParser(description="Display a saved LCB run.")
    parser.add_argument(
        "path",
        help="Path to *_eval_all.json, or the base path (without suffix)",
    )
    parser.add_argument(
        "--code", action="store_true", help="Also print the extracted code for each problem"
    )
    args = parser.parse_args()

    path = args.path
    # accept base path (strip known suffixes, then re-add)
    for suffix in ("_eval_all.json", "_eval.json", ".json", ""):
        if path.endswith(suffix):
            base = path[: len(path) - len(suffix)] if suffix else path
            break

    eval_all = base + "_eval_all.json"
    eval_sum = base + "_eval.json"

    if not os.path.exists(eval_all):
        print(f"Error: {eval_all} not found", file=sys.stderr)
        sys.exit(1)

    show(eval_all, eval_sum if os.path.exists(eval_sum) else None, show_code=args.code)


if __name__ == "__main__":
    main()
