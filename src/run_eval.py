"""
Eval runner: iterates over (test case × strategy) pairs, calls Claude, scores
the results, and writes a structured results file for reporting.

Usage:
    python -m src.run_eval --model claude-sonnet-4-20250514 --judge --out results/run_01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Allow running as both `python -m src.run_eval` and `python src/run_eval.py`
sys.path.insert(0, str(Path(__file__).parent))

from client import ClaudeClient  # noqa: E402
from evaluators import CaseScore, evaluate_case  # noqa: E402
from strategies import STRATEGIES  # noqa: E402


def run_eval(
    test_cases_path: Path,
    output_dir: Path,
    model: str,
    judge_model: str | None,
    strategies_to_run: list[str] | None = None,
    limit: int | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(test_cases_path.read_text(encoding="utf-8"))
    cases = data["cases"]
    if limit:
        cases = cases[:limit]

    strategies = strategies_to_run or list(STRATEGIES.keys())

    client = ClaudeClient(model=model)
    judge_client = ClaudeClient(model=judge_model) if judge_model else None

    all_scores: list[CaseScore] = []
    all_raw_outputs: list[dict] = []

    total_calls = len(cases) * len(strategies)
    call_idx = 0

    for strategy_name in strategies:
        strategy_fn = STRATEGIES[strategy_name]
        print(f"\n=== Strategy: {strategy_name} ===")

        for case in cases:
            call_idx += 1
            print(f"  [{call_idx}/{total_calls}] {case['id']}...", end=" ", flush=True)

            system, user = strategy_fn(case["ticket"])
            result = client.call(system, user)

            score = evaluate_case(case, strategy_name, result, judge_client)
            all_scores.append(score)

            all_raw_outputs.append({
                "case_id": case["id"],
                "strategy": strategy_name,
                "ticket": case["ticket"],
                "expected_category": case["expected_category"],
                "expected_escalate": case["expected_escalate"],
                "raw_response": result.raw_text,
                "parsed": result.parsed,
                "parse_error": result.parse_error,
            })

            status = "ok" if result.parsed else "PARSE_FAIL"
            print(f"{status} ({result.latency_ms}ms, ${result.cost_usd or 0:.4f})")

            # Gentle pacing to avoid rate limits on large test sets
            time.sleep(0.1)

    # -------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------
    summary = aggregate_results(all_scores, model, judge_model)

    # -------------------------------------------------------------------
    # Persist
    # -------------------------------------------------------------------
    scores_path = output_dir / "scores.json"
    raw_path = output_dir / "raw_outputs.json"
    summary_path = output_dir / "summary.json"

    scores_path.write_text(json.dumps([asdict(s) for s in all_scores], indent=2), encoding="utf-8")
    raw_path.write_text(json.dumps(all_raw_outputs, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWrote: {scores_path}")
    print(f"Wrote: {raw_path}")
    print(f"Wrote: {summary_path}")

    return summary


def aggregate_results(
    scores: list[CaseScore], model: str, judge_model: str | None
) -> dict:
    by_strategy: dict[str, list[CaseScore]] = {}
    for s in scores:
        by_strategy.setdefault(s.strategy, []).append(s)

    strategy_summaries = {}
    for strat, strat_scores in by_strategy.items():
        n = len(strat_scores)
        if n == 0:
            continue

        def mean(attr: str, skip_none: bool = False) -> float | None:
            vals = [getattr(s, attr) for s in strat_scores]
            if skip_none:
                vals = [v for v in vals if v is not None]
            if not vals:
                return None
            return sum(vals) / len(vals)

        total_cost = sum((s.cost_usd or 0) for s in strat_scores)
        total_latency = sum(s.latency_ms for s in strat_scores)

        strategy_summaries[strat] = {
            "n_cases": n,
            "classification_accuracy": mean("classification_correct"),
            "escalation_accuracy": mean("escalation_correct"),
            "completeness_mean": mean("completeness"),
            "no_hallucination_rate": mean("no_hallucination"),
            "parse_success_rate": mean("parse_succeeded"),
            "tone_score_mean": mean("tone_score", skip_none=True),
            "quality_score_mean": mean("quality_score", skip_none=True),
            "composite_mean": sum(s.composite for s in strat_scores) / n,
            "total_cost_usd": round(total_cost, 4),
            "avg_latency_ms": total_latency // n,
            "total_input_tokens": sum(s.input_tokens for s in strat_scores),
            "total_output_tokens": sum(s.output_tokens for s in strat_scores),
        }

    return {
        "model": model,
        "judge_model": judge_model,
        "strategies": strategy_summaries,
        "ranking_by_composite": sorted(
            strategy_summaries.keys(),
            key=lambda k: strategy_summaries[k]["composite_mean"],
            reverse=True,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Claude support-eval.")
    parser.add_argument(
        "--test-cases",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "test_cases.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "latest",
    )
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Enable LLM-as-judge scoring (uses ~2x API calls)",
    )
    parser.add_argument(
        "--judge-model",
        default="claude-sonnet-4-20250514",
        help="Model to use as judge (defaults to same as main model)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(STRATEGIES.keys()),
        help="Subset of strategies to run (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N test cases (useful for debugging)",
    )
    args = parser.parse_args()

    summary = run_eval(
        test_cases_path=args.test_cases,
        output_dir=args.out,
        model=args.model,
        judge_model=args.judge_model if args.judge else None,
        strategies_to_run=args.strategies,
        limit=args.limit,
    )

    print("\n\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
