"""
Stretch goal: compare text-based JSON output against the tool_use API.

In the main eval we ask Claude to return JSON as text, which we then parse. The
tool-use approach instead declares a `route_ticket` "tool" and lets Claude call
it with typed parameters — the API validates the structure on the way out.

This file provides a drop-in alternative to the main runner's call path so we
can measure:
  - Parse reliability (does text-JSON ever fail vs. tool-use never failing schema-wise?)
  - Latency/cost differences
  - Whether the constraint of a tool schema hurts quality of the response text

Usage:
    python -m src.run_tool_use_eval --out results/tool_use_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import anthropic

sys.path.insert(0, str(Path(__file__).parent))

from client import PRICING, CallResult, _compute_cost  # noqa: E402
from evaluators import evaluate_case  # noqa: E402
from strategies import PRODUCT_CONTEXT  # noqa: E402


ROUTE_TICKET_TOOL = {
    "name": "route_ticket",
    "description": (
        "Classify and route a customer support ticket. Call this exactly once "
        "after reading the ticket. The `response` field should be the customer-"
        "facing message that would be sent to the user."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": [
                    "billing",
                    "technical",
                    "feature_request",
                    "account_access",
                    "escalation",
                ],
                "description": "The support category that best describes this ticket.",
            },
            "escalate": {
                "type": "boolean",
                "description": (
                    "True if this ticket requires escalation to a specialized team "
                    "(security, legal, sales ops, on-call engineering, PR). Err toward "
                    "escalation for: security incidents, data loss, media inquiries, "
                    "legal/compliance, executive/VIP time pressure, billing disputes "
                    "with chargeback threats, or sustained customer frustration."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "1-2 sentences explaining category and escalation decision.",
            },
            "response": {
                "type": "string",
                "description": (
                    "The message to send to the customer. Be specific about next steps. "
                    "Do not invent facts, UI paths, error code meanings, feature existence, "
                    "or specific timelines. Route to the appropriate team when needed."
                ),
            },
        },
        "required": ["category", "escalate", "reasoning", "response"],
    },
}


TOOL_USE_SYSTEM = f"""You are a customer support assistant for CloudSync.

{PRODUCT_CONTEXT}

You MUST call the `route_ticket` tool exactly once with your classification and response."""


def call_with_tool(
    client: anthropic.Anthropic, model: str, ticket: str, max_tokens: int = 1024
) -> CallResult:
    start = time.perf_counter()
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=TOOL_USE_SYSTEM,
            tools=[ROUTE_TICKET_TOOL],
            tool_choice={"type": "tool", "name": "route_ticket"},
            messages=[{"role": "user", "content": f"Customer ticket:\n\n{ticket}"}],
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return CallResult(
            raw_text="",
            parsed=None,
            parse_error=f"API error: {type(e).__name__}: {e}",
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            cost_usd=None,
            model=model,
        )

    # Find the tool_use block
    tool_block = next(
        (b for b in msg.content if getattr(b, "type", None) == "tool_use"), None
    )

    parsed = None
    parse_error = None
    if tool_block is None:
        parse_error = "no tool_use block returned (model responded with text only)"
        raw_text = "".join(
            b.text for b in msg.content if getattr(b, "type", None) == "text"
        )
    else:
        parsed = dict(tool_block.input)  # already a dict from the SDK
        raw_text = json.dumps(parsed, indent=2)

    return CallResult(
        raw_text=raw_text,
        parsed=parsed,
        parse_error=parse_error,
        input_tokens=msg.usage.input_tokens,
        output_tokens=msg.usage.output_tokens,
        latency_ms=latency_ms,
        cost_usd=_compute_cost(model, msg.usage.input_tokens, msg.usage.output_tokens),
        model=model,
        stop_reason=msg.stop_reason,
    )


def run(
    test_cases_path: Path,
    output_dir: Path,
    model: str,
    limit: int | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(test_cases_path.read_text(encoding="utf-8"))
    cases = data["cases"][:limit] if limit else data["cases"]

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    scores = []
    raw_outputs = []

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}...", end=" ", flush=True)
        result = call_with_tool(client, model, case["ticket"])
        # Use strategy label "tool_use" so results slot into the same reporting
        score = evaluate_case(case, "tool_use", result, judge_client=None)
        scores.append(score)
        raw_outputs.append({
            "case_id": case["id"],
            "strategy": "tool_use",
            "ticket": case["ticket"],
            "expected_category": case["expected_category"],
            "expected_escalate": case["expected_escalate"],
            "raw_response": result.raw_text,
            "parsed": result.parsed,
            "parse_error": result.parse_error,
        })
        status = "ok" if result.parsed else "FAIL"
        print(f"{status} ({result.latency_ms}ms, ${result.cost_usd or 0:.4f})")
        time.sleep(0.1)

    # Minimal summary in the same shape as run_eval's output
    n = len(scores)
    summary = {
        "model": model,
        "judge_model": None,
        "strategies": {
            "tool_use": {
                "n_cases": n,
                "classification_accuracy": sum(s.classification_correct for s in scores) / n,
                "escalation_accuracy": sum(s.escalation_correct for s in scores) / n,
                "completeness_mean": sum(s.completeness for s in scores) / n,
                "no_hallucination_rate": sum(s.no_hallucination for s in scores) / n,
                "parse_success_rate": sum(s.parse_succeeded for s in scores) / n,
                "tone_score_mean": None,
                "quality_score_mean": None,
                "composite_mean": sum(s.composite for s in scores) / n,
                "total_cost_usd": round(sum((s.cost_usd or 0) for s in scores), 4),
                "avg_latency_ms": sum(s.latency_ms for s in scores) // n,
                "total_input_tokens": sum(s.input_tokens for s in scores),
                "total_output_tokens": sum(s.output_tokens for s in scores),
            }
        },
        "ranking_by_composite": ["tool_use"],
    }

    (output_dir / "scores.json").write_text(json.dumps([asdict(s) for s in scores], indent=2), encoding="utf-8")
    (output_dir / "raw_outputs.json").write_text(json.dumps(raw_outputs, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-cases",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "test_cases.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "tool_use",
    )
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    summary = run(args.test_cases, args.out, args.model, args.limit)
    print("\n=== Tool-use summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
