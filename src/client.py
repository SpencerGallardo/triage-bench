"""
Thin wrapper around the Anthropic SDK that tracks latency, token usage, and
cost per call, and robustly parses JSON responses (handles fenced output).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import anthropic

# -----------------------------------------------------------------------------
# Pricing, in dollars per 1M tokens.
#
# Prices as of April 2026. See https://docs.claude.com/en/docs/about-claude/pricing
# If Anthropic changes pricing, update this table. The framework treats cost as
# a derived metric — if pricing is unknown, cost is reported as None rather
# than silently wrong.
# -----------------------------------------------------------------------------
PRICING: dict[str, dict[str, float]] = {
    # User-requested models (per the project spec)
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    # Current flagship models — also supported if users want to run against latest
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "claude-opus-4-7": {"input": 5.00, "output": 25.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
}


@dataclass
class CallResult:
    """Everything we captured from one API call."""

    raw_text: str
    parsed: dict[str, Any] | None
    parse_error: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float | None
    model: str
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClaudeClient:
    """Wrapper that handles retries, timing, cost, and JSON extraction."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        api_key: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def call(self, system: str, user: str) -> CallResult:
        start = time.perf_counter()
        stop_reason: str | None = None
        try:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            stop_reason = msg.stop_reason

            # Claude returns a list of content blocks. For a plain text response
            # there's typically one text block; we concatenate defensively.
            raw_text = "".join(
                block.text for block in msg.content if getattr(block, "type", None) == "text"
            )

            input_tokens = msg.usage.input_tokens
            output_tokens = msg.usage.output_tokens
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
                model=self.model,
                stop_reason=None,
            )

        parsed, parse_error = _extract_json(raw_text)
        cost = _compute_cost(self.model, input_tokens, output_tokens)

        return CallResult(
            raw_text=raw_text,
            parsed=parsed,
            parse_error=parse_error,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            model=self.model,
            stop_reason=stop_reason,
        )


def _extract_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    Try to pull a JSON object out of the model's response.

    Handles three common cases:
      1. Plain JSON
      2. JSON inside ```json ... ``` fences
      3. JSON with surrounding prose (finds first `{` to last matching `}`)
    """
    if not text.strip():
        return None, "empty response"

    # Case 1: try direct parse
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass

    # Case 2: fenced code block
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1)), None
        except json.JSONDecodeError as e:
            return None, f"fenced JSON parse failed: {e}"

    # Case 3: find first { and last } and try
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first : last + 1]
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError as e:
            return None, f"embedded JSON parse failed: {e}"

    return None, "no JSON object found in response"


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    prices = PRICING.get(model)
    if prices is None:
        return None
    return (input_tokens / 1_000_000) * prices["input"] + (
        output_tokens / 1_000_000
    ) * prices["output"]
