"""
Scoring layer.

Each evaluator produces a dict of {metric_name: score in [0, 1]} plus notes.

We split scoring into:
  - Rule-based scorers (deterministic, fast, free): classification, escalation,
    completeness via keyword check, hallucination via forbidden-phrase check.
  - LLM-as-judge scorers (costlier but capture nuance): tone appropriateness
    and an overall response quality rating.

Keeping them separate lets us run the cheap scorers always and opt into the
judge when we want richer signal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from client import ClaudeClient


@dataclass
class CaseScore:
    case_id: str
    strategy: str

    # Rule-based scores (0 or 1 each, except completeness which is a fraction)
    classification_correct: int = 0
    escalation_correct: int = 0
    completeness: float = 0.0
    no_hallucination: int = 0

    # Judge-based scores (0.0 to 1.0)
    tone_score: float | None = None
    quality_score: float | None = None

    # Diagnostics
    parse_succeeded: int = 0
    notes: list[str] = field(default_factory=list)
    judge_rationale: str | None = None

    # Pass-through metadata
    latency_ms: int = 0
    cost_usd: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def composite(self) -> float:
        """
        Weighted aggregate score for ranking strategies.

        Weights reflect what actually matters for a support system:
          - Escalation judgment: 25% — getting escalation wrong is the highest-impact error
          - No hallucination: 25% — fabrications erode customer trust and create legal risk
          - Classification: 20% — drives downstream routing
          - Tone: 15% — the humanizing layer
          - Completeness: 10% — did we address the ask
          - Parsing: 5% — can we actually use the output in a pipeline
        """
        parts = [
            (0.25, float(self.escalation_correct)),
            (0.25, float(self.no_hallucination)),
            (0.20, float(self.classification_correct)),
            (0.15, self.tone_score if self.tone_score is not None else 0.5),
            (0.10, self.completeness),
            (0.05, float(self.parse_succeeded)),
        ]
        return sum(weight * score for weight, score in parts)


# ---------------------------------------------------------------------------
# Rule-based evaluators
# ---------------------------------------------------------------------------
def score_classification(expected: str, parsed: dict[str, Any] | None) -> int:
    if not parsed:
        return 0
    return int(str(parsed.get("category", "")).strip().lower() == expected.lower())


def score_escalation(expected: bool, parsed: dict[str, Any] | None) -> int:
    if not parsed:
        return 0
    actual = parsed.get("escalate")
    # Accept bools and common stringy equivalents
    if isinstance(actual, str):
        actual = actual.strip().lower() in ("true", "yes", "1")
    return int(bool(actual) == bool(expected))


def score_completeness(must_mention: list[str], parsed: dict[str, Any] | None) -> float:
    """
    Fraction of required concepts mentioned in the response.

    We check case-insensitively and do a loose substring match. This is
    intentionally a soft check — phrasing varies, but if none of the required
    concepts appear at all, the response is almost certainly incomplete.
    """
    if not parsed or not must_mention:
        return 1.0 if parsed else 0.0
    response_text = str(parsed.get("response", "")).lower()
    if not response_text:
        return 0.0
    hits = sum(1 for concept in must_mention if concept.lower() in response_text)
    return hits / len(must_mention)


def score_no_hallucination(must_not: list[str], parsed: dict[str, Any] | None) -> tuple[int, list[str]]:
    """
    Flag obvious hallucinations by checking for forbidden substrings and patterns.

    This is a coarse check — it catches the specific traps we seeded into the
    dataset (e.g., inventing what SYNC_ERR_503 means, claiming a Notion
    integration exists). The judge-based evaluator can catch subtler ones.
    """
    if not parsed:
        return 0, ["no parsed response"]

    response_text = str(parsed.get("response", "")).lower()
    if not response_text:
        return 0, ["empty response"]

    flagged: list[str] = []
    for forbidden in must_not:
        # We don't ban the exact phrase — we look for patterns suggesting
        # the model made an unverified claim. Keep it simple: substring match
        # of characteristic tokens from the forbidden claim.
        tokens = [t.lower() for t in forbidden.split() if len(t) > 3]
        if tokens and all(tok in response_text for tok in tokens):
            flagged.append(forbidden)

    return (0 if flagged else 1), flagged


# ---------------------------------------------------------------------------
# LLM-as-judge evaluator
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = """\
You are an expert customer support QA reviewer. You evaluate draft responses
from a support assistant on two dimensions:

1. TONE (0.0 to 1.0): Does the response match the tone the situation calls
   for? Key considerations:
   - Frustrated/angry customers need empathetic acknowledgment before solutions
   - Technical users want directness, not excessive apology
   - Security/legal/press issues warrant careful, measured language
   - Positive feedback deserves warmth, not over-effusive gratitude

2. QUALITY (0.0 to 1.0): Is this a genuinely good support response? Consider:
   - Does it move the ticket toward resolution?
   - Is it specific about next steps rather than vague?
   - Does it avoid over-promising (specific timelines, refund amounts,
     guaranteed outcomes) while still being helpful?
   - Does it avoid inventing facts (fake integrations, fabricated error
     meanings, imagined UI paths)?

Return ONLY a JSON object with this exact shape:
{"tone_score": <float 0-1>, "quality_score": <float 0-1>, "rationale": "<1-2 sentences>"}
"""


def judge_response(
    ticket: str,
    expected_tone: str,
    response_text: str,
    judge_client: ClaudeClient,
) -> tuple[float | None, float | None, str | None]:
    if not response_text:
        return 0.0, 0.0, "empty response, nothing to judge"

    user = f"""Customer ticket:
{ticket}

Expected tone: {expected_tone}

Support agent's response:
{response_text}

Evaluate and return the JSON object."""

    result = judge_client.call(JUDGE_SYSTEM, user)
    if result.parsed is None:
        return None, None, f"judge parse failed: {result.parse_error}"

    try:
        tone = float(result.parsed.get("tone_score", 0))
        quality = float(result.parsed.get("quality_score", 0))
        # Clamp to [0, 1] — judges occasionally return slightly out-of-range numbers
        tone = max(0.0, min(1.0, tone))
        quality = max(0.0, min(1.0, quality))
        rationale = str(result.parsed.get("rationale", ""))
        return tone, quality, rationale
    except (TypeError, ValueError) as e:
        return None, None, f"judge returned non-numeric scores: {e}"


# ---------------------------------------------------------------------------
# Full scoring for one case × strategy
# ---------------------------------------------------------------------------
def evaluate_case(
    case: dict[str, Any],
    strategy_name: str,
    call_result: Any,
    judge_client: ClaudeClient | None = None,
) -> CaseScore:
    parsed = call_result.parsed
    score = CaseScore(
        case_id=case["id"],
        strategy=strategy_name,
        parse_succeeded=int(parsed is not None),
        latency_ms=call_result.latency_ms,
        cost_usd=call_result.cost_usd,
        input_tokens=call_result.input_tokens,
        output_tokens=call_result.output_tokens,
    )

    if call_result.parse_error:
        score.notes.append(f"parse: {call_result.parse_error}")

    if parsed is None:
        return score  # everything else scored 0 by default

    score.classification_correct = score_classification(case["expected_category"], parsed)
    score.escalation_correct = score_escalation(case["expected_escalate"], parsed)
    score.completeness = score_completeness(case.get("must_mention", []), parsed)

    no_hall, flagged = score_no_hallucination(case.get("must_not_hallucinate", []), parsed)
    score.no_hallucination = no_hall
    if flagged:
        score.notes.append(f"hallucination flags: {flagged}")

    # Judge step — optional and expensive, so we only run it when a judge
    # client is provided
    if judge_client is not None:
        response_text = str(parsed.get("response", ""))
        tone, quality, rationale = judge_response(
            case["ticket"],
            case["expected_tone"],
            response_text,
            judge_client,
        )
        score.tone_score = tone
        score.quality_score = quality
        score.judge_rationale = rationale

    return score
