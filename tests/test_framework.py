"""
Unit tests for the evaluation framework itself.

These don't hit the Claude API — they test the scoring logic, JSON parsing,
and strategy construction so we can trust the numbers the harness produces.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from client import _compute_cost, _extract_json, PRICING
from evaluators import (
    CaseScore,
    score_classification,
    score_completeness,
    score_escalation,
    score_no_hallucination,
)
from strategies import STRATEGIES


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------
class TestExtractJson:
    def test_plain_json(self):
        parsed, err = _extract_json('{"category": "billing", "escalate": false}')
        assert parsed == {"category": "billing", "escalate": False}
        assert err is None

    def test_fenced_json(self):
        text = 'Here is the result:\n```json\n{"category": "billing"}\n```'
        parsed, err = _extract_json(text)
        assert parsed == {"category": "billing"}
        assert err is None

    def test_json_with_preamble(self):
        text = 'Sure! {"category": "billing", "escalate": true} done.'
        parsed, err = _extract_json(text)
        assert parsed == {"category": "billing", "escalate": True}
        assert err is None

    def test_empty_text(self):
        parsed, err = _extract_json("")
        assert parsed is None
        assert "empty" in err

    def test_no_json(self):
        parsed, err = _extract_json("This is just text, no structure.")
        assert parsed is None
        assert err is not None

    def test_malformed_json(self):
        # missing closing brace past the last `{`
        parsed, err = _extract_json('{"category": "billing"')
        assert parsed is None
        assert err is not None


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------
class TestCost:
    def test_known_model(self):
        # sonnet-4: $3/M input, $15/M output
        cost = _compute_cost("claude-sonnet-4-20250514", 1_000_000, 1_000_000)
        assert cost == pytest.approx(18.0)

    def test_small_call(self):
        cost = _compute_cost("claude-sonnet-4-20250514", 1000, 500)
        # (1000/1M * 3) + (500/1M * 15) = 0.003 + 0.0075 = 0.0105
        assert cost == pytest.approx(0.0105)

    def test_unknown_model(self):
        assert _compute_cost("made-up-model", 1000, 1000) is None


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
class TestClassification:
    def test_correct(self):
        assert score_classification("billing", {"category": "billing"}) == 1

    def test_case_insensitive(self):
        assert score_classification("billing", {"category": "Billing"}) == 1

    def test_wrong(self):
        assert score_classification("billing", {"category": "technical"}) == 0

    def test_none(self):
        assert score_classification("billing", None) == 0

    def test_missing_field(self):
        assert score_classification("billing", {}) == 0


class TestEscalation:
    def test_correct_true(self):
        assert score_escalation(True, {"escalate": True}) == 1

    def test_correct_false(self):
        assert score_escalation(False, {"escalate": False}) == 1

    def test_wrong(self):
        assert score_escalation(True, {"escalate": False}) == 0

    def test_string_true(self):
        # Some models emit strings
        assert score_escalation(True, {"escalate": "true"}) == 1

    def test_none(self):
        assert score_escalation(True, None) == 0


class TestCompleteness:
    def test_all_present(self):
        parsed = {"response": "I'll check your billing and refund the charge."}
        assert score_completeness(["billing", "refund"], parsed) == 1.0

    def test_partial(self):
        parsed = {"response": "I'll check your billing."}
        result = score_completeness(["billing", "refund"], parsed)
        assert result == pytest.approx(0.5)

    def test_none(self):
        parsed = {"response": "Sorry."}
        assert score_completeness(["billing", "refund"], parsed) == 0.0

    def test_case_insensitive(self):
        parsed = {"response": "I'll check your BILLING."}
        assert score_completeness(["billing"], parsed) == 1.0

    def test_empty_must_mention(self):
        # No requirements → trivially complete if there's a response
        assert score_completeness([], {"response": "hi"}) == 1.0


class TestHallucination:
    def test_clean_response(self):
        parsed = {"response": "I'll look into your billing issue and get back to you."}
        score, flags = score_no_hallucination(["specific refund amount"], parsed)
        assert score == 1
        assert flags == []

    def test_hallucination_detected(self):
        parsed = {"response": "I'll refund the specific amount of $42."}
        score, flags = score_no_hallucination(["specific refund amount"], parsed)
        assert score == 0
        assert len(flags) == 1

    def test_none(self):
        score, flags = score_no_hallucination(["anything"], None)
        assert score == 0


# ---------------------------------------------------------------------------
# Strategy construction
# ---------------------------------------------------------------------------
class TestStrategies:
    def test_all_strategies_exist(self):
        expected = {"zero_shot", "few_shot", "chain_of_thought", "structured"}
        assert expected <= set(STRATEGIES.keys())

    def test_user_custom_registered_when_file_exists(self):
        from strategies import _USER_PROMPT_PATH, STRATEGIES
        if _USER_PROMPT_PATH.exists():
            assert "user_custom" in STRATEGIES
            system, user = STRATEGIES["user_custom"]("Test ticket")
            assert len(system) > 20
            assert "Test ticket" in user
        else:
            assert "user_custom" not in STRATEGIES

    def test_all_return_system_and_user(self):
        for name, fn in STRATEGIES.items():
            system, user = fn("Test ticket")
            assert isinstance(system, str), f"{name} system not str"
            assert isinstance(user, str), f"{name} user not str"
            assert len(system) > 50, f"{name} system too short"
            assert "Test ticket" in user, f"{name} user must contain ticket"

    def test_few_shot_has_examples(self):
        system, _ = STRATEGIES["few_shot"]("Test")
        assert "Example 1" in system
        assert "Example 2" in system

    def test_chain_of_thought_has_reasoning_steps(self):
        system, _ = STRATEGIES["chain_of_thought"]("Test")
        assert "step" in system.lower()

    def test_structured_has_role(self):
        system, _ = STRATEGIES["structured"]("Test")
        assert "role" in system.lower() or "specialist" in system.lower()


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------
class TestComposite:
    def test_perfect_score(self):
        s = CaseScore(
            case_id="x",
            strategy="y",
            classification_correct=1,
            escalation_correct=1,
            completeness=1.0,
            no_hallucination=1,
            tone_score=1.0,
            parse_succeeded=1,
        )
        assert s.composite == pytest.approx(1.0)

    def test_zero_score(self):
        s = CaseScore(case_id="x", strategy="y", tone_score=0.0)
        assert s.composite == pytest.approx(0.0)

    def test_weights_sum_to_one(self):
        # If all components are 1, composite must be 1 (sanity check on weights)
        s = CaseScore(
            case_id="x",
            strategy="y",
            classification_correct=1,
            escalation_correct=1,
            completeness=1.0,
            no_hallucination=1,
            tone_score=1.0,
            parse_succeeded=1,
        )
        # If this drifts from 1.0, the weight tuple in composite() needs updating
        assert abs(s.composite - 1.0) < 1e-9

    def test_missing_tone_defaults_to_neutral(self):
        # When no judge is used, tone_score is None — we use 0.5 as neutral
        s = CaseScore(
            case_id="x",
            strategy="y",
            classification_correct=1,
            escalation_correct=1,
            completeness=1.0,
            no_hallucination=1,
            tone_score=None,
            parse_succeeded=1,
        )
        # Everything else is 1, tone is treated as 0.5, weight 0.15
        # → 0.85 + 0.15*0.5 = 0.925
        assert s.composite == pytest.approx(0.925)


# ---------------------------------------------------------------------------
# Dataset integrity
# ---------------------------------------------------------------------------
class TestDataset:
    @pytest.fixture
    def dataset(self):
        path = Path(__file__).parent.parent / "data" / "test_cases.json"
        return json.loads(path.read_text())

    def test_has_cases(self, dataset):
        assert len(dataset["cases"]) >= 30

    def test_all_cases_have_required_fields(self, dataset):
        required = {
            "id",
            "ticket",
            "expected_category",
            "expected_escalate",
            "expected_tone",
            "difficulty",
            "must_mention",
            "must_not_hallucinate",
        }
        for case in dataset["cases"]:
            missing = required - set(case.keys())
            assert not missing, f"Case {case.get('id')} missing {missing}"

    def test_categories_valid(self, dataset):
        valid = {"billing", "technical", "feature_request", "account_access", "escalation"}
        for case in dataset["cases"]:
            assert case["expected_category"] in valid

    def test_difficulties_valid(self, dataset):
        valid = {"easy", "medium", "hard"}
        for case in dataset["cases"]:
            assert case["difficulty"] in valid

    def test_unique_ids(self, dataset):
        ids = [c["id"] for c in dataset["cases"]]
        assert len(ids) == len(set(ids)), "Duplicate case IDs"

    def test_category_balance(self, dataset):
        # No single category should dominate — rough balance matters
        from collections import Counter
        counts = Counter(c["expected_category"] for c in dataset["cases"])
        total = sum(counts.values())
        for cat, count in counts.items():
            assert count / total < 0.5, f"{cat} is overrepresented ({count}/{total})"

    def test_escalation_balance(self, dataset):
        # We need both escalate=True and escalate=False cases to measure judgment
        esc_true = sum(1 for c in dataset["cases"] if c["expected_escalate"])
        esc_false = sum(1 for c in dataset["cases"] if not c["expected_escalate"])
        assert esc_true >= 5, "Need enough escalation=True cases to measure"
        assert esc_false >= 5, "Need enough escalation=False cases to measure"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
