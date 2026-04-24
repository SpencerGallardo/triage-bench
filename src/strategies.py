"""
Prompt strategies for evaluating Claude on customer support ticket handling.

Each strategy takes a ticket and returns a (system_prompt, user_prompt) tuple.
Strategies vary in how much context, structure, and reasoning guidance they
provide to the model.
"""

from pathlib import Path
from textwrap import dedent

# ---------------------------------------------------------------------------
# Shared product context — included in every strategy's system prompt so that
# differences in performance reflect differences in *prompting approach*, not
# differences in product knowledge.
# ---------------------------------------------------------------------------
PRODUCT_CONTEXT = dedent("""
    CloudSync is a B2B SaaS file synchronization and team collaboration platform.

    Plans:
    - Free: 2 users, 5 GB storage
    - Pro: $12/user/month, 100 GB storage, SSO
    - Business: $24/user/month, 1 TB storage, audit logs, admin API
    - Enterprise: custom pricing, dedicated support

    Support categories: billing, technical, feature_request, account_access, escalation

    Escalation criteria:
    - Security issues (credential exposure, unauthorized access, vulnerabilities)
    - Legal/compliance (GDPR, HIPAA, tax, regulatory)
    - Media or press inquiries
    - Sustained customer frustration (cancellation threats, chargebacks, public complaints)
    - Data loss reports
    - Anything involving executives or VIP accounts under time pressure
    - API rate limit and enterprise contract requests
""").strip()


OUTPUT_SCHEMA = dedent("""
    Respond with a JSON object containing these fields:
    {
      "category": "billing" | "technical" | "feature_request" | "account_access" | "escalation",
      "escalate": true | false,
      "reasoning": "<1-2 sentence explanation of your category and escalation decision>",
      "response": "<the message you would send to the customer>"
    }

    Return ONLY the JSON object, with no surrounding text or markdown fences.
""").strip()


# ---------------------------------------------------------------------------
# Strategy 1: Zero-shot
# ---------------------------------------------------------------------------
def zero_shot(ticket: str) -> tuple[str, str]:
    system = dedent(f"""
        You are a customer support assistant for CloudSync.

        {PRODUCT_CONTEXT}

        {OUTPUT_SCHEMA}
    """).strip()

    user = f"Customer ticket:\n\n{ticket}"
    return system, user


# ---------------------------------------------------------------------------
# Strategy 2: Few-shot
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = dedent("""
    Example 1:
    Ticket: "Can I get an invoice with my company name instead of my personal name?"
    Response: {"category": "billing", "escalate": false, "reasoning": "Standard billing admin request that support can help with by pointing to billing settings or the billing team.", "response": "Happy to help update your invoice details. You can change the billing name under Settings > Billing > Company Info, or I can route this to our billing team to update directly. Which would you prefer?"}

    Example 2:
    Ticket: "Our entire team just got logged out simultaneously and now nobody can sign in. We have a customer demo in 20 minutes."
    Response: {"category": "account_access", "escalate": true, "reasoning": "Widespread auth failure affecting an entire team under time pressure. Needs urgent engineering attention, not standard troubleshooting.", "response": "I'm escalating this to our on-call engineering team right now given the scale and urgency. While I do that: can you confirm the email domain of your workspace and whether SSO is enabled? I'll follow up here within 5 minutes."}

    Example 3:
    Ticket: "Does your Pro plan come with API access?"
    Response: {"category": "billing", "escalate": false, "reasoning": "Plan feature question — a factual answer about what's included. No escalation needed.", "response": "Great question. API access is available, but the admin API specifically is a Business plan feature. Pro includes standard read/write endpoints. Want me to share what's included at each tier?"}
""").strip()


def few_shot(ticket: str) -> tuple[str, str]:
    system = dedent(f"""
        You are a customer support assistant for CloudSync.

        {PRODUCT_CONTEXT}

        Here are examples of well-handled tickets:

        {FEW_SHOT_EXAMPLES}

        {OUTPUT_SCHEMA}
    """).strip()

    user = f"Customer ticket:\n\n{ticket}"
    return system, user


# ---------------------------------------------------------------------------
# Strategy 3: Chain-of-thought
# ---------------------------------------------------------------------------
def chain_of_thought(ticket: str) -> tuple[str, str]:
    system = dedent(f"""
        You are a customer support assistant for CloudSync.

        {PRODUCT_CONTEXT}

        Before responding, think through the ticket step by step:
        1. What is the customer actually asking for or experiencing?
        2. What category does this fall into, and why?
        3. Does this meet any escalation criteria? If borderline, err toward escalating for security, legal, and data-loss cases.
        4. What does the customer need to hear — both practically and emotionally?
        5. What facts would I need to verify vs. what can I safely say without verification? (Do not invent error code meanings, feature existence, timelines, or specific UI paths you are not certain about.)

        After reasoning, produce the final response.

        {OUTPUT_SCHEMA}

        Include your step-by-step reasoning in the "reasoning" field — keep it tight (3-4 sentences max).
    """).strip()

    user = f"Customer ticket:\n\n{ticket}"
    return system, user


# ---------------------------------------------------------------------------
# Strategy 4: Structured system prompt (role + constraints + anti-hallucination)
# ---------------------------------------------------------------------------
def structured(ticket: str) -> tuple[str, str]:
    system = dedent(f"""
        # Role
        You are a senior customer support specialist at CloudSync. You've been
        on the team for 3 years and your priority is accurate, empathetic help
        that doesn't over-promise.

        # Product
        {PRODUCT_CONTEXT}

        # Response principles
        - Lead with acknowledgment when the customer is frustrated or experiencing a problem.
        - Be specific about what you can do, not vague about what might happen.
        - Never invent facts. If you don't know the meaning of an error code, the existence of a feature, a specific timeline, a sales rep's commitments, or an exact UI path, say what you can verify and offer to route to someone who can confirm.
        - Match the customer's urgency without amplifying panic.
        - Prefer routing to the right team over attempting to resolve things outside your scope.

        # Escalation
        Escalate when any of the following are true:
        - Security incident (credential exposure, unauthorized access, vulnerability report)
        - Legal, compliance, or regulatory matter (GDPR, HIPAA, tax, contracts)
        - Media or press inquiry — do not confirm or deny anything, route to PR
        - Data loss or cross-account data leakage
        - Sustained frustration with cancellation, chargeback, or public complaint threats
        - Billing dispute with specific monetary impact claim
        - Time-critical executive access issues
        - API rate limit increases or enterprise contract questions

        # Output
        {OUTPUT_SCHEMA}
    """).strip()

    user = f"Customer ticket:\n\n{ticket}"
    return system, user


# ---------------------------------------------------------------------------
# Strategy 5: User-provided custom prompt
#
# Reads a user's own prompt from prompts/user_custom.txt and tests it alongside
# the four canonical strategies. If the file doesn't exist, this strategy is
# silently excluded from the registry so the framework still works out of the
# box for anyone who just clones the repo.
# ---------------------------------------------------------------------------
_USER_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "user_custom.txt"


def user_custom(ticket: str) -> tuple[str, str]:
    system = _USER_PROMPT_PATH.read_text(encoding="utf-8").strip()
    user = f"Customer ticket:\n\n{ticket}"
    return system, user


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
STRATEGIES = {
    "zero_shot": zero_shot,
    "few_shot": few_shot,
    "chain_of_thought": chain_of_thought,
    "structured": structured,
}

# Only register user_custom if a prompt file actually exists. This keeps the
# framework functional for anyone who clones the repo without providing their
# own prompt, while letting power users plug theirs in for comparison.
if _USER_PROMPT_PATH.exists():
    STRATEGIES["user_custom"] = user_custom
