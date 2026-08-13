from app.config import get_settings
from app.models.run import RunPlan
from app.planner.claude_planner import plan_with_claude
from app.planner.rule_based_fallback import plan_from_prompt


def plan(prompt: str) -> RunPlan:
    """Prompt -> RunPlan. Uses Claude (structured tool-output) if an API key is
    configured, otherwise falls back to deterministic rule-based routing. Also
    falls back if the Claude call fails for any reason, so plan creation never
    hard-fails just because the planning LLM is unavailable."""
    settings = get_settings()
    if settings.anthropic_api_key:
        try:
            return plan_with_claude(prompt)
        except Exception:
            pass
    return plan_from_prompt(prompt)
