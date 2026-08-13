from app.config import get_settings
from app.models.run import RunPlan
from app.planner.schema import PLANNER_SYSTEM_PROMPT, run_plan_tool_schema


class PlannerUnavailableError(RuntimeError):
    pass


def plan_with_claude(prompt: str) -> RunPlan:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise PlannerUnavailableError("no ANTHROPIC_API_KEY configured")

    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    tool = run_plan_tool_schema()

    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=2000,
        system=PLANNER_SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": "emit_run_plan"},
        messages=[{"role": "user", "content": prompt}],
    )

    for block in resp.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "emit_run_plan":
            return RunPlan.model_validate(block.input)

    raise PlannerUnavailableError("Claude did not return a tool_use block")
