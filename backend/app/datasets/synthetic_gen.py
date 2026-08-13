from typing import Any

from app.datasets.base import DatasetSource
from app.models.run import DatasetSpec


class SyntheticGenSource(DatasetSource):
    """Generates a training set from a natural-language description of the task.

    Uses Claude if ANTHROPIC_API_KEY is configured; otherwise falls back to a small
    number of templated instruction/response pairs so the pipeline never hard-fails
    for lack of an external key."""

    def build(self, spec: DatasetSpec) -> list[dict[str, Any]]:
        instructions = spec.generation_instructions or "Generate a helpful instruction-following example."

        from app.config import get_settings

        settings = get_settings()
        if settings.anthropic_api_key:
            examples = self._claude_generate(instructions, spec.n_train)
            if examples:
                return examples

        return self._template_generate(instructions, spec.n_train)

    def _claude_generate(self, instructions: str, n: int) -> list[dict[str, Any]] | None:
        try:
            import anthropic
        except ImportError:
            return None

        from app.config import get_settings

        client = anthropic.Anthropic(api_key=get_settings().anthropic_api_key)
        examples: list[dict[str, Any]] = []
        batch_size = 10
        while len(examples) < n:
            want = min(batch_size, n - len(examples))
            try:
                resp = client.messages.create(
                    model="claude-sonnet-5",
                    max_tokens=1500,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Generate {want} diverse (prompt, completion) training pairs for this "
                            f"fine-tuning goal: {instructions}\n\n"
                            "Respond ONLY with a JSON array of objects like "
                            '[{"prompt": "...", "completion": "..."}, ...]. No other text.'
                        ),
                    }],
                )
                text = "".join(b.text for b in resp.content if hasattr(b, "text"))
                import json

                batch = json.loads(text)
                examples.extend(
                    {"prompt": str(x["prompt"]), "completion": str(x["completion"])} for x in batch
                )
            except Exception:
                break
        return examples[:n] if examples else None

    def _template_generate(self, instructions: str, n: int) -> list[dict[str, Any]]:
        return [
            {
                "prompt": f"Task {i + 1}: {instructions}",
                "completion": f"Understood. Here is a response consistent with: {instructions}",
            }
            for i in range(n)
        ]
