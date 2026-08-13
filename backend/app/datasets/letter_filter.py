"""Builds a training set whose completions provably never contain the letter B.

Two generation paths:
  - Claude-backed (if ANTHROPIC_API_KEY is set): ask Claude to answer a prompt while
    avoiding the letter B, verify with regex, retry with feedback on failure.
  - Template path (always available, no external calls): compose sentences from a
    hand-curated vocabulary that excludes the letter B by construction.

Either way, every completion that leaves `build()` has already passed the same
regex check, so the guarantee ("zero B in training data") does not depend on which
path produced it.
"""

import random
import re

from app.models.run import DatasetSpec

_B_RE = re.compile(r"[bB]")

# Common English words that contain no letter 'b' (checked by hand + the regex
# guard below, which is what actually enforces the guarantee at build time).
_SUBJECTS = [
    "the sun", "a river", "my friend", "the old house", "our team", "the wind",
    "a young cat", "the garden", "her voice", "the mountain", "an artist",
    "the ocean", "a quiet town", "the moon", "our teacher", "the forest",
    "a small dog", "the city", "an old clock", "the rain",
]
_VERBS = [
    "moves slowly", "grows every year", "changes with the seasons", "shines at night",
    "runs across the field", "sings a soft song", "creates something new", "waits patiently",
    "travels far away", "opens a new door", "tells an old story", "watches the stars",
    "climbs the hill", "greets the morning", "returns at sunset", "explores the valley",
]
_DESCRIPTORS = [
    "calm and quiet", "full of color", "strong and steady", "sweet and gentle",
    "clear as glass", "warm and kind", "old and wise", "fresh and cool",
    "full of life", "soft as snow", "quick and clever", "deep and mysterious",
]
_CONNECTORS = ["and", "while", "as", "so", "yet still"]

_PROMPT_TEMPLATES = [
    "Describe {topic} in a few sentences.",
    "Write a short paragraph about {topic}.",
    "Tell me a little story involving {topic}.",
    "Explain what makes {topic} interesting.",
    "Give a poetic description of {topic}.",
    "Share your thoughts on {topic}.",
    "Compose a few lines about {topic}.",
    "What comes to mind when you think of {topic}?",
]
_TOPICS = [
    "a quiet morning", "the changing seasons", "a mountain village", "an old friendship",
    "a rainy afternoon", "a garden at dawn", "a long journey", "the night sky",
    "a peaceful lake", "an autumn forest", "a busy city street", "a warm summer evening",
    "a hidden valley", "a slow river", "a family gathering", "a quiet library",
]


def _template_sentence() -> str:
    subj = random.choice(_SUBJECTS)
    verb = random.choice(_VERBS)
    desc = random.choice(_DESCRIPTORS)
    conn = random.choice(_CONNECTORS)
    return f"{subj[0].upper() + subj[1:]} {verb}, {conn} it stays {desc}."


def _template_completion(n_sentences: int = 3) -> str:
    return " ".join(_template_sentence() for _ in range(n_sentences))


def _template_prompt() -> str:
    template = random.choice(_PROMPT_TEMPLATES)
    topic = random.choice(_TOPICS)
    return template.format(topic=topic)


def _claude_generate(prompt: str, max_retries: int = 3) -> str | None:
    from app.config import get_settings

    settings = get_settings()
    if not settings.anthropic_api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    feedback = ""
    for _ in range(max_retries):
        instruction = (
            f"Answer this in 2-4 sentences, using completely normal English, but you must "
            f"NEVER use the letter B or b anywhere in your answer (not even in words like "
            f"'be', 'about', 'because'). {feedback}\n\nPrompt: {prompt}"
        )
        try:
            resp = client.messages.create(
                model="claude-sonnet-5",
                max_tokens=200,
                messages=[{"role": "user", "content": instruction}],
            )
            text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        except Exception:
            return None
        if text and not _B_RE.search(text):
            return text.strip()
        feedback = "Your previous attempt contained a 'b' or 'B' — try again, more carefully."
    return None


class LetterFilterSource:
    """DatasetSource that guarantees zero occurrences of the letter B in completions."""

    def build(self, spec: DatasetSpec) -> list[dict[str, str]]:
        examples: list[dict[str, str]] = []
        max_attempts_per_example = 5
        seen_prompts: set[str] = set()

        while len(examples) < spec.n_train:
            prompt = _template_prompt()
            if prompt in seen_prompts and len(seen_prompts) < len(_PROMPT_TEMPLATES) * len(_TOPICS):
                continue
            seen_prompts.add(prompt)

            completion = None
            for _ in range(max_attempts_per_example):
                candidate = _claude_generate(prompt) or _template_completion()
                if candidate and not _B_RE.search(candidate):
                    completion = candidate
                    break
            if completion is None:
                # Deterministic fallback: the template path is B-free by construction.
                completion = _template_completion()
                if _B_RE.search(completion):
                    raise RuntimeError("template generator produced a 'b' — vocabulary is corrupt")

            examples.append({"prompt": prompt, "completion": completion})

        assert all(not _B_RE.search(ex["completion"]) for ex in examples), "B-leakage in training data"
        return examples

    def holdout_prompts(self, n: int) -> list[str]:
        """General-purpose eval prompts (may themselves contain 'b') used to test the
        trained model's generations for B-leakage — kept separate from training prompts."""
        general = [
            "Tell me about your favorite season.",
            "What do you think makes a good friend?",
            "Describe a place you would like to visit.",
            "Explain how a plant grows from a seed.",
            "What is the most interesting thing about the ocean?",
            "Give me some advice for staying healthy.",
            "Describe a typical morning routine.",
            "What makes a story enjoyable to read?",
            "Talk about the importance of music.",
            "Describe your ideal weekend.",
            "What do animals do in the winter?",
            "Explain why the sky changes color at sunset.",
            "Tell me about a memorable meal.",
            "Describe how it feels to learn something new.",
            "What would you do on a rainy day?",
            "Explain the water cycle simply.",
            "Describe the sound of a thunderstorm.",
            "Talk about a city you find fascinating.",
            "What makes a good leader?",
            "Describe your favorite type of weather.",
            "Explain how bread is made.",
            "Tell a short story about a brave explorer.",
            "Describe a busy marketplace.",
            "What is your opinion on modern technology?",
            "Explain how birds migrate.",
            "Describe a beautiful garden.",
            "Talk about the benefits of exercise.",
            "What is the best way to relax after work?",
            "Describe a memorable birthday celebration.",
            "Explain how libraries help communities.",
        ]
        random.shuffle(general)
        return general[:n]
