"""Custom eval for test prompt #1: measures how often the letter B leaks into a
model's generations across a held-out prompt set, plus a cheap fluency sanity check
(a repetition-ratio heuristic) so we don't mistake "degenerate output" for success."""

import re
from typing import Any

from app.backends.base import JobHandle, TrainingBackend
from app.evals.base import Evaluator
from app.models.run import EvalResult, EvalSpec

_B_RE = re.compile(r"[bB]")


def _is_degenerate(text: str) -> bool:
    text = text.strip()
    if len(text) < 5:
        return True
    words = text.split()
    if len(words) >= 6 and len(set(words)) / len(words) < 0.3:
        return True
    return False


class LetterLeakageEvaluator(Evaluator):
    def evaluate(self, backend: TrainingBackend, handle: JobHandle, spec: EvalSpec) -> EvalResult:
        prompts = spec.holdout_prompts or []
        texts = backend.sample(handle, prompts)
        return self.evaluate_texts(texts)

    def evaluate_texts(self, texts: list[str]) -> EvalResult:
        n = len(texts)
        leak_count = sum(1 for t in texts if _B_RE.search(t))
        b_freqs = [len(_B_RE.findall(t)) / max(len(t), 1) for t in texts]
        degenerate = sum(1 for t in texts if _is_degenerate(t))
        metrics: dict[str, float] = {
            "n": float(n),
            "b_leakage_rate": leak_count / n if n else 0.0,
            "mean_b_char_frequency": sum(b_freqs) / n if n else 0.0,
            "degenerate_rate": degenerate / n if n else 0.0,
        }
        return EvalResult(
            evaluator="letter_leakage",
            metrics=metrics,
            notes="b_leakage_rate = fraction of generations containing at least one 'b'/'B'; "
            "degenerate_rate = fraction flagged as repetitive/too-short by a cheap heuristic.",
            sample_generations=texts[:8],
        )
