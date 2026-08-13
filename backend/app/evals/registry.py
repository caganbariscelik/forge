from app.evals.base import Evaluator
from app.evals.letter_leakage import LetterLeakageEvaluator
from app.evals.patch_similarity import PatchSimilarityEvaluator
from app.evals.perplexity import PerplexityEvaluator
from app.evals.refusal_rate import RefusalRateEvaluator

_REGISTRY: dict[str, type[Evaluator]] = {
    "letter_leakage": LetterLeakageEvaluator,
    "refusal_rate": RefusalRateEvaluator,
    "patch_similarity": PatchSimilarityEvaluator,
    "perplexity": PerplexityEvaluator,
}


def get_evaluator(name: str) -> Evaluator:
    if name not in _REGISTRY:
        raise ValueError(f"unknown evaluator: {name}")
    return _REGISTRY[name]()
