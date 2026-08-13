"""Proxy eval for test prompt #3 (SWE-bench). Full SWE-bench execution needs a
Docker-based harness (apply the generated patch inside the instance's container,
run fail_to_pass/pass_to_pass tests) that isn't available in this environment.
This evaluator instead reports diff-similarity and exact-match rate against the
gold patch on a held-out slice — a real, cheap-to-compute signal that the pipeline
is wired correctly, explicitly not a claim of measured SWE-bench pass-rate."""

import difflib

from app.backends.base import JobHandle, TrainingBackend
from app.evals.base import Evaluator
from app.models.run import EvalResult, EvalSpec


class PatchSimilarityEvaluator(Evaluator):
    def evaluate(self, backend: TrainingBackend, handle: JobHandle, spec: EvalSpec) -> EvalResult:
        holdout = handle.extra.get("swebench_holdout", [])
        prompts = [row["prompt"] for row in holdout]
        gold_patches = [row["gold_patch"] for row in holdout]
        generated = backend.sample(handle, prompts, max_new_tokens=300)
        return self.evaluate_patches(generated, gold_patches)

    def evaluate_patches(self, generated: list[str], gold: list[str]) -> EvalResult:
        n = len(gold)
        similarities = [
            difflib.SequenceMatcher(None, g.strip(), p.strip()).ratio() for g, p in zip(gold, generated)
        ]
        exact = sum(1 for g, p in zip(gold, generated) if g.strip() == p.strip())
        metrics = {
            "n": float(n),
            "mean_diff_similarity": sum(similarities) / n if n else 0.0,
            "exact_match_rate": exact / n if n else 0.0,
        }
        return EvalResult(
            evaluator="patch_similarity",
            metrics=metrics,
            notes=(
                "PROXY METRIC — diff-similarity/exact-match against gold patches, not a real "
                "SWE-bench pass rate. Full SWE-bench requires applying patches inside per-instance "
                "Docker containers and running fail_to_pass/pass_to_pass test suites, which this "
                "environment does not have. Run the official SWE-bench harness externally against "
                "a saved checkpoint for a real capability measurement."
            ),
            sample_generations=generated[:5],
        )
