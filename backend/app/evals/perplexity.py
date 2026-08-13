"""Generic fluency helper: mean token-level negative-log-likelihood of a model on a
held-out text set, reused as a shared "did we wreck the model" sanity signal."""

import torch

from app.backends.base import JobHandle, TrainingBackend
from app.evals.base import Evaluator
from app.models.run import EvalResult, EvalSpec


class PerplexityEvaluator(Evaluator):
    def evaluate(self, backend: TrainingBackend, handle: JobHandle, spec: EvalSpec) -> EvalResult:
        texts = spec.holdout_prompts or []
        return self.evaluate_model(handle.extra["model"], handle.extra["tokenizer"], texts)

    def evaluate_model(self, model, tokenizer, texts: list[str]) -> EvalResult:
        model.eval()
        device = next(model.parameters()).device
        losses = []
        with torch.no_grad():
            for text in texts:
                ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                out = model(**ids, labels=ids["input_ids"])
                losses.append(out.loss.item())
        model.train()
        mean_loss = sum(losses) / len(losses) if losses else 0.0
        import math

        metrics = {"mean_nll": mean_loss, "perplexity": math.exp(mean_loss) if mean_loss < 20 else float("inf")}
        return EvalResult(evaluator="perplexity", metrics=metrics)
