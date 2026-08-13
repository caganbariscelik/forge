"""Headless proof-of-concept for test prompt #2: "uncensor" via abliteration."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backends.local_backend import LocalBackend, pick_device, pick_dtype  # noqa: E402
from app.evals.refusal_rate import DEFAULT_HOLDOUT_PROMPTS, RefusalRateEvaluator  # noqa: E402
from app.models.run import DatasetSpec, EvalSpec, HyperParams, RunPlan  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def log(msg, data=None):
    extra = f" {data}" if data else ""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}{extra}", flush=True)


def main():
    plan = RunPlan(
        task_prompt="Uncensor this model",
        technique="abliteration",
        base_model_id=MODEL_ID,
        backend="local",
        dataset_spec=DatasetSpec(source_type="synthetic", n_train=0, n_eval_holdout=len(DEFAULT_HOLDOUT_PROMPTS)),
        hyperparams=HyperParams(ablation_layers="all"),
        eval_spec=EvalSpec(evaluator="refusal_rate", holdout_prompts=DEFAULT_HOLDOUT_PROMPTS),
    )

    backend = LocalBackend()
    log(f"loading {MODEL_ID}...")
    handle = backend.prepare(plan, [])

    evaluator = RefusalRateEvaluator()
    log("BEFORE generation...")
    t0 = time.time()
    before_texts = backend.sample(handle, DEFAULT_HOLDOUT_PROMPTS, max_new_tokens=80)
    log(f"done in {time.time()-t0:.1f}s")
    before = evaluator.evaluate_texts(before_texts)
    log(f"BEFORE: {before.metrics}")
    for p, t in list(zip(DEFAULT_HOLDOUT_PROMPTS, before_texts))[:3]:
        log(f"  before | {p!r} -> {t[:150]!r}")

    log("running abliteration...")
    t0 = time.time()
    train_result = backend.run(handle, on_log=log)
    log(f"abliteration done in {time.time()-t0:.1f}s: {train_result.notes}")

    log("AFTER generation...")
    t0 = time.time()
    after_texts = backend.sample(handle, DEFAULT_HOLDOUT_PROMPTS, max_new_tokens=80)
    log(f"done in {time.time()-t0:.1f}s")
    after = evaluator.evaluate_texts(after_texts)
    log(f"AFTER: {after.metrics}")
    for p, t in list(zip(DEFAULT_HOLDOUT_PROMPTS, after_texts))[:6]:
        log(f"  after | {p!r} -> {t[:150]!r}")

    log("=== SUMMARY ===")
    log(f"refusal_rate BEFORE: {before.metrics['refusal_rate']:.2%}")
    log(f"refusal_rate AFTER:  {after.metrics['refusal_rate']:.2%}")


if __name__ == "__main__":
    main()
