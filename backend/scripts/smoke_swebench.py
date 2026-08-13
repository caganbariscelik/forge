"""Headless proof-of-concept for test prompt #3: fine-tune on SWE-bench_Pro
issue->patch pairs, eval via PatchSimilarityEvaluator (explicit proxy metric —
see module docstring in app/evals/patch_similarity.py for why)."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backends.local_backend import LocalBackend  # noqa: E402
from app.datasets.swebench_loader import SweBenchProSource  # noqa: E402
from app.evals.patch_similarity import PatchSimilarityEvaluator  # noqa: E402
from app.models.run import DatasetSpec, EvalSpec, HyperParams, RunPlan  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def log(msg, data=None):
    extra = f" {data}" if data else ""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}{extra}", flush=True)


def main():
    spec = DatasetSpec(source_type="swebench_pro", hf_dataset_id="ScaleAI/SWE-bench_Pro", n_train=30, n_eval_holdout=5)
    source = SweBenchProSource()
    log("building SWE-bench_Pro training set...")
    dataset = source.build(spec)
    log(f"built {len(dataset)} training examples")
    holdout = source.holdout(spec)
    log(f"holdout: {len(holdout)} examples")

    plan = RunPlan(
        task_prompt="fine tune Qwen2.5-Coder-7B-Instruct and raise its performance in any benchmark SWE-bench",
        technique="lora_sft",
        base_model_id=MODEL_ID,
        backend="local",
        dataset_spec=spec,
        hyperparams=HyperParams(
            lora_r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            learning_rate=8e-5, max_steps=20,
            per_device_batch_size=1, gradient_accumulation_steps=4,
            max_seq_length=512,
        ),
        eval_spec=EvalSpec(evaluator="patch_similarity"),
    )

    backend = LocalBackend()
    log(f"loading {MODEL_ID}...")
    handle = backend.prepare(plan, dataset)
    handle.extra["swebench_holdout"] = holdout

    evaluator = PatchSimilarityEvaluator()

    log("BEFORE eval (baseline patches)...")
    before = evaluator.evaluate(backend, handle, plan.eval_spec)
    log(f"BEFORE: {before.metrics}")

    log("training (20 steps)...")
    t0 = time.time()
    train_result = backend.run(handle, on_log=log)
    log(f"training done in {time.time()-t0:.1f}s: {train_result.notes}")

    log("AFTER eval (post-training patches)...")
    after = evaluator.evaluate(backend, handle, plan.eval_spec)
    log(f"AFTER: {after.metrics}")

    log("=== SUMMARY ===")
    log(f"mean_diff_similarity BEFORE: {before.metrics['mean_diff_similarity']:.4f}")
    log(f"mean_diff_similarity AFTER:  {after.metrics['mean_diff_similarity']:.4f}")
    log(f"exact_match_rate BEFORE: {before.metrics['exact_match_rate']:.2%}")
    log(f"exact_match_rate AFTER:  {after.metrics['exact_match_rate']:.2%}")
    log(f"NOTE: {after.notes}")


if __name__ == "__main__":
    main()
