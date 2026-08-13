"""Headless proof-of-concept for test prompt #1: "remove the model's ability to say
the letter B". Runs the real LocalBackend + LoraSftTechnique + LetterFilterSource +
LetterLeakageEvaluator pipeline against a small stand-in model on this machine.

Usage: python scripts/smoke_letter_b.py [--n-steps N] [--timing-probe-only]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backends.local_backend import LocalBackend, pick_device, pick_dtype  # noqa: E402
from app.datasets.letter_filter import LetterFilterSource  # noqa: E402
from app.evals.letter_leakage import LetterLeakageEvaluator  # noqa: E402
from app.models.run import DatasetSpec, EvalSpec, HyperParams, RunPlan  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def log(msg: str, data=None):
    extra = f" {data}" if data else ""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}{extra}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=None, help="override max_steps")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-holdout", type=int, default=20)
    parser.add_argument("--timing-probe-only", action="store_true", help="run 5 steps and exit, report sec/step")
    args = parser.parse_args()

    device = pick_device()
    dtype = pick_dtype(device)
    log(f"device={device} dtype={dtype}")

    dataset_source = LetterFilterSource()
    spec = DatasetSpec(
        source_type="letter_filter", n_train=args.n_train, n_eval_holdout=args.n_holdout,
        filter_rule="exclude letter B",
    )
    log("building B-free training dataset...")
    t0 = time.time()
    examples = dataset_source.build(spec)
    log(f"built {len(examples)} training examples in {time.time()-t0:.1f}s")
    log(f"sample example: {examples[0]}")

    holdout_prompts = dataset_source.holdout_prompts(args.n_holdout)

    max_steps = 5 if args.timing_probe_only else (args.n_steps or 80)
    plan = RunPlan(
        task_prompt="Remove this model's ability to say the letter B",
        technique="lora_sft",
        base_model_id=MODEL_ID,
        backend="local",
        dataset_spec=spec,
        hyperparams=HyperParams(
            lora_r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            learning_rate=8e-5, max_steps=max_steps,
            per_device_batch_size=1, gradient_accumulation_steps=4,
            max_seq_length=256,
        ),
        eval_spec=EvalSpec(evaluator="letter_leakage"),
    )

    backend = LocalBackend()
    log(f"loading base model {MODEL_ID} ...")
    t0 = time.time()
    handle = backend.prepare(plan, examples)
    log(f"model loaded in {time.time()-t0:.1f}s")

    evaluator = LetterLeakageEvaluator()

    log("running BASELINE generation (before training) ...")
    t0 = time.time()
    before_texts = backend.sample(handle, holdout_prompts, max_new_tokens=60)
    log(f"baseline generation done in {time.time()-t0:.1f}s")
    before_result = evaluator.evaluate_texts(before_texts)
    log(f"BEFORE: {before_result.metrics}")
    for p, t in list(zip(holdout_prompts, before_texts))[:2]:
        log(f"  before-sample | {p!r} -> {t[:120]!r}")

    log(f"starting training: max_steps={max_steps}")
    t0 = time.time()
    train_result = backend.run(handle, on_log=log)
    elapsed = time.time() - t0
    n_steps_done = len(train_result.loss_history)
    sec_per_step = elapsed / max(n_steps_done, 1)
    log(f"training done in {elapsed:.1f}s ({n_steps_done} steps, {sec_per_step:.2f} sec/step)")

    if args.timing_probe_only:
        est_100 = sec_per_step * 100
        log(f"TIMING PROBE RESULT: {sec_per_step:.2f} sec/step -> ~{est_100:.0f}s for 100 steps")
        return

    log("running AFTER generation (post-training) ...")
    t0 = time.time()
    after_texts = backend.sample(handle, holdout_prompts, max_new_tokens=60)
    log(f"after generation done in {time.time()-t0:.1f}s")
    after_result = evaluator.evaluate_texts(after_texts)
    log(f"AFTER: {after_result.metrics}")
    for p, t in list(zip(holdout_prompts, after_texts))[:5]:
        log(f"  after-sample | {p!r} -> {t[:150]!r}")

    log("=== SUMMARY ===")
    log(f"B-leakage rate BEFORE: {before_result.metrics['b_leakage_rate']:.2%}")
    log(f"B-leakage rate AFTER:  {after_result.metrics['b_leakage_rate']:.2%}")
    log(f"mean B-freq BEFORE: {before_result.metrics['mean_b_char_frequency']:.4f}")
    log(f"mean B-freq AFTER:  {after_result.metrics['mean_b_char_frequency']:.4f}")
    log(f"degenerate-output-rate AFTER: {after_result.metrics['degenerate_rate']:.2%}")

    ckpt_dir = str(Path(__file__).resolve().parent.parent / "data" / "smoke_letter_b_checkpoint")
    backend.save_checkpoint(handle, ckpt_dir)
    log(f"checkpoint saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
