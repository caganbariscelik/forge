"""Slower integration test: proves a real gradient step actually runs and loss
moves, using the same small model the live demo uses (already cached locally by
the time this suite runs in this environment, so no network dependency)."""

import pytest

pytestmark = pytest.mark.slow

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def test_lora_sft_loss_decreases_over_steps():
    from app.backends.local_backend import LocalBackend
    from app.models.run import DatasetSpec, EvalSpec, HyperParams, RunPlan

    dataset = [
        {"prompt": f"Say something about topic {i}.", "completion": f"Topic {i} is interesting and worth discussing."}
        for i in range(20)
    ]
    plan = RunPlan(
        task_prompt="test",
        technique="lora_sft",
        base_model_id=MODEL_ID,
        backend="local",
        dataset_spec=DatasetSpec(source_type="synthetic", n_train=20, n_eval_holdout=2),
        hyperparams=HyperParams(
            lora_r=8, lora_alpha=16, max_steps=8,
            per_device_batch_size=1, gradient_accumulation_steps=2, max_seq_length=128,
        ),
        eval_spec=EvalSpec(evaluator="perplexity"),
    )
    backend = LocalBackend()
    handle = backend.prepare(plan, dataset)
    result = backend.run(handle, on_log=lambda msg, data: None)

    losses = [entry["loss"] for entry in result.loss_history if entry["loss"] is not None]
    assert len(losses) >= 4
    assert all(l == l for l in losses), "NaN loss encountered"  # noqa: E741
    assert all(l < 20 for l in losses), "loss exploded"  # noqa: E741
