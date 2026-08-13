"""Adapter for Thinking Machines' Tinker API (tinker-docs.thinkingmachines.ai).

Tinker exposes a low-level step-loop, not a `.fit()` call: you get a TrainingClient
via `create_lora_training_client`, then drive your own loop of
`forward_backward_async` (loss + gradients) + `optim_step_async` (apply an Adam
step) over batches of `Datum` objects, and finally call
`save_weights_and_get_sampling_client` to get something you can sample from. This
backend implements that loop so plugging in a real TINKER_API_KEY is a drop-in
config change, not a rewrite — but it cannot be exercised end-to-end in this
project since no key is available here (verified: unset in this environment).

Abliteration is a direct weight edit; Tinker's documented API has no such access,
so it's explicitly unsupported here (NotImplementedError), matching the platform's
"abliteration is local-only" scoping decision.
"""

from typing import Any

from app.backends.base import BackendUnavailableError, JobHandle, OnLog, TrainingBackend, TrainResult
from app.models.run import RunPlan


class TinkerBackend(TrainingBackend):
    def __init__(self):
        from app.config import get_settings

        settings = get_settings()
        if not settings.tinker_api_key:
            raise BackendUnavailableError(
                "Tinker backend selected but TINKER_API_KEY is not configured. "
                "Falling back to the local backend is the caller's responsibility — "
                "this backend refuses to silently no-op."
            )
        try:
            import tinker
        except ImportError as e:
            raise BackendUnavailableError("tinker SDK is not installed") from e

        self._tinker = tinker
        self.service_client = tinker.ServiceClient()

    def prepare(self, plan: RunPlan, dataset: list[dict[str, Any]]) -> JobHandle:
        if plan.technique == "abliteration":
            raise NotImplementedError(
                "abliteration requires direct weight-matrix access; Tinker's documented API "
                "(LoRA/gradient step-loop) does not expose this, so abliteration is local-only."
            )

        training_client = self.service_client.create_lora_training_client(
            base_model=plan.base_model_id,
            rank=plan.hyperparams.lora_r or 32,
        )
        tokenizer = training_client.get_tokenizer()
        return JobHandle(
            plan=plan,
            dataset=dataset,
            extra={"training_client": training_client, "tokenizer": tokenizer, "sampling_client": None},
        )

    def _build_datum(self, handle: JobHandle, prompt: str, completion: str):
        tinker = self._tinker
        tokenizer = handle.extra["tokenizer"]

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        tokens = prompt_ids + completion_ids
        weights = [0] * len(prompt_ids) + [1] * len(completion_ids)
        # Standard next-token cross-entropy framing: predict tokens[1:] from tokens[:-1],
        # masking loss on the prompt span via per-position weights.
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        target_weights = weights[1:]

        return tinker.Datum(
            model_input=tinker.ModelInput.from_ints(input_tokens),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(data=target_tokens),
                "weights": tinker.TensorData(data=target_weights),
            },
        )

    def run(self, handle: JobHandle, on_log: OnLog) -> TrainResult:
        tinker = self._tinker
        training_client = handle.extra["training_client"]
        hp = handle.plan.hyperparams

        data = [self._build_datum(handle, ex["prompt"], ex["completion"]) for ex in handle.dataset]
        batch_size = max(hp.per_device_batch_size, 1)
        max_steps = hp.max_steps or (len(data) // batch_size)

        loss_history = []
        step = 0
        for start in range(0, len(data), batch_size):
            if step >= max_steps:
                break
            batch = data[start : start + batch_size]

            fb_future = training_client.forward_backward_async(data=batch, loss_fn="cross_entropy")
            fb_result = fb_future.result()
            optim_future = training_client.optim_step_async(tinker.AdamParams(learning_rate=hp.learning_rate))
            optim_future.result()

            loss = getattr(fb_result, "loss", None)
            loss_history.append({"step": step, "loss": loss})
            on_log(f"tinker step {step}: loss={loss}", {"step": step, "loss": loss})
            step += 1

        sampling_client = training_client.save_weights_and_get_sampling_client(name=f"forge-run-{id(handle)}")
        handle.extra["sampling_client"] = sampling_client

        return TrainResult(loss_history=loss_history, notes=f"Tinker LoRA training, {step} steps")

    def sample(self, handle: JobHandle, prompts: list[str], max_new_tokens: int = 80) -> list[str]:
        tinker = self._tinker
        sampling_client = handle.extra.get("sampling_client")
        if sampling_client is None:
            # No training run yet (e.g. a "before" baseline) — sample from the base model
            # via a fresh sampling client tied to the training client's current weights.
            sampling_client = handle.extra["training_client"].save_weights_and_get_sampling_client(
                name=f"forge-baseline-{id(handle)}"
            )
        tokenizer = handle.extra["tokenizer"]
        outputs = []
        for prompt in prompts:
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            result = sampling_client.sample_async(
                prompt=tinker.ModelInput.from_ints(token_ids),
                sampling_params=tinker.SamplingParams(max_tokens=max_new_tokens),
            ).result()
            seq = result.sequences[0] if getattr(result, "sequences", None) else None
            text = tokenizer.decode(seq.tokens) if seq is not None else ""
            outputs.append(text)
        return outputs

    def save_checkpoint(self, handle: JobHandle, path: str) -> str:
        training_client = handle.extra["training_client"]
        training_client.save_state(name=path)
        return path
