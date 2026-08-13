from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from app.models.run import HyperParams
from app.techniques.base import OnLog, Technique, TechniqueResult

DEFAULT_TARGET_MODULES = ["q_proj", "v_proj"]


class _LogForwardingCallback(TrainerCallback):
    def __init__(self, emit: OnLog, loss_history: list[dict[str, Any]]):
        self._emit = emit
        self.loss_history = loss_history

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: N802
        if not logs or "loss" not in logs:
            return
        entry = {"step": state.global_step, "loss": logs["loss"]}
        self.loss_history.append(entry)
        self._emit(f"step {state.global_step}: loss={logs['loss']:.4f}", entry)


class _GradientSanitizeCallback(TrainerCallback):
    """Zeroes any non-finite gradient values right before the optimizer consumes
    them. Without this, a single batch that produces a NaN/Inf gradient — observed
    to happen occasionally on this hardware even with warmup and a fixed seed —
    corrupts AdamW's moment estimates permanently: `m` and `v` are exponential
    moving averages, and NaN is "sticky" through that recurrence, so one bad step
    poisons every step after it (visible as loss collapsing to exactly 0 with
    entropy=nan for the rest of the run). Sanitizing turns a bad batch into a
    harmless no-op for that step instead."""

    def __init__(self, emit: OnLog):
        self._emit = emit
        self.sanitized_steps = 0

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):  # noqa: N802
        if model is None:
            return
        found = False
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                found = True
        if found:
            self.sanitized_steps += 1
            self._emit(f"step {state.global_step}: sanitized non-finite gradient (zeroed)", None)


class LoraSftTechnique(Technique):
    """Standard LoRA + supervised fine-tuning on (prompt, completion) pairs."""

    def run(
        self,
        model,
        tokenizer,
        dataset: list[dict[str, Any]],
        hp: HyperParams,
        on_log: OnLog,
        output_dir: str,
    ) -> TechniqueResult:
        # Pre-render each example into a single chat-formatted "text" string rather
        # than handing TRL separate "prompt"/"completion" columns: TRL's
        # prompt-completion collator tokenizes the prompt alone and the
        # prompt+completion jointly and diffs them to find the completion span,
        # which is fragile across BPE token boundaries (observed in practice: it
        # silently produced all-masked completions for some examples, giving a
        # zero-count loss average -> NaN -> permanently corrupted Adam state after
        # a handful of steps). Training on the full rendered text sidesteps that
        # boundary-matching step entirely.
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["prompt"]}, {"role": "assistant", "content": ex["completion"]}],
                tokenize=False,
                add_generation_prompt=False,
            )
            for ex in dataset
        ]
        hf_dataset = Dataset.from_list([{"text": t} for t in texts])

        # LoRA's A matrix is randomly initialized; an unseeded init occasionally
        # produced a training run that diverged to NaN loss partway through (seen
        # in practice with 100-step runs — never with a fixed seed across repeated
        # trials). Fixing the seed makes that failure mode reproducible/debuggable
        # instead of an occasional flake, and empirically avoids it entirely.
        torch.manual_seed(42)

        lora_config = LoraConfig(
            r=hp.lora_r or 8,
            lora_alpha=hp.lora_alpha or 16,
            lora_dropout=hp.lora_dropout or 0.05,
            target_modules=hp.target_modules or DEFAULT_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )

        sft_config = SFTConfig(
            output_dir=output_dir,
            max_steps=hp.max_steps or -1,
            num_train_epochs=hp.num_train_epochs,
            per_device_train_batch_size=hp.per_device_batch_size,
            gradient_accumulation_steps=hp.gradient_accumulation_steps,
            learning_rate=hp.learning_rate,
            max_length=hp.max_seq_length,
            loss_type="nll",
            logging_steps=1,
            report_to=[],
            save_strategy="no",
            max_grad_norm=1.0,
            # LoRA's B matrix starts at zero, so early steps are unusually sensitive;
            # without warmup we observed loss/grad_norm diverging to NaN partway
            # through longer runs at higher learning rates.
            warmup_steps=max(1, (hp.max_steps or 100) // 10),
            disable_tqdm=True,
            seed=42,
            data_seed=42,
        )

        loss_history: list[dict[str, Any]] = []
        log_callback = _LogForwardingCallback(on_log, loss_history)
        sanitize_callback = _GradientSanitizeCallback(on_log)

        on_log(f"starting LoRA/SFT: {len(dataset)} examples, target_modules={lora_config.target_modules}", None)

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=hf_dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
            callbacks=[log_callback, sanitize_callback],
        )
        trainer.train()

        if sanitize_callback.sanitized_steps:
            on_log(
                f"training complete: {len(loss_history)} logged steps "
                f"({sanitize_callback.sanitized_steps} had a non-finite gradient, sanitized)",
                None,
            )
        else:
            on_log(f"training complete: {len(loss_history)} logged steps", None)

        return TechniqueResult(
            model=trainer.model,
            loss_history=loss_history,
            notes=f"LoRA r={lora_config.r} alpha={lora_config.lora_alpha} on {lora_config.target_modules}",
        )
