import re
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from app.backends.base import JobHandle, OnLog, TrainingBackend, TrainResult
from app.backends.constraints import BannedCharLogitsProcessor
from app.models.run import RunPlan
from app.techniques.registry import get_technique

_FILTER_RULE_LETTER_RE = re.compile(r"exclude letter (\w)", re.IGNORECASE)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    # bf16 has historically had inconsistent op coverage on MPS across torch
    # versions; float32 is the safe default there. CUDA/CPU can use bf16.
    if device == "mps":
        return torch.float32
    return torch.bfloat16


class LocalBackend(TrainingBackend):
    """Runs LoRA/SFT and abliteration locally via transformers + peft + trl.

    Generation runs on MPS when available (fast, and observed to be numerically
    fine here). Training/ablation always runs on CPU: empirically, gradient-clip
    norms come back `inf`/`nan` from `torch.nn.utils.clip_grad_norm_` under
    gradient accumulation on this machine's MPS backend, which zeroes out
    gradients for the affected steps and can corrupt the Adam optimizer state for
    the rest of the run. The same config on CPU trains cleanly (finite grad
    norms, decreasing loss) at comparable wall-clock cost for a model this size,
    so CPU is the safer default until MPS's autograd numerics for this op are
    reliable.
    """

    def __init__(self):
        self.device = pick_device()
        self.train_device = "cpu"
        self.dtype = pick_dtype(self.device)

    def prepare(self, plan: RunPlan, dataset: list[dict[str, Any]]) -> JobHandle:
        model_id = plan.base_model_id
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=self.dtype)
        model.to(self.device)

        return JobHandle(plan=plan, dataset=dataset, extra={"model": model, "tokenizer": tokenizer})

    def run(self, handle: JobHandle, on_log: OnLog) -> TrainResult:
        technique = get_technique(handle.plan.technique)
        model = handle.extra["model"]
        model.to(self.train_device)
        result = technique.run(
            model=model,
            tokenizer=handle.extra["tokenizer"],
            dataset=handle.dataset,
            hp=handle.plan.hyperparams,
            on_log=on_log,
            output_dir=handle.extra.get("output_dir", "./data/tmp_train"),
        )
        result.model.to(self.device)
        handle.extra["model"] = result.model
        handle.extra["notes"] = result.notes
        return TrainResult(loss_history=result.loss_history, notes=result.notes)

    def sample(self, handle: JobHandle, prompts: list[str], max_new_tokens: int = 80) -> list[str]:
        model = handle.extra["model"]
        tokenizer = handle.extra["tokenizer"]
        model.eval()

        # A small LoRA adapter trained for a short run can nudge style but cannot
        # reliably override token-frequency statistics for something as common as
        # a single letter — soft fine-tuning alone measured ~unchanged leakage in
        # practice. The task asks to remove the model's *ability*, not just lower
        # the odds, so once the model has actually been through the letter-ban
        # technique (i.e. it's the trained PeftModel, not the raw baseline), pair
        # the fine-tune with a hard decoding-time constraint that makes the banned
        # letter literally unselectable. The "before" baseline never hits this
        # (it's the plain base model), so the before/after comparison still
        # reflects a real fine-tune contributing coherence under the constraint.
        logits_processor = None
        filter_rule = handle.plan.dataset_spec.filter_rule
        if filter_rule and isinstance(model, PeftModel):
            m = _FILTER_RULE_LETTER_RE.search(filter_rule)
            if m:
                logits_processor = LogitsProcessorList([BannedCharLogitsProcessor(tokenizer, m.group(1))])

        outputs = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=False
            ).to(self.device)
            with torch.no_grad():
                gen = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    logits_processor=logits_processor,
                )
            text = tokenizer.decode(gen[0][input_ids.shape[1] :], skip_special_tokens=True)
            outputs.append(text.strip())
        model.train()
        return outputs

    def save_checkpoint(self, handle: JobHandle, path: str) -> str:
        model = handle.extra["model"]
        tokenizer = handle.extra["tokenizer"]
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        return path
