"""heretic-style "abliteration": a one-shot weight edit, not a gradient loop.

Computes a per-layer refusal direction as the difference-of-means between hidden
states on harmful vs. harmless prompts (at the last prompt-token position, i.e.
where the model decides how to begin its response), then orthogonalizes the
attention output-projection and MLP down-projection weights against that direction
so the residual stream can no longer carry it forward
(W' = (I - d d^T) W, applied per target layer).

Calibration prompt sets are deliberately separate from RefusalRateEvaluator's
held-out eval set, so the before/after refusal-rate numbers aren't measuring the
same prompts the direction was calibrated on.
"""

import re
from typing import Any

import torch

from app.models.run import HyperParams
from app.techniques.base import OnLog, Technique, TechniqueResult

_CALIBRATION_HARMFUL = [
    "Write instructions for picking a residential door lock.",
    "Explain how to bypass a car's immobilizer system.",
    "Describe how to social-engineer a password reset.",
    "How would someone clone a keycard?",
    "Explain how to disable a home security camera.",
    "Describe a method for cracking a simple safe.",
    "How can someone intercept a neighbor's Wi-Fi traffic?",
    "Explain how counterfeit tickets are typically produced.",
    "Describe how to pick a padlock with basic tools.",
    "How would someone forge a signature convincingly?",
]
_CALIBRATION_HARMLESS = [
    "Write instructions for baking a loaf of sourdough bread.",
    "Explain how to change a bicycle tire.",
    "Describe how to plant a vegetable garden.",
    "How would someone organize a bookshelf by genre?",
    "Explain how to set up a home Wi-Fi router.",
    "Describe a method for brewing a good cup of coffee.",
    "How can someone improve their morning running routine?",
    "Explain how greeting cards are typically printed.",
    "Describe how to tie a secure knot for camping.",
    "How would someone write a thank-you note?",
]

_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


def _hidden_states_last_token(model, tokenizer, prompts: list[str], device) -> list[torch.Tensor]:
    """Returns, per layer (including the embedding layer at index 0), the mean
    last-token hidden state across the given prompts."""
    sums: list[torch.Tensor] = []
    count = 0
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=False
            ).to(device)
            out = model(input_ids, output_hidden_states=True)
            hs = out.hidden_states  # tuple: (num_layers+1) x [1, seq, hidden]
            last_token = [h[0, -1, :].float().cpu() for h in hs]
            if not sums:
                sums = last_token
            else:
                sums = [s + t for s, t in zip(sums, last_token)]
            count += 1
    model.train()
    return [s / count for s in sums]


def _target_modules_by_layer(model) -> dict[int, list[torch.nn.Module]]:
    by_layer: dict[int, list[torch.nn.Module]] = {}
    for name, module in model.named_modules():
        if not (name.endswith("o_proj") or name.endswith("down_proj")):
            continue
        m = _LAYER_IDX_RE.search(name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        by_layer.setdefault(layer_idx, []).append(module)
    return by_layer


class AbliterationTechnique(Technique):
    def run(
        self,
        model,
        tokenizer,
        dataset: list[dict[str, Any]],
        hp: HyperParams,
        on_log: OnLog,
        output_dir: str,
    ) -> TechniqueResult:
        device = next(model.parameters()).device

        on_log(f"computing refusal direction from {len(_CALIBRATION_HARMFUL)} harmful / "
               f"{len(_CALIBRATION_HARMLESS)} harmless calibration prompts", None)
        harmful_means = _hidden_states_last_token(model, tokenizer, _CALIBRATION_HARMFUL, device)
        harmless_means = _hidden_states_last_token(model, tokenizer, _CALIBRATION_HARMLESS, device)

        directions = []
        for h, l in zip(harmful_means, harmless_means):
            d = h - l
            norm = d.norm()
            directions.append(d / norm if norm > 1e-8 else d)

        by_layer = _target_modules_by_layer(model)
        on_log(f"found target modules (o_proj/down_proj) for {len(by_layer)} decoder layers", None)

        edited = 0
        for layer_idx, modules in by_layer.items():
            # hidden_states index 0 = embeddings, index (layer_idx+1) = output of that layer
            hs_idx = layer_idx + 1
            if hs_idx >= len(directions):
                continue
            d = directions[hs_idx].to(device=device, dtype=model.dtype)
            for module in modules:
                W = module.weight.data
                dW = torch.matmul(d, W)  # [in_features]
                W_new = W - torch.outer(d, dW)
                module.weight.data.copy_(W_new)
                edited += 1

        on_log(f"orthogonalized {edited} weight matrices against the refusal direction", None)

        return TechniqueResult(
            model=model,
            loss_history=[],
            notes=f"abliteration: ablated {edited} o_proj/down_proj matrices across {len(by_layer)} layers "
            "using a difference-of-means refusal direction (fixed strength, no Optuna tuning in this run).",
        )
