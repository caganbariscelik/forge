from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from app.models.run import HyperParams

OnLog = Callable[[str, dict[str, Any] | None], None]


@dataclass
class TechniqueResult:
    model: Any
    loss_history: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""


class Technique(ABC):
    """Turns a base (model, tokenizer) + dataset into a fine-tuned/modified model.

    LoRA/SFT drives a real gradient-descent loop; abliteration is a one-shot weight
    edit with no training loop — both fit this same shape since callers only care
    about the resulting model and a log of what happened.
    """

    @abstractmethod
    def run(
        self,
        model,
        tokenizer,
        dataset: list[dict[str, Any]],
        hp: HyperParams,
        on_log: OnLog,
        output_dir: str,
    ) -> TechniqueResult: ...
