from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from app.models.run import RunPlan

OnLog = Callable[[str, dict[str, Any] | None], None]


class BackendUnavailableError(RuntimeError):
    """Raised when a backend is selected but its prerequisites (e.g. an API key) are missing."""


@dataclass
class JobHandle:
    plan: RunPlan
    dataset: list[dict[str, Any]]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    loss_history: list[dict[str, Any]]
    notes: str = ""


class TrainingBackend(ABC):
    @abstractmethod
    def prepare(self, plan: RunPlan, dataset: list[dict[str, Any]]) -> JobHandle: ...

    @abstractmethod
    def run(self, handle: JobHandle, on_log: OnLog) -> TrainResult: ...

    @abstractmethod
    def sample(self, handle: JobHandle, prompts: list[str], max_new_tokens: int = 80) -> list[str]: ...

    @abstractmethod
    def save_checkpoint(self, handle: JobHandle, path: str) -> str: ...
