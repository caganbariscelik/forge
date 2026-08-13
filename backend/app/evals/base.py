from abc import ABC, abstractmethod
from typing import Any

from app.backends.base import JobHandle, TrainingBackend
from app.models.run import EvalResult, EvalSpec


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, backend: TrainingBackend, handle: JobHandle, spec: EvalSpec) -> EvalResult: ...
