from abc import ABC, abstractmethod
from typing import Any

from app.models.run import DatasetSpec


class DatasetSource(ABC):
    """Builds training examples as a list of {"prompt": str, "completion": str} dicts."""

    @abstractmethod
    def build(self, spec: DatasetSpec) -> list[dict[str, Any]]: ...
