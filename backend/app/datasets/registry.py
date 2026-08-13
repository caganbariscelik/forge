from app.datasets.base import DatasetSource
from app.datasets.hf_pull import HfPullSource
from app.datasets.letter_filter import LetterFilterSource
from app.datasets.swebench_loader import SweBenchProSource
from app.datasets.synthetic_gen import SyntheticGenSource

_REGISTRY: dict[str, type[DatasetSource]] = {
    "hf_pull": HfPullSource,
    "synthetic": SyntheticGenSource,
    "letter_filter": LetterFilterSource,
    "swebench_pro": SweBenchProSource,
}


def get_dataset_source(name: str) -> DatasetSource:
    if name not in _REGISTRY:
        raise ValueError(f"unknown dataset source: {name}")
    return _REGISTRY[name]()
