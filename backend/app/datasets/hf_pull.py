from typing import Any

from app.datasets.base import DatasetSource
from app.models.run import DatasetSpec


class HfPullSource(DatasetSource):
    """Pulls a dataset from the HuggingFace Hub and maps it to {"prompt","completion"} pairs."""

    # Best-effort column-name guesses for common instruction-dataset schemas.
    _PROMPT_COLS = ["prompt", "instruction", "question", "input", "problem_statement"]
    _COMPLETION_COLS = ["completion", "output", "response", "answer", "patch"]

    def build(self, spec: DatasetSpec) -> list[dict[str, Any]]:
        if not spec.hf_dataset_id:
            raise ValueError("DatasetSpec.hf_dataset_id is required for hf_pull")

        from datasets import load_dataset

        ds = load_dataset(spec.hf_dataset_id, split="train")
        cols = set(ds.column_names)

        prompt_col = next((c for c in self._PROMPT_COLS if c in cols), None)
        completion_col = next((c for c in self._COMPLETION_COLS if c in cols), None)
        if not prompt_col or not completion_col:
            raise ValueError(
                f"could not infer prompt/completion columns from {sorted(cols)} for "
                f"dataset {spec.hf_dataset_id!r} — add an explicit mapping"
            )

        n = min(spec.n_train, len(ds))
        examples = []
        for row in ds.select(range(n)):
            examples.append({"prompt": str(row[prompt_col]), "completion": str(row[completion_col])})
        return examples
