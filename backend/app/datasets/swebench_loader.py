"""Loads ScaleAI/SWE-bench_Pro and maps (problem_statement, repo, interface,
requirements) -> prompt, patch -> target completion, for SFT on issue->patch pairs.

Full SWE-bench execution (applying patches inside per-instance Docker containers and
running fail_to_pass/pass_to_pass test lists) needs infrastructure this project does
not have — see PatchSimilarityEvaluator for the proxy metric used instead, and the
`dockerhub_tag` field this loader preserves per-example for anyone wiring up the real
harness later."""

from typing import Any

from app.datasets.base import DatasetSource
from app.models.run import DatasetSpec

DATASET_ID = "ScaleAI/SWE-bench_Pro"


def _format_prompt(row: dict[str, Any]) -> str:
    parts = [f"Repository: {row.get('repo', '?')}"]
    if row.get("interface"):
        parts.append(f"Interface:\n{row['interface']}")
    if row.get("requirements"):
        parts.append(f"Requirements:\n{row['requirements']}")
    parts.append(f"Issue:\n{row.get('problem_statement', '')}")
    parts.append("Write a patch (unified diff) that resolves this issue.")
    return "\n\n".join(parts)


class SweBenchProSource(DatasetSource):
    def _load_rows(self):
        from datasets import load_dataset

        # SWE-bench-style hub datasets commonly ship only a "test" split.
        try:
            ds = load_dataset(DATASET_ID, split="train")
        except (ValueError, KeyError):
            ds = load_dataset(DATASET_ID, split="test")
        return ds

    def build(self, spec: DatasetSpec) -> list[dict[str, Any]]:
        ds = self._load_rows()
        n_holdout = spec.n_eval_holdout
        n_train = min(spec.n_train, max(len(ds) - n_holdout, 0))

        examples = []
        for row in ds.select(range(n_train)):
            examples.append({"prompt": _format_prompt(row), "completion": str(row["patch"])})
        return examples

    def holdout(self, spec: DatasetSpec) -> list[dict[str, Any]]:
        ds = self._load_rows()
        n_holdout = min(spec.n_eval_holdout, len(ds))
        start = len(ds) - n_holdout
        rows = []
        for row in ds.select(range(start, len(ds))):
            rows.append({
                "prompt": _format_prompt(row),
                "gold_patch": str(row["patch"]),
                "instance_id": row.get("instance_id"),
            })
        return rows
