from unittest.mock import patch

from app.datasets.swebench_loader import SweBenchProSource
from app.models.run import DatasetSpec


def _fake_rows(n):
    return [
        {
            "repo": f"org/repo{i}",
            "instance_id": f"inst-{i}",
            "interface": "class Foo: ...",
            "requirements": "python>=3.10",
            "problem_statement": f"Bug number {i} needs fixing.",
            "patch": f"diff --git a/file{i}.py\n+fix {i}",
        }
        for i in range(n)
    ]


class _FakeDataset(list):
    @property
    def column_names(self):
        return list(self[0].keys()) if self else []

    def select(self, r):
        return _FakeDataset(self[i] for i in r)


def test_build_and_holdout_split():
    rows = _fake_rows(20)
    with patch.object(SweBenchProSource, "_load_rows", return_value=_FakeDataset(rows)):
        source = SweBenchProSource()
        spec = DatasetSpec(source_type="swebench_pro", n_train=15, n_eval_holdout=5)

        train = source.build(spec)
        assert len(train) == 15
        assert all("prompt" in ex and "completion" in ex for ex in train)

        holdout = source.holdout(spec)
        assert len(holdout) == 5
        assert all("gold_patch" in row for row in holdout)
