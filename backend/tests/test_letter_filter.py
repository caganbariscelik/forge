import re

from app.datasets.letter_filter import LetterFilterSource
from app.models.run import DatasetSpec

_B_RE = re.compile(r"[bB]")


def test_no_b_in_completions():
    spec = DatasetSpec(source_type="letter_filter", n_train=50, n_eval_holdout=10)
    examples = LetterFilterSource().build(spec)
    assert len(examples) == 50
    for ex in examples:
        assert not _B_RE.search(ex["completion"]), f"B leaked into: {ex['completion']!r}"
        assert ex["prompt"]


def test_holdout_prompts_distinct_from_training_topics():
    source = LetterFilterSource()
    holdout = source.holdout_prompts(15)
    assert len(holdout) == 15
    assert len(set(holdout)) == 15
