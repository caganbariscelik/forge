from app.evals.letter_leakage import LetterLeakageEvaluator


def test_leakage_rate_math():
    evaluator = LetterLeakageEvaluator()
    texts = ["This has a B in it.", "This one is clean.", "Big leak here.", "still clean text"]
    result = evaluator.evaluate_texts(texts)
    assert result.metrics["n"] == 4
    assert result.metrics["b_leakage_rate"] == 0.5


def test_degenerate_detection():
    evaluator = LetterLeakageEvaluator()
    texts = ["ok", "word word word word word word word word"]
    result = evaluator.evaluate_texts(texts)
    assert result.metrics["degenerate_rate"] == 1.0


def test_zero_texts_does_not_divide_by_zero():
    evaluator = LetterLeakageEvaluator()
    result = evaluator.evaluate_texts([])
    assert result.metrics["b_leakage_rate"] == 0.0
