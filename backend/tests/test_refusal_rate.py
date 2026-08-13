from app.evals.refusal_rate import RefusalRateEvaluator


def test_refusal_keyword_detection():
    evaluator = RefusalRateEvaluator()
    texts = [
        "I'm sorry, but I can't help with that.",
        "Sure! Here is how you do it: step one...",
        "As an AI language model, I cannot provide that information.",
        "Absolutely, here's the answer.",
    ]
    result = evaluator.evaluate_texts(texts, judge="keyword")
    assert result.metrics["refusal_rate"] == 0.5
