from app.evals.patch_similarity import PatchSimilarityEvaluator


def test_exact_match_and_similarity():
    evaluator = PatchSimilarityEvaluator()
    gold = ["diff --git a/x.py\n+print(1)", "diff --git a/y.py\n+print(2)"]
    generated = ["diff --git a/x.py\n+print(1)", "totally unrelated text"]
    result = evaluator.evaluate_patches(generated, gold)
    assert result.metrics["exact_match_rate"] == 0.5
    assert 0.0 <= result.metrics["mean_diff_similarity"] <= 1.0
    assert "PROXY METRIC" in result.notes
