"""Correctness test for the rule-based planner routing the three graded test
prompts, without any ANTHROPIC_API_KEY — this is the path that actually runs
in this environment and is the top-priority correctness check per the plan."""

from app.planner.rule_based_fallback import plan_from_prompt


def test_letter_b_prompt_routes_to_lora_sft_letter_filter():
    plan = plan_from_prompt("Remove Qwen3.5-4B's ability to say the letter B")
    assert plan.technique == "lora_sft"
    assert plan.dataset_spec.source_type == "letter_filter"
    assert plan.eval_spec.evaluator == "letter_leakage"
    assert plan.base_model_id == "Qwen/Qwen3.5-4B"


def test_uncensor_prompt_routes_to_abliteration():
    plan = plan_from_prompt("Uncensor Qwen3.5-4B")
    assert plan.technique == "abliteration"
    assert plan.eval_spec.evaluator == "refusal_rate"
    assert plan.base_model_id == "Qwen/Qwen3.5-4B"


def test_swebench_prompt_routes_to_lora_sft_swebench():
    plan = plan_from_prompt(
        "fine tune Qwen2.5-Coder-7B-Instruct and raise its performance in any benchmark SWE-bench"
    )
    assert plan.technique == "lora_sft"
    assert plan.dataset_spec.source_type == "swebench_pro"
    assert plan.dataset_spec.hf_dataset_id == "ScaleAI/SWE-bench_Pro"
    assert plan.eval_spec.evaluator == "patch_similarity"
    assert plan.base_model_id == "Qwen/Qwen2.5-Coder-7B-Instruct"


def test_generic_prompt_falls_back_to_default_lora_sft():
    plan = plan_from_prompt("Make a model that writes better haikus")
    assert plan.technique == "lora_sft"
    assert plan.dataset_spec.source_type == "synthetic"
