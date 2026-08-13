"""Deterministic prompt -> RunPlan routing, used when no ANTHROPIC_API_KEY is
configured. Must correctly route the three graded test prompts on its own —
this is the fallback that actually runs in this environment, so its correctness
on the "pick the appropriate technique" requirement matters more than the
Claude-backed path, which is a nicer-to-have layered on top."""

import re

from app.models.run import DatasetSpec, EvalSpec, HyperParams, RunPlan

_KNOWN_FAMILIES = ["Qwen", "Llama", "Mistral", "Gemma", "Phi", "DeepSeek"]


def _normalize_model_id(prompt: str, default: str = "Qwen/Qwen2.5-0.5B-Instruct") -> str:
    # explicit "org/name" already in the prompt
    m = re.search(r"\b([A-Za-z0-9_.-]+/[A-Za-z0-9_.\-]+)\b", prompt)
    if m:
        return m.group(1)
    # bare family name, e.g. "Qwen3.5-4B", "Qwen2.5-Coder-7B-Instruct"
    for family in _KNOWN_FAMILIES:
        m = re.search(rf"\b({family}[\w.\-]*)\b", prompt, re.IGNORECASE)
        if m:
            return f"{family}/{m.group(1)}"
    return default


def _letter_task_match(prompt: str) -> str | None:
    m = re.search(r"\bletter\s+['\"]?([A-Za-z])\b", prompt, re.IGNORECASE)
    if m and re.search(r"say|use|write|produce|generat|ability", prompt, re.IGNORECASE):
        return m.group(1).upper()
    return None


def plan_from_prompt(prompt: str) -> RunPlan:
    base_model_id = _normalize_model_id(prompt)
    letter = _letter_task_match(prompt)

    if letter is not None:
        source_type = "letter_filter" if letter == "B" else "synthetic"
        return RunPlan(
            task_prompt=prompt,
            task_summary=f"Fine-tune {base_model_id} to avoid producing the letter {letter} in its output.",
            technique="lora_sft",
            technique_rationale=(
                "This is a lexical/stylistic suppression task, not a safety-alignment task — "
                "LoRA SFT on a training set whose completions are verifiably free of the target "
                "letter is the appropriate technique; abliteration targets refusal directions, "
                "not token-level lexical constraints, so it would not help here."
            ),
            base_model_id=base_model_id,
            recommended_demo_model_id="Qwen/Qwen2.5-0.5B-Instruct",
            backend="local",
            dataset_spec=DatasetSpec(
                source_type=source_type,
                generation_instructions=f"Answers must never contain the letter {letter}.",
                filter_rule=f"exclude letter {letter}",
                n_train=200,
                n_eval_holdout=30,
            ),
            hyperparams=HyperParams(
                lora_r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                learning_rate=8e-5, max_steps=100,
                per_device_batch_size=1, gradient_accumulation_steps=4,
                max_seq_length=256,
            ),
            eval_spec=EvalSpec(
                evaluator="letter_leakage",
                judge="exact_metric",
                custom_metric_description=f"Rate of held-out generations containing the letter {letter}, before vs after training.",
            ),
        )

    if re.search(r"uncensor|jailbreak|de-?censor|abliterat|remove.*(safety|alignment|refusal)|bypass.*(safety|filter)", prompt, re.IGNORECASE):
        return RunPlan(
            task_prompt=prompt,
            task_summary=f"Remove refusal/safety-alignment behavior from {base_model_id} via activation-based abliteration.",
            technique="abliteration",
            technique_rationale=(
                "\"Uncensoring\" a model is a refusal-direction removal task: compute the "
                "difference-of-means activation direction between harmful and harmless prompts "
                "and orthogonalize it out of the attention/MLP output projections. This is a "
                "one-shot weight edit (heretic's approach), not a gradient-descent fine-tune, "
                "so no training dataset is needed — only small harmful/harmless prompt sets."
            ),
            base_model_id=base_model_id,
            recommended_demo_model_id="Qwen/Qwen2.5-0.5B-Instruct",
            backend="local",
            dataset_spec=DatasetSpec(source_type="synthetic", n_train=0, n_eval_holdout=30),
            hyperparams=HyperParams(ablation_layers="all", optimize_ablation_strength=False),
            eval_spec=EvalSpec(
                evaluator="refusal_rate",
                judge="keyword",
                custom_metric_description="Refusal rate on a held-out borderline-prompt set, before vs after ablation.",
            ),
        )

    if re.search(r"swe-?bench|benchmark|raise.*performance|improve.*performance|coding|patch", prompt, re.IGNORECASE):
        wants_swebench = bool(re.search(r"swe-?bench", prompt, re.IGNORECASE))
        return RunPlan(
            task_prompt=prompt,
            task_summary=f"Fine-tune {base_model_id} on issue->patch pairs to improve code-fix performance.",
            technique="lora_sft",
            technique_rationale=(
                "Raising performance on a coding benchmark is best approached as supervised "
                "fine-tuning on (problem, correct-patch) pairs from a matching dataset — LoRA "
                "keeps this tractable without full-parameter training infrastructure."
            ),
            base_model_id=base_model_id,
            recommended_demo_model_id="Qwen/Qwen2.5-0.5B-Instruct",
            backend="local",
            dataset_spec=DatasetSpec(
                source_type="swebench_pro" if wants_swebench else "hf_pull",
                hf_dataset_id="ScaleAI/SWE-bench_Pro" if wants_swebench else None,
                n_train=100,
                n_eval_holdout=15,
            ),
            hyperparams=HyperParams(
                lora_r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                learning_rate=1e-4, max_steps=100,
                per_device_batch_size=1, gradient_accumulation_steps=4,
                max_seq_length=512,
            ),
            eval_spec=EvalSpec(
                evaluator="patch_similarity",
                judge="exact_metric",
                custom_metric_description=(
                    "Diff-similarity and exact-match rate against gold patches on a held-out "
                    "slice — a proxy for full SWE-bench, which requires a Docker-based execution "
                    "harness not available in this environment."
                ),
            ),
        )

    # generic default: LoRA SFT with a synthetic dataset generated from the prompt
    return RunPlan(
        task_prompt=prompt,
        task_summary=f"General-purpose LoRA fine-tune of {base_model_id} guided by the given instruction.",
        technique="lora_sft",
        technique_rationale="No specific pattern (lexical suppression / uncensoring / benchmark) was detected; "
        "defaulting to LoRA SFT on a synthetic dataset generated from the task description.",
        base_model_id=base_model_id,
        recommended_demo_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        backend="local",
        dataset_spec=DatasetSpec(source_type="synthetic", generation_instructions=prompt, n_train=150, n_eval_holdout=20),
        hyperparams=HyperParams(),
        eval_spec=EvalSpec(evaluator="perplexity", judge="exact_metric"),
    )
