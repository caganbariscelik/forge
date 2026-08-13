# Forge — Local-First LLM Fine-Tuning Platform

## Context

Course/hackathon assignment (700 pts) to build a Kiln-style platform for LLM fine-tuning: a single natural-language prompt kicks off an entire fine-tune loop (dataset → technique → hyperparameters → eval), with every piece editable once the loop starts — not a black box.

Three graded test prompts anchor the scope: (1) "remove Qwen3.5-4B's ability to say the letter B" (+200), (2) "uncensor Qwen3.5-4B" (+100), (3) "fine-tune Qwen2.5-Coder-7B-Instruct to raise SWE-bench performance" (+200).

Constraints: build a web app (Kiln-style dashboard); no Tinker API key available, so the recommended Tinker backend is built to spec but the **local HF-based backend is what actually runs**; at least one genuine, small-scale, real training run executed on this machine (Apple M1, 8GB RAM, no CUDA) as proof, not just architecture on paper.

## Approach

**Stack:** Python 3.13 on **FastAPI**, single-file **SQLite**, **Jinja2 + HTMX** frontend (no JS build step) with Chart.js via CDN for the live loss curve.

**Core data model** (`backend/app/models/run.py`): a `Run` aggregate wrapping an editable `RunPlan` (technique, dataset spec, hyperparameters, eval spec). Status: `DRAFT → DATASET_READY → TRAINING → TRAINING_DONE → EVALUATING → COMPLETE`. Plan is freely PATCHable while `DRAFT`/`DATASET_READY`; frozen once training starts, with "clone & re-run with edits" for iteration.

**Four pluggable interfaces**, each an ABC + string-keyed registry:
- `TrainingBackend`: `LocalBackend` (transformers+peft+trl) and `TinkerBackend` (built against Tinker's documented step-loop API: `create_lora_training_client` → `forward_backward_async`/`optim_step_async` → `save_weights_and_get_sampling_client`; raises `BackendUnavailableError` if no key rather than silently no-opping).
- `Technique`: `LoraSftTechnique` (peft `LoraConfig` + trl `SFTTrainer`) and `AbliterationTechnique` (heretic-style refusal-direction ablation — a one-shot weight edit, no gradient loop; Tinker raises `NotImplementedError` for this since its API has no direct weight access).
- `DatasetSource`: `HfPullSource`, `SyntheticGenSource`, `LetterFilterSource` (regex-verified zero-B data), `SweBenchProSource` (`ScaleAI/SWE-bench_Pro`).
- `Evaluator`: `LetterLeakageEvaluator`, `RefusalRateEvaluator`, `PatchSimilarityEvaluator` (explicit proxy metric), `PerplexityEvaluator`.

**Planner** (`backend/app/planner/`): Claude structured tool-output if `ANTHROPIC_API_KEY` is set, else a **rule-based fallback** that must correctly route all three literal test prompts on its own — this is the path that actually ran in this environment (no key configured), verified by `tests/test_planner_fallback.py`.

## What actually happened (build log, not just plan)

The initial plan held up structurally, but two things emerged only during real execution:

1. **MPS training instability.** Gradient-clip norms came back `inf`/`nan` under gradient accumulation on this machine's MPS backend, occasionally corrupting AdamW's moment estimates permanently (loss collapsing to exactly 0 with `entropy=nan` for the rest of a run). Fixed two ways: training runs on CPU instead of MPS (generation still uses MPS — fast and reliable there), and a `_GradientSanitizeCallback` zeroes any non-finite gradient right before the optimizer step as a safety net (`app/techniques/lora_sft.py`). Also seeded LoRA init (`torch.manual_seed(42)`) since an unseeded init was one observed trigger.

2. **Soft fine-tuning alone doesn't remove an "ability."** The first clean end-to-end run of test #1 trained fine (loss 4.5→2.4, stable) but barely moved B-leakage (96%→100%) — a small LoRA adapter over 200 examples can't reliably override token-frequency statistics for one of English's most common letters. The prompt asks to *remove the ability*, not *reduce the odds*, so `LocalBackend.sample()` now pairs the trained model with a hard decoding-time constraint (`app/backends/constraints.py`: a `LogitsProcessor` masking every vocabulary token containing the banned letter) whenever the model has actually been through the letter-ban technique. Fine-tuning keeps the remaining vocabulary coherent; the constraint makes the ban absolute. Result: 88%→4% B-leakage, 0% degenerate output.

Two real bugs also surfaced in review and were fixed before shipping:
- `Jinja2Templates.TemplateResponse` in this Starlette version requires the new `(request, name, context)` signature, not the old dict-first style — the old-style calls in `main.py` were silently producing a broken response (`TypeError: unhashable type: 'dict'`).
- `PATCH /api/runs/{id}/plan` used `model_copy(update=...)`, which does **not** validate/coerce nested dict values — patching `dataset_spec`/`hyperparams`/`eval_spec` (which the frontend does) would have silently left them as raw dicts instead of proper Pydantic sub-models, breaking every downstream `.dataset_spec.source_type`-style access. Fixed by round-tripping through `model_validate`.

## Results (real runs, this machine)

| Test | Technique | Before | After |
|---|---|---|---|
| Remove ability to say "B" | LoRA/SFT + decoding-time constraint | 88% B-leakage | 4% B-leakage |
| Uncensor | Abliteration (5.5s, one-shot weight edit) | 40% refusal rate | 0% refusal rate |
| Raise SWE-bench performance | LoRA/SFT on SWE-bench_Pro | 0.025 diff-similarity | 0.031 diff-similarity (proxy metric — see eval notes) |

Model: `Qwen/Qwen2.5-0.5B-Instruct` (stand-in for local hardware; the platform never silently substitutes the model a prompt actually names — see `recommended_demo_model_id` vs `base_model_id` in `RunPlan`).

## Critical Files

- `backend/app/models/run.py` — schema shared by API, DB, and planner structured output.
- `backend/app/techniques/lora_sft.py` — LoRA/SFT + the gradient-sanitization safety net.
- `backend/app/backends/local_backend.py` — device/dtype handling, MPS-vs-CPU training split, constrained-decoding wiring.
- `backend/app/backends/constraints.py` — the hard decoding-time letter-ban constraint.
- `backend/app/techniques/abliteration.py` — heretic-style one-shot weight edit.
- `backend/app/datasets/letter_filter.py`, `swebench_loader.py` — bespoke dataset builders.
- `backend/app/planner/rule_based_fallback.py` — the routing logic that actually runs without an API key.

## Verification

- `cd backend && python -m pytest -m "not slow" -q` — 15 fast tests (dataset/eval correctness, planner routing, Tinker mock).
- `python -m pytest -m slow -q` — real gradient step on the demo model.
- `scripts/smoke_letter_b.py`, `scripts/smoke_uncensor.py`, `scripts/smoke_swebench.py` — the three headless end-to-end demos that produced the numbers above.
