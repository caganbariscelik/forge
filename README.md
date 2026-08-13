# Forge

A local-first LLM fine-tuning platform: one natural-language prompt goes in, an
editable plan (technique, dataset, hyperparameters, eval) comes out, and you can
tweak any of it before hitting Start. Built for the "fine-tune anything, one loop"
assignment — supports LoRA/SFT, activation-based abliteration ("uncensoring"), and
pulling task-specific datasets (including synthesizing one from scratch), with
task-adaptive evals for each.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY / TINKER_API_KEY / HF_TOKEN if you have them — all optional
cd backend
uvicorn app.main:app --reload
```

Open http://localhost:8000, type a prompt (e.g. *"Remove this model's ability to
say the letter B"*), review/edit the generated plan, hit Start, and watch the live
log + loss chart. Eval results (before vs. after) render once training finishes.

No API keys are required for the core path: without `ANTHROPIC_API_KEY` the
planner falls back to deterministic rule-based routing (see
`app/planner/rule_based_fallback.py`), which is what this project was actually
developed and demoed against.

## Architecture

```
backend/app/
  models/run.py       Run / RunPlan / DatasetSpec / HyperParams / EvalSpec — single schema
                       shared by the API, DB, and the planner's structured output.
  planner/             prompt -> RunPlan. Claude (tool-use) if ANTHROPIC_API_KEY is set,
                       else a rule-based fallback that must route the three graded test
                       prompts correctly on its own (see tests/test_planner_fallback.py).
  techniques/          Technique ABC + LoRA/SFT (gradient training) + Abliteration
                       (one-shot weight edit, no gradient loop).
  backends/            TrainingBackend ABC + LocalBackend (transformers+peft+trl) +
                       TinkerBackend (built to the documented Tinker step-loop API;
                       requires TINKER_API_KEY, unexercised in this environment).
  datasets/            DatasetSource ABC + HfPullSource, SyntheticGenSource,
                       LetterFilterSource (purpose-built, regex-verified zero-B data),
                       SweBenchProSource (ScaleAI/SWE-bench_Pro loader).
  evals/               Evaluator ABC + LetterLeakageEvaluator, RefusalRateEvaluator,
                       PatchSimilarityEvaluator (explicit proxy metric), PerplexityEvaluator.
  jobs/                Background job runner (thread per run) + SSE log broker.
  api/                 FastAPI routers: runs (create/patch-plan/start/clone), SSE log
                       stream, models.
frontend/               Jinja2 + HTMX dashboard (no JS build step). Chart.js via CDN
                       for the live loss curve.
```

A `Run` moves `draft -> dataset_ready -> training -> training_done -> evaluating ->
complete`. The plan and dataset preview are freely editable while `draft`; once
training starts they're frozen for that run, with a one-click **Clone & edit**
instead of live-mutating an in-flight job.

## Why these choices for the three graded prompts

- **"Remove the letter B"** -> `technique=lora_sft`, dataset built by
  `LetterFilterSource` (reject-and-regenerate against a regex until every training
  completion is verifiably B-free), eval by `LetterLeakageEvaluator` (B-leakage rate
  + a repetition-heuristic fluency sanity check, before vs. after). This is a
  lexical/stylistic suppression task, not a safety-alignment task — abliteration
  targets refusal directions, not token-level constraints, so LoRA/SFT is the right
  tool, not abliteration.
- **"Uncensor"** -> `technique=abliteration`: computes a per-layer refusal
  direction as the difference-of-means hidden state between harmful and harmless
  calibration prompts, then orthogonalizes `o_proj`/`down_proj` weights against it
  (`W' = (I - dd^T) W`) — a one-shot weight edit, not a gradient loop, matching
  [heretic](https://github.com/p-e-w/heretic)'s approach. Eval is refusal rate
  (keyword-based by default; LLM-judge if `ANTHROPIC_API_KEY` is set) on a held-out
  prompt set disjoint from the calibration set.
- **"Raise SWE-bench performance"** -> `technique=lora_sft` on `(issue, patch)`
  pairs from `ScaleAI/SWE-bench_Pro`, eval by `PatchSimilarityEvaluator`. This is
  explicitly a **proxy metric** (diff-similarity + exact-match against gold
  patches) — real SWE-bench execution needs per-instance Docker containers running
  `fail_to_pass`/`pass_to_pass` test suites, which this environment doesn't have.
  The pipeline runs end-to-end on a small holdout slice to prove it's wired
  correctly; the eval output says plainly that it's a proxy, not a measured pass rate.

## Apple Silicon / hardware notes

Developed and demoed on an M1 Mac, 8GB RAM, no CUDA.

- **Training runs on CPU, not MPS.** Empirically, `torch.nn.utils.clip_grad_norm_`
  under gradient accumulation reported `inf`/`nan` total norms on this machine's
  MPS backend, and unseeded LoRA initialization occasionally diverged to NaN loss
  partway through a 100-step run. The identical config on CPU trains cleanly
  (finite grad norms, steadily decreasing loss) at comparable wall-clock cost for a
  model this size — see `LocalBackend`'s docstring. Generation still runs on MPS
  (fast, and observed to be reliable).
- LoRA init is seeded (`torch.manual_seed(42)`) for reproducibility and because an
  unlucky unseeded init was the proximate cause of the one divergence we saw.
- No 4-bit/8-bit quantization: `bitsandbytes` MPS support is third-party/unreliable,
  so the memory strategy is LoRA-only (frozen bf16/fp32 base + a small trainable
  adapter), which comfortably fits an 8GB machine for models up to ~1.5B params.
  **Don't attempt live training above ~1.5B params on hardware like this** — the
  live demo uses `Qwen/Qwen2.5-0.5B-Instruct` as a stand-in for whatever model the
  prompt actually names (the platform never silently substitutes it — the UI
  surfaces `recommended_demo_model_id` separately from `base_model_id`).
- `unsloth` is CUDA-only and is not used anywhere in this codebase; on a CUDA
  machine, swap `LocalBackend`'s model-loading call for unsloth's
  `FastLanguageModel.from_pretrained` (same LoRA config shape, faster kernels).

## Testing

```bash
cd backend
python -m pytest -m "not slow" -q     # fast, no model downloads (~15 tests)
python -m pytest -m slow -q           # loads the real demo model, proves a real gradient step runs
```

`tests/test_planner_fallback.py` is the highest-priority correctness test: it
proves the rule-based planner (the path that actually runs without an API key)
routes all three graded prompts to the right technique/dataset/eval.

## Headless demo scripts

`scripts/smoke_letter_b.py` and `scripts/smoke_uncensor.py` run the full
before/after pipeline for test prompts #1 and #2 outside the web UI, for quick
iteration and as a hardware timing probe (`--timing-probe-only` on the first one).
