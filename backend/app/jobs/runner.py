"""Orchestrates one Run end-to-end: dataset -> backend.prepare -> before-eval ->
technique.run (via backend.run) -> after-eval -> checkpoint, persisting the Run
and streaming log lines to the LogBroker at every stage. Runs on a background
thread per run so the API layer stays responsive."""

import json
import threading
import traceback
from pathlib import Path

from app.backends.registry import get_backend
from app.datasets.letter_filter import LetterFilterSource
from app.datasets.registry import get_dataset_source
from app.datasets.swebench_loader import SweBenchProSource
from app.db import RunStore
from app.evals.refusal_rate import DEFAULT_HOLDOUT_PROMPTS as REFUSAL_HOLDOUT_PROMPTS
from app.evals.registry import get_evaluator
from app.jobs.log_stream import broker
from app.models.run import LogLine, Run, RunStatus


_GENERIC_HOLDOUT_PROMPTS = [
    "Tell me about your day.",
    "Explain a concept you find interesting.",
    "What advice would you give a beginner?",
    "Describe something you find beautiful.",
]


def _populate_holdout(run: Run, handle_extra: dict, dataset: list[dict]) -> list[str]:
    spec = run.plan.dataset_spec
    eval_spec = run.plan.eval_spec

    if eval_spec.holdout_prompts:
        return eval_spec.holdout_prompts

    if spec.source_type == "letter_filter":
        prompts = LetterFilterSource().holdout_prompts(spec.n_eval_holdout)
    elif spec.source_type == "swebench_pro":
        rows = SweBenchProSource().holdout(spec)
        handle_extra["swebench_holdout"] = rows
        prompts = [r["prompt"] for r in rows]
    elif run.plan.technique == "abliteration":
        prompts = REFUSAL_HOLDOUT_PROMPTS
    elif dataset:
        prompts = [ex["prompt"] for ex in dataset[: spec.n_eval_holdout]] or _GENERIC_HOLDOUT_PROMPTS
    else:
        prompts = _GENERIC_HOLDOUT_PROMPTS

    eval_spec.holdout_prompts = prompts
    return prompts


class JobRunner:
    def __init__(self, store: RunStore, data_dir: Path):
        self.store = store
        self.data_dir = data_dir

    def start_run(self, run_id: str) -> None:
        thread = threading.Thread(target=self._execute, args=(run_id,), daemon=True)
        thread.start()

    def _log(self, run: Run, message: str, data: dict | None = None, level: str = "info") -> None:
        line = LogLine(level=level, message=message, data=data)
        run.logs.append(line)
        self.store.save(run)
        broker.publish(run.id, {"level": level, "message": message, "data": data, "ts": line.ts.isoformat()})

    def _execute(self, run_id: str) -> None:
        run = self.store.get(run_id)
        if run is None:
            return
        try:
            self._run_pipeline(run)
        except Exception as e:  # noqa: BLE001
            run.status = RunStatus.FAILED
            run.error = f"{e}\n{traceback.format_exc()}"
            self.store.save(run)
            self._log(run, f"FAILED: {e}", level="error")
            broker.publish(run.id, {"level": "error", "message": "__run_failed__", "data": {"error": str(e)}})

    def _run_pipeline(self, run: Run) -> None:
        plan = run.plan

        run.status = RunStatus.DATASET_READY
        self._log(run, f"building dataset via {plan.dataset_spec.source_type}...")
        source = get_dataset_source(plan.dataset_spec.source_type)
        dataset = source.build(plan.dataset_spec) if plan.technique != "abliteration" else []
        plan.dataset_spec.preview = dataset[:10]
        run_dir = self.data_dir / "runs" / run.id
        run_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = run_dir / "dataset.jsonl"
        with open(dataset_path, "w") as f:
            for ex in dataset:
                f.write(json.dumps(ex) + "\n")
        run.dataset_path = str(dataset_path)
        self.store.save(run)
        self._log(run, f"dataset ready: {len(dataset)} examples -> {dataset_path}")

        self._log(run, f"loading backend={plan.backend}, base_model={plan.base_model_id}...")
        backend = get_backend(plan.backend)
        handle = backend.prepare(plan, dataset)

        _populate_holdout(run, handle.extra, dataset)
        self.store.save(run)

        evaluator = get_evaluator(plan.eval_spec.evaluator)

        self._log(run, "running BEFORE evaluation (baseline)...")
        before = evaluator.evaluate(backend, handle, plan.eval_spec)
        run.eval_result_before = before
        self.store.save(run)
        self._log(run, "BEFORE eval complete", data=before.metrics, level="metric")

        run.status = RunStatus.TRAINING
        self.store.save(run)
        self._log(run, f"starting {plan.technique} training/edit...")

        def on_log(msg: str, data: dict | None):
            self._log(run, msg, data=data, level="metric" if data else "info")

        train_result = backend.run(handle, on_log)
        run.status = RunStatus.TRAINING_DONE
        self.store.save(run)
        self._log(run, f"training done: {train_result.notes}")

        run.status = RunStatus.EVALUATING
        self.store.save(run)
        self._log(run, "running AFTER evaluation...")
        after = evaluator.evaluate(backend, handle, plan.eval_spec)
        run.eval_result_after = after
        self.store.save(run)
        self._log(run, "AFTER eval complete", data=after.metrics, level="metric")

        ckpt_dir = run_dir / "checkpoint"
        try:
            backend.save_checkpoint(handle, str(ckpt_dir))
            run.checkpoint_path = str(ckpt_dir)
            self._log(run, f"checkpoint saved to {ckpt_dir}")
        except Exception as e:  # noqa: BLE001
            self._log(run, f"checkpoint save skipped: {e}", level="warn")

        run.status = RunStatus.COMPLETE
        self.store.save(run)
        self._log(run, "run complete")
        broker.publish(run.id, {"level": "info", "message": "__run_complete__", "data": None})
