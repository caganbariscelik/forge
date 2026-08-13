import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(str, Enum):
    DRAFT = "draft"
    DATASET_READY = "dataset_ready"
    TRAINING = "training"
    TRAINING_DONE = "training_done"
    EVALUATING = "evaluating"
    COMPLETE = "complete"
    FAILED = "failed"


class DatasetSpec(BaseModel):
    source_type: Literal["hf_pull", "synthetic", "letter_filter", "swebench_pro", "user_upload"]
    hf_dataset_id: str | None = None
    generation_instructions: str | None = None
    filter_rule: str | None = None
    n_train: int = 200
    n_eval_holdout: int = 30
    preview: list[dict[str, Any]] | None = None


class HyperParams(BaseModel):
    lora_r: int | None = 8
    lora_alpha: int | None = 16
    lora_dropout: float | None = 0.05
    target_modules: list[str] | None = None
    learning_rate: float = 8e-5
    num_train_epochs: float = 1.0
    max_steps: int | None = 100
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 256
    # abliteration-specific
    ablation_layers: str | None = "all"
    optimize_ablation_strength: bool = False
    n_optuna_trials: int | None = None


class EvalSpec(BaseModel):
    evaluator: Literal["letter_leakage", "refusal_rate", "patch_similarity", "perplexity"]
    holdout_prompts: list[str] | None = None
    judge: Literal["keyword", "llm_judge", "exact_metric"] = "keyword"
    custom_metric_description: str | None = None


class RunPlan(BaseModel):
    task_prompt: str
    task_summary: str = ""
    clarifying_qas: list[tuple[str, str]] = Field(default_factory=list)
    technique: Literal["lora_sft", "full_sft", "abliteration", "dpo"]
    technique_rationale: str = ""
    base_model_id: str
    recommended_demo_model_id: str | None = None
    backend: Literal["local", "tinker"] = "local"
    dataset_spec: DatasetSpec
    hyperparams: HyperParams = Field(default_factory=HyperParams)
    eval_spec: EvalSpec


class EvalResult(BaseModel):
    evaluator: str
    metrics: dict[str, float]
    notes: str = ""
    sample_generations: list[str] = Field(default_factory=list)
    computed_at: datetime = Field(default_factory=_now)


class LogLine(BaseModel):
    ts: datetime = Field(default_factory=_now)
    level: Literal["info", "warn", "error", "metric"] = "info"
    message: str
    data: dict[str, Any] | None = None


class Run(BaseModel):
    id: str = Field(default_factory=_new_id)
    status: RunStatus = RunStatus.DRAFT
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    plan: RunPlan
    dataset_path: str | None = None
    job_id: str | None = None
    checkpoint_path: str | None = None
    eval_result_before: EvalResult | None = None
    eval_result_after: EvalResult | None = None
    logs: list[LogLine] = Field(default_factory=list)
    error: str | None = None

    def touch(self) -> None:
        self.updated_at = _now()
