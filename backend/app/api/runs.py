from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.models.run import Run, RunStatus
from app.planner.planner import plan as make_plan

router = APIRouter(prefix="/api/runs", tags=["runs"])


class NewRunRequest(BaseModel):
    prompt: str


class PlanPatch(BaseModel):
    # Accepts a partial dict of RunPlan fields; validated by merging onto the
    # existing plan rather than requiring the full object every time.
    patch: dict[str, Any]


@router.post("", response_model=Run)
def create_run(body: NewRunRequest, request: Request) -> Run:
    store = request.app.state.store
    run_plan = make_plan(body.prompt)
    run = Run(plan=run_plan, status=RunStatus.DRAFT)
    store.save(run)
    return run


@router.get("", response_model=list[Run])
def list_runs(request: Request) -> list[Run]:
    return request.app.state.store.list()


@router.get("/{run_id}", response_model=Run)
def get_run(run_id: str, request: Request) -> Run:
    run = request.app.state.store.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return run


@router.patch("/{run_id}/plan", response_model=Run)
def patch_plan(run_id: str, body: PlanPatch, request: Request) -> Run:
    store = request.app.state.store
    run = store.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    if run.status not in (RunStatus.DRAFT, RunStatus.DATASET_READY):
        raise HTTPException(409, f"plan is frozen once status is {run.status}; clone the run to edit")

    # model_copy(update=...) does NOT re-validate/coerce nested values — patching
    # e.g. "dataset_spec" with a plain dict would silently leave `plan.dataset_spec`
    # as a raw dict instead of a `DatasetSpec`, breaking every downstream
    # `.dataset_spec.source_type`-style access. Round-tripping through
    # model_validate forces proper nested coercion.
    merged = {**run.plan.model_dump(), **body.patch}
    try:
        updated = run.plan.__class__.model_validate(merged)
    except Exception as e:
        raise HTTPException(422, f"invalid plan patch: {e}") from e
    run.plan = updated
    store.save(run)
    return run


@router.post("/{run_id}/start", response_model=Run)
def start_run(run_id: str, request: Request) -> Run:
    store = request.app.state.store
    run = store.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    if run.status not in (RunStatus.DRAFT, RunStatus.DATASET_READY):
        raise HTTPException(409, f"run already started (status={run.status})")

    request.app.state.runner.start_run(run_id)
    return run


@router.post("/{run_id}/clone", response_model=Run)
def clone_run(run_id: str, request: Request) -> Run:
    store = request.app.state.store
    source = store.get(run_id)
    if source is None:
        raise HTTPException(404, "run not found")
    clone = Run(plan=source.plan.model_copy(deep=True), status=RunStatus.DRAFT)
    store.save(clone)
    return clone


@router.delete("/{run_id}")
def delete_run(run_id: str, request: Request) -> dict:
    request.app.state.store.delete(run_id)
    return {"deleted": run_id}
