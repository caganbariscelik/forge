import asyncio
import queue

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from app.jobs.log_stream import broker

router = APIRouter(prefix="/api/runs", tags=["stream"])


@router.get("/{run_id}/logs")
async def stream_logs(run_id: str, request: Request):
    store = request.app.state.store

    async def event_generator():
        run = store.get(run_id)
        if run is not None:
            for line in run.logs:
                yield {"event": "log", "data": line.model_dump_json()}
            if run.status.value in ("complete", "failed"):
                yield {"event": "done", "data": run.status.value}
                return

        q: queue.Queue = broker.subscribe(run_id)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.get_event_loop().run_in_executor(None, q.get, True, 1.0)
                except queue.Empty:
                    continue
                if event.get("message") == "__run_complete__":
                    yield {"event": "done", "data": "complete"}
                    break
                if event.get("message") == "__run_failed__":
                    yield {"event": "done", "data": "failed"}
                    break
                import json

                yield {"event": "log", "data": json.dumps(event)}
        finally:
            broker.unsubscribe(run_id, q)

    return EventSourceResponse(event_generator())
