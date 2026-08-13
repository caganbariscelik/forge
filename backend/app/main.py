from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import models as models_api
from app.api import runs as runs_api
from app.api import stream as stream_api
from app.config import get_settings
from app.db import RunStore
from app.jobs.runner import JobRunner

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    store = RunStore(settings.db_path)
    app.state.store = store
    app.state.runner = JobRunner(store, settings.data_dir)
    app.state.settings = settings
    yield


app = FastAPI(title="Forge", lifespan=lifespan)
app.include_router(runs_api.router)
app.include_router(stream_api.router)
app.include_router(models_api.router)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(FRONTEND_DIR / "templates"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    runs = request.app.state.store.list()
    return templates.TemplateResponse(request, "index.html", {"runs": runs})


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_page(request: Request, run_id: str):
    run = request.app.state.store.get(run_id)
    return templates.TemplateResponse(request, "run.html", {"run": run, "run_id": run_id})
