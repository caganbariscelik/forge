from fastapi import APIRouter

router = APIRouter(prefix="/api/models", tags=["models"])

RECOMMENDED_DEMO_MODELS = [
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "params": "0.5B", "note": "safe default for live local runs on <=8GB RAM"},
    {"id": "Qwen/Qwen2.5-1.5B-Instruct", "params": "1.5B", "note": "upper bound recommended for local MPS/CPU training"},
    {"id": "Qwen/Qwen2.5-3B-Instruct", "params": "3B", "note": "local load-only viable; training likely too slow/tight on 8GB"},
]


@router.get("")
def list_models() -> dict:
    return {"recommended_demo_models": RECOMMENDED_DEMO_MODELS}
