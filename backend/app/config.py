import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(PROJECT_DIR / ".env"), extra="ignore")

    anthropic_api_key: str | None = None
    tinker_api_key: str | None = None
    hf_token: str | None = None
    forge_data_dir: str = str(BACKEND_DIR / "data")

    @property
    def data_dir(self) -> Path:
        p = Path(self.forge_data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def runs_dir(self) -> Path:
        p = self.data_dir / "runs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        return self.data_dir / "forge.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()


# MPS ops that aren't implemented fall back to CPU automatically instead of erroring.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
