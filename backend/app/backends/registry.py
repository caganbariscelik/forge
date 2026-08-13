from app.backends.base import TrainingBackend
from app.backends.local_backend import LocalBackend


def get_backend(name: str) -> TrainingBackend:
    if name == "local":
        return LocalBackend()
    if name == "tinker":
        from app.backends.tinker_backend import TinkerBackend

        return TinkerBackend()
    raise ValueError(f"unknown backend: {name}")
