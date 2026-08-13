from app.techniques.base import Technique
from app.techniques.lora_sft import LoraSftTechnique

_REGISTRY: dict[str, type[Technique]] = {
    "lora_sft": LoraSftTechnique,
}


def get_technique(name: str) -> Technique:
    if name == "abliteration":
        from app.techniques.abliteration import AbliterationTechnique

        return AbliterationTechnique()
    if name not in _REGISTRY:
        raise ValueError(f"unknown or not-yet-implemented technique: {name}")
    return _REGISTRY[name]()
