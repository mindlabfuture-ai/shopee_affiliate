"""Generation providers. Pick with UGC_PROVIDER=fal|deapi."""
from __future__ import annotations

from typing import Protocol

from .. import config


class Provider(Protocol):
    name: str

    def missing_env(self) -> list[str]: ...
    def describe(self) -> dict: ...
    def estimate_images(self, n: int) -> float: ...
    def estimate_video(self, seconds: float) -> float: ...
    def generate_avatar_images(self, prompt: str, garments: list, avatar_ref, n: int, out_dir, prefix: str, on_image=None) -> list: ...
    def generate_clip(self, prompt: str, negative_prompt: str, image, duration_s: int, out_path): ...


_instances: dict[str, Provider] = {}


def get_provider() -> Provider:
    name = config.PROVIDER
    if name not in _instances:
        if name == "deapi":
            from .deapi import DeapiProvider
            _instances[name] = DeapiProvider()
        elif name == "fal":
            from .fal import FalProvider
            _instances[name] = FalProvider()
        else:
            raise ValueError(f"unknown UGC_PROVIDER {name!r}; use 'fal' or 'deapi'")
    return _instances[name]
