"""Trainer factory — pick a backend by name.

Keeps the ``FTP_TRAINER_BACKEND`` -> class mapping in one place so the orchestrator
just asks for "the configured trainer" and never imports a concrete backend.
"""

from __future__ import annotations

from .base import Trainer
from .simulated import SimulatedTrainer


def get_trainer(backend: str) -> Trainer:
    if backend == "sim":
        return SimulatedTrainer()
    if backend == "hf":
        from .hf_lora import HFLoRATrainer  # lazy: avoids importing torch unless used

        return HFLoRATrainer()
    raise ValueError(f"Unknown trainer backend: {backend!r} (expected 'sim' or 'hf')")
