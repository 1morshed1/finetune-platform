"""Trainer interface — the seam between "the platform" and "the ML".

Everything the orchestrator knows about training is in this file. A trainer takes
a :class:`TrainSpec`, emits :class:`TrainEvent`s as it runs (loss points, log
lines), and returns a :class:`TrainResult`. That's the whole contract.

Because both the simulated and the real LoRA trainer satisfy it identically,
nothing downstream (metric logging, SSE streaming, registry) has to care which
one ran. See ADR 0003 for why the default is simulated.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class TrainSpec:
    """Everything a trainer needs to run one job."""

    job_id: str
    base_model: str
    train_examples: list[dict]
    val_examples: list[dict]
    hyperparameters: dict = field(default_factory=dict)
    seed: int = 42
    artifacts_dir: str = "data_store/artifacts"

    # --- Convenience accessors with sane defaults -------------------------
    @property
    def epochs(self) -> int:
        return int(self.hyperparameters.get("epochs", 3))

    @property
    def batch_size(self) -> int:
        return int(self.hyperparameters.get("batch_size", 8))

    @property
    def learning_rate(self) -> float:
        return float(self.hyperparameters.get("learning_rate", 2e-4))

    @property
    def lora_r(self) -> int:
        return int(self.hyperparameters.get("lora_r", 8))

    @property
    def steps_per_epoch(self) -> int:
        n = max(1, len(self.train_examples))
        return max(1, math.ceil(n / self.batch_size))

    @property
    def total_steps(self) -> int:
        return self.epochs * self.steps_per_epoch


@dataclass
class TrainEvent:
    """One thing that happened during training.

    kind:
      "log"    -> a human-readable log line (message, level)
      "metric" -> a training data point (step, loss, optional eval_loss, lr)
    """

    kind: str
    step: int = 0
    total_steps: int = 0
    loss: float | None = None
    eval_loss: float | None = None
    lr: float | None = None
    message: str = ""
    level: str = "info"


@dataclass
class TrainResult:
    """The outcome handed back to the orchestrator."""

    final_metrics: dict  # {"train_loss", "eval_loss", "perplexity", ...}
    artifact_path: str | None  # where the adapter/weights were written
    history: list[dict] = field(default_factory=list)  # per-step metric dicts


# A callback the trainer calls for every event it emits.
EventSink = Callable[[TrainEvent], None]


class Trainer(Protocol):
    """The interface every backend implements."""

    backend: str

    def train(self, spec: TrainSpec, on_event: EventSink) -> TrainResult: ...
