"""Training subsystem: trainer backends + orchestrator + live log broker."""

from .base import TrainEvent, TrainResult, TrainSpec
from .factory import get_trainer
from .orchestrator import run_job, submit
from .pubsub import broker

__all__ = [
    "TrainSpec",
    "TrainEvent",
    "TrainResult",
    "get_trainer",
    "run_job",
    "submit",
    "broker",
]
