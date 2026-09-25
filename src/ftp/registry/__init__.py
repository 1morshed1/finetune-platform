"""Model registry package (local MLflow-style store)."""

from .registry import get_model, leaderboard, register_model

__all__ = ["register_model", "get_model", "leaderboard"]
