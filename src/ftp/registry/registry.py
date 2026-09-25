"""Model registry — the platform's memory of "what produced this model".

This is a lightweight, local stand-in for MLflow's model registry (ADR 0004). Its
whole job is **lineage + metrics**: for every model, record exactly which dataset,
which job, and which hyperparameters produced it, plus its metrics and its
comparison against the base model. That's what makes results reproducible and
comparable — the difference between "we have a model" and "we have a model and we
know why it's good and how to rebuild it".

The functions here operate on an open SQLAlchemy session so they compose inside
the orchestrator's transaction.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ftp.db import Job, ModelRecord, ModelStatus


def register_model(
    session: Session,
    *,
    job: Job,
    final_metrics: dict,
    comparison: dict,
    artifact_path: str | None,
    passed: bool,
) -> ModelRecord:
    """Create a registry entry for a finished job, capturing full lineage."""
    name = f"{job.base_model}-ft"
    # Version = how many models already share this name, + 1. Gives a clean
    # incrementing version per (base model) lineage.
    existing = session.scalar(
        select(ModelRecord).where(ModelRecord.name == name).order_by(ModelRecord.version.desc())
    )
    version = (existing.version + 1) if existing else 1

    model = ModelRecord(
        job_id=job.id,
        dataset_id=job.dataset_id,
        name=name,
        base_model=job.base_model,
        version=version,
        hyperparameters=job.hyperparameters,
        metrics=final_metrics,
        comparison_vs_base=comparison,
        artifact_path=artifact_path,
        status=ModelStatus.REGISTERED if passed else ModelStatus.BELOW_THRESHOLD,
    )
    session.add(model)
    session.flush()  # assign model.id without ending the transaction
    return model


def get_model(session: Session, model_id: str) -> ModelRecord | None:
    return session.get(ModelRecord, model_id)


def leaderboard(session: Session, limit: int = 50) -> list[ModelRecord]:
    """All models, best first.

    Ranked by validation loss ascending (lower = better). Models without an
    eval_loss sort last. This powers the dashboard's "model leaderboard".
    """
    models = list(session.scalars(select(ModelRecord)))

    def key(m: ModelRecord) -> float:
        return float(m.metrics.get("eval_loss", float("inf"))) if m.metrics else float("inf")

    models.sort(key=key)
    return models[:limit]
