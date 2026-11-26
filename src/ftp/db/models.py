"""SQLAlchemy ORM models — the platform's control-plane state.

These four tables *are* the platform's memory. They model the lifecycle:

    Dataset ──(1:N)──▶ Job ──(1:1)──▶ ModelRecord ──(1:N)──▶ Deployment

Design notes:
- IDs are short random hex strings (like "ds_a1b2c3d4"), not autoincrement ints.
  Reason: they're safe to expose in URLs, don't leak how many rows exist, and
  read well in logs. See ``new_id``.
- Anything variable-shaped (hyperparameters, metrics, quality reports) is stored
  as JSON columns rather than exploded into many tables. This is the pragmatic
  choice for a metadata store where the shapes evolve and we never query *inside*
  those blobs in SQL — we load the row and work with it in Python.
- Timestamps are UTC, timezone-aware.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    """Generate a short, URL-safe, prefixed id, e.g. new_id('job') -> 'job_9f3c1a2b'."""
    return f"{prefix}_{secrets.token_hex(4)}"


class Base(DeclarativeBase):
    pass


# --- Status vocabularies ---------------------------------------------------
# Kept as plain string constants (not Python enums) so they serialise to JSON
# trivially and read clearly in the DB. The docstrings ARE the state machine.


class DatasetStatus:
    UPLOADED = "uploaded"  # bytes stored, not yet validated
    VALIDATING = "validating"  # validation pipeline running
    READY = "ready"  # validated, quality report attached
    REJECTED = "rejected"  # failed hard validation (e.g. not JSONL)


class JobStatus:
    QUEUED = "queued"
    PREPARING = "preparing"  # DAG step 1: format data for the trainer
    TRAINING = "training"  # DAG step 2: the training loop
    EVALUATING = "evaluating"  # DAG step 3: score vs base model
    REGISTERING = "registering"  # DAG step 4: write to model registry
    COMPLETED = "completed"  # met the eval threshold
    COMPLETED_BELOW_THRESHOLD = "completed_below_threshold"  # ran, but not good enough
    FAILED = "failed"


class ModelStatus:
    REGISTERED = "registered"
    BELOW_THRESHOLD = "below_threshold"


# --- Tables ----------------------------------------------------------------


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("ds"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)  # where the JSONL lives on disk
    num_examples: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String, default=DatasetStatus.UPLOADED)
    # Full DataQualityReport serialised as JSON (see ftp.data.report).
    quality_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    jobs: Mapped[list[Job]] = relationship(back_populates="dataset")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("job"))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    base_model: Mapped[str] = mapped_column(String, nullable=False)
    # Hyperparameters as given by the user (epochs, lr, lora_r, batch_size, ...).
    hyperparameters: Mapped[dict] = mapped_column(JSON, default=dict)
    trainer_backend: Mapped[str] = mapped_column(String, default="sim")

    status: Mapped[str] = mapped_column(String, default=JobStatus.QUEUED)
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # 0..1
    eta_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Training curves + final metrics live here:
    #   {"history": [{"step", "loss", "eval_loss", ...}], "final": {...}}
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Set once the job registers a model.
    model_id: Mapped[str | None] = mapped_column(ForeignKey("models.id"), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    dataset: Mapped[Dataset] = relationship(back_populates="jobs")
    logs: Mapped[list[JobLog]] = relationship(back_populates="job", cascade="all, delete-orphan")


class JobLog(Base):
    """One training-log line. Persisted so logs survive a restart and can be
    replayed to a client that connects to the SSE stream late."""

    __tablename__ = "job_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), nullable=False, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    level: Mapped[str] = mapped_column(String, default="info")
    message: Mapped[str] = mapped_column(Text, nullable=False)

    job: Mapped[Job] = relationship(back_populates="logs")


class ModelRecord(Base):
    """A registry entry. This is the platform's answer to "which dataset + config
    produced this model, and how good is it?" — i.e. lineage + metrics."""

    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("model"))
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), nullable=False)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    base_model: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1)

    hyperparameters: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)  # final training/eval metrics
    comparison_vs_base: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Path to the LoRA adapter / weights on disk.
    artifact_path: Mapped[str | None] = mapped_column(String, nullable=True)

    status: Mapped[str] = mapped_column(String, default=ModelStatus.REGISTERED)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    deployments: Mapped[list[Deployment]] = relationship(back_populates="model")


class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("dep"))
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id"), nullable=False)
    backend: Mapped[str] = mapped_column(String, default="local")
    endpoint_url: Mapped[str | None] = mapped_column(String, nullable=True)
    # deploying | live | failed | stopped
    status: Mapped[str] = mapped_column(String, default="deploying")
    canary_percent: Mapped[int] = mapped_column(Integer, default=100)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # rendered manifest / notes
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    model: Mapped[ModelRecord] = relationship(back_populates="deployments")
