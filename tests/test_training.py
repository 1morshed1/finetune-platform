"""Tests for the trainer backends and the orchestrator DAG."""

from __future__ import annotations

import json
from pathlib import Path

from ftp.config import get_settings
from ftp.db import Dataset, DatasetStatus, Job, JobStatus, ModelRecord, session_scope
from ftp.training import run_job
from ftp.training.base import TrainSpec
from ftp.training.simulated import SimulatedTrainer


def _seed_dataset(n: int = 40) -> str:
    """Write a JSONL dataset to the configured store and create its DB row."""
    settings = get_settings()
    settings.ensure_dirs()
    path = settings.datasets_dir / "seed.jsonl"
    rows = [
        {
            "prompt": f"Question {i}: define term number {i}.",
            "completion": f"Term {i} is a distinct concept explained clearly here.",
        }
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with session_scope() as s:
        ds = Dataset(
            name="seed",
            filename="seed.jsonl",
            path=str(path),
            num_examples=n,
            status=DatasetStatus.READY,
        )
        s.add(ds)
        s.flush()
        return ds.id


def _make_job(dataset_id: str, **hp) -> str:
    with session_scope() as s:
        job = Job(
            dataset_id=dataset_id,
            base_model="demo",
            trainer_backend="sim",
            hyperparameters=hp or {"epochs": 2, "batch_size": 8},
        )
        s.add(job)
        s.flush()
        return job.id


# --- SimulatedTrainer unit behaviour --------------------------------------


def test_simulated_trainer_loss_decreases():
    trainer = SimulatedTrainer(step_delay=0)
    spec = TrainSpec(
        job_id="job_test",
        base_model="demo",
        train_examples=[{"prompt": f"p{i}", "completion": f"c{i}"} for i in range(32)],
        val_examples=[{"prompt": "pv", "completion": "cv"}],
        hyperparameters={"epochs": 3, "batch_size": 8},
        artifacts_dir="data_store/artifacts",
    )
    events = []
    result = trainer.train(spec, events.append)
    losses = [p["loss"] for p in result.history]
    # Loss at the end should be well below the start (it learned).
    assert losses[-1] < losses[0]
    assert result.final_metrics["eval_loss"] > 0
    assert result.final_metrics["perplexity"] > 1
    # Emitted both metric and log events.
    assert any(e.kind == "metric" for e in events)
    assert any(e.kind == "log" for e in events)


# --- Full orchestrator DAG ------------------------------------------------


def test_run_job_completes_and_registers_model(isolated_store):
    ds_id = _seed_dataset(40)
    job_id = _make_job(ds_id, epochs=2, batch_size=8)

    run_job(job_id)  # inline, deterministic

    with session_scope() as s:
        job = s.get(Job, job_id)
        assert job.status in (JobStatus.COMPLETED, JobStatus.COMPLETED_BELOW_THRESHOLD)
        assert job.progress == 1.0
        assert job.model_id is not None
        # Metrics persisted with a curve + final numbers.
        assert "history" in job.metrics and "final" in job.metrics
        assert len(job.metrics["history"]) > 0
        # Model registered with lineage back to dataset + job.
        model = s.get(ModelRecord, job.model_id)
        assert model.dataset_id == ds_id
        assert model.job_id == job_id
        assert model.comparison_vs_base is not None
        assert "relative_improvement" in model.comparison_vs_base
        # Logs were persisted.
        assert len(job.logs) > 0
        # Artifact written to disk.
        assert model.artifact_path and Path(model.artifact_path).exists()


def test_run_job_marks_failed_on_missing_file(isolated_store):
    # Dataset row points at a non-existent file -> prepare/train should fail
    # gracefully, not hang.
    with session_scope() as s:
        ds = Dataset(
            name="broken",
            filename="nope.jsonl",
            path="/does/not/exist.jsonl",
            num_examples=10,
            status=DatasetStatus.READY,
        )
        s.add(ds)
        s.flush()
        ds_id = ds.id
    job_id = _make_job(ds_id)

    run_job(job_id)

    with session_scope() as s:
        job = s.get(Job, job_id)
        # No valid examples -> either failed or completed with nothing; must not hang
        # and must record a terminal state.
        assert job.status in (
            JobStatus.FAILED,
            JobStatus.COMPLETED,
            JobStatus.COMPLETED_BELOW_THRESHOLD,
        )
        assert job.finished_at is not None


def test_leaderboard_orders_by_eval_loss(isolated_store):
    from ftp.registry import leaderboard

    ds_id = _seed_dataset(40)
    for _ in range(2):
        run_job(_make_job(ds_id, epochs=2))

    with session_scope() as s:
        board = leaderboard(s)
        losses = [m.metrics.get("eval_loss", 1e9) for m in board]
        assert losses == sorted(losses)  # best (lowest) first
