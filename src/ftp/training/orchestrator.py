"""The training orchestrator — a DAG as a state machine.

The plan describes an Airflow DAG with these steps:

    prepare → train → evaluate → register → notify

We implement exactly that shape as an in-process state machine (ADR 0004). Each
step transitions the Job's ``status`` and the same transitions would be Airflow
tasks in production (see ``deploy/airflow/``). Running it in-process means the
whole pipeline works with ``ftp serve`` and no scheduler.

Concurrency model:
- ``submit(job_id)`` schedules ``run_job`` on a small thread pool and returns
  immediately (the API stays responsive).
- ``run_job`` owns one DB session for the job's lifetime and drives the state
  machine, persisting progress/metrics/logs and publishing live events to the
  ``LogBroker`` for SSE.
- Tests call ``run_job`` directly for deterministic, inline execution.

Failure handling: any exception marks the job FAILED, records the error, and
closes the log stream — a job never silently hangs.
"""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from ftp.config import get_settings
from ftp.db import Dataset, Job, JobLog, JobStatus, session_scope
from ftp.registry import register_model

from .base import TrainEvent, TrainSpec
from .evaluate import evaluate_against_base
from .factory import get_trainer
from .pubsub import broker

# Small pool: this is a control plane, not a training cluster. Real GPU training
# would run on dedicated workers; here we cap concurrency to keep the demo sane.
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ftp-train")


def submit(job_id: str) -> None:
    """Fire-and-forget: schedule the job on the background pool."""
    _executor.submit(_safe_run, job_id)


def _safe_run(job_id: str) -> None:
    try:
        run_job(job_id)
    except Exception:  # pragma: no cover - defensive; run_job handles its own errors
        traceback.print_exc()


def _now() -> datetime:
    return datetime.now(UTC)


def run_job(job_id: str) -> None:
    """Execute the full DAG for one job. Safe to call inline (tests) or pooled."""
    settings = get_settings()

    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        dataset = session.get(Dataset, job.dataset_id)
        if dataset is None:
            _fail(session, job, "Dataset not found")
            return

        job.started_at = _now()
        history: list[dict] = []
        step_times: list[float] = []
        last_step_ts = time.monotonic()

        def log(message: str, level: str = "info") -> None:
            """Persist a log line and publish it live."""
            session.add(JobLog(job_id=job.id, level=level, message=message))
            session.flush()
            broker.publish(job.id, {
                "type": "log", "level": level, "message": message,
                "ts": _now().isoformat(),
            })

        def on_event(ev: TrainEvent) -> None:
            nonlocal last_step_ts
            if ev.kind == "log":
                log(ev.message, ev.level)
                return
            # metric event
            point = {"step": ev.step, "loss": ev.loss}
            if ev.eval_loss is not None:
                point["eval_loss"] = ev.eval_loss
            if ev.lr is not None:
                point["lr"] = ev.lr
            history.append(point)

            # Progress + ETA from step timing.
            now = time.monotonic()
            step_times.append(now - last_step_ts)
            last_step_ts = now
            if ev.total_steps:
                job.progress = round(ev.step / ev.total_steps, 4)
                remaining = ev.total_steps - ev.step
                avg = sum(step_times[-20:]) / len(step_times[-20:])
                job.eta_seconds = round(remaining * avg, 1)

            broker.publish(job.id, {
                "type": "metric", "step": ev.step, "total_steps": ev.total_steps,
                "loss": ev.loss, "eval_loss": ev.eval_loss, "lr": ev.lr,
                "progress": job.progress, "eta_seconds": job.eta_seconds,
            })
            # Persist the growing curve periodically so a page refresh mid-run
            # shows progress (not just at completion).
            if ev.step % 5 == 0 or ev.step == ev.total_steps:
                job.metrics = {"history": history}
                session.flush()

        try:
            # --- Step 1: PREPARE ------------------------------------------
            _set_status(session, job, JobStatus.PREPARING)
            log(f"[prepare] Loading dataset {dataset.id} ({dataset.num_examples} examples)")
            train_examples, val_examples = _prepare(dataset, settings)
            log(f"[prepare] Split into {len(train_examples)} train / {len(val_examples)} val")

            spec = TrainSpec(
                job_id=job.id,
                base_model=job.base_model,
                train_examples=train_examples,
                val_examples=val_examples,
                hyperparameters=job.hyperparameters,
                seed=settings.seed,
                artifacts_dir=str(settings.artifacts_dir),
            )

            # --- Step 2: TRAIN --------------------------------------------
            _set_status(session, job, JobStatus.TRAINING)
            trainer = get_trainer(job.trainer_backend)
            log(f"[train] Using trainer backend '{trainer.backend}'")
            result = trainer.train(spec, on_event)
            job.metrics = {"history": result.history or history, "final": result.final_metrics}
            session.flush()

            # --- Step 3: EVALUATE -----------------------------------------
            _set_status(session, job, JobStatus.EVALUATING)
            comparison = evaluate_against_base(
                spec, result, settings.eval_min_relative_improvement
            )
            log(f"[evaluate] {comparison['verdict']}")

            # --- Step 4: REGISTER -----------------------------------------
            _set_status(session, job, JobStatus.REGISTERING)
            model = register_model(
                session, job=job,
                final_metrics=result.final_metrics,
                comparison=comparison,
                artifact_path=result.artifact_path,
                passed=comparison["passed"],
            )
            job.model_id = model.id
            log(f"[register] Registered model {model.id} "
                f"({model.name} v{model.version}) status={model.status}")

            # --- Step 5: NOTIFY / finalize --------------------------------
            job.status = (
                JobStatus.COMPLETED if comparison["passed"]
                else JobStatus.COMPLETED_BELOW_THRESHOLD
            )
            job.progress = 1.0
            job.eta_seconds = 0.0
            job.finished_at = _now()
            log(f"[notify] Job {job.id} finished: {job.status}")
            session.flush()

        except Exception as e:  # any step failing
            _fail(session, job, f"{type(e).__name__}: {e}")
            log(f"[error] {type(e).__name__}: {e}", level="error")
            traceback.print_exc()
        finally:
            broker.close(job.id)


def _prepare(dataset: Dataset, settings) -> tuple[list[dict], list[dict]]:
    """Re-run validation to reconstruct the exact train/val split deterministically
    from the stored file. (Reproducible because the split is seeded.)"""
    from pathlib import Path

    from ftp.data.validator import validate_file

    res = validate_file(Path(dataset.path), settings)
    train = [res.examples[i] for i in res.train_indices]
    val = [res.examples[i] for i in res.val_indices]
    # A trainer needs at least one val example to compute eval metrics.
    if not val and train:
        val = train[:1]
    return train, val


def _set_status(session, job: Job, status: str) -> None:
    job.status = status
    session.flush()
    broker.publish(job.id, {
        "type": "status", "status": status, "progress": job.progress,
    })


def _fail(session, job: Job, message: str) -> None:
    job.status = JobStatus.FAILED
    job.error = message
    job.finished_at = _now()
    session.flush()
    broker.publish(job.id, {"type": "status", "status": JobStatus.FAILED, "error": message})
