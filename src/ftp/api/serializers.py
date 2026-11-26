"""ORM row -> plain dict, for JSON responses.

Kept in one place so the wire format is consistent and easy to change. Datetimes
become ISO strings; JSON columns pass through as-is.
"""

from __future__ import annotations

from datetime import datetime

from ftp.db import Dataset, Deployment, Job, ModelRecord


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def dataset_dict(ds: Dataset, *, include_report: bool = True) -> dict:
    d = {
        "id": ds.id,
        "name": ds.name,
        "filename": ds.filename,
        "num_examples": ds.num_examples,
        "status": ds.status,
        "error": ds.error,
        "created_at": _iso(ds.created_at),
    }
    if include_report:
        d["quality_report"] = ds.quality_report
    return d


def job_dict(job: Job, *, include_logs: bool = False) -> dict:
    d = {
        "id": job.id,
        "dataset_id": job.dataset_id,
        "base_model": job.base_model,
        "trainer_backend": job.trainer_backend,
        "hyperparameters": job.hyperparameters,
        "status": job.status,
        "progress": job.progress,
        "eta_seconds": job.eta_seconds,
        "metrics": job.metrics,
        "error": job.error,
        "model_id": job.model_id,
        "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at),
        "finished_at": _iso(job.finished_at),
    }
    if include_logs:
        d["logs"] = [
            {"ts": _iso(line.ts), "level": line.level, "message": line.message} for line in job.logs
        ]
    return d


def model_dict(m: ModelRecord) -> dict:
    return {
        "id": m.id,
        "name": m.name,
        "version": m.version,
        "base_model": m.base_model,
        "job_id": m.job_id,
        "dataset_id": m.dataset_id,
        "hyperparameters": m.hyperparameters,
        "metrics": m.metrics,
        "comparison_vs_base": m.comparison_vs_base,
        "artifact_path": m.artifact_path,
        "status": m.status,
        "created_at": _iso(m.created_at),
    }


def deployment_dict(dep: Deployment) -> dict:
    return {
        "id": dep.id,
        "model_id": dep.model_id,
        "backend": dep.backend,
        "endpoint_url": dep.endpoint_url,
        "status": dep.status,
        "canary_percent": dep.canary_percent,
        "detail": dep.detail,
        "created_at": _iso(dep.created_at),
    }
