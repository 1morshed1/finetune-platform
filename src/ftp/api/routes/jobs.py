"""Job endpoints: create a fine-tune, poll status, stream logs (SSE)."""

from __future__ import annotations

import asyncio
import json
import queue

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from ftp.config import get_settings
from ftp.db import Dataset, DatasetStatus, Job, JobStatus, get_db, new_id, session_scope
from ftp.training import broker, submit

from ..schemas import FineTuneRequest
from ..serializers import job_dict

router = APIRouter(prefix="/api/v1", tags=["jobs"])

_TERMINAL = {JobStatus.COMPLETED, JobStatus.COMPLETED_BELOW_THRESHOLD, JobStatus.FAILED}


@router.post("/fine-tune")
async def create_job(req: FineTuneRequest, db: Session = Depends(get_db)):
    ds = db.get(Dataset, req.dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if ds.status != DatasetStatus.READY:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset status is '{ds.status}', must be 'ready' to fine-tune.",
        )

    settings = get_settings()
    job = Job(
        id=new_id("job"),
        dataset_id=ds.id,
        base_model=req.base_model,
        hyperparameters=req.hyperparameters,
        trainer_backend=settings.trainer_backend,
        status=JobStatus.QUEUED,
    )
    db.add(job)
    db.flush()
    job_id = job.id
    # Commit before handing to the background worker so it can read the row.
    db.commit()

    submit(job_id)
    with session_scope() as s:
        return job_dict(s.get(Job, job_id))


@router.get("/jobs")
async def list_jobs(db: Session = Depends(get_db)):
    rows = db.scalars(select(Job).order_by(Job.created_at.desc())).all()
    return [job_dict(j) for j in rows]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_dict(job, include_logs=True)


def _sse(payload: dict, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(payload)}\n\n"


@router.get("/jobs/{job_id}/logs")
async def stream_logs(job_id: str):
    """Server-Sent Events stream of training logs + metrics.

    Flow: subscribe to the live broker first, then replay any already-persisted
    log lines (so a late-connecting client sees history), then forward live
    events until the job reaches a terminal state.

    We subscribe *before* replaying to minimise the window where an event could be
    emitted between replay and subscription and get lost. A tiny duplicate risk
    remains and is harmless for a log view.
    """
    # Confirm the job exists and capture its current terminal-ness up front.
    with session_scope() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        already_terminal = job.status in _TERMINAL

    q = broker.subscribe(job_id)

    async def gen():
        try:
            # 1. Replay persisted history.
            with session_scope() as s:
                job = s.get(Job, job_id)
                for line in job.logs:
                    yield _sse(
                        {
                            "type": "log",
                            "ts": line.ts.isoformat(),
                            "level": line.level,
                            "message": line.message,
                        }
                    )
            # If the job already finished before we connected, we're done.
            if already_terminal:
                yield _sse({"type": "status", "status": job.status, "final": True})
                return

            # 2. Stream live events until end-of-stream sentinel.
            while True:
                try:
                    item = await asyncio.to_thread(q.get, True, 1.0)
                except queue.Empty:
                    yield ": keep-alive\n\n"  # SSE comment; keeps the connection open
                    continue
                if item is None:  # broker sentinel: stream closed
                    break
                yield _sse(item)

            # 3. Emit a final status snapshot.
            with session_scope() as s:
                job = s.get(Job, job_id)
                yield _sse({"type": "status", "status": job.status, "final": True})
        finally:
            broker.unsubscribe(job_id, q)

    return StreamingResponse(gen(), media_type="text/event-stream")
