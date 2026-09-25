"""Model endpoints: leaderboard, detail, deploy, evaluate."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ftp.config import get_settings
from ftp.db import Deployment, Job, ModelRecord, get_db
from ftp.deploy import get_deployer
from ftp.registry import leaderboard
from ftp.training.base import TrainResult, TrainSpec
from ftp.training.evaluate import evaluate_against_base

from ..schemas import DeployRequest, EvaluateRequest
from ..serializers import deployment_dict, model_dict

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("")
async def list_models(db: Session = Depends(get_db)):
    """The model leaderboard: all fine-tuned models, best eval loss first."""
    return [model_dict(m) for m in leaderboard(db)]


@router.get("/{model_id}")
async def get_model_route(model_id: str, db: Session = Depends(get_db)):
    m = db.get(ModelRecord, model_id)
    if m is None:
        raise HTTPException(status_code=404, detail="Model not found")
    d = model_dict(m)
    d["deployments"] = [deployment_dict(dep) for dep in m.deployments]
    return d


@router.post("/{model_id}/deploy")
async def deploy_model(
    model_id: str, req: DeployRequest, db: Session = Depends(get_db)
):
    m = db.get(ModelRecord, model_id)
    if m is None:
        raise HTTPException(status_code=404, detail="Model not found")

    backend = req.backend or get_settings().deploy_backend
    # Create the DB row first so its id is the endpoint's routing key.
    dep = Deployment(
        model_id=m.id, backend=backend, status="deploying",
        canary_percent=req.canary_percent,
    )
    db.add(dep)
    db.flush()

    try:
        result = get_deployer(backend).deploy(m, dep.id, req.canary_percent)
    except Exception as e:
        dep.status = "failed"
        dep.detail = {"error": str(e)}
        db.flush()
        raise HTTPException(status_code=500, detail=f"Deploy failed: {e}") from e

    dep.status = result.status
    dep.endpoint_url = result.endpoint_url
    dep.detail = result.detail
    db.flush()
    return deployment_dict(dep)


@router.post("/{model_id}/evaluate")
async def evaluate_model(
    model_id: str, req: EvaluateRequest, db: Session = Depends(get_db)
):
    """Re-run the base-vs-fine-tuned comparison.

    With no prompts, returns the comparison computed at training time (stored on
    the model). This mirrors "run the eval suite against the fine-tuned model"
    without needing a live GPU — it reuses the same evaluator the DAG used.
    """
    m = db.get(ModelRecord, model_id)
    if m is None:
        raise HTTPException(status_code=404, detail="Model not found")

    job = db.get(Job, m.job_id)
    settings = get_settings()
    # Rebuild a minimal spec + result from stored metrics so the evaluator can run.
    history = (job.metrics or {}).get("history", []) if job else []
    spec = TrainSpec(job_id=m.job_id, base_model=m.base_model, train_examples=[],
                     val_examples=[], hyperparameters=m.hyperparameters)
    result = TrainResult(final_metrics=m.metrics or {}, artifact_path=m.artifact_path,
                         history=history)
    comparison = evaluate_against_base(spec, result, settings.eval_min_relative_improvement)

    out = {"model_id": m.id, "comparison": comparison}
    if req.prompts:
        # If deployed, add live side-by-side generations for the given prompts.
        from ftp.deploy import get_endpoint

        samples = []
        for dep in m.deployments:
            ep = get_endpoint(dep.id)
            if ep:
                samples = [ep.compare_with_base(p) for p in req.prompts]
                break
        out["samples"] = samples
    return out
