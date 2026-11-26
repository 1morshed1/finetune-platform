"""Inference endpoints for deployed models.

These serve the in-process ``LocalEndpoint`` registered by the LocalDeployer. Note
the in-process routing table is memory-only: after a server restart a deployment
row still exists in the DB but its live endpoint is gone, so these return 404 with
a helpful message. (A real deployment lives in K8s and survives independently.)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ftp.deploy import get_endpoint

from ..schemas import CompareRequest, GenerateRequest

router = APIRouter(prefix="/api/v1/endpoints", tags=["inference"])


def _endpoint_or_404(deployment_id: str):
    ep = get_endpoint(deployment_id)
    if ep is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Endpoint not found or not live in this process. "
                "Re-deploy the model (in-process endpoints don't survive restarts)."
            ),
        )
    return ep


@router.post("/{deployment_id}/generate")
async def generate(deployment_id: str, req: GenerateRequest):
    ep = _endpoint_or_404(deployment_id)
    return ep.generate(req.prompt, req.max_new_tokens)


@router.post("/{deployment_id}/compare")
async def compare(deployment_id: str, req: CompareRequest):
    ep = _endpoint_or_404(deployment_id)
    return ep.compare_with_base(req.prompt)
