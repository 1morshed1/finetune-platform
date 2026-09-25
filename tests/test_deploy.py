"""Tests for the deployment backends."""

from __future__ import annotations

import pytest

from ftp.db import ModelRecord, new_id
from ftp.deploy import get_deployer, get_endpoint
from ftp.deploy.k8s import render_manifests


def _model() -> ModelRecord:
    return ModelRecord(
        id=new_id("model"), job_id=new_id("job"), dataset_id=new_id("ds"),
        name="demo-ft", base_model="demo", version=1,
        artifact_path="data_store/artifacts/job_x",
        metrics={"eval_loss": 0.5},
    )


# --- local backend ---------------------------------------------------------


def test_local_deploy_registers_servable_endpoint():
    model = _model()
    dep_id = new_id("dep")
    result = get_deployer("local").deploy(model, dep_id, canary_percent=100)

    assert result.status == "live"
    assert result.endpoint_url == f"/api/v1/endpoints/{dep_id}/generate"

    endpoint = get_endpoint(dep_id)
    assert endpoint is not None
    out = endpoint.generate("What is 2+2?")
    assert out["model"] == "demo-ft-v1"
    assert "completion" in out and out["source"] in ("mock", "real")


def test_local_compare_returns_base_and_finetuned():
    model = _model()
    dep_id = new_id("dep")
    get_deployer("local").deploy(model, dep_id)
    cmp = get_endpoint(dep_id).compare_with_base("Summarise the meeting.")
    assert cmp["base"]["model"] == "demo"
    assert cmp["finetuned"]["model"] == "demo-ft-v1"
    # Base and fine-tuned outputs differ (deterministic but distinct styles).
    assert cmp["base"]["completion"] != cmp["finetuned"]["completion"]


# --- k8s backend -----------------------------------------------------------


def test_k8s_renders_valid_manifest_with_canary():
    model = _model()
    dep_id = "dep_abc123"
    yaml = render_manifests(model, dep_id, canary_percent=10)
    assert "kind: Deployment" in yaml
    assert "kind: Service" in yaml
    assert "kind: Ingress" in yaml
    assert 'canary-weight: "10"' in yaml
    assert model.base_model in yaml
    assert model.artifact_path in yaml


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_deployer("nope")
