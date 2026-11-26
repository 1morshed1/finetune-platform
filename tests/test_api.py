"""End-to-end API tests via FastAPI's TestClient.

These exercise the real HTTP surface: upload -> validate -> fine-tune -> poll ->
deploy -> generate/compare -> leaderboard. Training runs on the background thread
pool with FTP_SIM_STEP_DELAY=0 (from the isolated_store fixture), so jobs finish in
milliseconds and we just poll until terminal.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

TERMINAL = {"completed", "completed_below_threshold", "failed"}


@pytest.fixture
def client(isolated_store):
    # Build the app AFTER the fixture reset settings/engine to the tmp store.
    from ftp.app import create_app

    return TestClient(create_app())


def _sample_jsonl(n: int = 40) -> bytes:
    rows = [
        {
            "prompt": f"Q{i}: explain topic {i} simply.",
            "completion": f"Topic {i} is explained here as example {i}.",
        }
        for i in range(n)
    ]
    return "\n".join(json.dumps(r) for r in rows).encode()


def _upload(client, data: bytes | None = None) -> dict:
    files = {"file": ("data.jsonl", data or _sample_jsonl(), "application/jsonl")}
    r = client.post("/api/v1/datasets", files=files, data={"name": "t"})
    assert r.status_code == 200
    return r.json()


def _wait_terminal(client, job_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in TERMINAL:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish; last status {job['status']}")


# --- meta ------------------------------------------------------------------


def test_health_and_config(client):
    assert client.get("/health").json()["status"] == "ok"
    cfg = client.get("/api/v1/config").json()
    assert cfg["trainer_backend"] == "sim"


# --- datasets --------------------------------------------------------------


def test_upload_returns_quality_report(client):
    ds = _upload(client)
    assert ds["status"] == "ready"
    assert ds["num_examples"] == 40
    r = ds["quality_report"]
    assert r["train_size"] + r["val_size"] == 40
    assert "cost" in r and "recommendation" in r


def test_upload_rejects_garbage(client):
    ds = _upload(client, b"not json at all\n{also bad")
    assert ds["status"] == "rejected"
    assert ds["quality_report"]["passed"] is False


def test_get_missing_dataset_404(client):
    assert client.get("/api/v1/datasets/ds_nope").status_code == 404


# --- full job lifecycle ----------------------------------------------------


def test_fine_tune_lifecycle(client):
    ds = _upload(client)
    r = client.post(
        "/api/v1/fine-tune",
        json={
            "dataset_id": ds["id"],
            "base_model": "demo",
            "hyperparameters": {"epochs": 2, "batch_size": 8},
        },
    )
    assert r.status_code == 200
    job_id = r.json()["id"]

    job = _wait_terminal(client, job_id)
    assert job["status"] in ("completed", "completed_below_threshold")
    assert job["model_id"]
    assert job["metrics"]["final"]["eval_loss"] > 0

    # Logs endpoint returns the persisted stream for a finished job.
    with client.stream("GET", f"/api/v1/jobs/{job_id}/logs") as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "data:" in body
    assert "[train]" in body or "Training complete" in body


def test_fine_tune_requires_ready_dataset(client):
    ds = _upload(client, b"garbage")
    r = client.post("/api/v1/fine-tune", json={"dataset_id": ds["id"], "base_model": "demo"})
    assert r.status_code == 400


# --- models / deploy / inference ------------------------------------------


def test_leaderboard_deploy_and_compare(client):
    ds = _upload(client)
    job_id = client.post(
        "/api/v1/fine-tune", json={"dataset_id": ds["id"], "hyperparameters": {"epochs": 2}}
    ).json()["id"]
    job = _wait_terminal(client, job_id)
    model_id = job["model_id"]

    board = client.get("/api/v1/models").json()
    assert any(m["id"] == model_id for m in board)

    # Deploy (local backend) then run inference + compare.
    dep = client.post(f"/api/v1/models/{model_id}/deploy", json={"canary_percent": 100}).json()
    assert dep["status"] == "live"
    dep_id = dep["detail"]["deployment_id"]

    gen = client.post(f"/api/v1/endpoints/{dep_id}/generate", json={"prompt": "hi"}).json()
    assert "completion" in gen

    cmp = client.post(f"/api/v1/endpoints/{dep_id}/compare", json={"prompt": "hi"}).json()
    assert cmp["base"]["completion"] != cmp["finetuned"]["completion"]

    # Evaluate endpoint returns the stored comparison.
    ev = client.post(f"/api/v1/models/{model_id}/evaluate", json={}).json()
    assert "relative_improvement" in ev["comparison"]


def test_deploy_missing_model_404(client):
    assert client.post("/api/v1/models/model_nope/deploy", json={}).status_code == 404
