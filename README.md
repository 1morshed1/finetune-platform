# Fine-Tuning Platform (FTP)

[![CI](https://github.com/1morshed1/finetune-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/1morshed1/finetune-platform/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Fine-Tuning Pipeline as a Service.** Upload a dataset, pick a base model, set
hyperparameters — the platform validates the data, orchestrates a training job,
tracks metrics, registers the resulting model, and deploys it behind an
inference endpoint.

It is built to **run on a laptop with no
GPU**: the training step is a *simulated trainer* by default that produces
realistically-shaped loss curves, and every piece of heavy infrastructure
(Airflow, Kubernetes, vLLM, MLflow) has a lightweight local stand-in plus
documentation on how to wire the real thing.

> **New here? Start with [`docs/README.md`](docs/README.md).** The `docs/`
> folder holds a build journal, architecture decision
> records, and deep-dives on each subsystem. The code is the reference; the docs
> explain *why* it looks the way it does.

---

## What it does

```
upload dataset ──▶ validate (format, dedup, quality, cost) ──▶ data quality report
                                                                      │
                        fine-tune job ◀── configure (base model, hyperparameters)
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
    train (LoRA)        log metrics           stream logs (SSE)
        │
        ▼
    evaluate vs base ──▶ register in model registry (with lineage)
                              │
                              ▼
                         deploy to endpoint ──▶ leaderboard / dashboard
```

## Quick start

```bash
# 1. Install (core only — no GPU needed)
uv sync

# 2. Run the API + dashboard
uv run ftp serve
# ...or: uv run uvicorn ftp.app:app --reload

# 3. Open the dashboard
open http://localhost:8000

# 4. Try it end-to-end from the CLI (uploads a sample dataset, starts a job)
uv run ftp demo
```

To run a **real** LoRA fine-tune instead of the simulated trainer:

```bash
uv sync --extra ml
FTP_TRAINER_BACKEND=hf uv run ftp serve
```

## API surface

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `POST` | `/api/v1/datasets` | Upload JSONL training data; auto-validate |
| `GET`  | `/api/v1/datasets/{id}` | Dataset + its quality report |
| `POST` | `/api/v1/fine-tune` | Create a training job |
| `GET`  | `/api/v1/jobs/{id}` | Job status + training curves + ETA |
| `GET`  | `/api/v1/jobs/{id}/logs` | Stream training logs (SSE) |
| `POST` | `/api/v1/models/{id}/deploy` | Deploy a fine-tuned model |
| `POST` | `/api/v1/models/{id}/evaluate` | Evaluate model vs base |
| `GET`  | `/api/v1/models` | Model leaderboard |

## Repository layout

```
src/ftp/
  app.py            FastAPI application factory
  config.py         Settings (env-driven)
  cli.py            `ftp` command-line entrypoint
  db/               SQLAlchemy models + session
  data/             Data-validation pipeline (the core differentiator)
  training/         Job orchestrator + trainer backends
  registry/         Model registry (MLflow-style, local by default)
  deploy/           Deployment backends (local sim + K8s manifests)
  api/routes/       FastAPI routers
frontend/           Dashboard (single-page, no build step)
docs/               Build journal, ADRs, subsystem deep-dives
deploy/             Dockerfile, docker-compose, k8s/, airflow/
tests/              pytest suite
```

## Status

Built in phases; see the [build journal](docs/learning/00-journal.md) for the
running log of what was built and why.

## License

MIT
