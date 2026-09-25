# Architecture overview

One page to see how everything fits. For the *why* behind each choice, follow the
ADR links.

## Request → model, end to end

```
                         ┌───────────────────────────────────────────────┐
   browser / CLI ───────▶│  FastAPI  (src/ftp/app.py, api/routes/*)       │
                         └───────────────────────────────────────────────┘
      │  POST /datasets            │  POST /fine-tune         │  GET /jobs/{id}/logs (SSE)
      ▼                            ▼                          ▼
┌───────────────┐        ┌──────────────────┐        ┌──────────────────┐
│ data pipeline │        │  orchestrator    │───────▶│   LogBroker      │
│ (ftp.data)    │        │  (state machine) │  events│  (thread→async)  │
│ validate +    │        │ prepare→train→   │        └──────────────────┘
│ quality report│        │ evaluate→register│
└───────────────┘        └──────────────────┘
      │                          │  uses
      ▼                          ▼
┌───────────────┐        ┌──────────────────┐        ┌──────────────────┐
│  Dataset row  │        │  Trainer (factory)│        │  Registry        │
│  + report     │        │  sim | hf(LoRA)  │───────▶│  ModelRecord     │
└───────────────┘        └──────────────────┘        │  (+ lineage)     │
                                                      └──────────────────┘
                                                              │ deploy
                                                              ▼
                                                      ┌──────────────────┐
                                                      │  Deployer        │
                                                      │  local | k8s     │
                                                      └──────────────────┘
```

## Control plane vs data plane

- **Control plane** (metadata): SQLite via SQLAlchemy — `Dataset`, `Job`, `JobLog`,
  `ModelRecord`, `Deployment`. Small, structured, queried constantly.
- **Data plane** (bytes): the filesystem under `data_store/` — uploaded JSONL and
  model artifacts. Referenced by path from the control plane rows.

This mirrors how production platforms split a metadata DB from object storage; we
use a file and a folder instead of Postgres + S3. Swapping either is localized:
the DB URL in `db/session.py`, the artifact paths from `config.py`.

## The three seams (interfaces) that make it swappable

Every place where "the platform" meets "heavy/optional machinery" is an interface
with a local default and a real backend:

| Seam | Interface | Local default | Real backend |
| ---- | --------- | ------------- | ------------ |
| Training | `training/base.py::Trainer` | `SimulatedTrainer` | `HFLoRATrainer` ([ml]) |
| Serving | `deploy/base.py::Deployer` | `LocalDeployer` | `K8sDeployer` |
| Orchestration | `orchestrator.run_job` | in-process | Airflow DAG |
| Registry | `registry/registry.py` | SQLite | MLflow |
| Streaming | `pubsub.py::LogBroker` | in-process queues | Redis/Kafka |

Pick the backend with an env var; nothing else changes. This is the single most
important design idea in the project — see
[ADR 0003](adr/0003-simulated-trainer-by-default.md) and
[ADR 0004](adr/0004-local-first-infrastructure.md).

## Job state machine

```
QUEUED → PREPARING → TRAINING → EVALUATING → REGISTERING → COMPLETED
                                                          ↘ COMPLETED_BELOW_THRESHOLD
   (any step) ─────────────────────────────────────────── → FAILED
```

Defined in `db/models.py::JobStatus`; driven by `training/orchestrator.py`; the
same states are the Airflow DAG's tasks.
```
