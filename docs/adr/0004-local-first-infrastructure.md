# ADR 0004 — Local-first infrastructure with documented "real" backends

- **Status:** Accepted
- **Date:** 2026-09-25

## Context

The plan names specific infrastructure: **Airflow** for orchestration, **MLflow**
for the registry, **Kubernetes + vLLM** for deployment. Standing all of that up
is a multi-day devops exercise and needs a cluster. It would drown the actual
learning (how does a fine-tuning platform *work*?) in YAML.

## Decision

For each piece of infrastructure, ship a **local stand-in that implements the
same concept**, plus documentation/manifests for the real thing:

| Concept | Local default | Real backend (documented) |
| ------- | ------------- | ------------------------- |
| Orchestration | In-process job runner (`ftp.training.orchestrator`) with a state machine identical to a DAG's | Airflow DAG in `deploy/airflow/` |
| Metric tracking / registry | SQLite-backed registry with runs, params, metrics, artifacts, lineage | MLflow (`[mlflow]` extra + `deploy/` notes) |
| Deployment | `LocalDeployer` that loads the model into the running process and exposes it | K8s manifests in `deploy/k8s/` (Deployment/Service/Ingress) |

The local versions are not toys — they model the same states, data, and lineage
the real systems track. Reading them teaches the *concept*; the manifests show
the *production wiring*.

## Consequences

- `git clone && uv sync && ftp serve` gives a working platform. No cluster.
- The mapping table above is the study guide: "here is the idea, here is how it
  scales up."
- Cost: we maintain two mental models per subsystem. Mitigated by keeping the
  local implementations small and the real backends documentation-only.

## Alternatives considered

- **docker-compose with real Airflow + MLflow + Postgres** — provided as an
  *optional* `deploy/docker-compose.yml`, but not the default path, because it's
  heavy and hides the concepts behind service config.
