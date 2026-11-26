# Build Journal — index

The platform was built in phases. Each phase has its own entry with the
concepts, the code, and the tradeoffs. Read them in order.

| Phase | Entry | What got built |
| ----- | ----- | -------------- |
| 1 | [01-project-scaffold](01-project-scaffold.md) | Repo layout, packaging with `uv`, config, DB models, the "local-first" philosophy |
| 2 | [02-data-validation](02-data-validation.md) | The data-quality pipeline: format checks, MinHash dedup, token/cost analysis, the quality report |
| 3 | [03-orchestration-and-training](03-orchestration-and-training.md) | Job orchestrator, the pluggable trainer (simulated + real LoRA), metric logging |
| 4 | [04-registry-and-deploy](04-registry-and-deploy.md) | Model registry with lineage, evaluation vs base, deployment backends |
| 5 | [05-api-and-dashboard](05-api-and-dashboard.md) | FastAPI routers, SSE log streaming, the dashboard |
| 6 | [06-infrastructure](06-infrastructure.md) | Docker, Airflow DAG, K8s manifests — the real backends the local ones stand in for |

See also [`../architecture.md`](../architecture.md) for the one-page system map.

## The one big idea

A **platform** is not a script. The spec says this explicitly. The whole project
is an exercise in the difference:

- A *script* fine-tunes one model on one dataset.
- A *platform* accepts *any* dataset from *any* user, tells them whether it's
  good enough **before** spending money, orchestrates the job, tracks lineage so
  results are reproducible, and serves the result.

Everywhere you see extra code that a script wouldn't have — validation,
reports, registries, lineage, status tracking — that is the platform tax, and
it is the whole point.

## Reading order for the concepts

If you're new to fine-tuning itself, read
[`../concepts-fine-tuning.md`](../concepts-fine-tuning.md) first. It explains
full fine-tuning vs LoRA/QLoRA, why we log loss curves, and what "evaluate vs
base" means — without which the rest won't click.
