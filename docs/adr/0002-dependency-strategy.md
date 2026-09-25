# ADR 0002 — Lightweight core, heavy ML as an optional extra

- **Status:** Accepted
- **Date:** 2026-09-25

## Context

A fine-tuning platform *sounds* like it must depend on PyTorch, transformers,
peft, CUDA, MLflow, Airflow, Kubernetes. Installing all of that is gigabytes,
needs a GPU to be useful, and makes the project impossible to clone-and-run for
someone learning from it.

But 90% of the *interesting platform logic* — validation, cost estimation,
orchestration, lineage, the API, the dashboard — has nothing to do with GPUs.

## Decision

Split dependencies:

- **Core** (`pip install .`): FastAPI, SQLAlchemy, numpy, pydantic. No ML libs.
  The full platform runs and is testable with just these.
- **`[ml]` extra**: torch, transformers, peft, datasets, tiktoken. Only needed
  to run a *real* fine-tune.
- **`[mlflow]` extra**: real experiment tracking backend.
- **`[dev]` extra**: pytest, ruff, httpx.

Any code that touches an optional dependency imports it **lazily** (inside the
function, not at module top) and degrades gracefully with a clear message if
it's missing.

## Consequences

- `uv sync` is fast and works on any machine. Reviewers can run the whole thing.
- The ML path is opt-in and clearly separated, which is also good architecture:
  the trainer is behind an interface (see [ADR 0003](0003-simulated-trainer-by-default.md)).
- Cost: a small amount of lazy-import boilerplate, and we must make sure tests
  cover both the "extra present" and "extra absent" paths.

## Alternatives considered

- **Everything in core** — too heavy, not runnable for learners, slow CI.
- **Two separate packages** — overkill for a portfolio project; extras are the
  standard Python idiom for exactly this.
