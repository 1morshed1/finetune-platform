# Phase 1 — Project scaffold

**Goal:** a runnable skeleton with the shape of a real platform: packaging,
config, a database for control-plane state, and the documentation habit.

## What got built

```
pyproject.toml        packaging + dependency extras
src/ftp/
  __init__.py
  config.py           typed settings from env vars
  db/
    models.py         ORM: Dataset, Job, JobLog, ModelRecord, Deployment
    session.py        engine + transactional session scope
docs/                 this journal, ADRs, concept primers
```

## Concepts & decisions

### Packaging with a `src/` layout and `uv`
The code lives under `src/ftp/` rather than `ftp/` at the repo root. The `src`
layout prevents a classic bug: without it, `import ftp` can accidentally pick up
the *source* directory instead of the *installed* package, so your tests pass
against uninstalled code and then break for real users. With `src/`, you must
install the package (even in editable mode) to import it — so you test what you
ship. `uv` handles the virtualenv and lockfile; `uv sync` is the one command to
reproduce the environment.

### Dependency extras — see [ADR 0002](../adr/0002-dependency-strategy.md)
Core install has **no ML libraries**. Torch/transformers/peft are in the `[ml]`
extra. This keeps the project clone-and-runnable and forces good architecture:
the heavy stuff sits behind interfaces.

### Config as one typed object
`config.py` uses `pydantic-settings`. Every knob is an `FTP_`-prefixed env var
with a default and a type, validated at startup. The alternative — scattering
`os.environ.get(...)` calls through the code — hides the full set of knobs and
fails late. One `Settings` object is the platform's control panel.

### The database *is* the control plane
An ML platform has two planes:
- **Control plane** — metadata: which datasets exist, what jobs ran, which model
  came from which dataset+config. Small, structured, queried constantly.
- **Data plane** — the bulk bytes: the datasets themselves, the model weights.
  Big, opaque, streamed.

We put the control plane in **SQLite** (`db/models.py`) and the data plane on the
**filesystem** under `data_store/` (paths referenced from the DB rows). This
separation is exactly what the big platforms do (a metadata DB + object storage);
we just use a file and a folder instead of Postgres + S3. See
[ADR 0004](../adr/0004-local-first-infrastructure.md).

The five tables map directly to the lifecycle in the spec:

```
Dataset ──▶ Job ──▶ ModelRecord ──▶ Deployment
                └──▶ JobLog (streamed training logs)
```

### Why JSON columns for hyperparameters / metrics / reports
Those shapes change often and we never filter *inside* them in SQL — we load the
row and use Python. Exploding them into their own tables would be premature
normalisation. Storing them as JSON columns is the pragmatic, honest choice for a
metadata store. (If we ever needed `WHERE metrics.eval_loss < 0.5` in SQL, we'd
revisit.)

### Prefixed random IDs
IDs look like `ds_a1b2c3d4`, `job_9f3c1a2b`. Benefits over autoincrement ints:
safe in URLs, don't leak row counts, and self-describe their type in logs.

## Try it

```bash
uv sync
uv run python -c "from ftp.db import get_engine, Base; get_engine(); print(list(Base.metadata.tables))"
# -> ['datasets', 'jobs', 'job_logs', 'models', 'deployments']
```

## Next
[Phase 2 — the data-validation pipeline](02-data-validation.md): the code that
decides whether a dataset is worth spending money to train on.
