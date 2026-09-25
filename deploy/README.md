# Deployment & infrastructure

This folder holds the "real infrastructure" artifacts. The platform runs fully
**without** any of it (that's the point of [ADR 0004](../docs/adr/0004-local-first-infrastructure.md));
these show how each local stand-in scales up.

## The mapping

| Concept | Local default (what runs now) | Real backend (here) |
| ------- | ----------------------------- | ------------------- |
| Orchestration | `ftp.training.orchestrator.run_job` (in-process state machine) | [`airflow/fine_tune_dag.py`](airflow/fine_tune_dag.py) |
| Metric tracking / registry | SQLite `ModelRecord` table | MLflow (compose service, commented in `docker-compose.yml`) |
| Model serving | `LocalDeployer` (in-process endpoint) | [`k8s/model-serving.example.yaml`](k8s/model-serving.example.yaml), rendered by `ftp.deploy.k8s` |
| The platform itself | `ftp serve` | [`k8s/platform.yaml`](k8s/platform.yaml) + [`Dockerfile`](../Dockerfile) |

The important property: the local and real backends run the **same `ftp` code**.
Only the scheduling/serving layer changes. There is no duplicate implementation to
keep in sync — the Airflow tasks call the same trainer/evaluator/registry
functions the local orchestrator calls.

## Run it in Docker

```bash
docker build -t finetune-platform:latest .
docker run -p 8000:8000 -v ftp_data:/data finetune-platform:latest
# open http://localhost:8000
```

or with compose (persistent volume, restart policy):

```bash
cd deploy && docker compose up --build
```

## Deploy the platform to Kubernetes

```bash
# Build + push the image to a registry your cluster can pull, then:
kubectl apply -f deploy/k8s/platform.yaml
kubectl -n finetune-platform port-forward svc/ftp-api 8000:80
```

Notes:
- The control plane runs a **single** replica because its state is SQLite (single
  writer). To scale it out you'd swap SQLite for Postgres — a one-line change in
  `ftp/db/session.py` (the URL) since we use no SQLite-specific SQL.
- Model-serving deployments are **generated per model** at deploy time by
  `ftp.deploy.k8s.render_manifests`. `model-serving.example.yaml` is a sample of
  that output: a vLLM pod serving a base model with the LoRA adapter mounted, a
  Service, and an Ingress carrying the canary weight.

## Airflow orchestration

`airflow/fine_tune_dag.py` is the DAG form of the orchestrator. To use it:
1. Install Airflow and put the file in your `dags/` folder.
2. Make the platform *trigger* the DAG (Airflow REST API) instead of calling
   `submit()` in-process.
3. Run `train` as a `KubernetesPodOperator` on a GPU node pool.

The DAG's tasks (`prepare → train → evaluate → register → notify`) are the exact
states in `ftp.db.models.JobStatus`.

## Real training (`hf` backend)

```bash
uv sync --extra ml
FTP_TRAINER_BACKEND=hf uv run ftp serve
```

Runs an actual LoRA fine-tune via transformers + peft. On CPU it uses a tiny model
(distilgpt2) and is slow; point `base_model` at a real model and run on a GPU host
for anything serious.
