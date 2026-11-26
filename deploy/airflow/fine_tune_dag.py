"""Airflow DAG: the production form of ftp.training.orchestrator.

This is the "real" orchestration backend documented in ADR 0004. The local
in-process orchestrator (`ftp.training.orchestrator.run_job`) walks a job through
the exact same five steps this DAG defines — they are deliberately 1:1, so reading
one teaches the other:

    prepare >> train >> evaluate >> register >> notify

To actually run this you'd install Airflow, drop this file in `dags/`, and have
the platform *trigger* the DAG (via the Airflow REST API) instead of calling
`submit()` in-process. Each task reuses the same `ftp` code the local orchestrator
uses, so there is no second implementation to keep in sync — only the scheduling
layer changes.

Not imported by the app; it's a reference artifact. Guarded so the repo doesn't
need Airflow installed to be tested.
"""

from __future__ import annotations

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:  # Airflow not installed — this file is documentation here.
    DAG = None  # type: ignore


# --- Task callables: thin wrappers over the same ftp code ------------------
# In production these push/pull small values via XCom (the dataset id, the job id,
# metric summaries). The heavy artifacts (weights) go to object storage, not XCom.


def prepare(**ctx):
    from pathlib import Path

    from ftp.config import get_settings
    from ftp.data.validator import validate_file
    from ftp.db import Dataset, session_scope

    job_id = ctx["dag_run"].conf["job_id"]
    with session_scope() as s:
        from ftp.db import Job

        job = s.get(Job, job_id)
        ds = s.get(Dataset, job.dataset_id)
        res = validate_file(Path(ds.path), get_settings())
    ctx["ti"].xcom_push(key="n_train", value=len(res.train_indices))
    return job_id


def train(**ctx):
    # Real training runs here on a GPU worker (KubernetesPodOperator in practice).
    # For parity with the local path we call the same trainer factory.
    job_id = ctx["dag_run"].conf["job_id"]
    # ... build TrainSpec from the job + dataset, call get_trainer(backend).train(...)
    # ... stream metrics to MLflow; store the adapter in object storage.
    return job_id


def evaluate(**ctx):
    # Reuse ftp.training.evaluate.evaluate_against_base against the stored metrics.
    return ctx["dag_run"].conf["job_id"]


def register(**ctx):
    # Reuse ftp.registry.register_model (or push to a real MLflow registry).
    return ctx["dag_run"].conf["job_id"]


def notify(**ctx):
    # Send a webhook / email with the result. See the spec's "notify" step.
    return ctx["dag_run"].conf["job_id"]


if DAG is not None:
    default_args = {
        "owner": "ftp",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    }
    with DAG(
        dag_id="fine_tune_job",
        description="Fine-tune a model: prepare -> train -> evaluate -> register -> notify",
        schedule=None,  # triggered by the platform, not on a cron
        start_date=datetime(2024, 1, 1),
        catchup=False,
        default_args=default_args,
        tags=["ftp", "fine-tuning"],
    ) as dag:
        t_prepare = PythonOperator(task_id="prepare", python_callable=prepare)
        t_train = PythonOperator(task_id="train", python_callable=train)
        t_evaluate = PythonOperator(task_id="evaluate", python_callable=evaluate)
        t_register = PythonOperator(task_id="register", python_callable=register)
        t_notify = PythonOperator(task_id="notify", python_callable=notify)

        t_prepare >> t_train >> t_evaluate >> t_register >> t_notify
