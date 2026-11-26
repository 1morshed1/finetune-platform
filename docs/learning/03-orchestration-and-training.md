# Phase 3 — orchestration & the pluggable trainer

**Goal:** turn a validated dataset into a registered model by running the spec's
DAG — `prepare → train → evaluate → register → notify` — with a trainer that runs
on any laptop but can be swapped for real LoRA.

## What got built

```
src/ftp/training/
  base.py          Trainer interface: TrainSpec / TrainEvent / TrainResult
  simulated.py     SimulatedTrainer  (default, no GPU, real loss-curve shape)
  hf_lora.py       HFLoRATrainer     (real LoRA via transformers+peft, [ml] extra)
  factory.py       get_trainer(backend)
  evaluate.py      evaluate_against_base()  -> deploy/no-deploy gate
  pubsub.py        LogBroker  (thread→async fan-out for live logs)
  orchestrator.py  run_job()  -> the DAG state machine
src/ftp/registry/
  registry.py      register_model / get_model / leaderboard  (lineage store)
tests/test_training.py   4 new tests
```

## The big ideas

### 1. Program to an interface, inject the implementation (ADR 0003)
`base.py` defines a three-type contract: give the trainer a `TrainSpec`, it emits
`TrainEvent`s and returns a `TrainResult`. The orchestrator only knows this
contract. So the *simulated* trainer and the *real LoRA* trainer are
interchangeable — `FTP_TRAINER_BACKEND=hf` swaps them with zero changes anywhere
else. This is dependency inversion, and it's why the platform is both runnable-
anywhere and capable of real training.

Read `simulated.py` to see how a believable loss curve is manufactured:
`L(step) = (L0 − floor)·e^(−k·step) + floor + noise`, with `k` reacting to
learning rate and LoRA rank, and an eval curve whose gap to train loss *widens*
on small datasets — i.e. it renders overfitting correctly. The point isn't to
fake results; it's to exercise every downstream system with honest-shaped data
(and the API labels the backend `sim` so nobody's fooled).

### 2. A DAG is just a state machine
`orchestrator.py::run_job` walks the Job through
`PREPARING → TRAINING → EVALUATING → REGISTERING → COMPLETED`. In production each
arrow is an Airflow task (`deploy/airflow/`); here they're sequential steps in one
function. Same states, same data, no scheduler required (ADR 0004). Modelling it
as an explicit state machine means the dashboard can show exactly where a job is,
and a failure at any step lands in a clean `FAILED` state with the error recorded
— **a job never silently hangs** (there's a test for the failure path).

### 3. Crossing the thread/async boundary — `LogBroker`
Training runs in a background **thread**; the SSE log endpoint is **async**.
`pubsub.py` is a tiny thread-safe fan-out: each subscriber (SSE request) gets its
own queue, so a slow browser can't stall the trainer or other viewers. A `None`
sentinel means "stream done". This is the in-process version of Redis/Kafka pub-
sub, behind an interface so it can be swapped later.

### 4. Lineage is the registry's reason to exist
`registry.py` records, for every model, the **dataset + job + hyperparameters**
that produced it, plus metrics and the base-model comparison. That triple is what
makes a result reproducible and a leaderboard meaningful. Storing a model without
its lineage is how teams end up with a "best_model_final_v2.bin" nobody can
rebuild.

### 5. The honesty gate — `evaluate_against_base`
A fine-tune that doesn't beat the base model is not a success. The evaluator
computes relative improvement in validation loss and, if it's below
`eval_min_relative_improvement`, the job ends `completed_below_threshold` and the
model is registered **but flagged**. The platform refuses to call a
non-improvement a win. (In `sim`, the "base" number is the step-1 loss, when the
LoRA adapter is still inert — a faithful before/after proxy, labelled as such.)

## Mistakes / decisions worth noting
- **One session per job vs per step.** I hold a single DB session for the whole
  `run_job` and flush at transitions, rather than opening a session per log line.
  Simpler and transactional; fine for SQLite at this scale.
- **Persisting the curve mid-run.** Metrics are written to the DB every 5 steps
  (not just at the end) so a page refresh during training shows progress.
- **Determinism for tests.** `FTP_SIM_STEP_DELAY=0` (set in `conftest.py`) makes
  simulated training instant, so the full DAG runs inline in a unit test.

## Try it
```bash
uv run pytest tests/test_training.py -v
```

## Next
[Phase 4 — registry & deployment](04-registry-and-deploy.md): serving a model and
the K8s story.
