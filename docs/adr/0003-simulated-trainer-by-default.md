# ADR 0003 — A pluggable trainer, simulated by default

- **Status:** Accepted
- **Date:** 2026-09-25

## Context

The headline feature is "fine-tune a model". But:

1. Real fine-tuning needs a GPU to finish in reasonable time.
2. It needs gigabytes of model weights downloaded.
3. It is non-deterministic and slow, which makes it a nightmare to test and to
   demo.

Yet the *platform* behaviour we want to show — a job that moves through states,
emits loss curves that go down, streams logs, produces metrics, and registers a
model — does not actually require real gradients to be demonstrated or tested.

## Decision

Define a `Trainer` interface (`ftp.training.base.Trainer`) with one job:

```python
def train(self, spec: TrainSpec, on_event: Callable[[TrainEvent], None]) -> TrainResult
```

Provide two implementations behind a factory keyed on `FTP_TRAINER_BACKEND`:

- **`sim`** (default) — a `SimulatedTrainer` that runs a real optimisation-shaped
  loop: loss decays like `L(step) = L0 * exp(-k·step) + noise + floor`, eval
  metrics improve, logs stream, and it respects the configured epochs/steps. No
  ML dependency. Deterministic given a seed.
- **`hf`** — a `HFLoRATrainer` that runs an actual LoRA/QLoRA fine-tune via
  transformers + peft. Imported lazily; only works with the `[ml]` extra.

Both emit the *same* `TrainEvent` stream, so everything downstream (metric
logging, SSE, registry) is identical regardless of backend.

## Consequences

- The platform is fully demonstrable and testable on any laptop.
- Swapping to real training is a one-env-var change, and the real trainer only
  has to satisfy the same interface — no changes anywhere else.
- The simulated curves are *honest about being simulated*: the API marks
  `backend: "sim"` on the job so nobody mistakes demo output for real results.

## Alternatives considered

- **Only real training** — not runnable/testable without a GPU; kills the
  learning goal.
- **Only simulated** — then the project can't claim to actually fine-tune. The
  interface + real backend keeps the claim honest.
- **Tiny real model (e.g. distilgpt2 on CPU)** — still slow and flaky for tests;
  we keep it available via the `hf` backend but don't make it the default.
