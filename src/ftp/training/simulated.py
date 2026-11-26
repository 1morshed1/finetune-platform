"""SimulatedTrainer — real training *shape*, no GPU.

Why this exists: see ADR 0003. We want to demonstrate and test the whole platform
— a job that streams logs, produces a loss curve that goes down, computes eval
metrics, and yields an artifact — without a GPU, model weights, or non-determinism.

The physics we imitate: in real fine-tuning, training loss follows roughly an
**exponential decay to a floor**:

    L(step) = (L0 - floor) * exp(-k * step) + floor  + small noise

- ``L0``   — starting loss (a fresh, un-adapted model on your data).
- ``floor``— the best loss achievable given the data (never reaches 0).
- ``k``    — how fast it learns; higher learning rate / more capacity => larger k.

We also model the **eval** curve, which tracks train loss but sits slightly above
it — and if the dataset is small, the gap *widens* over time. That widening gap is
exactly what overfitting looks like on a real chart, so the simulation teaches the
right intuition.

Everything is seeded, so a given (spec, seed) always produces the same curve.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

from .base import TrainEvent, TrainResult, TrainSpec


class SimulatedTrainer:
    backend = "sim"

    def __init__(self, step_delay: float | None = None) -> None:
        # Small per-step sleep so log/metric streaming is observable in the UI.
        # Set FTP_SIM_STEP_DELAY=0 in tests to make it instant.
        if step_delay is None:
            step_delay = float(os.environ.get("FTP_SIM_STEP_DELAY", "0.05"))
        self.step_delay = step_delay

    def _curve_params(self, spec: TrainSpec) -> tuple[float, float, float]:
        """Derive (L0, floor, k) from the spec so the curve reacts to the inputs."""
        n = max(1, len(spec.train_examples))
        # Bigger, cleaner datasets reach a lower floor.
        floor = max(0.25, 1.2 - 0.15 * math.log10(n + 1))
        L0 = 2.6
        # Learning speed grows with learning rate and LoRA rank (more capacity),
        # normalised so a typical config decays sensibly over the run.
        lr_factor = spec.learning_rate / 2e-4
        cap_factor = 1.0 + math.log2(max(1, spec.lora_r)) / 8
        k = (3.0 / max(1, spec.total_steps)) * lr_factor * cap_factor
        return L0, floor, k

    def train(self, spec: TrainSpec, on_event) -> TrainResult:
        rng = random.Random(spec.seed)
        L0, floor, k = self._curve_params(spec)
        total = spec.total_steps
        # Overfitting pressure: small datasets overfit, large ones don't.
        overfit = max(0.0, 0.5 - 0.1 * math.log10(len(spec.train_examples) + 1))

        on_event(
            TrainEvent(
                kind="log",
                message=(
                    f"Starting LoRA fine-tune of {spec.base_model} | "
                    f"{len(spec.train_examples)} train / {len(spec.val_examples)} val examples | "
                    f"epochs={spec.epochs} batch={spec.batch_size} lr={spec.learning_rate} "
                    f"lora_r={spec.lora_r} | {total} steps"
                ),
            )
        )

        history: list[dict] = []
        eval_interval = max(1, total // 10)  # ~10 eval points across the run
        last_eval_loss = None

        for step in range(1, total + 1):
            base = (L0 - floor) * math.exp(-k * step) + floor
            loss = round(base + rng.uniform(-0.03, 0.03), 4)

            point = {"step": step, "loss": loss, "lr": spec.learning_rate}

            # Evaluate periodically and at the final step.
            if step % eval_interval == 0 or step == total:
                # Eval sits above train loss; the gap grows with overfit pressure
                # as training progresses (fraction of run completed).
                progress_frac = step / total
                gap = 0.05 + overfit * progress_frac
                last_eval_loss = round(loss + gap + rng.uniform(0, 0.02), 4)
                point["eval_loss"] = last_eval_loss
                on_event(
                    TrainEvent(
                        kind="log",
                        level="info",
                        message=f"step {step}/{total}  loss={loss:.4f}  eval_loss={last_eval_loss:.4f}",
                    )
                )

            history.append(point)
            on_event(
                TrainEvent(
                    kind="metric",
                    step=step,
                    total_steps=total,
                    loss=loss,
                    eval_loss=point.get("eval_loss"),
                    lr=spec.learning_rate,
                )
            )
            if self.step_delay:
                time.sleep(self.step_delay)

        # Write a stand-in adapter artifact so the registry has something real to
        # point at. In the hf backend this would be the actual peft adapter dir.
        art_dir = Path(spec.artifacts_dir) / spec.job_id
        art_dir.mkdir(parents=True, exist_ok=True)
        adapter = {
            "type": "lora_adapter",
            "backend": "sim",
            "base_model": spec.base_model,
            "lora_r": spec.lora_r,
            "note": "Simulated adapter — no real weights. Produced by SimulatedTrainer.",
        }
        (art_dir / "adapter.json").write_text(json.dumps(adapter, indent=2))

        final_train = history[-1]["loss"]
        final_eval = last_eval_loss if last_eval_loss is not None else final_train
        result = TrainResult(
            final_metrics={
                "train_loss": final_train,
                "eval_loss": final_eval,
                "perplexity": round(math.exp(min(final_eval, 20)), 3),
                "steps": total,
            },
            artifact_path=str(art_dir),
            history=history,
        )
        on_event(
            TrainEvent(
                kind="log",
                message=f"Training complete. final eval_loss={final_eval:.4f} "
                f"perplexity={result.final_metrics['perplexity']:.2f}",
            )
        )
        return result
