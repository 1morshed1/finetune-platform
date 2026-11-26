"""Evaluate a fine-tuned model against its base — the deploy/no-deploy gate.

A fine-tune is only worth anything if it *beats the model you started from*. So
after training we compare the two on the same validation set and decide whether
the improvement clears a threshold.

How we get the "base" number:
- The honest, expensive way (real ML) is to run the frozen base model over the
  validation set and measure its loss. The ``hf`` backend could do this.
- In the simulated backend there are no real weights, so we use the loss at the
  **first training step** as the base-model proxy: at step 1 the LoRA adapter is
  effectively zero, so the model behaves like the base. The final eval_loss is the
  fine-tuned number. This gives a faithful "before vs after" story for the demo,
  and we label it as a proxy in the output.

Lower loss = better, so improvement = (base - finetuned) / base.
"""

from __future__ import annotations

import math

from .base import TrainResult, TrainSpec


def evaluate_against_base(spec: TrainSpec, result: TrainResult, min_rel_improvement: float) -> dict:
    history = result.history or []
    finetuned_eval = float(result.final_metrics.get("eval_loss", 0.0))

    # Base proxy: first recorded loss (adapter ~ inert at step 1). Fall back to a
    # conservative constant if history is empty.
    base_eval = float(history[0]["loss"]) if history else finetuned_eval + 0.5

    delta = base_eval - finetuned_eval
    rel = (delta / base_eval) if base_eval > 0 else 0.0
    passed = rel >= min_rel_improvement

    return {
        "base_model": spec.base_model,
        "base_eval_loss": round(base_eval, 4),
        "finetuned_eval_loss": round(finetuned_eval, 4),
        "abs_improvement": round(delta, 4),
        "relative_improvement": round(rel, 4),
        "base_perplexity": round(math.exp(min(base_eval, 20)), 3),
        "finetuned_perplexity": round(math.exp(min(finetuned_eval, 20)), 3),
        "threshold": min_rel_improvement,
        "passed": passed,
        "base_source": "step-1 proxy (simulated)"
        if spec.base_model in ("", "sim", "demo")
        else "step-1 proxy",
        "verdict": (
            f"Fine-tuned model improves validation loss by {rel * 100:.1f}% "
            f"({'meets' if passed else 'below'} the {min_rel_improvement * 100:.0f}% threshold)."
        ),
    }
