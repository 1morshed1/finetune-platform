"""HFLoRATrainer — a *real* LoRA fine-tune via HuggingFace transformers + peft.

Only used when ``FTP_TRAINER_BACKEND=hf`` and the ``[ml]`` extra is installed. It
emits the exact same TrainEvent stream as the simulated trainer, so the
orchestrator, SSE, and registry don't change at all — that's the payoff of the
Trainer interface (ADR 0003).

This runs on CPU with a tiny model (distilgpt2) so it *can* complete without a
GPU, just slowly. For anything serious you'd point ``base_model`` at a real model
and run on a GPU box. All heavy imports are lazy so importing this module never
drags in torch.
"""

from __future__ import annotations

import math
from pathlib import Path

from .base import TrainEvent, TrainResult, TrainSpec


class HFLoRATrainer:
    backend = "hf"

    def __init__(self, default_model: str = "distilgpt2") -> None:
        self.default_model = default_model

    def train(self, spec: TrainSpec, on_event) -> TrainResult:  # pragma: no cover - needs [ml]
        try:
            import torch  # noqa: F401  (imported to verify the [ml] extra is present)
            from datasets import Dataset
            from peft import LoraConfig, get_peft_model
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                Trainer,
                TrainerCallback,
                TrainingArguments,
            )
        except ImportError as e:
            raise RuntimeError(
                "The 'hf' trainer backend requires the [ml] extra. "
                "Install with: uv sync --extra ml"
            ) from e

        # Fall back to a small CPU-friendly model if the spec didn't name a real
        # HF model id (e.g. the "demo" placeholder used by the simulated path).
        model_name = spec.base_model
        if spec.base_model in ("", "sim", "demo"):
            model_name = self.default_model

        on_event(TrainEvent(kind="log", message=f"Loading base model {model_name} ..."))
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)

        lora = LoraConfig(
            r=spec.lora_r,
            lora_alpha=int(spec.hyperparameters.get("lora_alpha", 2 * spec.lora_r)),
            lora_dropout=float(spec.hyperparameters.get("lora_dropout", 0.05)),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        on_event(TrainEvent(kind="log", message="Wrapped model with LoRA adapters."))

        def to_text(ex: dict) -> dict:
            return {"text": f"{ex['prompt']}\n{ex['completion']}{tok.eos_token}"}

        def tokenize(batch):
            out = tok(batch["text"], truncation=True, max_length=512, padding="max_length")
            out["labels"] = out["input_ids"].copy()
            return out

        train_ds = Dataset.from_list([to_text(e) for e in spec.train_examples]).map(tokenize)
        val_ds = Dataset.from_list([to_text(e) for e in spec.val_examples]).map(tokenize)

        art_dir = Path(spec.artifacts_dir) / spec.job_id
        art_dir.mkdir(parents=True, exist_ok=True)

        history: list[dict] = []

        class _EventCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                if not logs:
                    return
                step = int(state.global_step)
                loss = logs.get("loss")
                eval_loss = logs.get("eval_loss")
                if loss is not None or eval_loss is not None:
                    point = {"step": step, "loss": loss, "eval_loss": eval_loss,
                             "lr": logs.get("learning_rate", spec.learning_rate)}
                    history.append(point)
                    on_event(TrainEvent(
                        kind="metric", step=step, total_steps=spec.total_steps,
                        loss=loss, eval_loss=eval_loss, lr=point["lr"],
                    ))
                    on_event(TrainEvent(
                        kind="log",
                        message=f"step {step}/{spec.total_steps} " +
                                (f"loss={loss:.4f} " if loss else "") +
                                (f"eval_loss={eval_loss:.4f}" if eval_loss else ""),
                    ))

        args = TrainingArguments(
            output_dir=str(art_dir / "checkpoints"),
            num_train_epochs=spec.epochs,
            per_device_train_batch_size=spec.batch_size,
            learning_rate=spec.learning_rate,
            logging_steps=1,
            eval_strategy="epoch",
            save_strategy="no",
            report_to=[],
            seed=spec.seed,
        )
        trainer = Trainer(
            model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
            callbacks=[_EventCallback()],
        )
        trainer.train()
        eval_metrics = trainer.evaluate()

        model.save_pretrained(str(art_dir))  # saves just the LoRA adapter
        eval_loss = float(eval_metrics.get("eval_loss", history[-1].get("loss", 1.0)))
        result = TrainResult(
            final_metrics={
                "train_loss": history[-1].get("loss") if history else None,
                "eval_loss": eval_loss,
                "perplexity": round(math.exp(min(eval_loss, 20)), 3),
                "steps": int(getattr(trainer.state, "global_step", spec.total_steps)),
            },
            artifact_path=str(art_dir),
            history=history,
        )
        on_event(TrainEvent(kind="log", message=f"Training complete. eval_loss={eval_loss:.4f}"))
        return result
