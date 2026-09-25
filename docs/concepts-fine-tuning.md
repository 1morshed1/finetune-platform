# Concepts: what fine-tuning actually is

If the ML vocabulary here is new, read this once. Everything else in the repo
assumes these ideas.

## The one-paragraph version

A large language model is a giant function with billions of numbers (weights)
inside it, trained on the whole internet to predict the next token. **Fine-tuning**
means: take that pre-trained model and nudge its weights using *your* small,
specific dataset so it gets better at *your* task (your tone, your format, your
domain). You are not training from scratch — you're specialising something that
already knows language.

## Full fine-tuning vs LoRA vs QLoRA

Updating *all* billions of weights (full fine-tuning) needs enormous GPU memory
(you must store the weights, their gradients, and optimizer state — roughly
4× the model size). Most people can't afford that.

**LoRA (Low-Rank Adaptation)** is the trick that made fine-tuning accessible:

- Freeze the original weights. Don't touch them.
- Next to each big weight matrix `W` (shape `d × d`), add two tiny matrices
  `A` (`d × r`) and `B` (`r × d`) where the *rank* `r` is small (e.g. 8 or 16).
- The model now computes `W·x + B·A·x`. Only `A` and `B` are trained.
- `A` and `B` together have `2·d·r` parameters instead of `d²` — often **<1%**
  of the model. Tiny to train, tiny to store (a few MB "adapter").

The output of LoRA training is that little **adapter**, not a whole new model.
At inference you load the base model once and snap different adapters onto it —
which is exactly why the plan mentions "LoRA adapter hot-loading".

**QLoRA** = LoRA + the frozen base model quantized to 4-bit, so it fits in even
less GPU memory. Same idea, smaller footprint.

> In this platform, `lora_r` (the rank `r` above) is a hyperparameter you pass to
> a fine-tune job. The `hf` trainer backend implements real LoRA via HuggingFace
> `peft`; the default `sim` backend just mimics the resulting loss curve.

## Why we obsess over the loss curve

**Loss** is a number measuring how wrong the model's predictions are on the
training data. Training repeatedly:

1. shows the model a batch of examples,
2. measures the loss,
3. nudges the trainable weights to reduce it.

Over many **steps** the loss goes *down* and (usually) *flattens*. Plotting loss
vs step is the single most important diagnostic:

- Going down and flattening → learning, then converged. Good.
- Not going down → learning rate wrong, or data broken.
- Training loss down but **eval** loss going *up* → **overfitting** (memorising
  the training set, getting worse at unseen data). This is why we always hold out
  a validation split and plot *both* curves.

The simulated trainer models loss as `L(step) = (L0 − floor)·e^(−k·step) + floor`
plus noise — the classic exponential-decay-to-a-floor shape real training shows.

## Epoch, step, batch

- **Batch**: a handful of examples processed together (e.g. 8).
- **Step**: one weight update = one batch.
- **Epoch**: one full pass over the whole dataset.

So `steps_per_epoch = ceil(num_examples / batch_size)` and
`total_steps = epochs · steps_per_epoch`. The platform uses exactly this to
estimate ETA and to size the loss curve.

## "Evaluate vs base"

A fine-tuned model is only worth deploying if it beats the model you started
from. So after training we run the **validation set** through both the base model
and the fine-tuned model and compare metrics (loss/perplexity, and BLEU/ROUGE for
generation tasks). If the fine-tuned model isn't meaningfully better, the job is
marked `completed_below_threshold` and the model is registered but flagged — the
platform refuses to pretend a non-improvement is a win. That honesty *is* the
platform.

## Perplexity, BLEU, ROUGE (30-second definitions)

- **Perplexity**: `exp(loss)`. "How surprised is the model, on average?" Lower is
  better. A perplexity of 10 ≈ the model is as unsure as if picking among 10
  equally-likely next tokens.
- **BLEU / ROUGE**: overlap metrics between generated text and reference text
  (BLEU leans precision, ROUGE leans recall). Used for summarisation/translation-
  style tasks where there's a "right answer" to compare against.
