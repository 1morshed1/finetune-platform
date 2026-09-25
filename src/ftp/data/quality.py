"""Per-example quality scoring.

The plan calls for "quality scoring (use perplexity from a small model)". That is
the *ideal*: run each example through a small LM and use its perplexity — high
perplexity means the text is weird/noisy/low-quality. But that needs the ``[ml]``
extra and a model download, so it can't be the default (ADR 0002).

So we provide two scorers behind one function:

- **heuristic** (default, no deps): a blend of cheap signals that correlate with
  quality — emptiness, extreme length, character-junk ratio, and lexical
  repetition. Fast, offline, deterministic.
- **perplexity** (``[ml]`` extra): the real thing, using a small model.

Both return a score in ``[0, 1]`` where higher = better, so the rest of the
pipeline treats them identically. An example scoring below a threshold (default
0.3, from the plan) is flagged "low quality" for human review — never deleted
automatically. A platform advises; the user decides.
"""

from __future__ import annotations

import math
import re

from .tokenizer import count_tokens

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# "Printable-ish": letters, digits, common punctuation, whitespace. Anything else
# (control chars, mojibake, stray binary) counts as junk.
_JUNK_RE = re.compile(r"[^\w\s.,!?;:'\"()\-–—/%$&@#*+=\[\]{}<>|~`^\\]", re.UNICODE)


def _repetition_penalty(text: str) -> float:
    """1.0 = all tokens distinct; →0 = highly repetitive.

    Uses the type/token ratio (unique tokens / total tokens). Degenerate data like
    "yes yes yes yes ..." scores near 0. Short texts are exempt (ratio is
    meaningless with few tokens)."""
    tokens = _TOKEN_RE.findall(text.lower())
    if len(tokens) < 8:
        return 1.0
    return len(set(tokens)) / len(tokens)


def _junk_penalty(text: str) -> float:
    """1.0 = clean; →0 = lots of non-printable / encoding-junk characters."""
    if not text:
        return 0.0
    junk = len(_JUNK_RE.findall(text))
    return max(0.0, 1.0 - junk / len(text))


def _length_score(prompt_tokens: int, completion_tokens: int) -> float:
    """Penalise the extremes: empty examples and absurdly long ones.

    A healthy supervised example has a non-trivial prompt and completion and fits
    a normal context window. We use smooth ramps rather than hard cutoffs so the
    score degrades gracefully."""
    if prompt_tokens == 0 or completion_tokens == 0:
        return 0.0
    total = prompt_tokens + completion_tokens
    # Too short: <4 tokens total is barely an example.
    short = min(1.0, total / 4.0)
    # Too long: start penalising past ~4k tokens (context-window pressure).
    long = 1.0 if total <= 4096 else max(0.2, 4096 / total)
    return short * long


def heuristic_score(prompt: str, completion: str) -> float:
    """Blend the cheap signals into a single [0,1] quality score."""
    pt, ct = count_tokens(prompt), count_tokens(completion)
    length = _length_score(pt, ct)
    rep = _repetition_penalty(prompt + "\n" + completion)
    junk = _junk_penalty(prompt + "\n" + completion)
    # Geometric-ish mean: any single signal being terrible drags the score down,
    # which is what we want (one broken dimension = a bad example).
    return round(length * (0.5 + 0.5 * rep) * (0.5 + 0.5 * junk), 4)


def score_quality(examples: list[dict], method: str = "auto") -> tuple[list[float], str]:
    """Score every example. Returns (scores, method_used).

    method: "heuristic", "perplexity", or "auto" (perplexity if the [ml] extra is
    importable, else heuristic).
    """
    if method == "auto":
        method = "perplexity" if _perplexity_available() else "heuristic"

    if method == "perplexity":
        try:
            return _perplexity_scores(examples), "perplexity"
        except Exception:
            method = "heuristic"  # graceful fallback if the model can't load

    return [heuristic_score(ex["prompt"], ex["completion"]) for ex in examples], "heuristic"


def _perplexity_available() -> bool:
    try:
        import torch  # noqa: F401,PLC0415
        import transformers  # noqa: F401,PLC0415

        return True
    except Exception:
        return False


def _perplexity_scores(examples: list[dict]) -> list[float]:
    """Real quality via language-model perplexity (needs the [ml] extra).

    We run each example through a small causal LM (distilgpt2), compute perplexity
    = exp(loss), and map it to [0,1] with a squashing function: low perplexity
    (fluent, predictable text) -> high score. Lazy-imported by design.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.eval()

    scores: list[float] = []
    for ex in examples:
        text = f"{ex['prompt']}\n{ex['completion']}"
        ids = tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            loss = model(**ids, labels=ids["input_ids"]).loss.item()
        ppl = math.exp(min(loss, 20))  # clamp to avoid overflow
        # Map perplexity to [0,1]: ppl<=20 is great (~1.0), ppl>=200 is poor (~0).
        lo, hi = math.log(20), math.log(200)
        score = max(0.0, min(1.0, 1.0 - (math.log(ppl) - lo) / (hi - lo)))
        scores.append(round(score, 4))
    return scores
