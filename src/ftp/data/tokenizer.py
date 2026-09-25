"""Token counting.

Why this matters: fine-tuning is billed and time-boxed **per token**, so the
platform must estimate token counts to (a) estimate cost, (b) estimate training
time, and (c) warn about examples that exceed a model's context window.

Two strategies, chosen automatically:

1. **tiktoken** (accurate) — the real BPE tokenizer used by GPT models. Available
   only with the ``[ml]`` extra, and its first use downloads a small vocab file.
2. **regex approximation** (always available) — no dependency, no network. Good
   enough for *estimates*; we are explicit that it is an approximation.

The whole platform stays runnable offline (ADR 0002) because strategy 2 is the
fallback. Everything imports :func:`count_tokens` and never cares which ran.
"""

from __future__ import annotations

import re
from functools import lru_cache

# One regex, compiled once. It splits text into "word-ish" units and individual
# punctuation marks — roughly the granularity a BPE tokenizer starts from.
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Empirical fudge factor: real BPE tokenizers split longer/rare words into
# multiple sub-word tokens, so the raw word+punct count undercounts. ~1.3 lines
# the approximation up with cl100k_base on typical English prose. It is a
# deliberate over-simplification; see the module docstring.
_SUBWORD_FACTOR = 1.3


@lru_cache(maxsize=1)
def _tiktoken_encoder():
    """Return a tiktoken encoder, or None if unavailable/offline.

    Cached so we attempt the (possibly network-hitting) load exactly once.
    """
    try:
        import tiktoken  # noqa: PLC0415  (lazy import by design — ADR 0002)

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        # tiktoken not installed, or the vocab couldn't be fetched offline.
        return None


def approximate_token_count(text: str) -> int:
    """Dependency-free estimate. Words + punctuation, scaled for sub-word splits."""
    if not text:
        return 0
    units = len(_WORD_RE.findall(text))
    return max(1, round(units * _SUBWORD_FACTOR))


def count_tokens(text: str) -> int:
    """Count tokens in ``text``, using tiktoken if available, else the estimate."""
    enc = _tiktoken_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return approximate_token_count(text)


def tokenizer_name() -> str:
    """Which strategy is active — surfaced in the quality report so the reader
    knows whether token counts are exact or approximate."""
    return "tiktoken/cl100k_base" if _tiktoken_encoder() is not None else "regex-approx"
