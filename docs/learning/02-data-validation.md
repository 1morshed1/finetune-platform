# Phase 2 — the data-validation pipeline

**Goal:** implement the subsystem the plan calls out as what "differentiates a
platform from a script" — decide whether a dataset is worth training on, before
spending a cent.

## What got built

```
src/ftp/data/
  tokenizer.py   token counting (tiktoken or offline regex fallback)
  minhash.py     near-duplicate detection (MinHash + LSH, from scratch)
  quality.py     per-example quality scoring (heuristic or perplexity)
  report.py      DataQualityReport + Stats + recommendation generator
  validator.py   orchestrates the 7-step pipeline
tests/test_data_validation.py   12 tests, all green
```

## The mental model I want you to take away

A dataset goes through a funnel:

```
raw JSONL
  │  parse + format-check (per line, collect all errors)
  ▼
valid examples ──▶ dedup (exact set + MinHash/LSH near-dup)
                ──▶ quality score each example, flag < 0.3
                ──▶ count tokens -> distribution stats
                ──▶ estimate cost (OpenAI vs local GPU)
                ──▶ 90/10 train/val split (stratified if labels)
                ▼
          DataQualityReport  (+ recommendation in English)
```

## Concepts I learned / re-derived writing this

### MinHash + LSH — the star of the phase
The full derivation is in `minhash.py`'s docstring and
[`../data-validation.md`](../data-validation.md). The "aha": you can replace an
O(N²) all-pairs Jaccard comparison with (a) K min-hashes per doc that *estimate*
Jaccard by counting integer matches, and (b) LSH banding that only ever compares
docs likely to be similar. This exact pattern is how people dedup billion-document
pre-training corpora. Implementing it by hand (instead of `pip install datasketch`)
was the highest-value part of this phase.

### Right-skewed distributions → report the median
Token counts aren't normally distributed; a few giant examples drag the mean up.
So `Stats` reports mean **and** median **and** p95. Whenever you summarise a
real-world distribution with one number, ask "is this skewed?" first.

### Graceful degradation as an architecture pattern
Both `tokenizer.py` (tiktoken → regex) and `quality.py` (perplexity → heuristic)
try the good-but-heavy path and silently fall back to the always-available path,
then **record which one ran** in the output. The platform stays runnable offline
(ADR 0002) and stays honest about the fidelity of its own numbers.

### Advise, don't mutate
Big call: only a hard format failure *rejects* a dataset. Everything else is a
warning. The platform never deletes or silently "cleans" the user's rows — it
reports and lets them decide. Trust > cleverness.

## Mistakes / things that bit me
- **Line length in comment-heavy files.** Ruff's E501 fired on explanatory
  comments. Fixed by moving trailing comments to their own line rather than
  loosening the limit — the lint check stays useful.
- **Near-dup counting is fuzzy.** "How many near-duplicates" has no single right
  answer (is a cluster of 3 similar docs 2 dupes or 3?). I count *distinct docs
  that appear as the second element of a similar pair* as an upper bound on
  removable rows, and I'm explicit about that in the code.

## Try it
```bash
uv run pytest tests/test_data_validation.py -v
```

## Next
[Phase 3 — orchestration & training](03-orchestration-and-training.md): the job
state machine and the pluggable trainer.
