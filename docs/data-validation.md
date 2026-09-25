# Deep-dive: the data-validation pipeline

Code: `src/ftp/data/`. This is the subsystem that turns "here's a script that
fine-tunes" into "here's a platform".

## Why validation is the whole game

Fine-tuning is *garbage-in, garbage-out*, and it costs real money and hours. The
single highest-leverage thing a platform can do is tell you, **before** you spend
anything, whether your data is worth training on. Every check here maps to a way
fine-tunes fail in practice:

| Check | The failure it prevents |
| ----- | ----------------------- |
| Format validation | A job crashing 20 minutes in on a malformed line |
| Exact + near dedup | Overfitting + paying to train on the same example twice |
| Quality scoring | The model learning from empty/garbage/repetitive rows |
| Token analysis | An example silently truncated because it exceeds context |
| Cost estimation | A surprise bill |
| Train/val split | Having no honest way to measure if training helped |

## Module map

```
data/
  tokenizer.py   count_tokens()          — cost & length, tiktoken or regex fallback
  minhash.py     find_near_duplicates()  — MinHash + LSH, hand-rolled
  quality.py     score_quality()         — heuristic or perplexity
  report.py      DataQualityReport, Stats, generate_recommendation()
  validator.py   validate_file()         — orchestrates all 7 steps
```

## The MinHash + LSH bit (the interesting algorithm)

Read the module docstring in `minhash.py` for the full derivation. The short
version:

- Naive near-dup detection is **O(N²)** pairwise comparisons.
- **MinHash** turns "how similar are these two sets of shingles?" into "how many
  of these K small integers match?" — cheap, and provably an unbiased estimate of
  Jaccard similarity.
- **LSH banding** avoids comparing all pairs: only documents that collide in some
  band get compared. Near-linear.

`_choose_bands` picks the band/row split whose theoretical threshold
`(1/b)^(1/r)` best matches the similarity threshold you asked for — so requesting
`threshold=0.85` actually tunes the LSH to be sensitive around 0.85.

This is worth understanding deeply: the same MinHash/LSH pattern shows up in
recommendation systems, plagiarism detection, and large-scale dedup of
pre-training corpora.

## Quality scoring: honest about its limits

The plan says "use perplexity from a small model". We do — *if* the `[ml]` extra
is installed (`quality.py::_perplexity_scores` runs distilgpt2). Otherwise we fall
back to a **heuristic blend** (length sanity × lexical repetition × character
junk). The heuristic is not as good as real perplexity, and the code says so. The
report always records which method ran (`quality_method` field) so a reader knows
whether to trust the scores. Being explicit about the fidelity of your own signals
is a platform-maturity thing.

## The report is the product

`DataQualityReport` (in `report.py`) is what the user actually sees. Note the
statistics choice: we report **median and p95**, not just the mean, because token
distributions are right-skewed — a handful of huge examples distort the mean. The
median is the typical example; p95/max is the worst case that might blow the
context window.

`generate_recommendation` turns numbers into sentences. Every threshold it checks
comes from `Settings`, not a magic constant, because "how many duplicates is too
many" is **policy**, and an operator should tune policy without editing code.

## Design principle: advise, don't delete

Only a *hard* failure (not JSONL, zero valid rows) rejects a dataset. Duplicates,
low-quality rows, small size — all become **warnings** in the report. The platform
never silently drops the user's data, because only they know whether a 40% near-dup
rate is fine for their use case. Automated systems that quietly mutate user data
destroy trust.

## See it run

```bash
uv run pytest tests/test_data_validation.py -v
```

The tests are written to read like a spec of what "good data" means here.
