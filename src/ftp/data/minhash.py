"""Near-duplicate detection with MinHash + LSH — implemented from scratch.

## The problem
Exact duplicates are trivial (`set()`). But training data is full of *near*
duplicates: same example with a typo fixed, whitespace changed, one word swapped.
Near-dupes cause **overfitting** (the model sees the same thing many times) and
waste money (you pay to train on redundant tokens). A platform should flag them.

Comparing every pair of N documents is O(N²) — 10k examples = 50M comparisons.
MinHash + LSH gets this to roughly O(N).

## The two ideas

**1. MinHash — estimate set similarity cheaply.**
Represent each document as a *set of shingles* (overlapping word n-grams). The
similarity of two documents is the **Jaccard similarity** of their shingle sets:
`|A ∩ B| / |A ∪ B|`. Computing that directly is expensive per pair.

MinHash trick: pick K hash functions. For each document, its "signature" is, for
each hash function, the *minimum* hash value over all its shingles. Key theorem:

> P(min-hash of A == min-hash of B, for one hash fn) == Jaccard(A, B).

So the fraction of signature positions where two documents agree is an unbiased
estimate of their Jaccard similarity — comparing K small ints instead of two big
sets.

**2. LSH — avoid comparing all pairs.**
Split each K-length signature into `b` bands of `r` rows (`K = b·r`). Hash each
band. Two documents become *candidates* if they collide in **any** band. Similar
documents share bands with high probability; dissimilar ones almost never do. The
approximate threshold where the collide-probability curve steepens is
`(1/b)^(1/r)`. We only compute the exact-ish MinHash estimate for candidate pairs.

This file is intentionally hand-rolled (not `datasketch`) because implementing it
is the whole point of the learning exercise. It's small and readable.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# A large Mersenne prime, > any 32-bit hash, used as the modulus for the
# universal hash family h(x) = (a*x + b) mod P.
_MERSENNE_P = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def shingles(text: str, k: int = 3) -> set[int]:
    """Turn text into a set of hashed k-word shingles.

    k=3 (word trigrams) is a good default: small enough to catch near-dupes,
    large enough that random short overlaps don't create false similarity.
    Documents shorter than k fall back to their single token set.
    """
    words = _TOKEN_RE.findall(text.lower())
    if len(words) < k:
        grams = words or [""]
    else:
        grams = [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]
    out: set[int] = set()
    for g in grams:
        # Stable 32-bit hash of each shingle string.
        h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=4).digest(), "big")
        out.add(h)
    return out


class MinHasher:
    """Builds K-permutation MinHash signatures with a universal hash family."""

    def __init__(self, num_perm: int = 128, seed: int = 42) -> None:
        self.num_perm = num_perm
        import random

        rng = random.Random(seed)
        # K independent hash functions h_i(x) = (a_i * x + b_i) mod P.
        self._a = [rng.randrange(1, _MERSENNE_P) for _ in range(num_perm)]
        self._b = [rng.randrange(0, _MERSENNE_P) for _ in range(num_perm)]

    def signature(self, text: str, k: int = 3) -> list[int]:
        sh = shingles(text, k)
        if not sh:
            return [0] * self.num_perm
        sig = []
        for a, b in zip(self._a, self._b, strict=True):
            m = min(((a * x + b) % _MERSENNE_P) & _MAX_HASH for x in sh)
            sig.append(m)
        return sig


def estimate_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    """Fraction of signature positions that agree ≈ Jaccard similarity."""
    if not sig_a:
        return 0.0
    matches = sum(1 for x, y in zip(sig_a, sig_b, strict=True) if x == y)
    return matches / len(sig_a)


def _choose_bands(num_perm: int, threshold: float) -> tuple[int, int]:
    """Pick (bands b, rows r) with b*r == num_perm whose LSH threshold
    (1/b)^(1/r) is closest to the requested similarity threshold."""
    best = (num_perm, 1)
    best_err = float("inf")
    for b in range(1, num_perm + 1):
        if num_perm % b:
            continue
        r = num_perm // b
        t = (1.0 / b) ** (1.0 / r)
        err = abs(t - threshold)
        if err < best_err:
            best_err, best = err, (b, r)
    return best


def find_near_duplicates(
    texts: list[str],
    threshold: float = 0.85,
    num_perm: int = 128,
    k: int = 3,
    seed: int = 42,
) -> list[tuple[int, int, float]]:
    """Return ``(i, j, similarity)`` for pairs with estimated Jaccard >= threshold.

    Uses LSH banding to generate candidate pairs (near-linear), then verifies each
    candidate with the full MinHash estimate. ``i < j`` always.
    """
    hasher = MinHasher(num_perm=num_perm, seed=seed)
    sigs = [hasher.signature(t, k) for t in texts]
    bands, rows = _choose_bands(num_perm, threshold)

    # LSH: bucket documents by each band's tuple; same bucket => candidate pair.
    candidates: set[tuple[int, int]] = set()
    for band in range(bands):
        start = band * rows
        buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for idx, sig in enumerate(sigs):
            buckets[tuple(sig[start : start + rows])].append(idx)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for a_i in range(len(members)):
                for b_i in range(a_i + 1, len(members)):
                    candidates.add((members[a_i], members[b_i]))

    # Verify candidates with the actual estimate.
    out: list[tuple[int, int, float]] = []
    for i, j in candidates:
        sim = estimate_jaccard(sigs[i], sigs[j])
        if sim >= threshold:
            out.append((i, j, round(sim, 4)))
    out.sort()
    return out
