"""The Data Quality Report — the platform's headline artifact.

This is the object the plan highlights as "the part that differentiates a platform
from a script". Before a user spends money training, the platform hands them a
report: how many examples, how many duplicates, token distribution, cost estimate,
which examples look bad, and a plain-English recommendation.

These are plain dataclasses with a ``to_dict`` so they serialise straight into the
``Dataset.quality_report`` JSON column and out through the API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Stats:
    """Summary statistics for a distribution (e.g. prompt token counts).

    We report median and p95 alongside the mean because token distributions are
    usually **right-skewed** — a few very long examples pull the mean up. The
    median tells you the typical example; p95 and max tell you the worst case that
    might blow the context window.
    """

    mean: float
    median: float
    p95: float
    min: int
    max: int

    @classmethod
    def from_values(cls, values: list[int]) -> Stats:
        if not values:
            return cls(0.0, 0.0, 0.0, 0, 0)
        arr = np.array(values)
        return cls(
            mean=round(float(arr.mean()), 2),
            median=round(float(np.median(arr)), 2),
            p95=round(float(np.percentile(arr, 95)), 2),
            min=int(arr.min()),
            max=int(arr.max()),
        )


@dataclass
class CostEstimate:
    total_tokens: int
    openai_usd: float          # cost if fine-tuned via the OpenAI API
    local_gpu_usd: float       # cost if trained on rented/spot GPU
    training_time_minutes: float
    notes: str = "Estimates only. Actual cost depends on epochs, model size, and provider rates."


@dataclass
class DataQualityReport:
    total_examples: int
    unique_examples: int
    exact_duplicates: int
    near_duplicates: int
    near_duplicate_pairs: list[tuple[int, int, float]]
    prompt_token_stats: Stats
    completion_token_stats: Stats
    low_quality_examples: list[int]
    quality_method: str
    tokenizer: str
    cost: CostEstimate
    train_size: int
    val_size: int
    recommendation: str
    passed: bool = True
    format_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def generate_recommendation(
    *,
    total: int,
    unique: int,
    low_quality: int,
    near_dupes: int,
    min_examples: int,
    max_examples_warn: int,
    max_duplicate_ratio: float,
    max_low_quality_ratio: float,
) -> str:
    """Turn the raw numbers into actionable advice.

    Adapted from the plan's ``generate_recommendation``, but every threshold comes
    from config (a platform operator tunes policy without editing code) and we also
    account for near-duplicates, not just exact ones.
    """
    issues: list[str] = []

    dup_count = (total - unique) + near_dupes
    if total and dup_count / total > max_duplicate_ratio:
        pct = dup_count / total * 100
        issues.append(
            f"High duplication ({pct:.0f}%: {total - unique} exact + {near_dupes} near). "
            "Remove duplicates to avoid overfitting and wasted spend."
        )
    if total and low_quality / total > max_low_quality_ratio:
        issues.append(
            f"{low_quality} examples ({low_quality / total * 100:.0f}%) flagged low quality. "
            "Review and remove or improve them."
        )
    if total < min_examples:
        issues.append(
            f"Dataset is small ({total} < {min_examples}). Add more examples for better "
            "generalisation, or expect the fine-tune to overfit."
        )
    if total > max_examples_warn:
        issues.append(
            f"Large dataset ({total} examples). Consider a subset for initial experiments "
            "to reduce cost before committing to the full run."
        )

    if not issues:
        return "Dataset looks good. Ready for fine-tuning."
    return "Issues found:\n" + "\n".join(f"- {i}" for i in issues)
