"""The data-validation pipeline — orchestrates every check into one report.

This is the code that runs on dataset upload. It implements the 7 steps from the
plan, in order:

  1. Format validation  (valid JSONL? required fields? encoding?)
  2. Deduplication      (exact via set, near via MinHash+LSH)
  3. Quality scoring    (heuristic or perplexity; flag outliers)
  4. Token analysis     (per-example counts -> distribution stats)
  5. Cost estimation    (OpenAI API vs local GPU)
  6. Train/val split    (90/10, stratified if a 'label' field is present)
  7. Report generation  (the DataQualityReport)

Design choice: a **hard failure** (not JSONL, no valid rows, missing required
fields on every row) → the dataset is *rejected*. Everything else is a *warning*
captured in the report — the platform advises, the user decides (only they know if
a 40% near-dup rate is acceptable for their use case).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ftp.config import Settings

from .minhash import find_near_duplicates
from .quality import score_quality
from .report import CostEstimate, DataQualityReport, Stats, generate_recommendation
from .tokenizer import count_tokens, tokenizer_name

REQUIRED_FIELDS = ("prompt", "completion")


@dataclass
class ValidationResult:
    status: str  # "ready" | "rejected"
    report: DataQualityReport
    examples: list[dict] = field(default_factory=list)  # parsed, valid rows
    train_indices: list[int] = field(default_factory=list)
    val_indices: list[int] = field(default_factory=list)


def _parse_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    """Parse a JSONL file. Returns (valid_examples, errors).

    We validate *per line* and collect errors rather than aborting on the first
    bad row — a user wants to know all their problems at once, and one malformed
    line shouldn't discard 9,999 good ones.
    """
    examples: list[dict] = []
    errors: list[str] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        return [], [f"File is not valid UTF-8: {e}"]

    for lineno, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue  # tolerate blank lines
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {lineno}: invalid JSON ({e.msg})")
            continue
        if not isinstance(obj, dict):
            errors.append(f"Line {lineno}: expected a JSON object, got {type(obj).__name__}")
            continue
        missing = [f for f in REQUIRED_FIELDS if f not in obj]
        if missing:
            errors.append(f"Line {lineno}: missing required field(s): {', '.join(missing)}")
            continue
        if not isinstance(obj["prompt"], str) or not isinstance(obj["completion"], str):
            errors.append(f"Line {lineno}: 'prompt' and 'completion' must be strings")
            continue
        examples.append(obj)

    # Cap the number of listed errors so a totally-broken file doesn't produce a
    # megabyte of messages; the count in the report still reflects reality.
    if len(errors) > 50:
        extra = len(errors) - 50
        errors = errors[:50] + [f"... and {extra} more format errors"]
    return examples, errors


def _stratified_split(
    examples: list[dict], val_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """90/10 train/val split. Stratified by 'label' if present, else random.

    Stratification keeps the label distribution the same in train and val, so the
    validation metric isn't skewed by an accidentally lopsided split.
    """
    rng = random.Random(seed)
    n = len(examples)
    indices = list(range(n))

    has_labels = all("label" in ex for ex in examples) and n > 0
    if not has_labels:
        rng.shuffle(indices)
        cut = int(round(n * (1 - val_ratio)))
        return indices[:cut], indices[cut:]

    # Group by label, take val_ratio from each group.
    groups: dict[object, list[int]] = {}
    for i in indices:
        groups.setdefault(examples[i]["label"], []).append(i)
    train, val = [], []
    for members in groups.values():
        rng.shuffle(members)
        cut = int(round(len(members) * (1 - val_ratio)))
        train.extend(members[:cut])
        val.extend(members[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _estimate_cost(total_tokens: int, settings: Settings) -> CostEstimate:
    """Cost + time estimates. Formulas are deliberately simple and labelled as
    estimates in the report (see report.CostEstimate.notes)."""
    openai_usd = round(total_tokens / 1_000_000 * settings.openai_ft_usd_per_1m_tokens, 2)
    # Rough throughput assumption: ~50k tokens/minute on a modest GPU (from plan).
    training_minutes = round(total_tokens / 50_000, 2)
    gpu_hours = training_minutes / 60
    local_usd = round(gpu_hours * settings.local_gpu_usd_per_hour, 2)
    return CostEstimate(
        total_tokens=total_tokens,
        openai_usd=openai_usd,
        local_gpu_usd=local_usd,
        training_time_minutes=training_minutes,
    )


def validate_examples(
    examples: list[dict], settings: Settings, format_errors: list[str]
) -> ValidationResult:
    """Run steps 2–7 over already-parsed examples. Split out from file parsing so
    it's unit-testable without touching disk."""
    total = len(examples)

    # If nothing parsed, reject early with an empty-but-valid report shell.
    if total == 0:
        report = DataQualityReport(
            total_examples=0,
            unique_examples=0,
            exact_duplicates=0,
            near_duplicates=0,
            near_duplicate_pairs=[],
            prompt_token_stats=Stats.from_values([]),
            completion_token_stats=Stats.from_values([]),
            low_quality_examples=[],
            quality_method="n/a",
            tokenizer=tokenizer_name(),
            cost=_estimate_cost(0, settings),
            train_size=0,
            val_size=0,
            recommendation="No valid examples found. Fix the format errors and re-upload.",
            passed=False,
            format_errors=format_errors,
        )
        return ValidationResult(status="rejected", report=report)

    prompts = [ex["prompt"] for ex in examples]

    # Step 2: deduplication.
    unique_prompts = len(set(prompts))
    exact_dupes = total - unique_prompts
    near_pairs = find_near_duplicates(
        prompts, threshold=settings.near_duplicate_threshold, seed=settings.seed
    )
    # Count distinct examples involved in a near-dup relationship (upper bound on
    # removable rows), excluding ones already counted as exact dupes.
    near_dupe_ids = {j for _, j, _ in near_pairs}
    near_dupes = len(near_dupe_ids)

    # Step 3: quality scoring.
    scores, method = score_quality(examples, method="auto")
    low_quality = [i for i, s in enumerate(scores) if s < 0.3]

    # Step 4: token analysis.
    prompt_tokens = [count_tokens(ex["prompt"]) for ex in examples]
    completion_tokens = [count_tokens(ex["completion"]) for ex in examples]
    total_tokens = sum(prompt_tokens) + sum(completion_tokens)

    # Step 5: cost.
    cost = _estimate_cost(total_tokens, settings)

    # Step 6: split.
    train_idx, val_idx = _stratified_split(examples, val_ratio=0.10, seed=settings.seed)

    # Step 7: report.
    recommendation = generate_recommendation(
        total=total,
        unique=unique_prompts,
        low_quality=len(low_quality),
        near_dupes=near_dupes,
        min_examples=settings.min_examples,
        max_examples_warn=settings.max_examples_warn,
        max_duplicate_ratio=settings.max_duplicate_ratio,
        max_low_quality_ratio=settings.max_low_quality_ratio,
    )
    report = DataQualityReport(
        total_examples=total,
        unique_examples=unique_prompts,
        exact_duplicates=exact_dupes,
        near_duplicates=near_dupes,
        near_duplicate_pairs=near_pairs[:100],  # cap payload size
        prompt_token_stats=Stats.from_values(prompt_tokens),
        completion_token_stats=Stats.from_values(completion_tokens),
        low_quality_examples=low_quality[:200],
        quality_method=method,
        tokenizer=tokenizer_name(),
        cost=cost,
        train_size=len(train_idx),
        val_size=len(val_idx),
        recommendation=recommendation,
        passed=True,
        format_errors=format_errors,
    )
    return ValidationResult(
        status="ready",
        report=report,
        examples=examples,
        train_indices=train_idx,
        val_indices=val_idx,
    )


def validate_file(path: Path, settings: Settings) -> ValidationResult:
    """Entry point: parse a JSONL file from disk, then validate."""
    examples, format_errors = _parse_jsonl(path)
    return validate_examples(examples, settings, format_errors)
