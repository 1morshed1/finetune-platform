"""Tests for the data-validation pipeline.

These double as executable documentation: read them to see exactly what the
platform considers a good vs bad dataset.
"""

from __future__ import annotations

import json

from ftp.config import Settings
from ftp.data import count_tokens, find_near_duplicates
from ftp.data.validator import validate_examples, validate_file


def _settings(**over) -> Settings:
    base = dict(min_examples=5, max_examples_warn=1000)
    base.update(over)
    return Settings(**base)


def _ex(prompt: str, completion: str, **extra) -> dict:
    return {"prompt": prompt, "completion": completion, **extra}


# --- tokenizer -------------------------------------------------------------


def test_count_tokens_monotonic():
    assert count_tokens("") == 0
    assert count_tokens("hello") >= 1
    assert count_tokens("hello world foo bar baz") > count_tokens("hello")


# --- near-duplicate detection ---------------------------------------------


def test_find_near_duplicates_catches_small_edits():
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the lazy dog!",  # near-dup of 0
        "completely different sentence about databases and indexing",
    ]
    pairs = find_near_duplicates(texts, threshold=0.6)
    dup_indices = {i for i, _, _ in pairs} | {j for _, j, _ in pairs}
    assert 0 in dup_indices and 1 in dup_indices
    assert 2 not in dup_indices


def test_find_near_duplicates_none_when_all_distinct():
    texts = [f"unique sentence number {i} about topic {i * 7}" for i in range(10)]
    assert find_near_duplicates(texts, threshold=0.85) == []


# --- full pipeline ---------------------------------------------------------


def test_good_dataset_is_ready():
    examples = [
        _ex(f"Question {i}: what is {i}+{i}?", f"The answer is {i + i}.") for i in range(20)
    ]
    res = validate_examples(examples, _settings(), format_errors=[])
    assert res.status == "ready"
    assert res.report.total_examples == 20
    assert res.report.exact_duplicates == 0
    # 90/10 split
    assert res.report.train_size + res.report.val_size == 20
    assert res.report.val_size == 2
    assert "good" in res.report.recommendation.lower()


def test_exact_duplicates_flagged():
    examples = [_ex("same prompt", "same completion") for _ in range(10)]
    res = validate_examples(examples, _settings(), format_errors=[])
    assert res.report.exact_duplicates == 9  # 10 rows, 1 unique
    assert "duplication" in res.report.recommendation.lower()


def test_small_dataset_warned():
    examples = [_ex(f"p{i}", f"c{i}") for i in range(3)]
    res = validate_examples(examples, _settings(min_examples=100), format_errors=[])
    assert "small" in res.report.recommendation.lower()


def test_low_quality_flagged():
    good = [
        _ex(f"Explain concept {i} in detail please.", f"Concept {i} means a useful thing.")
        for i in range(10)
    ]
    junk = [_ex("", "")]  # empty -> length_score 0 -> low quality
    res = validate_examples(good + junk, _settings(), format_errors=[])
    assert len(res.report.low_quality_examples) >= 1


def test_stratified_split_preserves_labels():
    # 20 label-A, 20 label-B. A 10% val split should draw from both.
    examples = [_ex(f"a{i}", f"ca{i}", label="A") for i in range(20)] + [
        _ex(f"b{i}", f"cb{i}", label="B") for i in range(20)
    ]
    res = validate_examples(examples, _settings(), format_errors=[])
    val_labels = {examples[i]["label"] for i in res.val_indices}
    assert val_labels == {"A", "B"}


def test_cost_estimate_scales_with_tokens():
    small = validate_examples([_ex("hi", "there")], _settings(), format_errors=[])
    big = validate_examples(
        [_ex("word " * 100, "word " * 100) for _ in range(10)], _settings(), format_errors=[]
    )
    assert big.report.cost.total_tokens > small.report.cost.total_tokens
    assert big.report.cost.openai_usd >= small.report.cost.openai_usd


def test_empty_dataset_rejected():
    res = validate_examples([], _settings(), format_errors=["Line 1: invalid JSON"])
    assert res.status == "rejected"
    assert res.report.passed is False


# --- file parsing ----------------------------------------------------------


def test_validate_file_collects_format_errors(tmp_path):
    p = tmp_path / "data.jsonl"
    lines = [
        json.dumps(_ex("valid prompt one", "valid completion one")),
        "{not json",  # bad JSON
        json.dumps({"prompt": "missing completion"}),  # missing field
        json.dumps(_ex("valid prompt two", "valid completion two")),
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    res = validate_file(p, _settings())
    assert res.report.total_examples == 2
    assert len(res.report.format_errors) == 2
    assert any("invalid JSON" in e for e in res.report.format_errors)
    assert any("missing required field" in e for e in res.report.format_errors)


def test_report_serialises_to_dict():
    res = validate_examples([_ex("hello there", "general kenobi")], _settings(), format_errors=[])
    d = res.report.to_dict()
    assert isinstance(d, dict)
    # nested dataclasses become nested dicts (JSON-ready for the DB column)
    assert isinstance(d["prompt_token_stats"], dict)
    assert isinstance(d["cost"], dict)
    json.dumps(d)  # must be JSON-serialisable
