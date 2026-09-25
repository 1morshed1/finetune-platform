"""Data-validation pipeline.

Public surface:
- validate_file / validate_examples  -> run the pipeline
- ValidationResult, DataQualityReport -> the outputs
- count_tokens, find_near_duplicates  -> reusable primitives
"""

from .minhash import estimate_jaccard, find_near_duplicates
from .quality import heuristic_score, score_quality
from .report import CostEstimate, DataQualityReport, Stats
from .tokenizer import count_tokens, tokenizer_name
from .validator import ValidationResult, validate_examples, validate_file

__all__ = [
    "validate_file",
    "validate_examples",
    "ValidationResult",
    "DataQualityReport",
    "CostEstimate",
    "Stats",
    "count_tokens",
    "tokenizer_name",
    "find_near_duplicates",
    "estimate_jaccard",
    "score_quality",
    "heuristic_score",
]
