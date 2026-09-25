"""Request/response schemas for the API.

We use pydantic models for **request bodies** (so FastAPI validates and documents
them) and for a few structured responses. Most GET responses are serialised from
ORM rows by ``serializers.py`` into plain dicts — simpler than mirroring every
column in a pydantic model that would just drift out of sync.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FineTuneRequest(BaseModel):
    dataset_id: str
    base_model: str = Field(default="demo", description="Base model id (HF id, or 'demo' for sim)")
    # Free-form so users can pass any hyperparameter; the trainer picks what it
    # understands (epochs, batch_size, learning_rate, lora_r, lora_alpha, ...).
    hyperparameters: dict = Field(default_factory=dict)


class DeployRequest(BaseModel):
    backend: str | None = Field(default=None, description="'local' or 'k8s'; default from settings")
    canary_percent: int = Field(default=100, ge=1, le=100)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=64, ge=1, le=1024)


class CompareRequest(BaseModel):
    prompt: str


class EvaluateRequest(BaseModel):
    # Optional prompts for a quick side-by-side; if omitted, returns the stored
    # base-vs-finetuned comparison from training.
    prompts: list[str] = Field(default_factory=list)
