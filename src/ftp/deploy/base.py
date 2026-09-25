"""Deployer interface.

Deploying a fine-tuned model means "make it answer requests at a URL". Like the
trainer, we hide the *how* behind an interface so a local in-process deployment
and a real Kubernetes deployment are interchangeable (ADR 0004).

A deployer takes a registered model + a canary percentage and returns a
:class:`DeployResult` describing the live endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ftp.db import ModelRecord


@dataclass
class DeployResult:
    backend: str
    status: str                     # "live" | "failed"
    endpoint_url: str | None
    canary_percent: int = 100
    detail: dict = field(default_factory=dict)  # rendered manifest / notes / errors


class Deployer(Protocol):
    backend: str

    def deploy(
        self, model: ModelRecord, deployment_id: str, canary_percent: int = 100
    ) -> DeployResult:
        ...
