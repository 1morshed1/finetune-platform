"""Deployer factory — pick a deployment backend by name."""

from __future__ import annotations

from .base import Deployer


def get_deployer(backend: str) -> Deployer:
    if backend == "local":
        from .local import LocalDeployer

        return LocalDeployer()
    if backend == "k8s":
        from .k8s import K8sDeployer

        return K8sDeployer()
    raise ValueError(f"Unknown deploy backend: {backend!r} (expected 'local' or 'k8s')")
