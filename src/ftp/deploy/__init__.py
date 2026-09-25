"""Deployment subsystem: local in-process serving + K8s manifest rendering."""

from .base import DeployResult
from .factory import get_deployer
from .local import get_endpoint

__all__ = ["DeployResult", "get_deployer", "get_endpoint"]
