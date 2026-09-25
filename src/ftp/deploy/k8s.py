"""K8sDeployer — render the Kubernetes manifests for a real deployment.

This is the documented "real" backend (ADR 0004). It does **not** require a
cluster: by default it *renders* the Deployment + Service + Ingress YAML that
*would* serve the model (base model + LoRA adapter behind vLLM) and returns them,
writing a copy under the artifacts dir. Reading the rendered YAML is the lesson:
this is exactly what the local in-process endpoint stands in for.

If ``kubectl`` is available and ``FTP_K8S_APPLY=1`` is set, it will actually apply
the manifests. That path is opt-in and untested in CI.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from ftp.config import get_settings
from ftp.db import ModelRecord

from .base import DeployResult


def render_manifests(model: ModelRecord, deployment_id: str, canary_percent: int) -> str:
    """Return the multi-document YAML for this model's deployment.

    The canary weight is expressed as an Ingress annotation — the mechanism the
    plan describes: route N% of traffic to the fine-tuned model, watch error
    rates, roll back if they spike.
    """
    name = f"ft-{deployment_id.replace('_', '-')}"
    return f"""# Rendered by ftp.deploy.k8s for model {model.id} ({model.name} v{model.version})
# Base model: {model.base_model} | LoRA adapter: {model.artifact_path}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  labels:
    app: {name}
    ftp/model-id: "{model.id}"
    ftp/base-model: "{model.base_model}"
spec:
  replicas: 2
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
    spec:
      containers:
        - name: inference
          # vLLM serving the base model with the LoRA adapter mounted.
          image: vllm/vllm-openai:latest
          args:
            - "--model={model.base_model}"
            - "--enable-lora"
            - "--lora-modules"
            - "ft={model.artifact_path}"
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: "1"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 20
---
apiVersion: v1
kind: Service
metadata:
  name: {name}
spec:
  selector:
    app: {name}
  ports:
    - port: 80
      targetPort: 8000
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {name}
  annotations:
    # Canary: send {canary_percent}% of traffic here; the rest stays on the
    # incumbent model. Watch error rate, then ramp to 100 or roll back.
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "{canary_percent}"
spec:
  rules:
    - host: {name}.models.local
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: {name}
                port:
                  number: 80
"""


class K8sDeployer:
    backend = "k8s"

    def deploy(
        self, model: ModelRecord, deployment_id: str, canary_percent: int = 100
    ) -> DeployResult:
        manifests = render_manifests(model, deployment_id, canary_percent)
        out = get_settings().artifacts_dir / model.job_id / f"{deployment_id}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(manifests)

        applied = False
        if os.environ.get("FTP_K8S_APPLY") == "1" and shutil.which("kubectl"):
            try:  # pragma: no cover - needs a cluster
                subprocess.run(
                    ["kubectl", "apply", "-f", str(out)], check=True, capture_output=True
                )
                applied = True
            except Exception as e:
                return DeployResult(
                    backend="k8s", status="failed", endpoint_url=None,
                    canary_percent=canary_percent,
                    detail={"manifest_path": str(out), "error": str(e)},
                )

        name = f"ft-{deployment_id.replace('_', '-')}"
        return DeployResult(
            backend="k8s",
            status="live" if applied else "rendered",
            endpoint_url=f"http://{name}.models.local/" if applied else None,
            canary_percent=canary_percent,
            detail={
                "manifest_path": str(out),
                "applied": applied,
                "manifest": manifests,
                "note": "Set FTP_K8S_APPLY=1 with a reachable cluster to apply.",
            },
        )
