"""LocalDeployer — serve a model inside this process.

The spec's deployment step loads "base model + LoRA adapter into an inference
server (vLLM/Ollama)" on Kubernetes. Locally we do the conceptually-identical
thing without a cluster: register an in-process **endpoint** that can answer
generation requests, and expose it at a URL the API serves
(`/api/v1/endpoints/{id}/generate`).

Inference itself:
- With the ``[ml]`` extra and a *real* adapter on disk, we lazily load the base
  model + LoRA adapter with peft and actually generate. (Best-effort; falls back
  if the model can't be loaded.)
- Otherwise we return a **mock** generation. It's deterministic and clearly
  labelled as a mock — enough to demo the "base vs fine-tuned side-by-side"
  dashboard panel without pretending to be a real model.

An in-process dict maps deployment_id -> LocalEndpoint. This is the local stand-in
for a service mesh's routing table.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ftp.db import ModelRecord

from .base import DeployResult

# deployment_id -> LocalEndpoint. Process-wide "routing table".
_ENDPOINTS: dict[str, LocalEndpoint] = {}


class LocalEndpoint:
    """A tiny in-process model server."""

    def __init__(self, deployment_id: str, model: ModelRecord) -> None:
        self.deployment_id = deployment_id
        self.model_id = model.id
        self.model_name = f"{model.name}-v{model.version}"
        self.base_model = model.base_model
        self.artifact_path = model.artifact_path
        self._real = None  # lazily-loaded (base_model + adapter) if [ml] present

    # -- inference ---------------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int = 64) -> dict:
        text = self._try_real_generate(prompt, max_new_tokens)
        source = "real"
        if text is None:
            text = self._mock_generate(prompt, finetuned=True)
            source = "mock"
        return {"model": self.model_name, "source": source, "completion": text}

    def compare_with_base(self, prompt: str) -> dict:
        """Return base-model vs fine-tuned output for the same prompt — powers the
        dashboard's side-by-side panel."""
        return {
            "prompt": prompt,
            "base": {
                "model": self.base_model,
                "completion": self._mock_generate(prompt, finetuned=False),
            },
            "finetuned": {
                "model": self.model_name,
                "completion": self._mock_generate(prompt, finetuned=True),
            },
            "note": "Mock generations for demonstration unless the [ml] extra is installed.",
        }

    def _try_real_generate(self, prompt: str, max_new_tokens: int) -> str | None:
        """Load base+adapter and generate for real, if possible. Returns None to
        signal 'fall back to mock'."""
        if not self.artifact_path:
            return None
        adapter_dir = Path(self.artifact_path)
        # The simulated trainer writes adapter.json (not real weights) — skip.
        if (adapter_dir / "adapter.json").exists():
            return None
        try:  # pragma: no cover - needs [ml] + real adapter
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self._real is None:
                tok = AutoTokenizer.from_pretrained(self.base_model)
                base = AutoModelForCausalLM.from_pretrained(self.base_model)
                model = PeftModel.from_pretrained(base, str(adapter_dir))
                model.eval()
                self._real = (tok, model)
            tok, model = self._real
            ids = tok(prompt, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=max_new_tokens)
            return tok.decode(out[0], skip_special_tokens=True)
        except Exception:
            return None

    def _mock_generate(self, prompt: str, *, finetuned: bool) -> str:
        """Deterministic pseudo-generation. NOT a real model — a stand-in whose
        only job is to be stable and visibly different between base and fine-tuned
        so the UI has something to show."""
        seed = hashlib.blake2b(
            (("ft" if finetuned else "base") + prompt).encode(), digest_size=4
        ).hexdigest()
        who = self.model_name if finetuned else self.base_model
        style = (
            "answers in the fine-tuned domain style, concise and on-format"
            if finetuned
            else "gives a generic, pre-training-style answer"
        )
        return f"[{who} · {style} · sig:{seed}] Response to: {prompt[:80]}"


def get_endpoint(deployment_id: str) -> LocalEndpoint | None:
    return _ENDPOINTS.get(deployment_id)


class LocalDeployer:
    backend = "local"

    def deploy(
        self, model: ModelRecord, deployment_id: str, canary_percent: int = 100
    ) -> DeployResult:
        _ENDPOINTS[deployment_id] = LocalEndpoint(deployment_id, model)
        endpoint_url = f"/api/v1/endpoints/{deployment_id}/generate"
        return DeployResult(
            backend="local",
            status="live",
            endpoint_url=endpoint_url,
            canary_percent=canary_percent,
            detail={
                "deployment_id": deployment_id,
                "compare_url": f"/api/v1/endpoints/{deployment_id}/compare",
                "served_in_process": True,
            },
        )
