# Phase 4 — deployment backends

**Goal:** make a registered model *serveable*. Same pattern as the trainer: an
interface with a local stand-in and a documented real (Kubernetes) backend.

## What got built

```
src/ftp/deploy/
  base.py     Deployer interface + DeployResult
  local.py    LocalDeployer + LocalEndpoint (in-process serving + mock inference)
  k8s.py      K8sDeployer (renders Deployment/Service/Ingress YAML; optional apply)
  factory.py  get_deployer(backend)
tests/test_deploy.py   4 tests
```

(The **registry** — the other half of this phase's title — was built in Phase 3
because the DAG's `register` step needs it. See `registry/registry.py`.)

## Ideas

### Deploying = "answer requests at a URL", behind an interface
`LocalDeployer` registers an in-process `LocalEndpoint` in a module-level dict —
the local stand-in for a service mesh's routing table — and the API serves it at
`/api/v1/endpoints/{id}/generate`. `K8sDeployer` renders the real thing: a
`Deployment` running vLLM with the LoRA adapter mounted, a `Service`, and an
`Ingress` with the **canary weight** annotation the plan calls for. Both satisfy
the same `Deployer.deploy(model, deployment_id, canary) -> DeployResult`.

The key detail: the API creates the `Deployment` DB row id and *passes it in*, so
the routing key and the database id are the same value. Small thing, but it means
"the endpoint I can call" and "the deployment I can query" are never out of sync.

### Mock inference, labelled as mock
Without the `[ml]` extra there are no real weights, so `LocalEndpoint._mock_generate`
returns a *deterministic* pseudo-response that is visibly different between base
and fine-tuned. That's enough to drive the dashboard's "base vs fine-tuned
side-by-side" panel. It is clearly labelled `source: "mock"`; with `[ml]` + a real
adapter, `_try_real_generate` loads base+adapter via peft and generates for real.
Same honesty rule as the simulated trainer: demonstrate the mechanism, never fake
the claim.

### Canary as an Ingress annotation
Read the rendered YAML in `k8s.py::render_manifests`. The canary strategy — "route
10% of traffic to the fine-tuned model, watch error rate, ramp or roll back" — is
literally two nginx-ingress annotations (`canary: true`, `canary-weight: N`). The
local backend doesn't need traffic splitting (one process), but rendering the real
manifest is the lesson: *this* is what "canary deploy" compiles down to.

### Rendered vs applied
`K8sDeployer` **renders** manifests by default (writes YAML to the artifacts dir,
returns it) and only **applies** them if `FTP_K8S_APPLY=1` and a cluster is
reachable. So you can study — and diff, and code-review — exactly what would be
deployed without ever touching a cluster.

## Try it
```bash
uv run pytest tests/test_deploy.py -v
```

## Next
[Phase 5 — API & dashboard](05-api-and-dashboard.md): wire it all behind FastAPI,
stream logs over SSE, and put a face on it.
