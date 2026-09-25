# Phase 5 — API & dashboard

**Goal:** put the whole platform behind an HTTP API and a dashboard, including
live training logs over Server-Sent Events. This is where the parts become a
product.

## What got built

```
src/ftp/
  app.py               FastAPI factory: routers + static dashboard + /health, /config
  cli.py               `ftp serve | demo | sample`
  api/
    schemas.py         pydantic request bodies
    serializers.py     ORM row -> JSON dict
    routes/
      datasets.py      POST /datasets (upload+validate), GET list/detail
      jobs.py          POST /fine-tune, GET jobs, GET job, GET job/logs (SSE)
      models.py        GET leaderboard/detail, POST deploy, POST evaluate
      endpoints.py     POST endpoints/{id}/generate, compare
frontend/              index.html + styles.css + app.js (no build step)
tests/test_api.py      8 end-to-end HTTP tests
```

## Ideas

### Synchronous validation on upload — a UX decision
`POST /datasets` runs the *entire* validation pipeline before responding, and
returns the quality report in the same call. Uploading exists **to get the
report**, so making the user poll for it would be silly. Training, by contrast, is
long, so it's asynchronous (below). Match the API shape to the operation's
duration.

### Server-Sent Events for live logs (`jobs.py::stream_logs`)
Training happens on a background thread; the browser wants a live feed. We use
**SSE** — a one-way `text/event-stream` over plain HTTP — rather than WebSockets,
because the traffic is strictly server→client and SSE auto-reconnects and needs no
handshake. The endpoint:
1. subscribes to the `LogBroker` first,
2. replays persisted log lines (so a late viewer sees history),
3. forwards live events until the job's terminal `status` event,
4. sends `: keep-alive` comments on idle so proxies don't drop the connection.

Bridging the thread→async gap uses `await asyncio.to_thread(q.get, ...)` so a
blocking queue read doesn't stall the event loop. That one line is the crux of
serving a thread-produced stream from an async endpoint.

### Routing key == database id
`POST /models/{id}/deploy` creates the `Deployment` row first, then passes its id
into the deployer as the endpoint's routing key. So "the row I can query" and "the
URL I can call" are the same identifier — no mapping table, no drift.

### The dashboard draws its own chart
`frontend/app.js` has a hand-rolled `<canvas>` loss chart (`drawLoss`) — no
charting library. It plots train loss and eval loss vs step, rescaling live as SSE
metric events arrive. Writing it by hand (coordinate transforms, gridlines, two
series) is ~40 lines and worth understanding once; you see exactly how a line
chart maps data-space to pixel-space. The "base vs fine-tuned" playground and the
leaderboard round out the plan's dashboard requirements.

### Static frontend, mounted last
The dashboard is plain files served by `StaticFiles(html=True)`, mounted at `/`
**after** the API routers so it never shadows `/api/...`. No bundler, no npm — open
the page and it works. Fitting for a local-first tool.

## Testing the whole thing over HTTP
`test_api.py` uses FastAPI's `TestClient` to drive the real routes: upload →
fine-tune → poll to terminal → deploy → generate/compare → leaderboard. Because
`FTP_SIM_STEP_DELAY=0`, a job finishes in milliseconds, so the async path is
testable with a simple poll loop. 28 tests total, all green.

## Try it
```bash
uv run ftp serve          # then open http://localhost:8000
uv run ftp demo           # or watch the whole pipeline in the terminal
```

## Next
[Phase 6 — infrastructure](../README.md): Dockerfile, docker-compose, and the
documented Airflow/K8s manifests that the local backends stand in for.
