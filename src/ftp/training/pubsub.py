"""In-process log/metric broker for live streaming (SSE).

The orchestrator runs a job in a background **thread** and emits events; the API
serves the log stream from an **async** request. We need to hand events across
that thread/async boundary. This is a tiny thread-safe fan-out broker:

    publisher (job thread)  ──▶  broker  ──▶  N subscribers (each an SSE request)

Each subscriber gets its own queue, so a slow client can't block the trainer or
other clients. A ``None`` sentinel signals "stream finished".

Why in-process (not Redis/Kafka)? Local-first, single node (ADR 0004). The broker
sits behind a trivial interface, so swapping in Redis pub/sub later touches only
this file.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

_SENTINEL = None  # pushed to signal end-of-stream


class LogBroker:
    def __init__(self) -> None:
        self._subs: dict[str, set[queue.SimpleQueue]] = {}
        self._lock = threading.Lock()

    def subscribe(self, job_id: str) -> queue.SimpleQueue:
        q: queue.SimpleQueue = queue.SimpleQueue()
        with self._lock:
            self._subs.setdefault(job_id, set()).add(q)
        return q

    def unsubscribe(self, job_id: str, q: queue.SimpleQueue) -> None:
        with self._lock:
            subs = self._subs.get(job_id)
            if subs:
                subs.discard(q)
                if not subs:
                    self._subs.pop(job_id, None)

    def publish(self, job_id: str, item: dict) -> None:
        with self._lock:
            subs = list(self._subs.get(job_id, ()))
        for q in subs:
            q.put(item)

    def close(self, job_id: str) -> None:
        """Tell every subscriber of this job the stream is over."""
        with self._lock:
            subs = list(self._subs.get(job_id, ()))
        for q in subs:
            q.put(_SENTINEL)

    def stream(self, job_id: str, timeout: float = 0.5) -> Iterator[dict | None]:
        """Blocking generator over a subscription; yields None on idle timeouts so
        the caller can send SSE keep-alives and check for client disconnects."""
        q = self.subscribe(job_id)
        try:
            while True:
                try:
                    item = q.get(timeout=timeout)
                except queue.Empty:
                    yield None  # idle tick
                    continue
                if item is _SENTINEL:
                    return
                yield item
        finally:
            self.unsubscribe(job_id, q)


# Process-wide singleton broker.
broker = LogBroker()
