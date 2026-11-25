"""Thread-safe telemetry queue with batched drain."""

from __future__ import annotations

import contextlib
import logging
import queue
from typing import Any

logger = logging.getLogger("amund")

# Sentinel to signal background thread shutdown
SHUTDOWN = object()


class TelemetryQueue:
    """Bounded, thread-safe queue for RunRecord dicts.

    Accepts events via put() (non-blocking, <0.1ms) and provides
    drain() to pull batches for flushing. Uses stdlib queue.Queue
    which is thread-safe without external locks.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_size)

    def put(self, event: dict[str, Any]) -> bool:
        """Enqueue an event. Returns True if accepted, False if dropped."""
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            # SDKDIAG-09: previously a local WARNING and nothing else, so
            # sustained telemetry loss was invisible to us. Now counted.
            from amund._internal import diagnostics

            diagnostics.record_drop(diagnostics.REASON_QUEUE_FULL, "trace")
            return False

    def drain(self, max_batch: int = 100) -> list[Any]:
        """Drain up to max_batch items from the queue.

        Returns list of events. May include SHUTDOWN sentinel.
        """
        batch: list[Any] = []
        while len(batch) < max_batch:
            try:
                item = self._queue.get_nowait()
                batch.append(item)
            except queue.Empty:
                break
        return batch

    def signal_shutdown(self) -> None:
        """Put the shutdown sentinel into the queue."""
        with contextlib.suppress(queue.Full):
            self._queue.put(SHUTDOWN, timeout=1.0)

    @property
    def empty(self) -> bool:
        return self._queue.empty()

    @property
    def size(self) -> int:
        return self._queue.qsize()
