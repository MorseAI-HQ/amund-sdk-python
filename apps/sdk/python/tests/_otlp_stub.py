"""Answer the OTLP exporter's HTTP attempts without a network (MHQ-1922).

The stub is a ``requests`` transport adapter mounted on the session the
exporter sends through. Everything above it runs for real: the installed
version of the stock ``OTLPSpanExporter``, its retry loop, and Amund's
``_StatusRecordingSession``. Earlier tests replaced ``exporter._session``
instead, which only exists up to opentelemetry-exporter-otlp-proto-http 1.44,
so on 1.45 they silently stopped exercising the code under test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import requests
from requests.adapters import BaseAdapter

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# (status, headers) for an answer, or None for a connection error.
Answer = tuple[int, dict[str, str]] | None


class _ScriptedAdapter(BaseAdapter):
    def __init__(self, answer: Callable[[], Answer]) -> None:
        super().__init__()
        self._answer = answer
        self.calls = 0

    def send(self, request: Any, **_kwargs: Any) -> requests.Response:  # type: ignore[override]
        self.calls += 1
        answer = self._answer()
        if answer is None:
            raise requests.exceptions.ConnectionError("no route to host")
        status, headers = answer
        response = requests.Response()
        response.status_code = status
        response.headers.update(headers)
        response.reason = "stub"
        response._content = b""
        response.url = request.url
        response.request = request
        return response

    def close(self) -> None:
        pass


def script_exporter(exporter: Any, answers: Sequence[Answer]) -> _ScriptedAdapter:
    """Make ``exporter``'s HTTP attempts answer ``answers`` in order.

    The last entry repeats once the list is used up, so ``[(401, {})]`` means
    "401 forever". Returns the adapter, whose ``calls`` counts attempts.
    """
    queue = list(answers)

    def _next() -> Answer:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    adapter = _ScriptedAdapter(_next)
    session = exporter._amund_session
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return adapter
