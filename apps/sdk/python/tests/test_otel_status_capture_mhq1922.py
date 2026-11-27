"""A rejected key must be named on every supported OTel exporter (MHQ-1922).

The regression: on opentelemetry-exporter-otlp-proto-http 1.45 a 401 was
reported as ``dropped N span (send_failed) — could not reach the Amund API``.
The HTTP status used to be captured by overriding the exporter's private
``_export()``; 1.45 sends through a new HTTP client and never calls it, so the
status stayed None.

These tests run the installed exporter version against a real local HTTP
server, with no stub anywhere between the exporter and the socket, so they fail
on whichever version the capture does not work on. CI runs them against the
locked version; run them again under another version with e.g.
``uv run --with 'opentelemetry-exporter-otlp-proto-http==1.45.2' pytest ...``.
"""

from __future__ import annotations

import http.server
import logging
import socket
import threading
from importlib.metadata import version
from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext, TraceFlags

from amund import otel_provider
from amund._internal import diagnostics

if TYPE_CHECKING:
    from collections.abc import Iterator

OTEL_EXPORTER_VERSION = version("opentelemetry-exporter-otlp-proto-http")


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    diagnostics._reset_for_testing()
    yield
    diagnostics._reset_for_testing()


@pytest.fixture
def own_drops(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Drops recorded by this thread only.

    The counter is process-wide, and a BatchSpanProcessor leaked by an earlier
    test can record its own drop while this one runs. export() is synchronous,
    so a drop recorded on this thread is ours.
    """
    me = threading.current_thread()
    drops: list[tuple] = []
    real = diagnostics.record_drop

    def _record(*args, **kwargs):  # noqa: ANN002, ANN003
        if threading.current_thread() is me:
            drops.append((args, kwargs))
            real(*args, **kwargs)

    monkeypatch.setattr(diagnostics, "record_drop", _record)
    return drops


def _span() -> ReadableSpan:
    return ReadableSpan(
        name="s",
        context=SpanContext(
            trace_id=0x1234567890ABCDEF1234567890ABCDEF,
            span_id=0x1234567890ABCDEF,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        ),
        start_time=1,
        end_time=2,
    )


def _serve(status: int) -> tuple[http.server.HTTPServer, list[str]]:
    seen_auth: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            seen_auth.append(self.headers.get("Authorization", ""))
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen_auth


def _exporter(url: str) -> otel_provider._DropReportingOTLPSpanExporter:
    return otel_provider._DropReportingOTLPSpanExporter(
        endpoint=url,
        headers={"Authorization": "Bearer amund_invalid_test_key"},
        timeout=2,
    )


def _warnings(caplog: pytest.LogCaptureFixture) -> str:
    return " ".join(
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    )


@pytest.mark.req("MHQ-1922")
@pytest.mark.parametrize("status", [401, 403])
def test_rejected_key_is_named_over_a_real_socket(
    status: int, caplog: pytest.LogCaptureFixture, own_drops: list[tuple]
) -> None:
    server, seen_auth = _serve(status)
    try:
        port = server.server_address[1]
        exporter = _exporter(f"http://127.0.0.1:{port}/api/v1/otlp/v1/traces")
        with caplog.at_level(logging.WARNING, logger="amund"):
            result = exporter.export([_span(), _span()])
    finally:
        server.shutdown()
        server.server_close()

    assert result is SpanExportResult.FAILURE
    assert seen_auth == ["Bearer amund_invalid_test_key"], (
        "the Authorization header must still reach the server"
    )
    ((args, kwargs),) = own_drops
    assert args[0] == diagnostics.REASON_AUTH_REJECTED, (
        f"otel exporter {OTEL_EXPORTER_VERSION}: status not captured"
    )
    assert kwargs.get("status") == status
    text = _warnings(caplog)
    assert f"Amund rejected the API key ({status})" in text
    assert "could not reach" not in text


@pytest.mark.req("MHQ-1922")
def test_endpoint_rejection_is_not_reported_as_a_network_fault(
    own_drops: list[tuple],
) -> None:
    server, _ = _serve(404)
    try:
        port = server.server_address[1]
        exporter = _exporter(f"http://127.0.0.1:{port}/wrong/path")
        exporter.export([_span()])
    finally:
        server.shutdown()
        server.server_close()

    ((args, kwargs),) = own_drops
    assert args[0] == diagnostics.REASON_ENDPOINT_REJECTED
    assert kwargs.get("status") == 404


@pytest.mark.req("MHQ-1922")
def test_refused_connection_keeps_could_not_reach_wording(
    caplog: pytest.LogCaptureFixture, own_drops: list[tuple]
) -> None:
    # A port nothing listens on: bind, read the number, close.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    exporter = otel_provider._DropReportingOTLPSpanExporter(
        endpoint=f"http://127.0.0.1:{port}/api/v1/otlp/v1/traces",
        headers={"Authorization": "Bearer amund_invalid_test_key"},
        timeout=0.5,
    )
    with caplog.at_level(logging.WARNING, logger="amund"):
        result = exporter.export([_span()])

    assert result is SpanExportResult.FAILURE
    ((args, kwargs),) = own_drops
    assert args[0] in (
        diagnostics.REASON_SEND_FAILED,
        diagnostics.REASON_RETRY_EXHAUSTED,
    )
    assert kwargs.get("status") is None
    text = _warnings(caplog)
    assert "could not reach the Amund API" in text
    assert "rejected the API key" not in text


@pytest.mark.req("MHQ-1922")
def test_the_exporter_sends_through_the_recording_session() -> None:
    """Canary for a future exporter that stops using the session it is given.

    If this fails, the status capture above has lost its seam; the socket tests
    will then report every rejection as a network fault.
    """
    exporter = otel_provider._DropReportingOTLPSpanExporter(
        endpoint="http://127.0.0.1:1/x", timeout=0.2
    )
    assert isinstance(exporter._amund_session, otel_provider._StatusRecordingSession)
    calls: list[str] = []
    real_request = exporter._amund_session.request

    def _spy(method, url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        calls.append(url)
        return real_request(method, url, *args, **kwargs)

    exporter._amund_session.request = _spy  # type: ignore[method-assign]
    exporter.export([_span()])
    assert calls, f"otel exporter {OTEL_EXPORTER_VERSION} did not send through session="
