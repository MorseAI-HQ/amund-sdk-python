"""ADR-019 F2.3 — streaming-safety proof for in-process capture.

The dogfood Copilot (and any customer agent on a streaming endpoint) runs the
graph inside a generator that yields between steps. OTel auto-instrumentation
ORPHANS the http/db spans emitted during that generator because the request's
OTel context has already been torn down by the time the generator iterates.

The Morse SDK capture does NOT have that problem: it keys on the SDK's
own `_active_trace` ContextVar, which the callback sets INSIDE the generator
body (run().__enter__()). ContextVars propagate through generator execution,
so DB/HTTP calls made mid-stream land on the trace correctly.

These tests pin that behavior so we can honestly claim streaming support.
"""

import sqlite3

import httpx
import pytest

from morse_ai.infra_capture import (
    disable_db_capture,
    disable_http_capture,
    enable_db_capture,
    enable_http_capture,
)
from morse_ai.tracing import _get_last_trace, run


@pytest.fixture(autouse=True)
def _restore():
    yield
    disable_db_capture()
    disable_http_capture()


def test_db_capture_survives_generator_streaming():
    """Mimics the SSE pattern: trace opened inside a generator, DB calls made
    between yields, generator driven externally (as StreamingHttpResponse does)."""
    enable_db_capture()

    def sse_like_stream():
        # The callback opens the trace INSIDE the generator body.
        with run("streaming-agent"):
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE TABLE s (a INTEGER)")
            yield "event: chunk1"
            conn.execute("INSERT INTO s VALUES (1)")  # DB call mid-stream
            yield "event: chunk2"
            conn.execute("SELECT a FROM s")
            yield "event: done"
            conn.close()

    # Drive the generator the way Django's StreamingHttpResponse iterator does.
    chunks = list(sse_like_stream())
    assert chunks == ["event: chunk1", "event: chunk2", "event: done"]

    trace = _get_last_trace()
    verbs = [s["name"] for s in trace["spans"] if s["type"] == "db"]
    # All three DB ops — including the ones AFTER the first yield — correlate.
    assert "CREATE" in verbs
    assert "INSERT" in verbs
    assert "SELECT" in verbs


def test_http_capture_survives_generator_streaming():
    enable_http_capture()
    transport = httpx.MockTransport(lambda req: httpx.Response(200))

    def sse_like_stream():
        with run("streaming-http-agent"):
            client = httpx.Client(transport=transport)
            client.get("https://api.example.com/start")
            yield "chunk1"
            client.get("https://api.example.com/mid")  # mid-stream HTTP
            yield "chunk2"
            client.close()

    list(sse_like_stream())
    trace = _get_last_trace()
    names = [s["name"] for s in trace["spans"] if s["type"] == "http"]
    assert "GET api.example.com" in names
    assert names.count("GET api.example.com") == 2  # both calls captured


def test_no_cross_generator_trace_bleed():
    """Two independent streamed traces must not bleed spans into each other —
    the ContextVar is reset on each run() exit."""
    enable_db_capture()

    def stream(tag: str):
        with run(f"agent-{tag}"):
            conn = sqlite3.connect(":memory:")
            conn.execute(f"CREATE TABLE t_{tag} (a INTEGER)")
            yield tag
            conn.close()

    list(stream("a"))
    trace_a = _get_last_trace()
    list(stream("b"))
    trace_b = _get_last_trace()

    # Each trace has exactly its own one CREATE span, no leakage.
    assert len([s for s in trace_a["spans"] if s["type"] == "db"]) == 1
    assert len([s for s in trace_b["spans"] if s["type"] == "db"]) == 1
    assert trace_a["trace"]["agent"] == "agent-a"
    assert trace_b["trace"]["agent"] == "agent-b"
