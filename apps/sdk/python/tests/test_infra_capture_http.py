"""Tests for ADR-019 F2.2 — in-process HTTP auto-capture.

Uses httpx MockTransport so no real network is touched. Covers the
LLM-host-exclusion rule that prevents double-counting the `llm` spans the
adapters already emit.
"""

import httpx
import pytest

from morse_ai.infra_capture import (
    _host_excluded,
    add_excluded_host,
    disable_http_capture,
    enable_http_capture,
)
from morse_ai.tracing import _get_last_trace, run


@pytest.fixture(autouse=True)
def _restore():
    yield
    disable_http_capture()


def _http_spans(trace: dict) -> list[dict]:
    return [s for s in trace["spans"] if s["type"] == "http"]


def _mock_client() -> httpx.Client:
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"ok": True}))
    return httpx.Client(transport=transport)


def test_enable_wraps_httpx_and_requests():
    wrapped = enable_http_capture()
    assert "httpx" in wrapped
    assert "requests" in wrapped


def test_httpx_call_inside_trace_emits_http_span():
    enable_http_capture()
    with run("agent-http"):
        client = _mock_client()
        client.get("https://example.com/data")
        client.post("https://api.internal.example.com/v1/do")
        client.close()

    trace = _get_last_trace()
    names = [s["name"] for s in _http_spans(trace)]
    assert "GET example.com" in names
    assert "POST api.internal.example.com" in names


def test_llm_provider_hosts_excluded():
    """httpx calls to api.anthropic.com / api.openai.com must NOT become http
    spans (they're already llm spans via the adapters)."""
    enable_http_capture()
    with run("agent-llm-host"):
        # Point the mock at an LLM host — should be skipped.
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
        client = httpx.Client(transport=transport)
        client.post("https://api.anthropic.com/v1/messages")
        client.post("https://api.openai.com/v1/chat/completions")
        client.get("https://example.com/ok")  # this one IS captured
        client.close()

    trace = _get_last_trace()
    names = [s["name"] for s in _http_spans(trace)]
    assert names == ["GET example.com"]


def test_subdomain_of_excluded_host_is_excluded():
    assert _host_excluded("eu.api.anthropic.com") is True
    assert _host_excluded("api.anthropic.com") is True
    assert _host_excluded("example.com") is False


def test_hf_router_hosts_excluded_but_dedicated_endpoints_captured():
    """HF Inference Providers / serverless router hosts are reached via the
    OpenAI-compatible adapter (already `llm` spans) so must NOT also become http
    spans (double-count → unpriced `hf` sibling). Dedicated Endpoints
    (*.endpoints.huggingface.cloud) have no adapter span, so they ARE captured."""
    # Router/serverless → excluded (adapter-handled).
    assert _host_excluded("router.huggingface.co") is True
    assert _host_excluded("api-inference.huggingface.co") is True
    # Dedicated Endpoints → NOT excluded (genuine raw call, honest hf visibility).
    assert _host_excluded("abc123.us-east-1.aws.endpoints.huggingface.cloud") is False


def test_extra_excluded_host_runtime():
    enable_http_capture(extra_excluded_hosts=["internal.mycorp.com"])
    with run("agent-extra-exclude"):
        transport = httpx.MockTransport(lambda req: httpx.Response(200))
        client = httpx.Client(transport=transport)
        client.get("https://internal.mycorp.com/health")  # excluded
        client.get("https://example.com/x")  # captured
        client.close()
    trace = _get_last_trace()
    names = [s["name"] for s in _http_spans(trace)]
    assert names == ["GET example.com"]


def test_no_http_span_when_no_active_trace():
    enable_http_capture()
    client = _mock_client()
    resp = client.get("https://example.com/data")  # outside any run()
    client.close()
    assert resp.status_code == 200  # request still works


def test_disable_restores_httpx_send():
    enable_http_capture()
    assert httpx.Client.send.__name__ == "send"  # our wrapper is also named send
    disable_http_capture()
    # After restore, calling still works and emits nothing inside a trace-less ctx.
    client = _mock_client()
    assert client.get("https://example.com").status_code == 200
    client.close()


def test_add_excluded_host_helper():
    add_excluded_host("Foo.Example.COM")
    assert _host_excluded("foo.example.com") is True


def test_morse_ai_endpoint_host_auto_excluded(monkeypatch):
    """The SDK's own ingest endpoint host must be excluded so telemetry POSTs
    don't self-trace as http spans."""
    monkeypatch.setenv(
        "MORSE_ENDPOINT", "https://ingest.morsehq.dev/api/v1/traces/batch"
    )
    enable_http_capture()
    assert _host_excluded("ingest.morsehq.dev") is True


@pytest.mark.anyio
async def test_async_httpx_capture():
    enable_http_capture()
    with run("agent-async-http"):
        transport = httpx.MockTransport(lambda req: httpx.Response(200))
        async with httpx.AsyncClient(transport=transport) as client:
            await client.get("https://async.example.com/ping")
    trace = _get_last_trace()
    names = [s["name"] for s in _http_spans(trace)]
    assert "GET async.example.com" in names


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# Step 1b: response-status capture — 5xx/429 dependency failures must NOT be
# recorded as "completed". status_code is always recorded for correlation.
# ---------------------------------------------------------------------------


def _span_for(trace: dict) -> dict:
    spans = _http_spans(trace)
    assert len(spans) == 1, f"expected 1 http span, got {len(spans)}"
    return spans[0]


@pytest.mark.parametrize(
    "code,expected_status",
    [
        (200, "completed"),
        (404, "completed"),  # client error — caller's bug, not a dependency outage
        (401, "completed"),
        (429, "failed"),  # rate-limited dependency
        (500, "failed"),
        (502, "failed"),
        (503, "failed"),
        (504, "failed"),
        (408, "failed"),  # server-returned request timeout
    ],
)
def test_http_span_status_follows_response_code(code, expected_status):
    enable_http_capture()
    with run("agent-http-status"):
        transport = httpx.MockTransport(lambda req: httpx.Response(code))
        client = httpx.Client(transport=transport)
        client.get("https://dep.example.com/x")
        client.close()

    span = _span_for(_get_last_trace())
    assert span["status"] == expected_status
    assert (span["metadata"] or {}).get("http.status_code") == code


@pytest.mark.anyio
async def test_async_http_5xx_marked_failed():
    enable_http_capture()
    with run("agent-async-5xx"):
        transport = httpx.MockTransport(lambda req: httpx.Response(503))
        async with httpx.AsyncClient(transport=transport) as client:
            await client.get("https://async.dep.example.com/x")

    span = _span_for(_get_last_trace())
    assert span["status"] == "failed"
    assert (span["metadata"] or {}).get("http.status_code") == 503
