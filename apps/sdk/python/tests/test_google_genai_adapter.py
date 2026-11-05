"""google-genai adapter (MHQ-1506).

Real ``google-genai`` SDK, HTTP mocked with ``httpx.MockTransport``. No key, no
network. Covers the Gemini API and Vertex AI clients, sync and async,
``generate_content`` and ``generate_content_stream``, thinking/cached tokens,
failed calls, function calls (manual and automatic function calling), the
framework double-record guard, and the auto-detect wiring.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

genai = pytest.importorskip("google.genai")
httpx = pytest.importorskip("httpx")

from google.auth import credentials as google_credentials  # noqa: E402
from google.genai import types  # noqa: E402

from morse_ai import tracing  # noqa: E402
from morse_ai._internal import auto_detect  # noqa: E402
from morse_ai._internal.active_framework import (  # noqa: E402
    reset_llm_owner,
    set_llm_owner,
)
from morse_ai.adapters import google_genai as av_genai  # noqa: E402
from morse_ai.pricing import calculate_cost  # noqa: E402

_MODEL = "gemini-2.5-flash-lite"
_FAKE_KEY = "test-not-a-real-key"  # pragma: allowlist secret


class _FakeCredentials(google_credentials.Credentials):
    """Always-valid Vertex credentials; never touches the network."""

    def __init__(self) -> None:
        super().__init__()
        self.token = "fake-token"  # noqa: S105

    def refresh(self, request: Any) -> None:  # noqa: ARG002
        self.token = "fake-token"  # noqa: S105


def _usage(**over: int) -> dict[str, int]:
    usage = {
        "promptTokenCount": 120,
        "candidatesTokenCount": 8,
        "thoughtsTokenCount": 30,
        "cachedContentTokenCount": 0,
        "totalTokenCount": 158,
    }
    usage.update(over)
    return usage


def _body(
    *, parts: list[dict[str, Any]] | None = None, usage: dict | None = None
) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": parts or [{"text": "Hello"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": usage if usage is not None else _usage(),
        "modelVersion": _MODEL,
        "responseId": "resp-1",
    }


def _sse(events: list[dict[str, Any]]) -> httpx.Response:
    data = b"".join(b"data: " + json.dumps(e).encode() + b"\r\n\r\n" for e in events)
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=data
    )


def _stream_events() -> list[dict[str, Any]]:
    first = {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "Hel"}]}}],
        "usageMetadata": {"promptTokenCount": 120, "totalTokenCount": 120},
        "modelVersion": _MODEL,
        "responseId": "resp-s",
    }
    last = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "lo"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": _usage(),
        "modelVersion": _MODEL,
        "responseId": "resp-s",
    }
    return [first, last]


def _client(handler: Any, *, vertex: bool = False, is_async: bool = False) -> Any:
    transport = httpx.MockTransport(handler)
    http: dict[str, Any] = {"base_url": "https://genai.test"}
    if is_async:
        http["httpx_async_client"] = httpx.AsyncClient(transport=transport)
    else:
        http["httpx_client"] = httpx.Client(transport=transport)
    opts = types.HttpOptions(**http)
    if vertex:
        return genai.Client(
            vertexai=True,
            project="test-project",
            location="us-central1",
            credentials=_FakeCredentials(),
            http_options=opts,
        )
    return genai.Client(api_key=_FAKE_KEY, http_options=opts)


@pytest.fixture
def installed():
    token = tracing._active_trace.set(None)
    av_genai.uninstall()
    assert av_genai.install() is True
    try:
        yield
    finally:
        av_genai.uninstall()
        tracing._active_trace.reset(token)


def _spans(kind: str) -> list[dict[str, Any]]:
    trace = tracing._get_last_trace()
    assert trace is not None
    return [s for s in trace["spans"] if s["type"] == kind]


@pytest.mark.req("MHQ-1506")
def test_generate_content_records_llm_span_with_thinking_tokens_and_cost(installed):
    client = _client(lambda r: httpx.Response(200, json=_body()))
    with tracing.run("gemini"):
        resp = client.models.generate_content(
            model=_MODEL,
            contents="Say hello",
            config=types.GenerateContentConfig(system_instruction="Be brief."),
        )
    assert resp.text == "Hello"
    (llm,) = _spans("llm")
    assert llm["name"] == "google_genai.models.generate_content"
    assert llm["model"] == _MODEL
    assert llm["input_tokens"] == 120
    # Thinking tokens are billed as output: candidates (8) + thoughts (30).
    assert llm["output_tokens"] == 38
    assert llm["reasoning_tokens"] == 30
    assert llm["cached_input_tokens"] == 0
    assert llm["total_tokens"] == 158
    expected = calculate_cost(input_tokens=120, output_tokens=38, model=_MODEL)
    assert llm["cost_usd"] == pytest.approx(round(expected, 6))
    assert llm["metadata"]["provider"] == "gemini"
    assert "location" not in llm["metadata"]
    assert llm["metadata"]["finish_reason"] == "STOP"
    assert llm["output_data"] == {"content": "Hello"}
    assert llm["input_data"]["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Say hello"},
    ]
    assert llm["status"] == "completed"


@pytest.mark.req("MHQ-1506")
def test_cached_tokens_reduce_cost(installed):
    usage = _usage(cachedContentTokenCount=100, thoughtsTokenCount=0)
    client = _client(lambda r: httpx.Response(200, json=_body(usage=usage)))
    with tracing.run("gemini-cache"):
        client.models.generate_content(model=_MODEL, contents="hi")
    (llm,) = _spans("llm")
    assert llm["cached_input_tokens"] == 100
    assert llm.get("reasoning_tokens") is None
    expected = calculate_cost(
        input_tokens=120, output_tokens=8, model=_MODEL, cached_input_tokens=100
    )
    assert llm["cost_usd"] == pytest.approx(round(expected, 6))


@pytest.mark.req("MHQ-1506")
def test_vertex_client_records_provider_and_location(installed):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json=_body())

    client = _client(handler, vertex=True)
    with tracing.run("vertex"):
        client.models.generate_content(model=_MODEL, contents="hi")
    assert "publishers/google/models" in seen[0]
    (llm,) = _spans("llm")
    assert llm["model"] == _MODEL
    assert llm["metadata"]["provider"] == "vertex_ai"
    assert llm["metadata"]["location"] == "us-central1"


@pytest.mark.req("MHQ-1506")
def test_model_falls_back_to_normalized_request_model(installed):
    body = _body()
    del body["modelVersion"]
    client = _client(lambda r: httpx.Response(200, json=body))
    with tracing.run("no-version"):
        client.models.generate_content(model=f"models/{_MODEL}", contents="hi")
    (llm,) = _spans("llm")
    assert llm["model"] == _MODEL
    assert llm["cost_usd"] and llm["cost_usd"] > 0


@pytest.mark.req("MHQ-1506")
def test_call_outside_a_trace_opens_its_own(installed):
    client = _client(lambda r: httpx.Response(200, json=_body()))
    client.models.generate_content(model=_MODEL, contents="hi")
    (llm,) = _spans("llm")
    assert llm["model"] == _MODEL


@pytest.mark.req("MHQ-1506")
def test_failed_call_records_failed_span(installed):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "message": "bad request",
                    "status": "INVALID_ARGUMENT",
                }
            },
        )

    client = _client(handler)
    with tracing.run("fail"), pytest.raises(genai.errors.ClientError):
        client.models.generate_content(model=_MODEL, contents="hi")
    (llm,) = _spans("llm")
    assert llm["status"] == "failed"
    assert llm["model"] == _MODEL
    assert "ClientError" in llm["error_message"]
    assert llm.get("cost_usd") is None


@pytest.mark.req("MHQ-1506")
def test_stream_aggregates_usage_and_text(installed):
    client = _client(lambda r: _sse(_stream_events()))
    with tracing.run("stream"):
        text = "".join(
            c.text or ""
            for c in client.models.generate_content_stream(model=_MODEL, contents="hi")
        )
    assert text == "Hello"
    (llm,) = _spans("llm")
    assert llm["name"] == "google_genai.models.generate_content_stream"
    assert llm["input_tokens"] == 120
    assert llm["output_tokens"] == 38
    assert llm["cost_usd"] and llm["cost_usd"] > 0
    assert llm["output_data"] == {"content": "Hello"}
    assert llm["metadata"]["streamed"] is True
    assert llm["metadata"]["time_to_first_chunk_ms"] >= 0


@pytest.mark.req("MHQ-1506")
def test_stream_outside_a_trace_records_when_it_ends(installed):
    client = _client(lambda r: _sse(_stream_events()))
    for _ in client.models.generate_content_stream(model=_MODEL, contents="hi"):
        pass
    (llm,) = _spans("llm")
    assert llm["output_tokens"] == 38


@pytest.mark.req("MHQ-1506")
def test_async_generate_content_and_stream(installed):
    async def main() -> None:
        client = _client(lambda r: httpx.Response(200, json=_body()), is_async=True)
        with tracing.run("async"):
            await client.aio.models.generate_content(model=_MODEL, contents="hi")
        (llm,) = _spans("llm")
        assert llm["output_tokens"] == 38

        sclient = _client(lambda r: _sse(_stream_events()), is_async=True)
        with tracing.run("async-stream"):
            chunks = [
                c
                async for c in await sclient.aio.models.generate_content_stream(
                    model=_MODEL, contents="hi"
                )
            ]
        assert len(chunks) == 2
        (sllm,) = _spans("llm")
        assert sllm["metadata"]["streamed"] is True
        assert sllm["input_tokens"] == 120

    asyncio.run(main())


@pytest.mark.req("MHQ-1506")
def test_manual_function_call_loop_yields_tool_span(installed):
    call = {"functionCall": {"name": "get_price", "args": {"ticker": "AAPL"}}}
    bodies = [_body(parts=[call]), _body()]
    client = _client(lambda r: httpx.Response(200, json=bodies.pop(0)))
    with tracing.run("tools"):
        first = client.models.generate_content(model=_MODEL, contents="price?")
        fc = first.function_calls[0]
        client.models.generate_content(
            model=_MODEL,
            contents=[
                types.Content(role="user", parts=[types.Part.from_text(text="price?")]),
                first.candidates[0].content,
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=fc.name, response={"price": 1}
                        )
                    ],
                ),
            ],
        )
    llms = _spans("llm")
    assert len(llms) == 2
    assert llms[0]["output_data"]["content"].startswith("[tool_call:get_price]")
    (tool,) = _spans("tool")
    assert tool["name"] == "get_price"
    assert tool["input_data"] == {"arguments": {"ticker": "AAPL"}}
    assert json.loads(tool["output_data"]["content"]) == {"price": 1}
    assert tool["metadata"]["timing"] == "inferred"
    roles = [m["role"] for m in llms[1]["input_data"]["messages"]]
    assert roles == ["user", "assistant", "user"]


@pytest.mark.req("MHQ-1506")
def test_automatic_function_calling_records_one_span_per_request(installed):
    def get_price(ticker: str) -> dict:
        """Return the price for a ticker."""
        return {"price": 42}

    call = {"functionCall": {"name": "get_price", "args": {"ticker": "AAPL"}}}
    bodies = [_body(parts=[call]), _body()]
    client = _client(lambda r: httpx.Response(200, json=bodies.pop(0)))
    with tracing.run("afc"):
        resp = client.models.generate_content(
            model=_MODEL,
            contents="price?",
            config=types.GenerateContentConfig(tools=[get_price]),
        )
    assert resp.text == "Hello"
    llms = _spans("llm")
    # Both billed requests are priced, not just the last one.
    assert len(llms) == 2
    assert all(s["output_tokens"] == 38 for s in llms)
    (tool,) = _spans("tool")
    assert tool["name"] == "get_price"


@pytest.mark.req("MHQ-1506")
def test_framework_owned_call_is_not_recorded_twice(installed):
    client = _client(lambda r: httpx.Response(200, json=_body()))
    with tracing.run("owned"):
        token = set_llm_owner("some-framework")
        try:
            client.models.generate_content(model=_MODEL, contents="hi")
        finally:
            reset_llm_owner(token)
    assert _spans("llm") == []


@pytest.mark.req("MHQ-1506")
def test_call_inside_framework_llm_span_is_not_recorded_twice(installed):
    client = _client(lambda r: httpx.Response(200, json=_body()))
    with (
        tracing.run("lc"),
        tracing.span("ChatGoogleGenerativeAI", span_type="llm"),
    ):
        client.models.generate_content(model=_MODEL, contents="hi")
    llms = _spans("llm")
    assert [s["name"] for s in llms] == ["ChatGoogleGenerativeAI"]


@pytest.mark.req("MHQ-1506")
def test_install_is_idempotent_and_uninstall_restores():
    from google.genai import models as genai_models

    original = genai_models.Models.__dict__["_generate_content"]
    av_genai.uninstall()
    assert av_genai.install() is True
    assert av_genai.install() is True
    patched = genai_models.Models.__dict__["_generate_content"]
    assert patched.__wrapped__ is original
    av_genai.uninstall()
    assert genai_models.Models.__dict__["_generate_content"] is original
    assert av_genai.is_installed() is False


@pytest.mark.req("MHQ-1506")
def test_auto_detect_targets_google_genai():
    assert auto_detect._adapter_targets()["google.genai"] == (
        "morse_ai.adapters.google_genai"
    )


def test_extract_usage_handles_missing_fields():
    assert av_genai.extract_usage(None, _MODEL) == {}
    usage = types.GenerateContentResponseUsageMetadata(prompt_token_count=5)
    out = av_genai.extract_usage(usage, "unknown-model")
    assert out["input_tokens"] == 5
    assert out["output_tokens"] == 0
    assert "cost_usd" not in out


@pytest.mark.req("MHQ-1506")
def test_afc_outside_a_trace_keeps_both_requests_and_the_tool_in_one_trace(
    installed,
):
    def get_price(ticker: str) -> dict:
        """Return the price for a ticker."""
        return {"price": 42}

    call = {"functionCall": {"name": "get_price", "args": {"ticker": "AAPL"}}}
    bodies = [_body(parts=[call]), _body()]
    client = _client(lambda r: httpx.Response(200, json=bodies.pop(0)))
    client.models.generate_content(
        model=_MODEL,
        contents="price?",
        config=types.GenerateContentConfig(tools=[get_price]),
    )
    assert len(_spans("llm")) == 2
    assert [s["name"] for s in _spans("tool")] == ["get_price"]
