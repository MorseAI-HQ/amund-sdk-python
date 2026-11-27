"""MHQ-1944: OpenAI-client calls to Groq, Gemini ... record the real provider.

Intern testers (2026-11-25) called Groq and Gemini through ``OpenAI(base_url=...)``
and both showed as OpenAI. Real ``openai`` SDK, HTTP mocked with
``httpx.MockTransport``; no key, no network.
"""

from __future__ import annotations

from typing import Any

import pytest

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")

from amund import tracing  # noqa: E402
from amund.adapters import openai as av_openai  # noqa: E402

HOSTS = [
    ("https://api.groq.com/openai/v1", "api.groq.com", "groq"),
    (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "generativelanguage.googleapis.com",
        "gemini",
    ),
    ("https://openrouter.ai/api/v1", "openrouter.ai", "openrouter"),
    ("https://api.together.xyz/v1", "api.together.xyz", "together"),
    ("https://api.mistral.ai/v1", "api.mistral.ai", "mistral"),
    ("https://api.deepseek.com/v1", "api.deepseek.com", "deepseek"),
    ("https://api.fireworks.ai/inference/v1", "api.fireworks.ai", "fireworks"),
]


def _chat_body(model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 1000,
            "total_tokens": 2000,
        },
    }


def _client(base_url: str, model: str, *, is_async: bool = False):  # noqa: ANN202
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_body(model))

    kw: dict[str, Any] = {
        "api_key": "compat-test-not-a-real-key",  # pragma: allowlist secret
        "base_url": base_url,
        "max_retries": 0,
    }
    if is_async:
        return openai.AsyncOpenAI(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)), **kw
        )
    return openai.OpenAI(
        http_client=httpx.Client(transport=httpx.MockTransport(handle)), **kw
    )


@pytest.fixture(autouse=True)
def _clean():
    token = tracing._active_trace.set(None)
    av_openai.uninstall()
    av_openai.install()
    try:
        yield
    finally:
        av_openai.uninstall()
        tracing._active_trace.reset(token)


def _llm() -> dict[str, Any]:
    trace = tracing._get_last_trace()
    assert trace is not None
    spans = [s for s in trace["spans"] if s["type"] == "llm"]
    assert len(spans) == 1
    return spans[0]


@pytest.mark.req("MHQ-1944")
@pytest.mark.parametrize(("base_url", "host", "provider"), HOSTS)
def test_base_url_records_provider_and_server_address(base_url, host, provider):
    client = _client(base_url, "some-model")
    client.chat.completions.create(
        model="some-model", messages=[{"role": "user", "content": "hi"}]
    )
    metadata = _llm()["metadata"]
    assert metadata["provider"] == provider
    assert metadata["gen_ai.system"] == provider
    assert metadata["server.address"] == host


@pytest.mark.req("MHQ-1944")
def test_async_client_records_the_provider():
    import asyncio

    client = _client(
        "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", is_async=True
    )
    asyncio.run(
        client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    assert _llm()["metadata"]["provider"] == "groq"


@pytest.mark.req("MHQ-1944")
def test_resale_host_reports_no_sdk_cost_for_another_vendors_rate():
    # gpt-oss-120b has a bundled (Bedrock) rate; Groq's price is its own.
    client = _client("https://api.groq.com/openai/v1", "openai/gpt-oss-120b")
    client.chat.completions.create(
        model="openai/gpt-oss-120b", messages=[{"role": "user", "content": "hi"}]
    )
    assert _llm().get("cost_usd") is None


@pytest.mark.req("MHQ-1944")
def test_plain_openai_client_is_unchanged():
    client = _client("https://api.openai.com/v1", "gpt-4o-mini")
    client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
    )
    span = _llm()
    assert "provider" not in (span.get("metadata") or {})
    assert span["cost_usd"] and span["cost_usd"] > 0


@pytest.mark.req("MHQ-1944")
def test_unknown_host_is_not_guessed():
    client = _client("https://llm-proxy.internal.example/v1", "gpt-4o-mini")
    client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
    )
    assert "provider" not in (_llm().get("metadata") or {})
