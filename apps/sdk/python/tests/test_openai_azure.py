"""MHQ-1741: Azure OpenAI spans are priced on the canonical model.

Real ``openai`` SDK (``AzureOpenAI`` / ``AsyncAzureOpenAI``), HTTP mocked with
``httpx.MockTransport``. No key, no network. Shapes follow what Azure returned
on 2026-11-04 (deployment ``morse-hr-mini`` -> ``gpt-4.1-mini-2025-04-14``,
api-version ``2025-04-01-preview``):

- Responses API: ``response.model`` is the *deployment name* (also on every
  streamed event, and on the ``/openai/v1`` path).
- Chat Completions: ``model`` is the canonical dated model; a stream's first
  chunk (prompt-filter results) has ``model: ""``.
- Every response carries ``x-ms-served-model: gpt-4.1-mini-2025-04-14``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")

from amund import tracing  # noqa: E402
from amund.adapters import openai as av_openai  # noqa: E402

_DEPLOYMENT = "morse-hr-mini"
_SERVED = "gpt-4.1-mini-2025-04-14"
_ENDPOINT = "https://morse-testbed.openai.azure.com"
_API_VERSION = "2025-04-01-preview"
_HEADERS = {"x-ms-served-model": _SERVED, "x-ms-region": "East US 2"}


def _responses_body(model: str = _DEPLOYMENT) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi!", "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 1000,
            "total_tokens": 2000,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _chat_body(model: str = _SERVED) -> dict[str, Any]:
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


def _sse(events: list[dict[str, Any]]) -> bytes:
    out = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
    return out + b"data: [DONE]\n\n"


def _responses_events() -> list[dict[str, Any]]:
    created = {**_responses_body(), "status": "in_progress", "output": []}
    created["usage"] = None
    return [
        {"type": "response.created", "sequence_number": 0, "response": created},
        {
            "type": "response.completed",
            "sequence_number": 1,
            "response": _responses_body(),
        },
    ]


def _chat_chunks() -> list[dict[str, Any]]:
    base = {"id": "chatcmpl-s", "object": "chat.completion.chunk", "created": 0}
    return [
        # Azure's prompt-filter chunk: empty model, no choices.
        {**base, "model": "", "choices": [], "prompt_filter_results": []},
        {
            **base,
            "model": _SERVED,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}}],
        },
        {
            **base,
            "model": _SERVED,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            **base,
            "model": _SERVED,
            "choices": [],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 1000,
                "total_tokens": 2000,
            },
        },
    ]


def _handler(seen: list[str], *, headers: dict[str, str] | None = None):  # noqa: ANN202
    """Route like Azure: /responses or /deployments/<d>/chat/completions."""
    hdrs = _HEADERS if headers is None else headers

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        body = json.loads(request.content or b"{}")
        streaming = bool(body.get("stream"))
        if request.url.path.endswith("/responses"):
            if streaming:
                return httpx.Response(
                    200,
                    headers={**hdrs, "content-type": "text/event-stream"},
                    content=_sse(_responses_events()),
                )
            return httpx.Response(200, headers=hdrs, json=_responses_body())
        if streaming:
            return httpx.Response(
                200,
                headers={**hdrs, "content-type": "text/event-stream"},
                content=_sse(_chat_chunks()),
            )
        return httpx.Response(200, headers=hdrs, json=_chat_body())

    return handle


def _azure(handler, *, is_async: bool = False, **extra: Any):  # noqa: ANN001, ANN202
    kw: dict[str, Any] = {
        "api_key": "azure-test-not-a-real-key",  # pragma: allowlist secret
        "azure_endpoint": _ENDPOINT,
        "api_version": _API_VERSION,
        "max_retries": 0,
        **extra,
    }
    if is_async:
        return openai.AsyncAzureOpenAI(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kw
        )
    return openai.AzureOpenAI(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kw
    )


@pytest.fixture(autouse=True)
def _clean():
    token = tracing._active_trace.set(None)
    av_openai.uninstall()
    try:
        yield
    finally:
        av_openai.uninstall()
        tracing._active_trace.reset(token)


@pytest.fixture
def installed():
    av_openai.install()
    yield


def _llm() -> dict[str, Any]:
    trace = tracing._get_last_trace()
    assert trace is not None
    spans = [s for s in trace["spans"] if s["type"] == "llm"]
    assert len(spans) == 1
    return spans[0]


def _assert_canonical(span: dict[str, Any], source: str = "served_model_header"):
    assert span["model"] == _SERVED
    assert span["cost_usd"] and span["cost_usd"] > 0
    assert span["metadata"]["provider"] == "azure_openai"
    assert span["metadata"]["azure_deployment"] == _DEPLOYMENT
    assert span["metadata"]["azure_model_source"] == source


# ---------------------------------------------------------------------------
# Responses API — the reported bug (deployment name in response.model)
# ---------------------------------------------------------------------------


@pytest.mark.req("MHQ-1741")
def test_install_async_responses_prices_the_served_model(installed):
    seen: list[str] = []
    client = _azure(_handler(seen), is_async=True)

    async def go() -> Any:
        with tracing.run("azure-search"):
            return await client.responses.create(model=_DEPLOYMENT, input="hi")

    response = asyncio.run(go())
    assert response.model == _DEPLOYMENT  # the caller's object is untouched
    assert seen == ["/openai/responses"]
    _assert_canonical(_llm())


@pytest.mark.req("MHQ-1741")
def test_install_sync_responses_prices_the_served_model(installed):
    client = _azure(_handler([]))
    with tracing.run("azure"):
        client.responses.create(model=_DEPLOYMENT, input="hi")
    _assert_canonical(_llm())


@pytest.mark.req("MHQ-1741")
def test_wrap_sync_responses_prices_the_served_model():
    client = av_openai.wrap(_azure(_handler([])))
    with tracing.run("azure"):
        client.responses.create(model=_DEPLOYMENT, input="hi")
    _assert_canonical(_llm())


@pytest.mark.req("MHQ-1741")
def test_wrap_async_responses_stream_prices_the_served_model():
    client = av_openai.wrap(_azure(_handler([]), is_async=True))

    async def go() -> None:
        with tracing.run("azure-stream"):
            stream = await client.responses.create(
                model=_DEPLOYMENT, input="hi", stream=True
            )
            async for _ in stream:
                pass

    asyncio.run(go())
    span = _llm()
    _assert_canonical(span)
    assert span["metadata"]["streamed"] is True
    assert span["input_tokens"] == 1000


@pytest.mark.req("MHQ-1741")
def test_install_sync_responses_stream_outside_run_prices_the_served_model(
    installed,
):
    client = _azure(_handler([]))
    stream = client.responses.create(model=_DEPLOYMENT, input="hi", stream=True)
    for _ in stream:
        pass
    _assert_canonical(_llm())


@pytest.mark.req("MHQ-1741")
def test_responses_without_header_falls_back_to_the_deployment(installed):
    client = _azure(_handler([], headers={}))
    with tracing.run("azure"):
        client.responses.create(model=_DEPLOYMENT, input="hi")
    span = _llm()
    # Nothing better is knowable client-side: the previous behaviour, labelled.
    assert span["model"] == _DEPLOYMENT
    assert span["metadata"]["provider"] == "azure_openai"
    assert span["metadata"]["azure_deployment"] == _DEPLOYMENT
    assert span["metadata"]["azure_model_source"] == "deployment"


# ---------------------------------------------------------------------------
# Chat Completions — canonical in the body; the deployment used to be lost
# ---------------------------------------------------------------------------


@pytest.mark.req("MHQ-1741")
def test_install_sync_chat_keeps_the_deployment(installed):
    seen: list[str] = []
    client = _azure(_handler(seen))
    with tracing.run("azure-chat"):
        client.chat.completions.create(
            model=_DEPLOYMENT, messages=[{"role": "user", "content": "hi"}]
        )
    assert seen == [f"/openai/deployments/{_DEPLOYMENT}/chat/completions"]
    _assert_canonical(_llm())


@pytest.mark.req("MHQ-1741")
def test_chat_without_header_uses_the_response_model(installed):
    client = _azure(_handler([], headers={}))
    with tracing.run("azure-chat"):
        client.chat.completions.create(
            model=_DEPLOYMENT, messages=[{"role": "user", "content": "hi"}]
        )
    _assert_canonical(_llm(), source="response")


@pytest.mark.req("MHQ-1741")
def test_wrap_async_chat_stream_prices_the_served_model():
    client = av_openai.wrap(_azure(_handler([]), is_async=True))

    async def go() -> None:
        with tracing.run("azure-chat-stream"):
            stream = await client.chat.completions.create(
                model=_DEPLOYMENT,
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
                stream_options={"include_usage": True},
            )
            async for _ in stream:
                pass

    asyncio.run(go())
    span = _llm()
    _assert_canonical(span)
    assert span["output_data"] == {"content": "Hi"}


@pytest.mark.req("MHQ-1741")
def test_install_with_raw_response_litellm_path(installed):
    """LiteLLM's Azure provider calls ``chat.completions.with_raw_response``."""
    client = _azure(_handler([]))
    with tracing.run("litellm"):
        raw = client.chat.completions.with_raw_response.create(
            model=_DEPLOYMENT, messages=[{"role": "user", "content": "hi"}]
        )
        parsed = raw.parse()
    assert parsed.model == _SERVED
    _assert_canonical(_llm())


@pytest.mark.req("MHQ-1741")
def test_azure_deployment_on_the_client_wins_over_model_kwarg(installed):
    seen: list[str] = []
    client = _azure(_handler(seen), azure_deployment=_DEPLOYMENT)
    with tracing.run("azure-chat"):
        client.chat.completions.create(
            model="ignored-by-azure", messages=[{"role": "user", "content": "hi"}]
        )
    assert seen == [f"/openai/deployments/{_DEPLOYMENT}/chat/completions"]
    _assert_canonical(_llm())


@pytest.mark.req("MHQ-1741")
def test_plain_openai_client_on_the_azure_v1_endpoint_is_detected(installed):
    client = openai.OpenAI(
        api_key="azure-test-not-a-real-key",  # pragma: allowlist secret
        base_url=f"{_ENDPOINT}/openai/v1/",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(_handler([]))),
    )
    with tracing.run("azure-v1"):
        client.responses.create(model=_DEPLOYMENT, input="hi")
    _assert_canonical(_llm())


# ---------------------------------------------------------------------------
# Safety: other clients untouched, the customer's hooks keep running, and
# concurrent calls do not cross-attribute.
# ---------------------------------------------------------------------------


@pytest.mark.req("MHQ-1741")
def test_non_azure_client_gets_no_azure_metadata_or_hook(installed):
    http = httpx.Client(transport=httpx.MockTransport(_handler([])))
    client = openai.OpenAI(
        api_key="sk-test-not-a-real-key",  # pragma: allowlist secret
        base_url="https://api.openai.test/v1",
        max_retries=0,
        http_client=http,
    )
    with tracing.run("openai"):
        client.chat.completions.create(
            model="gpt-4.1-mini", messages=[{"role": "user", "content": "hi"}]
        )
    span = _llm()
    assert span["model"] == _SERVED  # the body model, as before
    assert "provider" not in (span.get("metadata") or {})
    assert http.event_hooks["response"] == []


@pytest.mark.req("MHQ-1741")
def test_customer_response_hooks_are_kept_and_hook_is_added_once(installed):
    calls: list[int] = []
    http = httpx.Client(
        transport=httpx.MockTransport(_handler([])),
        event_hooks={"response": [lambda r: calls.append(r.status_code)]},
    )
    client = openai.AzureOpenAI(
        api_key="azure-test-not-a-real-key",  # pragma: allowlist secret
        azure_endpoint=_ENDPOINT,
        api_version=_API_VERSION,
        max_retries=0,
        http_client=http,
    )
    with tracing.run("azure"):
        client.responses.create(model=_DEPLOYMENT, input="hi")
        client.responses.create(model=_DEPLOYMENT, input="again")
    assert calls == [200, 200]
    assert len(http.event_hooks["response"]) == 2


@pytest.mark.req("MHQ-1741")
def test_concurrent_async_calls_keep_their_own_served_model(installed):
    served = {"dep-a": "gpt-4.1-mini-2025-04-14", "dep-b": "gpt-4.1-2025-04-14"}

    async def handle(request: httpx.Request) -> httpx.Response:
        dep = json.loads(request.content)["model"]
        await asyncio.sleep(0.02 if dep == "dep-a" else 0)
        return httpx.Response(
            200,
            headers={"x-ms-served-model": served[dep]},
            json=_responses_body(model=dep),
        )

    client = _azure(handle, is_async=True)
    models: dict[str, str] = {}

    async def one(dep: str) -> None:
        with tracing.run(dep):
            await client.responses.create(model=dep, input="hi")
            state = tracing.get_active_trace()
            (span,) = [s for s in state.completed_spans if s.type == "llm"]
            models[dep] = span.model

    async def go() -> None:
        await asyncio.gather(one("dep-a"), one("dep-b"))

    asyncio.run(go())
    assert models == served
