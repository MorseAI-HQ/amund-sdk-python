"""Launch-verification repros for the Anthropic findings (group I, 2026-10-18).

Real ``anthropic`` SDK, HTTP layer mocked (``httpx.MockTransport`` / ``respx``),
recorded response shapes from the dogfood round. No key, no network.

- MHQ-1415 item 2: a failed ``messages.create`` (401) under the zero-code
  install must leave a failed ``llm`` span carrying the requested model.
- MHQ-1415 item 3: LangChain-Anthropic with ``init()`` (zero-code Anthropic
  install) plus ``MorseCallbackHandler`` must give ONE ``llm`` span per call,
  not an extra empty ``anthropic.messages.create`` span.
- MHQ-1414 (LangChain path): cache-write tokens reach the span as
  ``cache_write_tokens`` instead of being priced as plain input.
"""

# ruff: noqa: E501
from __future__ import annotations

from typing import Any

import pytest

from amund import tracing
from amund.adapters import anthropic as av_anthropic

_MODEL = "claude-haiku-4-5-20251001"
_KW = {
    "model": _MODEL,
    "max_tokens": 32,
    "messages": [{"role": "user", "content": "hi"}],
}


def _message_body(cache_write: int = 0, cache_read: int = 0) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": _MODEL,
        "content": [{"type": "text", "text": "Hello"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 9,
            "output_tokens": 3,
            "cache_creation_input_tokens": cache_write,
            "cache_read_input_tokens": cache_read,
        },
    }


@pytest.fixture
def installed():
    token = tracing._active_trace.set(None)
    av_anthropic.install()
    try:
        yield
    finally:
        av_anthropic.uninstall()
        tracing._active_trace.reset(token)


def _llm(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in trace["spans"] if s["type"] == "llm"]


@pytest.mark.req("MHQ-1415")
def test_mhq1415_failed_create_leaves_a_failed_llm_span(installed):
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx")

    def handler(request):  # noqa: ANN001, ANN202
        return httpx.Response(
            401,
            json={
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "invalid x-api-key",
                },
            },
        )

    client = anthropic.Anthropic(
        api_key="test",  # pragma: allowlist secret
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    with pytest.raises(anthropic.AuthenticationError), tracing.run("anthropic-fail"):
        client.messages.create(**_KW)

    trace = tracing._get_last_trace()
    assert trace is not None
    llm = _llm(trace)
    assert len(llm) == 1, [(s["type"], s["name"]) for s in trace["spans"]]
    assert llm[0]["status"] == "failed"
    assert llm[0]["model"] == _MODEL
    assert "invalid x-api-key" in (llm[0].get("error_message") or "")
    assert llm[0].get("duration_ms") is not None
    # A call that returned an error has no usage: never price it.
    assert not llm[0].get("cost_usd")


@pytest.mark.req("MHQ-1415")
def test_mhq1415_langchain_anthropic_with_init_and_handler_one_span_per_call(installed):
    pytest.importorskip("langchain_anthropic")
    respx = pytest.importorskip("respx")
    httpx = pytest.importorskip("httpx")
    from langchain_anthropic import ChatAnthropic

    from amund.adapters.langchain import MorseCallbackHandler

    with respx.mock(assert_all_called=False) as mock:
        mock.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=_message_body())
        )
        llm = ChatAnthropic(
            model=_MODEL,
            api_key="test",  # pragma: allowlist secret
            max_retries=0,
        )
        handler = MorseCallbackHandler(agent_name="lc-anthropic")
        with tracing.run("lc-anthropic"):
            llm.invoke("hi", config={"callbacks": [handler]})
            llm.invoke("again", config={"callbacks": [handler]})

    trace = tracing._get_last_trace()
    assert trace is not None
    spans = _llm(trace)
    # Before the fix: 2 calls -> 4 llm spans (each with an extra empty
    # anthropic.messages.create sibling).
    assert len(spans) == 2, [(s["name"], s.get("input_tokens")) for s in spans]
    for s in spans:
        assert s["input_tokens"] and s["output_tokens"]
        assert s["model"] == _MODEL


@pytest.mark.req("MHQ-1414")
def test_mhq1414_langchain_cache_write_tokens_reach_the_span():
    pytest.importorskip("langchain_core")
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from amund.adapters.langchain import MorseCallbackHandler

    class _CacheWriteChat(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "fake-anthropic"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
            # langchain-anthropic's usage_metadata for a cache WRITE call:
            # input_tokens includes the cache tokens.
            msg = AIMessage(
                content="ok",
                usage_metadata={
                    "input_tokens": 7714,
                    "output_tokens": 20,
                    "total_tokens": 7734,
                    "input_token_details": {"cache_creation": 7705, "cache_read": 0},
                },
                response_metadata={"model_name": _MODEL},
            )
            return ChatResult(generations=[ChatGeneration(message=msg)])

    token = tracing._active_trace.set(None)
    try:
        with tracing.run("cache-write"):
            _CacheWriteChat().invoke(
                "x",
                config={"callbacks": [MorseCallbackHandler(agent_name="cache-write")]},
            )
    finally:
        tracing._active_trace.reset(token)
    trace = tracing._get_last_trace()
    assert trace is not None
    (span,) = _llm(trace)
    assert span["input_tokens"] == 7714
    assert span["cache_write_tokens"] == 7705
    assert not span.get("cached_input_tokens")


@pytest.mark.req("MHQ-1236")
def test_mhq1236_no_cache_activity_is_zero_not_null(anthropic_response):
    """Usage reported with no cache hit is 0; NULL means "not reported"."""
    telemetry = av_anthropic.AnthropicAdapter().extract(anthropic_response)
    assert telemetry["cached_input_tokens"] == 0
    assert telemetry["cache_write_tokens"] == 0
