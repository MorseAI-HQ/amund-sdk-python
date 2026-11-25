"""Tests for the OpenAI adapter's context_segment pre-segmentation.

Tests the ``wrap()`` and ``install()`` patterns and verify that
``chat.completions.create`` calls emit ``llm`` spans with context_segments.
These tests mock the OpenAI SDK — no real API calls are made.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import amund
from amund import tracing
from amund.adapters import openai as av_openai
from amund.adapters.openai import OpenAIAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_response(
    model: str = "gpt-4o-mini", prompt_tokens: int = 10, completion_tokens: int = 5
) -> Any:
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )


def _by_source(segs, source):
    return [s for s in segs if s["source"] == source]


@pytest.fixture(autouse=True)
def _reset():
    from amund.tracing import _active_trace

    _active_trace.set(None)
    amund._reset_for_testing()
    av_openai.uninstall()
    yield
    _active_trace.set(None)
    amund._reset_for_testing()
    av_openai.uninstall()


# ---------------------------------------------------------------------------
# OpenAIAdapter.extract_segments unit tests
# ---------------------------------------------------------------------------


@pytest.mark.req("CTX-11")
def test_openai_adapter_extract_segments_basic():
    """extract_segments returns context segments for chat completions kwargs."""
    adapter = OpenAIAdapter()
    kwargs = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hello"},
        ],
    }
    segs = adapter.extract_segments(kwargs, _fake_response())
    assert len(segs) >= 2
    sys_segs = _by_source(segs, "system_prompt")
    cur_segs = _by_source(segs, "current_input")
    assert len(sys_segs) == 1
    assert len(cur_segs) == 1


@pytest.mark.req("CTX-11")
def test_openai_adapter_extract_segments_with_tools():
    adapter = OpenAIAdapter()
    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "get weather"}],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    }
    segs = adapter.extract_segments(kwargs, _fake_response())
    td = _by_source(segs, "tool_descriptions")
    assert len(td) == 1
    assert td[0]["attrs"]["tool_count"] == 1


@pytest.mark.req("CTX-11")
def test_openai_adapter_extract_segments_empty_messages():
    adapter = OpenAIAdapter()
    segs = adapter.extract_segments({}, _fake_response())
    assert segs == []


# ---------------------------------------------------------------------------
# wrap() integration tests — span emission
# ---------------------------------------------------------------------------


class FakeCompletions:
    """Fake openai.chat.completions with a configurable create() side-effect."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.called_kwargs: dict[str, Any] = {}

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self.called_kwargs = kwargs
        return self._response


class FakeChat:
    def __init__(self, response: Any) -> None:
        self.completions = FakeCompletions(response)


class FakeOpenAIClient:
    def __init__(self, response: Any | None = None) -> None:
        self.chat = FakeChat(response or _fake_response())


@pytest.mark.req("CTX-11")
def test_wrap_emits_llm_span_with_context_segments():
    client = av_openai.wrap(FakeOpenAIClient())
    kwargs = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a coding assistant"},
            {"role": "user", "content": "write a hello world"},
        ],
    }
    with tracing.run("test-agent"):
        client.chat.completions.create(**kwargs)

    trace = tracing._get_last_trace()
    assert trace is not None
    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    assert len(llm_spans) == 1

    segs = llm_spans[0].get("context_segments") or []
    assert len(segs) >= 2
    sys_segs = _by_source(segs, "system_prompt")
    cur_segs = _by_source(segs, "current_input")
    assert len(sys_segs) == 1
    assert len(cur_segs) == 1
    assert sys_segs[0]["content"] == "You are a coding assistant"
    assert cur_segs[0]["content"] == "write a hello world"


@pytest.mark.req("CTX-11")
def test_wrap_emits_telemetry_tokens_and_model():
    response = _fake_response(model="gpt-4o", prompt_tokens=25, completion_tokens=10)
    client = av_openai.wrap(FakeOpenAIClient(response))
    with tracing.run("test-agent"):
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )

    trace = tracing._get_last_trace()
    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    assert len(llm_spans) == 1
    span = llm_spans[0]
    assert span["model"] == "gpt-4o"
    assert span["input_tokens"] == 25
    assert span["output_tokens"] == 10


@pytest.mark.req("CTX-11")
def test_wrap_started_at_is_before_call_not_after():
    """Regression test: SpanData.started_at defaults to "now" at construction
    time — which happens AFTER response = client.chat.completions.create(...)
    already returned. The server derives duration_ms from the
    started_at/ended_at wall-clock pair (not the SDK's own monotonic-based
    value), so every LLM span showed ~0ms regardless of true latency until
    started_at_iso was threaded through from the call site (captured before
    the API call). Same root cause as MHQ-771 (Anthropic adapter)."""
    import time
    from datetime import UTC, datetime

    class _SlowCompletions(FakeCompletions):
        def create(self, *args: Any, **kwargs: Any) -> Any:
            time.sleep(0.05)
            return super().create(*args, **kwargs)

    client = FakeOpenAIClient()
    client.chat.completions = _SlowCompletions(_fake_response())
    wrapped = av_openai.wrap(client)

    before = datetime.now(UTC)
    with tracing.run("test-agent"):
        wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
    after = datetime.now(UTC)

    trace = tracing._get_last_trace()
    llm_span = next(s for s in trace["spans"] if s["type"] == "llm")
    started_at = datetime.fromisoformat(llm_span["started_at"])
    ended_at = datetime.fromisoformat(llm_span["ended_at"])

    # started_at must be captured BEFORE the (slow) call, not after.
    assert before <= started_at <= after
    # The wall-clock gap must reflect the real ~50ms call latency, not ~0.
    assert (ended_at - started_at).total_seconds() >= 0.04


@pytest.mark.req("CTX-11")
def test_wrap_async_started_at_is_before_call_not_after():
    """Same regression, async path (_WrappedAsyncCompletions.create)."""
    import asyncio
    from datetime import UTC, datetime

    class _SlowAsyncCompletions:
        def __init__(self, response: Any) -> None:
            self._response = response

        async def create(self, *args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0.05)
            return self._response

    class _AsyncClient:
        def __init__(self, response: Any) -> None:
            self.chat = SimpleNamespace(completions=_SlowAsyncCompletions(response))

    fake = _AsyncClient(_fake_response())
    type(fake).__name__ = "AsyncOpenAI"
    wrapped = av_openai.wrap(fake)

    async def call() -> tuple[Any, Any]:
        with tracing.run("test-agent"):
            before = datetime.now(UTC)
            await wrapped.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )
            after = datetime.now(UTC)
        return before, after

    before, after = asyncio.run(call())

    trace = tracing._get_last_trace()
    llm_span = next(s for s in trace["spans"] if s["type"] == "llm")
    started_at = datetime.fromisoformat(llm_span["started_at"])
    ended_at = datetime.fromisoformat(llm_span["ended_at"])

    assert before <= started_at <= after
    assert (ended_at - started_at).total_seconds() >= 0.04


@pytest.mark.req("CTX-11")
def test_wrap_is_idempotent():
    client = FakeOpenAIClient()
    wrapped_once = av_openai.wrap(client)
    wrapped_twice = av_openai.wrap(wrapped_once)
    assert wrapped_once is wrapped_twice


@pytest.mark.req("CTX-11")
def test_wrap_outside_trace_creates_auto_trace():
    """wrap() auto-creates a trace when there's no active trace context."""
    client = av_openai.wrap(FakeOpenAIClient())
    tracing._last_trace = None

    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )

    trace = tracing._get_last_trace()
    assert trace is not None
    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    assert len(llm_spans) == 1


@pytest.mark.req("CTX-11")
def test_wrap_with_tool_results_emits_tool_observation():
    client = av_openai.wrap(FakeOpenAIClient())
    with tracing.run("test-agent"):
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "get weather"},
                {"role": "assistant", "content": None},
                {"role": "tool", "content": "sunny, 25°C", "tool_call_id": "call_1"},
                {"role": "user", "content": "thanks"},
            ],
        )

    trace = tracing._get_last_trace()
    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    segs = llm_spans[0].get("context_segments") or []
    obs = _by_source(segs, "tool_observation")
    assert len(obs) == 1
    assert "sunny, 25°C" in obs[0]["content"]


# ---------------------------------------------------------------------------
# LangGraph on_chat_model_start decomposition test
# ---------------------------------------------------------------------------


@pytest.mark.req("CTX-11")
def test_langgraph_on_chat_model_start_decomposes_messages():
    """LangGraph handler decomposes LangChain messages into context_segments."""
    import uuid

    from amund.adapters.langgraph import LangGraphCallbackHandler

    class FakeHuman:
        type = "human"
        content = "tell me a joke"

    class FakeSystem:
        type = "system"
        content = "You are funny"

    class MockLLMResult:
        generations = [[type("Gen", (), {"text": "why did the chicken..."})()]]
        llm_output = {}

    handler = LangGraphCallbackHandler("test-agent")
    run_id = uuid.uuid4()

    with tracing.run("test-agent"):
        handler.on_chat_model_start(
            {"id": ["langchain", "chat_models", "ChatOpenAI"], "name": "ChatOpenAI"},
            [[FakeSystem(), FakeHuman()]],
            run_id=run_id,
        )
        handler.on_llm_end(MockLLMResult(), run_id=run_id)

    trace = tracing._get_last_trace()
    assert trace is not None
    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    assert len(llm_spans) == 1

    segs = llm_spans[0].get("context_segments") or []
    sys_segs = _by_source(segs, "system_prompt")
    cur_segs = _by_source(segs, "current_input")
    assert len(sys_segs) == 1, (
        f"expected system_prompt segment, got: {[s['source'] for s in segs]}"
    )
    assert len(cur_segs) == 1
    assert sys_segs[0]["content"] == "You are funny"
    assert cur_segs[0]["content"] == "tell me a joke"


# ---------------------------------------------------------------------------
# MHQ-774: input_data / output_data capture (previously always null on this
# adapter — context_segments were captured, but the raw request/response
# content the trace-detail INPUT/OUTPUT panels read was not). Mirrors the
# MHQ-772 fix already shipped for the raw Anthropic adapter.
# ---------------------------------------------------------------------------


def _fake_completion_response(
    content: str | None = "hello back",
    tool_calls: list[Any] | None = None,
    model: str = "gpt-4o-mini",
) -> Any:
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
    )


def test_wrap_captures_input_and_output_data():
    client = av_openai.wrap(FakeOpenAIClient(_fake_completion_response()))

    with tracing.run("test-agent"):
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
        )

    trace = tracing._get_last_trace()
    llm_span = next(s for s in trace["spans"] if s["type"] == "llm")

    assert llm_span["input_data"] is not None
    assert llm_span["input_data"]["messages"] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    assert llm_span["output_data"] is not None
    assert llm_span["output_data"]["content"] == "hello back"


def test_wrap_captures_tool_call_output():
    """A response that only calls a tool (no text content) still gets a
    meaningful output_data value instead of showing nothing."""
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="get_weather", arguments='{"city": "Paris"}')
    )
    response = _fake_completion_response(content=None, tool_calls=[tool_call])
    client = av_openai.wrap(FakeOpenAIClient(response))

    with tracing.run("test-agent"):
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "weather in Paris?"}],
        )

    trace = tracing._get_last_trace()
    llm_span = next(s for s in trace["spans"] if s["type"] == "llm")
    assert "get_weather" in llm_span["output_data"]["content"]
    assert "Paris" in llm_span["output_data"]["content"]


def test_wrap_input_data_truncates_long_message_content():
    client = av_openai.wrap(FakeOpenAIClient(_fake_completion_response()))

    long_content = "x" * 10_000  # well over _CONTENT_BYTE_CAP (4096)
    with tracing.run("test-agent"):
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": long_content}],
        )

    trace = tracing._get_last_trace()
    llm_span = next(s for s in trace["spans"] if s["type"] == "llm")
    captured_content = llm_span["input_data"]["messages"][0]["content"]
    assert len(captured_content) < len(long_content)
    assert len(captured_content.encode("utf-8")) <= av_openai._CONTENT_BYTE_CAP


def test_wrap_no_messages_gives_null_input_data():
    """create() with no `messages` kwarg (shouldn't happen in practice, but
    the adapter must not raise) yields input_data=None rather than a crash
    or an empty-but-misleading dict."""
    client = av_openai.wrap(FakeOpenAIClient(_fake_completion_response()))

    with tracing.run("test-agent"):
        client.chat.completions.create(model="gpt-4o-mini")

    trace = tracing._get_last_trace()
    llm_span = next(s for s in trace["spans"] if s["type"] == "llm")
    assert llm_span.get("input_data") is None


def test_wrap_no_content_or_tool_calls_gives_null_output_data():
    """A response with neither text content nor tool_calls (edge case, e.g.
    a refusal-only message) yields output_data=None rather than an empty dict."""
    response = _fake_completion_response(content=None, tool_calls=None)
    client = av_openai.wrap(FakeOpenAIClient(response))

    with tracing.run("test-agent"):
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )

    trace = tracing._get_last_trace()
    llm_span = next(s for s in trace["spans"] if s["type"] == "llm")
    assert llm_span.get("output_data") is None
