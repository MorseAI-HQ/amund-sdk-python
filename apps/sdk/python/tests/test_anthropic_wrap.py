"""Tests for the Anthropic adapter's ``wrap()`` public API.

``wrap(client)`` is the RECOMMENDED public entry point — mirroring the
TypeScript SDK's ``wrapAnthropic(client, options)``. It returns a
delegating wrapper that intercepts ``messages.create`` to emit Morse
spans while leaving every other attribute of the client untouched.

These tests use a hand-rolled fake Anthropic client to avoid pulling in
the upstream ``anthropic`` package at unit-test time.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from morse_ai.adapters import anthropic as av_anthropic
from morse_ai.spans import SpanType
from morse_ai.tracing import run

# ---------------------------------------------------------------------------
# Fakes — minimal Anthropic-shaped doubles
# ---------------------------------------------------------------------------


def _fake_response(*, model: str = "claude-haiku-4-5"):
    return SimpleNamespace(
        id="msg_test",
        model=model,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        content=[SimpleNamespace(type="text", text="hello back")],
    )


class _SyncMessages:
    """Sync ``client.messages`` look-alike with a recorded call log."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []
        # Extra attribute so tests can verify __getattr__ delegation.
        self.batches = SimpleNamespace(create=lambda **_: "batch-created")

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


class _SyncClient:
    """Anthropic.Anthropic look-alike."""

    def __init__(self, response: Any) -> None:
        self.messages = _SyncMessages(response)
        self.api_key = "sk-ant-test"
        self.base_url = "https://api.anthropic.com"


class _AsyncMessages:
    """``client.messages`` for the async client. ``create`` is a coroutine."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


class _AsyncAnthropicClient:
    """AsyncAnthropic look-alike — name starts with 'Async' for wrapper detection."""

    def __init__(self, response: Any) -> None:
        self.messages = _AsyncMessages(response)
        self.api_key = "sk-ant-async-test"


# Force the class-name-prefix detection path to work without a name change.
_AsyncAnthropicClient.__name__ = "AsyncAnthropic"


@pytest.fixture
def captured_traces():
    """Capture trace entries as they're emitted.

    MHQ-668: run()/trace_agent's primary emission path is real OTel spans
    over OTLP (``tracing._finish_trace`` -> ``otel_provider.emit_trace_entry``),
    not ``BatchTransport.track()``. Patch ``_finish_trace`` itself instead of
    the retired transport mock, and reproduce the historical
    ``{"traces": [entry]}`` capture shape so the rest of this file's
    assertions (``captured_traces[0]["traces"][0][...]``) need no changes.
    """
    from morse_ai import tracing as _tracing

    captured: list[dict[str, Any]] = []
    original = _tracing._finish_trace

    def _capture(entry: dict[str, Any]) -> None:
        _tracing._capture_last_trace(entry)
        captured.append({"traces": [entry]})

    _tracing._finish_trace = _capture
    try:
        yield captured
    finally:
        _tracing._finish_trace = original


# ---------------------------------------------------------------------------
# Core wrap() behavior
# ---------------------------------------------------------------------------


def test_wrap_returns_drop_in_replacement_for_messages_create(captured_traces):
    """The wrapped client's messages.create call delegates AND emits an llm span."""
    fake = _SyncClient(_fake_response())
    client = av_anthropic.wrap(fake)

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
    )

    # Delegation: the inner client got the call exactly once with the same kwargs.
    assert len(fake.messages.calls) == 1
    assert fake.messages.calls[0]["model"] == "claude-haiku-4-5"

    # The wrapper returns whatever the inner returned, untouched.
    assert response.id == "msg_test"
    assert response.model == "claude-haiku-4-5"

    # And exactly one trace got emitted with one llm span.
    assert len(captured_traces) == 1
    spans = captured_traces[0]["traces"][0]["spans"]
    llm = [s for s in spans if s["type"] == SpanType.LLM.value]
    assert len(llm) == 1
    assert llm[0]["model"] == "claude-haiku-4-5"


def test_wrap_preserves_non_messages_attributes():
    """Attributes other than ``messages`` are delegated to the inner client."""
    fake = _SyncClient(_fake_response())
    client = av_anthropic.wrap(fake)

    # Field delegation:
    assert client.api_key == "sk-ant-test"
    assert client.base_url == "https://api.anthropic.com"


def test_wrap_preserves_non_create_methods_on_messages():
    """messages.batches and similar siblings delegate through to the inner messages."""
    fake = _SyncClient(_fake_response())
    client = av_anthropic.wrap(fake)

    # ``batches`` is on the inner _SyncMessages and must surface through the wrapper.
    assert client.messages.batches.create() == "batch-created"


def test_wrap_is_idempotent():
    """Wrapping an already-wrapped client returns the same instance."""
    fake = _SyncClient(_fake_response())
    once = av_anthropic.wrap(fake)
    twice = av_anthropic.wrap(once)
    assert once is twice


def test_wrap_none_client_is_noop():
    """wrap(None) returns None rather than crashing — defensive."""
    assert av_anthropic.wrap(None) is None  # type: ignore[func-returns-value]


def test_wrap_inherits_existing_active_trace(captured_traces):
    """Calls inside an existing ``run`` block reuse the surrounding trace."""
    fake = _SyncClient(_fake_response())
    client = av_anthropic.wrap(fake)

    with run("outer-agent"):
        client.messages.create(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "hi"}],
        )
        client.messages.create(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "again"}],
        )

    # One outer trace with both llm spans (no per-call auto-traces).
    assert len(captured_traces) == 1
    spans = captured_traces[0]["traces"][0]["spans"]
    llm_count = sum(1 for s in spans if s["type"] == SpanType.LLM.value)
    assert llm_count == 2


def test_wrap_auto_trace_when_no_active_run(captured_traces):
    """A call outside any ``run`` block auto-opens a trace so the span has a home."""
    fake = _SyncClient(_fake_response())
    client = av_anthropic.wrap(fake)

    client.messages.create(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert len(captured_traces) == 1
    spans = captured_traces[0]["traces"][0]["spans"]
    # One synthesized agent span + one llm span.
    assert any(s["type"] == SpanType.AGENT.value for s in spans)
    assert any(s["type"] == SpanType.LLM.value for s in spans)


def test_wrap_emits_context_segments(captured_traces):
    """Pre-segmentation runs through ``wrap()`` — same CTX-08 wire shape."""
    fake = _SyncClient(_fake_response())
    client = av_anthropic.wrap(fake)

    client.messages.create(
        model="claude-haiku-4-5",
        system="You are helpful",
        messages=[{"role": "user", "content": "hello"}],
    )

    spans = captured_traces[0]["traces"][0]["spans"]
    llm = next(s for s in spans if s["type"] == SpanType.LLM.value)
    segs = llm.get("context_segments")
    assert segs is not None
    sources = [s["source"] for s in segs]
    assert "system_prompt" in sources
    assert "current_input" in sources


# ---------------------------------------------------------------------------
# Async client wrap
# ---------------------------------------------------------------------------


def test_wrap_async_client_emits_llm_span(captured_traces):
    """AsyncAnthropic-shaped clients are wrapped via the async messages path."""
    fake = _AsyncAnthropicClient(_fake_response())
    client = av_anthropic.wrap(fake)

    async def call() -> Any:
        return await client.messages.create(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "hello"}],
        )

    # Match the rest of the SDK test suite — use asyncio.run rather than
    # pulling in pytest-asyncio.
    response = asyncio.run(call())

    assert len(fake.messages.calls) == 1
    assert response.id == "msg_test"

    assert len(captured_traces) == 1
    spans = captured_traces[0]["traces"][0]["spans"]
    llm = [s for s in spans if s["type"] == SpanType.LLM.value]
    assert len(llm) == 1


def test_wrap_async_client_detected_via_class_name(captured_traces):
    """Wrapper picks the async branch when the inner class name starts with 'Async'."""
    fake = _AsyncAnthropicClient(_fake_response())
    wrapped = av_anthropic.wrap(fake)
    # The wrapper's messages object must be the async flavour.
    assert wrapped.messages.__class__.__name__ == "_WrappedAsyncMessages"


# ---------------------------------------------------------------------------
# Duration bug: started_at must be captured before the API call, not after
# ---------------------------------------------------------------------------


def test_wrap_started_at_is_before_call_not_after(captured_traces):
    """Regression test: SpanData.started_at defaults to "now" at construction
    time — which happens AFTER response = client.messages.create(...) already
    returned. The server derives duration_ms from the started_at/ended_at
    wall-clock pair (not the SDK's own monotonic-based value), so every LLM
    span showed ~0ms regardless of true latency until started_at_iso was
    threaded through from the call site (captured before the API call)."""
    import time
    from datetime import UTC, datetime

    class _SlowMessages(_SyncMessages):
        def create(self, **kwargs: Any) -> Any:
            time.sleep(0.05)
            return super().create(**kwargs)

    fake = _SyncClient(_fake_response())
    fake.messages = _SlowMessages(_fake_response())
    client = av_anthropic.wrap(fake)

    before = datetime.now(UTC)
    client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
    )
    after = datetime.now(UTC)

    spans = captured_traces[0]["traces"][0]["spans"]
    llm_span = next(s for s in spans if s["type"] == SpanType.LLM.value)
    started_at = datetime.fromisoformat(llm_span["started_at"])
    ended_at = datetime.fromisoformat(llm_span["ended_at"])

    # started_at must be captured BEFORE the (slow) call, not after.
    assert before <= started_at <= after
    # The wall-clock gap must reflect the real ~50ms call latency, not ~0.
    assert (ended_at - started_at).total_seconds() >= 0.04


def test_wrap_async_started_at_is_before_call_not_after(captured_traces):
    """Same regression, async path (_WrappedAsyncMessages.create)."""
    from datetime import UTC, datetime

    class _SlowAsyncMessages(_AsyncMessages):
        async def create(self, **kwargs: Any) -> Any:
            await asyncio.sleep(0.05)
            return await super().create(**kwargs)

    fake = _AsyncAnthropicClient(_fake_response())
    fake.messages = _SlowAsyncMessages(_fake_response())
    client = av_anthropic.wrap(fake)

    async def call() -> tuple[Any, Any]:
        before = datetime.now(UTC)
        await client.messages.create(
            model="claude-haiku-4-5", messages=[{"role": "user", "content": "hi"}]
        )
        after = datetime.now(UTC)
        return before, after

    before, after = asyncio.run(call())

    spans = captured_traces[0]["traces"][0]["spans"]
    llm_span = next(s for s in spans if s["type"] == SpanType.LLM.value)
    started_at = datetime.fromisoformat(llm_span["started_at"])
    ended_at = datetime.fromisoformat(llm_span["ended_at"])

    assert before <= started_at <= after
    assert (ended_at - started_at).total_seconds() >= 0.04


# ---------------------------------------------------------------------------
# MHQ-772: input_data / output_data capture (previously always null)
# ---------------------------------------------------------------------------


def test_wrap_captures_input_and_output_data(captured_traces):
    fake = _SyncClient(_fake_response())
    client = av_anthropic.wrap(fake)

    client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    spans = captured_traces[0]["traces"][0]["spans"]
    llm_span = next(s for s in spans if s["type"] == SpanType.LLM.value)

    assert llm_span["input_data"] is not None
    assert llm_span["input_data"]["system"] == "You are a helpful assistant."
    assert llm_span["input_data"]["messages"] == [
        {"role": "user", "content": "What is the capital of France?"}
    ]

    assert llm_span["output_data"] is not None
    assert llm_span["output_data"]["content"] == "hello back"


def test_wrap_captures_tool_use_output(captured_traces):
    """A response that only calls a tool (no text block) still gets a
    meaningful output_data value instead of showing nothing."""
    response = SimpleNamespace(
        id="msg_tool",
        model="claude-haiku-4-5",
        usage=SimpleNamespace(
            input_tokens=50,
            output_tokens=10,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        content=[
            SimpleNamespace(
                type="tool_use", name="get_weather", input={"city": "Paris"}
            )
        ],
    )
    fake = _SyncClient(response)
    client = av_anthropic.wrap(fake)

    client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": "weather in Paris?"}],
    )

    spans = captured_traces[0]["traces"][0]["spans"]
    llm_span = next(s for s in spans if s["type"] == SpanType.LLM.value)
    assert "get_weather" in llm_span["output_data"]["content"]
    assert "Paris" in llm_span["output_data"]["content"]


def test_wrap_input_data_truncates_long_message_content(captured_traces):
    fake = _SyncClient(_fake_response())
    client = av_anthropic.wrap(fake)

    long_content = "x" * 10_000  # well over _CONTENT_BYTE_CAP (4096)
    client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": long_content}],
    )

    spans = captured_traces[0]["traces"][0]["spans"]
    llm_span = next(s for s in spans if s["type"] == SpanType.LLM.value)
    captured_content = llm_span["input_data"]["messages"][0]["content"]
    assert len(captured_content) < len(long_content)
    assert len(captured_content.encode("utf-8")) <= av_anthropic._CONTENT_BYTE_CAP


def test_wrap_no_messages_gives_null_input_data(captured_traces):
    """messages.create with no `messages` kwarg (shouldn't happen in practice,
    but the adapter must not raise) yields input_data=None rather than a
    crash or an empty-but-misleading dict."""
    fake = _SyncClient(_fake_response())
    client = av_anthropic.wrap(fake)

    client.messages.create(model="claude-haiku-4-5", max_tokens=10)

    spans = captured_traces[0]["traces"][0]["spans"]
    llm_span = next(s for s in spans if s["type"] == SpanType.LLM.value)
    assert llm_span.get("input_data") is None
