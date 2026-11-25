"""Tests for the Anthropic adapter's summarize hook (MEM-11).

When the customer wraps a Claude call with ``av_anthropic.summarize()``, the
adapter emits the LLM span as normal (CTX-08 behavior) AND a sibling
``memory`` span with:

  - type = "memory"
  - parent_span_id = the LLM span's span_id (server stores as producer_span_id)
  - metadata.memory_op_type = "consolidate"
  - metadata.summary_token_count = response.usage.output_tokens
  - metadata.summary_content_hash = SHA-256 of the response's first text block

The flag is opt-in only — when it's absent, only the LLM span emits (CTX-08
existing behavior). Mirrors the requirement scenarios in
``openspec/changes/memory-profiler/specs/sdk/spec.md`` (MEM-11).
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

import pytest

from amund.adapters import anthropic as av_anthropic
from amund.adapters.anthropic import (
    _record_llm_span_for_response,
    _summarize_mode,
)
from amund.spans import SpanType
from amund.tracing import run

# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _fake_response(
    *,
    summary_text: str = "this is a summary of the conversation",
    output_tokens: int = 42,
    input_tokens: int = 1000,
):
    """Mimic an anthropic.types.Message with a single text content block."""
    return SimpleNamespace(
        id="msg_summary",
        model="claude-haiku-4-5",
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        content=[SimpleNamespace(type="text", text=summary_text)],
    )


def _basic_kwargs() -> dict[str, Any]:
    return {
        "model": "claude-haiku-4-5",
        "system": "Summarize this conversation in 200 words.",
        "messages": [
            {"role": "user", "content": "Long transcript turn 1..."},
            {"role": "assistant", "content": "Long transcript turn 2..."},
            {"role": "user", "content": "Long transcript turn 3..."},
        ],
    }


@pytest.fixture
def captured_traces():
    """Capture trace entries as they're emitted.

    MHQ-668: run()/trace_agent's primary emission path is real OTel spans
    over OTLP (``tracing._finish_trace`` -> ``otel_provider.emit_trace_entry``),
    not ``BatchTransport.track()``. Patch ``_finish_trace`` itself instead of
    the retired transport mock, and reproduce the historical
    ``{"traces": [entry]}`` capture shape so ``_spans_of()`` below and the
    rest of this file's assertions need no changes.
    """
    from amund import tracing as _tracing

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


def _spans_of(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return payload["traces"][0]["spans"]


# ---------------------------------------------------------------------------
# MEM-11 — explicit scenarios
# ---------------------------------------------------------------------------


@pytest.mark.req("MEM-11")
def test_summarize_true_emits_sibling_memory_span(captured_traces):
    """summarize=True → one llm span + one sibling memory span (consolidate)."""
    with run("agent-using-summarize"), av_anthropic.summarize():
        _record_llm_span_for_response(
            "anthropic.messages.create",
            _fake_response(),
            start_mono=0.0,
            kwargs=_basic_kwargs(),
        )

    assert len(captured_traces) == 1
    spans = _spans_of(captured_traces[0])
    # 1 root agent span + 1 llm span + 1 memory span
    llm_spans = [s for s in spans if s["type"] == SpanType.LLM.value]
    memory_spans = [s for s in spans if s["type"] == SpanType.MEMORY.value]
    assert len(llm_spans) == 1, f"expected 1 llm span; got {spans}"
    assert len(memory_spans) == 1, f"expected 1 memory span; got {spans}"

    memory = memory_spans[0]
    assert memory["name"] == "anthropic.summarize"
    assert memory["metadata"]["memory_op_type"] == "consolidate"


@pytest.mark.req("MEM-11")
def test_summarize_true_links_via_parent_span_id(captured_traces):
    """The memory span's parent_span_id MUST equal the LLM span's span_id."""
    with run("agent-using-summarize"), av_anthropic.summarize():
        _record_llm_span_for_response(
            "anthropic.messages.create",
            _fake_response(),
            start_mono=0.0,
            kwargs=_basic_kwargs(),
        )

    spans = _spans_of(captured_traces[0])
    llm = next(s for s in spans if s["type"] == SpanType.LLM.value)
    memory = next(s for s in spans if s["type"] == SpanType.MEMORY.value)

    # MEM-11 — server-side this becomes producer_span_id via the selector.
    assert memory["parent_span_id"] == llm["span_id"]


@pytest.mark.req("MEM-11")
def test_summarize_true_sets_consolidate_op_type(captured_traces):
    """metadata.memory_op_type must be exactly "consolidate" (server enum)."""
    with run("agent-using-summarize"), av_anthropic.summarize():
        _record_llm_span_for_response(
            "anthropic.messages.create",
            _fake_response(output_tokens=200),
            start_mono=0.0,
            kwargs=_basic_kwargs(),
        )

    spans = _spans_of(captured_traces[0])
    memory = next(s for s in spans if s["type"] == SpanType.MEMORY.value)
    meta = memory["metadata"]
    assert meta["memory_op_type"] == "consolidate"
    # Token count flows from response.usage.output_tokens
    assert meta["summary_token_count"] == 200
    # And the content hash is non-empty hex of the first text block
    expected_hash = hashlib.sha256(b"this is a summary of the conversation").hexdigest()
    assert meta["summary_content_hash"] == expected_hash


@pytest.mark.req("MEM-11")
def test_summarize_false_preserves_ctx08_behavior(captured_traces):
    """Without the flag, no memory span emits — only the existing CTX-08 LLM span."""
    with run("agent-no-summarize"):
        # NOTE: no `with summarize()` wrapper.
        _record_llm_span_for_response(
            "anthropic.messages.create",
            _fake_response(),
            start_mono=0.0,
            kwargs=_basic_kwargs(),
        )

    spans = _spans_of(captured_traces[0])
    llm_spans = [s for s in spans if s["type"] == SpanType.LLM.value]
    memory_spans = [s for s in spans if s["type"] == SpanType.MEMORY.value]
    assert len(llm_spans) == 1
    assert memory_spans == []


@pytest.mark.req("MEM-11")
def test_summarize_default_is_false():
    """The summarize-mode ContextVar must default to False outside any context."""
    # Outside `with summarize()`, the ContextVar reports False.
    assert _summarize_mode.get() is False

    with av_anthropic.summarize():
        assert _summarize_mode.get() is True

    # And restored on exit.
    assert _summarize_mode.get() is False


# ---------------------------------------------------------------------------
# Additional safety checks (non-MEM-11 but adjacent)
# ---------------------------------------------------------------------------


def test_summarize_no_op_outside_active_trace(captured_traces):
    """summarize() inside no `amund.run` must NOT crash + must emit nothing."""
    with av_anthropic.summarize():
        # No active trace state — recorder silently drops.
        _record_llm_span_for_response(
            "anthropic.messages.create",
            _fake_response(),
            start_mono=0.0,
            kwargs=_basic_kwargs(),
        )

    # No trace was opened, so transport saw nothing.
    assert captured_traces == []


def test_summarize_handles_empty_content_gracefully(captured_traces):
    """A response with no text content yields a hash of the empty string — no crash."""
    response = SimpleNamespace(
        id="msg_empty",
        model="claude-haiku-4-5",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        content=[],
    )

    with run("agent-empty-content"), av_anthropic.summarize():
        _record_llm_span_for_response(
            "anthropic.messages.create",
            response,
            start_mono=0.0,
            kwargs=_basic_kwargs(),
        )

    spans = _spans_of(captured_traces[0])
    memory = next(s for s in spans if s["type"] == SpanType.MEMORY.value)
    # SHA-256 of "" is the well-known empty-string digest.
    assert memory["metadata"]["summary_content_hash"] == hashlib.sha256(b"").hexdigest()
    assert memory["metadata"]["summary_token_count"] == 0


# ---------------------------------------------------------------------------
# MEM-11 — per-client summarize=True via wrap()
# ---------------------------------------------------------------------------


class _FakeMessages:
    """Sync ``messages`` resource returning a canned response."""

    def __init__(self, response):
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeAnthropicClient:
    """Minimal anthropic.Anthropic look-alike with a ``messages`` attribute."""

    def __init__(self, response):
        self.messages = _FakeMessages(response)
        self.api_key = "sk-ant-test"


@pytest.mark.req("MEM-11")
def test_wrap_with_summarize_true_emits_memory_span(captured_traces):
    """wrap(client, summarize=True) emits the MEM-11 sibling memory span."""
    fake = _FakeAnthropicClient(_fake_response(output_tokens=64))
    client = av_anthropic.wrap(fake, summarize=True)

    # Driving a real wrapped call instead of poking the private recorder.
    client.messages.create(**_basic_kwargs())

    assert len(captured_traces) == 1
    spans = _spans_of(captured_traces[0])
    llm_spans = [s for s in spans if s["type"] == SpanType.LLM.value]
    memory_spans = [s for s in spans if s["type"] == SpanType.MEMORY.value]
    assert len(llm_spans) == 1
    assert len(memory_spans) == 1
    assert memory_spans[0]["metadata"]["memory_op_type"] == "consolidate"
    assert memory_spans[0]["metadata"]["summary_token_count"] == 64
    assert memory_spans[0]["parent_span_id"] == llm_spans[0]["span_id"]


@pytest.mark.req("MEM-11")
def test_wrap_default_summarize_false_emits_no_memory_span(captured_traces):
    """wrap(client) without summarize=True preserves CTX-08-only behavior."""
    fake = _FakeAnthropicClient(_fake_response())
    client = av_anthropic.wrap(fake)  # summarize defaults to False

    client.messages.create(**_basic_kwargs())

    spans = _spans_of(captured_traces[0])
    llm_spans = [s for s in spans if s["type"] == SpanType.LLM.value]
    memory_spans = [s for s in spans if s["type"] == SpanType.MEMORY.value]
    assert len(llm_spans) == 1
    assert memory_spans == []


@pytest.mark.req("MEM-11")
def test_summarize_context_overrides_wrap_false(captured_traces):
    """The `summarize()` context manager wins even on a wrap(summarize=False) client."""
    fake = _FakeAnthropicClient(_fake_response())
    client = av_anthropic.wrap(fake, summarize=False)

    with av_anthropic.summarize():
        client.messages.create(**_basic_kwargs())

    spans = _spans_of(captured_traces[0])
    memory_spans = [s for s in spans if s["type"] == SpanType.MEMORY.value]
    assert len(memory_spans) == 1


@pytest.mark.req("MEM-11")
def test_wrap_summarize_true_still_emits_with_context_off(captured_traces):
    """wrap(summarize=True) does not require the context manager to be active."""
    fake = _FakeAnthropicClient(_fake_response())
    client = av_anthropic.wrap(fake, summarize=True)

    # NB: no `with summarize():` here — the per-client default carries.
    client.messages.create(**_basic_kwargs())

    spans = _spans_of(captured_traces[0])
    memory_spans = [s for s in spans if s["type"] == SpanType.MEMORY.value]
    assert len(memory_spans) == 1
