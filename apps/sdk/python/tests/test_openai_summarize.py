"""Tests for the OpenAI adapter's summarize hook (OI-MEM-01).

When the customer wraps an OpenAI call with ``av_openai.summarize()``, the
adapter emits the LLM span as normal AND a sibling ``memory`` span with:

  - type = "memory"
  - parent_span_id = the LLM span's span_id
  - name = "openai.summarize"
  - metadata.memory_op_type = "consolidate"
  - metadata.summary_token_count = response.usage.completion_tokens
  - metadata.summary_content_hash = SHA-256 of response.choices[0].message.content

Opt-in only — without the context manager, only the LLM span emits.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

import pytest

from morse_ai.adapters import openai as av_openai
from morse_ai.adapters.openai import (
    _record_llm_span_for_response,
    _summarize_mode,
)
from morse_ai.spans import SpanType
from morse_ai.tracing import run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_response(
    *,
    summary_text: str = "This is a summary of the conversation so far.",
    completion_tokens: int = 55,
    prompt_tokens: int = 800,
):
    """Mimic an openai.types.chat.ChatCompletion response."""
    return SimpleNamespace(
        id="chatcmpl-test",
        model="gpt-4o-mini",
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=summary_text, role="assistant"),
                finish_reason="stop",
                index=0,
            )
        ],
    )


def _basic_kwargs() -> dict[str, Any]:
    return {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Summarize the conversation below."},
            {"role": "user", "content": "Long transcript turn 1..."},
            {"role": "assistant", "content": "Long transcript turn 2..."},
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


def _spans_of(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return payload["traces"][0]["spans"]


# ---------------------------------------------------------------------------
# OI-MEM-01 — context manager scenarios
# ---------------------------------------------------------------------------


@pytest.mark.req("OI-MEM-01", "OI-SDK-02")
def test_summarize_emits_sibling_memory_span(captured_traces):
    """summarize() context manager → LLM span + sibling memory span."""
    with run("openai-summarize-agent"), av_openai.summarize():
        _record_llm_span_for_response(
            "openai.chat.completions.create",
            _fake_response(),
            start_mono=0.0,
            kwargs=_basic_kwargs(),
        )

    assert len(captured_traces) == 1
    spans = _spans_of(captured_traces[0])
    llm_spans = [s for s in spans if s["type"] == SpanType.LLM.value]
    memory_spans = [s for s in spans if s["type"] == SpanType.MEMORY.value]
    assert len(llm_spans) == 1, f"expected 1 llm span; got {spans}"
    assert len(memory_spans) == 1, f"expected 1 memory span; got {spans}"
    assert memory_spans[0]["name"] == "openai.summarize"


@pytest.mark.req("OI-MEM-01")
def test_summarize_links_via_parent_span_id(captured_traces):
    """memory span's parent_span_id must equal the LLM span's span_id."""
    with run("openai-link-test"), av_openai.summarize():
        _record_llm_span_for_response(
            "openai.chat.completions.create",
            _fake_response(),
            start_mono=0.0,
            kwargs=_basic_kwargs(),
        )

    spans = _spans_of(captured_traces[0])
    llm = next(s for s in spans if s["type"] == SpanType.LLM.value)
    memory = next(s for s in spans if s["type"] == SpanType.MEMORY.value)
    assert memory["parent_span_id"] == llm["span_id"]


@pytest.mark.req("OI-MEM-01")
def test_summarize_sets_consolidate_metadata(captured_traces):
    """Verify all required memory metadata fields."""
    summary_text = "This is a summary of the conversation so far."
    with run("openai-metadata-test"), av_openai.summarize():
        _record_llm_span_for_response(
            "openai.chat.completions.create",
            _fake_response(summary_text=summary_text, completion_tokens=77),
            start_mono=0.0,
            kwargs=_basic_kwargs(),
        )

    spans = _spans_of(captured_traces[0])
    memory = next(s for s in spans if s["type"] == SpanType.MEMORY.value)
    meta = memory["metadata"]
    assert meta["memory_op_type"] == "consolidate"
    assert meta["memory_scope"] == "agent"
    assert meta["summary_token_count"] == 77
    expected_hash = hashlib.sha256(summary_text.encode()).hexdigest()
    assert meta["summary_content_hash"] == expected_hash


@pytest.mark.req("OI-MEM-01")
def test_no_summarize_preserves_llm_only(captured_traces):
    """Without the context manager, only the LLM span emits."""
    with run("openai-no-summarize"):
        _record_llm_span_for_response(
            "openai.chat.completions.create",
            _fake_response(),
            start_mono=0.0,
            kwargs=_basic_kwargs(),
        )

    spans = _spans_of(captured_traces[0])
    llm_spans = [s for s in spans if s["type"] == SpanType.LLM.value]
    memory_spans = [s for s in spans if s["type"] == SpanType.MEMORY.value]
    assert len(llm_spans) == 1
    assert memory_spans == []


@pytest.mark.req("OI-MEM-01")
def test_summarize_default_is_false():
    """ContextVar must default to False outside any summarize() block."""
    assert _summarize_mode.get() is False

    with av_openai.summarize():
        assert _summarize_mode.get() is True

    assert _summarize_mode.get() is False


def test_summarize_no_op_outside_active_trace(captured_traces):
    """summarize() outside an active trace must not crash and emit nothing."""
    with av_openai.summarize():
        _record_llm_span_for_response(
            "openai.chat.completions.create",
            _fake_response(),
            start_mono=0.0,
            kwargs=_basic_kwargs(),
        )

    assert captured_traces == []


def test_summarize_handles_no_choices_gracefully(captured_traces):
    """A response with no choices should hash the empty string — no crash."""
    response = SimpleNamespace(
        id="chatcmpl-empty",
        model="gpt-4o-mini",
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=0,
            total_tokens=10,
        ),
        choices=[],
    )

    with run("openai-empty-choices"), av_openai.summarize():
        _record_llm_span_for_response(
            "openai.chat.completions.create",
            response,
            start_mono=0.0,
            kwargs=_basic_kwargs(),
        )

    spans = _spans_of(captured_traces[0])
    memory = next(s for s in spans if s["type"] == SpanType.MEMORY.value)
    assert memory["metadata"]["summary_content_hash"] == hashlib.sha256(b"").hexdigest()
    assert memory["metadata"]["summary_token_count"] == 0


def test_summarize_tokens_evicted_matches_input_tokens(captured_traces):
    """memory_tokens_evicted must equal the LLM span's input_tokens."""
    with run("openai-evicted-test"), av_openai.summarize():
        _record_llm_span_for_response(
            "openai.chat.completions.create",
            _fake_response(prompt_tokens=1200, completion_tokens=40),
            start_mono=0.0,
            kwargs=_basic_kwargs(),
        )

    spans = _spans_of(captured_traces[0])
    llm = next(s for s in spans if s["type"] == SpanType.LLM.value)
    memory = next(s for s in spans if s["type"] == SpanType.MEMORY.value)
    assert memory["metadata"]["memory_tokens_evicted"] == llm["input_tokens"]
    assert memory["metadata"]["memory_tokens_injected"] == 40
