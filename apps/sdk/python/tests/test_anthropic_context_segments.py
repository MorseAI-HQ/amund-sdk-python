"""Tests for the Anthropic adapter's context_segments pre-segmentation (CTX-08).

The Anthropic adapter pre-segments the *request* kwargs (system, tools, messages)
into the wire-format `context_segments` list — so the server gets high-fidelity
tagging that doesn't require role-walking, and we can pick up Anthropic-specific
prompt-cache hints (`cache_control` blocks).

Wire shape (matches `ContextSegmentIn` schema in the backend):
  - position: int
  - source: str (system_prompt | tool_descriptions | chat_history |
                 current_input | tool_observation | ...)
  - content_hash: str (sha256 of FULL content, pre-truncation)
  - content: str | None (truncated to 4096 bytes when oversized)
  - content_truncated: bool
  - tokens: None  (server tokenizes — CTX-04)
  - tokens_estimated: bool
  - cache_state: str | None (cold | cache_write | cache_hit)
  - attrs: dict | None
"""

# ruff: noqa: E501, SIM117
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from amund.adapters.anthropic import (
    AnthropicAdapter,
    _extract_anthropic_segments,
)


def _fake_response(*, cache_read: int = 0, cache_creation: int = 0):
    """Mimic an anthropic.types.Message just enough for the extractor."""
    return SimpleNamespace(
        id="msg_test",
        model="claude-sonnet-4-6-20260217",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        ),
    )


def _by_source(segments, source):
    return [s for s in segments if s["source"] == source]


def test_segments_emitted_for_basic_anthropic_call():
    """system_prompt at position 0, current_input at position 1; both hashed; tokens=None."""
    kwargs = {
        "model": "claude-sonnet-4-6-20260217",
        "system": "You are helpful",
        "messages": [{"role": "user", "content": "hi"}],
    }
    segments = AnthropicAdapter().extract_segments(kwargs, _fake_response())

    assert len(segments) == 2

    sys_seg = segments[0]
    assert sys_seg["position"] == 0
    assert sys_seg["source"] == "system_prompt"
    assert sys_seg["content"] == "You are helpful"
    assert sys_seg["content_truncated"] is False
    assert sys_seg["content_hash"] == hashlib.sha256(b"You are helpful").hexdigest()
    assert sys_seg["tokens"] is None  # server tokenizes — CTX-04
    assert sys_seg["tokens_estimated"] is False

    user_seg = segments[1]
    assert user_seg["position"] == 1
    assert user_seg["source"] == "current_input"
    assert user_seg["content"] == "hi"
    assert user_seg["content_hash"] == hashlib.sha256(b"hi").hexdigest()
    assert user_seg["tokens"] is None


def test_segments_with_tools_have_tool_descriptions_first():
    """When tools are passed, tool_descriptions is the first segment with deterministic JSON."""
    tools = [
        {
            "name": "web_search",
            "description": "search the web",
            "input_schema": {"type": "object"},
        },
        {
            "name": "calculator",
            "description": "do math",
            "input_schema": {"type": "object"},
        },
    ]
    kwargs = {
        "system": "sys",
        "messages": [{"role": "user", "content": "go"}],
        "tools": tools,
    }
    segments = AnthropicAdapter().extract_segments(kwargs, _fake_response())

    first = segments[0]
    assert first["position"] == 0
    assert first["source"] == "tool_descriptions"
    # Deterministic — sort_keys=True so the same tools list always hashes the same.
    expected_json = json.dumps(tools, sort_keys=True)
    assert first["content"] == expected_json
    assert first["content_hash"] == hashlib.sha256(expected_json.encode()).hexdigest()
    assert first.get("attrs", {}).get("tool_count") == 2

    # And later segments preserve canonical ordering.
    sources = [s["source"] for s in segments]
    assert sources == ["tool_descriptions", "system_prompt", "current_input"]


def test_messages_with_tool_result_become_tool_observation():
    """A tool_result block lifts out into a `tool_observation` segment."""
    kwargs = {
        "system": "sys",
        "messages": [
            {"role": "user", "content": "what's the weather?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "get_weather",
                        "input": {"city": "SF"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "result body 72F sunny",
                    }
                ],
            },
            {"role": "user", "content": "thanks, what about LA?"},
        ],
    }
    segments = AnthropicAdapter().extract_segments(kwargs, _fake_response())

    obs = _by_source(segments, "tool_observation")
    assert len(obs) == 1
    assert "result body 72F sunny" in obs[0]["content"]

    # The pure tool_result message should NOT also be classified as
    # current_input — that designation goes to the user's actual question
    # ("thanks, what about LA?").
    cur = _by_source(segments, "current_input")
    assert len(cur) == 1
    assert cur[0]["content"] == "thanks, what about LA?"


def test_last_user_message_classified_as_current_input():
    """Multi-turn convo: only the LAST user message is current_input."""
    kwargs = {
        "system": "sys",
        "messages": [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "follow up"},
            {"role": "assistant", "content": "second answer"},
            {"role": "user", "content": "third and final"},
        ],
    }
    segments = AnthropicAdapter().extract_segments(kwargs, _fake_response())

    history = _by_source(segments, "chat_history")
    current = _by_source(segments, "current_input")

    history_texts = [s["content"] for s in history]
    assert "first question" in history_texts
    assert "first answer" in history_texts
    assert "follow up" in history_texts
    assert "second answer" in history_texts

    assert len(current) == 1
    assert current[0]["content"] == "third and final"


def test_anthropic_block_form_system_field_concatenated():
    """system=[{type:text, text:'a'}, {type:text, text:'b'}] -> one merged segment."""
    kwargs = {
        "system": [
            {"type": "text", "text": "part 1"},
            {"type": "text", "text": "part 2"},
        ],
        "messages": [{"role": "user", "content": "hi"}],
    }
    segments = AnthropicAdapter().extract_segments(kwargs, _fake_response())

    sys_segs = _by_source(segments, "system_prompt")
    assert len(sys_segs) == 1
    assert sys_segs[0]["content"] == "part 1part 2"


def test_cache_control_marks_cache_write_state():
    """A system block with cache_control={'type':'ephemeral'} -> cache_state='cache_write'."""
    kwargs = {
        "system": [
            {
                "type": "text",
                "text": "stable preamble that should be cached",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": "hi"}],
    }
    # cache_creation_input_tokens > 0 is the response-side confirmation, but
    # the request-side cache_control marker is what we key off of.
    segments = AnthropicAdapter().extract_segments(
        kwargs, _fake_response(cache_creation=500)
    )

    sys_seg = _by_source(segments, "system_prompt")[0]
    assert sys_seg["cache_state"] == "cache_write"


def test_cache_read_input_tokens_marks_cache_hit_on_prefix():
    """When response.usage.cache_read_input_tokens > 0, the cached prefix
    (system_prompt + tool_descriptions) gets cache_state='cache_hit'."""
    tools = [{"name": "x", "description": "y", "input_schema": {}}]
    kwargs = {
        "tools": tools,
        "system": "sys",
        "messages": [{"role": "user", "content": "go"}],
    }
    segments = AnthropicAdapter().extract_segments(
        kwargs, _fake_response(cache_read=500)
    )

    tool_seg = _by_source(segments, "tool_descriptions")[0]
    sys_seg = _by_source(segments, "system_prompt")[0]
    cur_seg = _by_source(segments, "current_input")[0]

    assert tool_seg["cache_state"] == "cache_hit"
    assert sys_seg["cache_state"] == "cache_hit"
    # Non-prefix segments are NOT auto-tagged — they default to cold.
    assert cur_seg.get("cache_state") in (None,)


def test_content_truncated_at_4096_bytes():
    """Large message body -> content_truncated=True, content body <= 4096 bytes."""
    big = "x" * 10_000
    kwargs = {
        "system": "sys",
        "messages": [{"role": "user", "content": big}],
    }
    segments = AnthropicAdapter().extract_segments(kwargs, _fake_response())

    cur = _by_source(segments, "current_input")[0]
    assert cur["content_truncated"] is True
    assert len(cur["content"].encode("utf-8")) <= 4096
    # The hash should still cover the FULL content, not the truncated prefix.
    assert cur["content_hash"] == hashlib.sha256(big.encode()).hexdigest()


def test_extract_segments_swallows_errors():
    """Pathological input must not raise — keep adapter non-invasive."""

    class Boom:
        def __getattr__(self, name):
            raise RuntimeError("nope")

    # kwargs not a dict at all -> empty list, no raise.
    assert AnthropicAdapter().extract_segments("not-a-dict", _fake_response()) == []  # type: ignore[arg-type]
    # And a response object whose usage.cache_read_input_tokens raises is tolerated.
    kwargs = {"system": "sys", "messages": [{"role": "user", "content": "hi"}]}
    response = SimpleNamespace(usage=Boom())
    out = AnthropicAdapter().extract_segments(kwargs, response)
    # Should still return [] (or some segments) without raising.
    assert isinstance(out, list)


def test_helper_function_is_addressable_for_langchain_adapter():
    """`AnthropicAdapter.extract_segments` mirrors the passive `extract()` shape
    so the LangChain callback handler can call it directly without monkey-patching.
    """
    adapter = AnthropicAdapter()
    assert callable(adapter.extract_segments)
    out = adapter.extract_segments(
        {"system": "s", "messages": [{"role": "user", "content": "u"}]},
        _fake_response(),
    )
    # And the underscore helper is also exposed for unit-level callers.
    out2 = _extract_anthropic_segments(
        {"system": "s", "messages": [{"role": "user", "content": "u"}]},
        _fake_response(),
    )
    assert out == out2


def test_context_composition_present_in_llm_span_metadata(monkeypatch):
    """LLM span metadata contains context_composition dict when segments + tokens exist."""
    from types import SimpleNamespace

    from amund import tracing as _tracing
    from amund.adapters.anthropic import (
        _record_llm_span_for_response,
        install,
        uninstall,
    )
    from amund.tracing import run

    install()

    # MHQ-668: run()'s primary emission path is real OTel spans over OTLP
    # (tracing._finish_trace -> otel_provider.emit_trace_entry), not
    # BatchTransport.track(). Patch _finish_trace itself to capture the
    # emitted spans instead of mocking the retired transport.
    captured_spans = []
    original_finish_trace = _tracing._finish_trace

    def _capture(entry):
        _tracing._capture_last_trace(entry)
        captured_spans.extend(entry.get("spans") or [])

    _tracing._finish_trace = _capture

    try:
        mock_response = SimpleNamespace(
            id="msg_test",
            model="claude-3-5-sonnet-20241022",
            usage=SimpleNamespace(
                input_tokens=500,
                output_tokens=80,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            content=[],
            stop_reason="end_turn",
        )

        with run("test-agent"):
            _record_llm_span_for_response(
                "anthropic.messages.create",
                mock_response,
                start_mono=0.0,
                kwargs={
                    "model": "claude-3-5-sonnet-20241022",
                    "system": "You are a helpful assistant.",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1024,
                },
            )

        llm_spans = [s for s in captured_spans if s.get("type") == "llm"]
        assert llm_spans, f"Expected at least one LLM span, got: {captured_spans}"
        meta = llm_spans[0].get("metadata") or {}
        comp = meta.get("context_composition")
        assert isinstance(comp, dict), f"Expected context_composition dict, got: {meta}"
        assert "system" in comp
        assert "history" in comp
        # Total should equal input_tokens
        assert sum(comp.values()) == 500
    finally:
        _tracing._finish_trace = original_finish_trace
        uninstall()
