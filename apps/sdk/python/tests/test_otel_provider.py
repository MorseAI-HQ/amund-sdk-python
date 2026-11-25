"""Tests for the MHQ-668 OTel transport engine (``otel_provider.py``).

Covers the pieces that are new in the OTel rebuild and aren't exercised by
the (adapted) legacy test suite: sampler ratio, byte-cap truncation,
export-time redaction, blocking flush, and fork safety.
"""

from __future__ import annotations

import json
import os
import sys
import uuid

import pytest

from morse_ai._internal.redaction import RedactionConfig
from morse_ai.otel_provider import (
    TRUNCATION_MARKER,
    cap_bytes,
    emit_trace_entry,
    get_engine,
    json_capped,
)


def _spans_for(engine, trace_id_hex: str) -> list:
    """Recorded spans (in the test in-memory exporter) matching a trace id."""
    trace_id_int = int(trace_id_hex, 16)
    assert engine.recorder is not None
    return [s for s in engine.recorder.spans if s.context.trace_id == trace_id_int]


def _fresh_entry(trace_id_hex: str, *, input_data=None, metadata=None) -> dict:
    """Build a minimal well-formed trace entry (the ``_get_last_trace()``/
    ``_build_trace_dict()`` shape) with a single root agent span."""
    span_id_hex = uuid.uuid4().hex
    span: dict = {
        "span_id": span_id_hex,
        "trace_id": trace_id_hex,
        "parent_span_id": None,
        "type": "agent",
        "name": "test-agent",
        "status": "completed",
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:00:01+00:00",
        "duration_ms": 1000,
    }
    if input_data is not None:
        span["input_data"] = input_data
    if metadata is not None:
        span["metadata"] = metadata
    return {
        "trace": {
            "trace_id": trace_id_hex,
            "agent": "test-agent",
            "status": "completed",
            "started_at": "2026-01-01T00:00:00+00:00",
        },
        "spans": [span],
    }


@pytest.fixture
def engine():
    """A freshly-configured engine with in-memory capture — no real network
    calls happen in these tests."""
    eng = get_engine()
    eng.configure(
        endpoint="http://127.0.0.1:1/api/v1/otlp/v1/traces",  # never actually dialed
        api_key="av_test",  # pragma: allowlist secret
        sample_rate=1.0,
        schedule_delay_millis=50,
        max_queue_size=4096,
        max_export_batch_size=512,
        max_field_bytes=65536,
        timeout=1.0,
        redaction_config=RedactionConfig.build(enabled=True),
        capture_for_tests=True,
    )
    try:
        yield eng
    finally:
        eng.reset_for_testing()


# ---------------------------------------------------------------------------
# Byte-cap truncation (MORSE_MAX_FIELD_BYTES)
# ---------------------------------------------------------------------------


class TestByteCap:
    def test_cap_bytes_noop_under_limit(self):
        assert cap_bytes("short", 100) == "short"

    def test_cap_bytes_truncates_and_marks(self):
        long_value = "x" * 200
        capped = cap_bytes(long_value, 50)
        marker_len = len(TRUNCATION_MARKER.encode("utf-8"))
        assert len(capped.encode("utf-8")) <= 50 + marker_len
        assert capped.endswith(TRUNCATION_MARKER)
        assert len(capped) < len(long_value)

    def test_cap_bytes_never_splits_multibyte_utf8(self):
        # Each "é" is 2 UTF-8 bytes — an off-by-one cap must not produce a
        # decode error, and must still round-trip as valid UTF-8.
        value = "é" * 100
        capped = cap_bytes(value, 51)
        capped.encode("utf-8")  # must not raise

    def test_json_capped_string_stays_valid_json(self):
        long_value = "y" * 1000
        encoded = json_capped(long_value, 100)
        assert encoded is not None
        # Truncating the raw string BEFORE encoding keeps the JSON valid.
        decoded = json.loads(encoded)
        assert decoded.endswith(TRUNCATION_MARKER)

    def test_json_capped_none_passthrough(self):
        assert json_capped(None, 100) is None

    def test_emit_trace_entry_caps_oversized_input_data(self, engine):
        """A span's input_data larger than MORSE_MAX_FIELD_BYTES is
        byte-capped at attribute-set time (span_dict -> OTel attributes)."""
        trace_id = uuid.uuid4().hex
        entry = _fresh_entry(trace_id, input_data="z" * 200_000)
        emit_trace_entry(entry)
        engine.flush()

        matching = _spans_for(engine, trace_id)
        assert matching, "expected the emitted span to be captured"
        raw = matching[0].attributes.get("morse.input_data")
        assert raw is not None
        marker_len = len(TRUNCATION_MARKER.encode("utf-8"))
        assert len(raw.encode("utf-8")) <= 65536 + marker_len + 2


# ---------------------------------------------------------------------------
# Sampler — deterministic trace-ID ratio, whole-trace, parent-based
# ---------------------------------------------------------------------------


class TestSampler:
    def test_ratio_approximately_matches_configured_rate(self):
        eng = get_engine()
        eng.configure(
            endpoint="http://127.0.0.1:1/api/v1/otlp/v1/traces",
            api_key="av_test",  # pragma: allowlist secret
            sample_rate=0.1,
            schedule_delay_millis=50,
            max_queue_size=8192,
            max_export_batch_size=2048,
            max_field_bytes=65536,
            timeout=1.0,
            redaction_config=None,
            capture_for_tests=True,
        )
        try:
            n = 2000
            sampled = sum(
                1 for _ in range(n) if eng.resolve_sampled(int(uuid.uuid4().hex, 16))
            )
            ratio = sampled / n
            # Binomial std dev at p=0.1, n=2000 is ~0.67% — 5 percentage
            # points is a generous, non-flaky bound.
            assert 0.05 <= ratio <= 0.15, f"sampled ratio {ratio} outside [0.05, 0.15]"
        finally:
            eng.reset_for_testing()

    def test_rate_1_always_samples(self):
        eng = get_engine()
        eng.configure(
            endpoint="http://127.0.0.1:1/api/v1/otlp/v1/traces",
            api_key="av_test",  # pragma: allowlist secret
            sample_rate=1.0,
            schedule_delay_millis=50,
            max_queue_size=8192,
            max_export_batch_size=2048,
            max_field_bytes=65536,
            timeout=1.0,
            redaction_config=None,
            capture_for_tests=True,
        )
        try:
            assert all(
                eng.resolve_sampled(int(uuid.uuid4().hex, 16)) for _ in range(50)
            )
        finally:
            eng.reset_for_testing()

    def test_rate_0_never_samples(self):
        eng = get_engine()
        eng.configure(
            endpoint="http://127.0.0.1:1/api/v1/otlp/v1/traces",
            api_key="av_test",  # pragma: allowlist secret
            sample_rate=0.0,
            schedule_delay_millis=50,
            max_queue_size=8192,
            max_export_batch_size=2048,
            max_field_bytes=65536,
            timeout=1.0,
            redaction_config=None,
            capture_for_tests=True,
        )
        try:
            assert not any(
                eng.resolve_sampled(int(uuid.uuid4().hex, 16)) for _ in range(50)
            )
        finally:
            eng.reset_for_testing()

    def test_sampled_out_trace_never_reaches_the_exporter(self, engine):
        """Whole-trace head sampling: with rate 0.0, emit_trace_entry() must
        produce zero recorded spans — not a partial/truncated batch."""
        engine.configure(
            endpoint="http://127.0.0.1:1/api/v1/otlp/v1/traces",
            api_key="av_test",  # pragma: allowlist secret
            sample_rate=0.0,
            schedule_delay_millis=50,
            max_queue_size=4096,
            max_export_batch_size=512,
            max_field_bytes=65536,
            timeout=1.0,
            redaction_config=None,
            capture_for_tests=True,
        )
        trace_id = uuid.uuid4().hex
        emit_trace_entry(_fresh_entry(trace_id))
        engine.flush()
        assert engine.recorder is not None
        assert engine.recorder.spans == []


# ---------------------------------------------------------------------------
# Redaction at export time (single pass, off the hot path)
# ---------------------------------------------------------------------------


class TestExportTimeRedaction:
    def test_pii_in_input_data_is_redacted_in_the_exported_copy(self, engine):
        trace_id = uuid.uuid4().hex
        entry = _fresh_entry(
            trace_id, input_data={"message": "contact me at jane@example.com"}
        )
        emit_trace_entry(entry)
        engine.flush()

        matching = _spans_for(engine, trace_id)
        assert matching
        raw = matching[0].attributes.get("morse.input_data")
        assert raw is not None
        assert "jane@example.com" not in raw
        assert "[REDACTED:email]" in raw

    def test_redaction_disabled_passes_through(self):
        eng = get_engine()
        eng.configure(
            endpoint="http://127.0.0.1:1/api/v1/otlp/v1/traces",
            api_key="av_test",  # pragma: allowlist secret
            sample_rate=1.0,
            schedule_delay_millis=50,
            max_queue_size=4096,
            max_export_batch_size=512,
            max_field_bytes=65536,
            timeout=1.0,
            redaction_config=RedactionConfig.build(enabled=False),
            capture_for_tests=True,
        )
        try:
            trace_id = uuid.uuid4().hex
            entry = _fresh_entry(trace_id, input_data={"message": "jane@example.com"})
            emit_trace_entry(entry)
            eng.flush()
            matching = [
                s for s in eng.recorder.spans if s.context.trace_id == int(trace_id, 16)
            ]
            assert matching
            assert "jane@example.com" in matching[0].attributes.get(
                "morse.input_data", ""
            )
        finally:
            eng.reset_for_testing()


# ---------------------------------------------------------------------------
# flush() blocks until export completes
# ---------------------------------------------------------------------------


class TestFlushBlocks:
    def test_flush_blocks_until_span_is_exported(self, engine):
        """force_flush() must not return before the in-memory recorder (which
        sits behind the SAME BatchSpanProcessor pipeline as the real OTLP
        exporter) has actually received the span — proving flush() is a
        real blocking drain, not a fire-and-forget poll."""
        trace_id = uuid.uuid4().hex
        emit_trace_entry(_fresh_entry(trace_id))
        engine.flush()  # should BLOCK until the batch processor exports
        matching = _spans_for(engine, trace_id)
        assert matching, "flush() returned before the span was exported"


# ---------------------------------------------------------------------------
# Fork safety (POSIX only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="os.fork() is POSIX-only")
class TestForkSafety:
    def test_child_process_emits_without_crash_or_deadlock(self, engine):
        """A forked child that emits a span must not crash or hang. OTel's
        BatchSpanProcessor doesn't auto-restart its worker thread across
        fork() (a known upstream limitation, not something this SDK papers
        over) — this test only asserts the child process itself doesn't
        crash or deadlock while doing so, per the MHQ-668 test plan."""
        pid = os.fork()
        if pid == 0:
            # Child
            try:
                trace_id = uuid.uuid4().hex
                emit_trace_entry(_fresh_entry(trace_id))
            except Exception:  # noqa: BLE001
                os._exit(1)
            os._exit(0)
        else:
            _, status = os.waitpid(pid, 0)
            assert os.WIFEXITED(status)
            assert os.WEXITSTATUS(status) == 0


@pytest.mark.req("MHQ-1158")
def test_spans_carry_the_sdk_version_on_the_instrumentation_scope(engine):
    """The server stamps traces.sdk_version from the scope version on the OTLP
    path; with none, every trace read ``otlp/http`` (rc2 acceptance)."""
    from morse_ai._version import __version__

    trace_id = uuid.uuid4().hex
    emit_trace_entry(_fresh_entry(trace_id))
    engine.flush()

    scope = _spans_for(engine, trace_id)[0].instrumentation_scope
    assert (scope.name, scope.version) == ("morse_ai", __version__)


def _mk_span(span_id: str, parent: str | None, name: str, trace_id: str) -> dict:
    return {
        "span_id": span_id,
        "trace_id": trace_id,
        "parent_span_id": parent,
        "type": "agent",
        "name": name,
        "status": "completed",
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:00:01+00:00",
        "duration_ms": 1000,
    }


@pytest.mark.req("MHQ-1412")
def test_children_ending_before_their_parent_keep_their_parent_link(engine):
    """MHQ-1412: spans arrive in completion order (child before parent). The
    replay must still parent each span on its real parent, not on the synthetic
    0x1 placeholder that is stored as 0000...0001."""
    trace_id = uuid.uuid4().hex
    root, chain, prompt, llm = (uuid.uuid4().hex[:16] for _ in range(4))
    entry = {
        "trace": {
            "trace_id": trace_id,
            "agent": "lcel",
            "status": "completed",
            "started_at": "2026-01-01T00:00:00+00:00",
        },
        # completion order: leaves first, root last
        "spans": [
            _mk_span(prompt, chain, "ChatPromptTemplate", trace_id),
            _mk_span(llm, chain, "ChatOpenAI", trace_id),
            _mk_span(chain, root, "chain", trace_id),
            _mk_span(root, None, "root", trace_id),
        ],
    }
    emit_trace_entry(entry)
    engine.flush()

    by_name = {s.name: s for s in _spans_for(engine, trace_id)}
    chain_id = by_name["chain"].context.span_id
    assert by_name["chain"].parent.span_id == by_name["root"].context.span_id
    assert by_name["ChatPromptTemplate"].parent.span_id == chain_id
    assert by_name["ChatOpenAI"].parent.span_id == chain_id


@pytest.mark.req("MHQ-1412")
def test_parents_first_orders_parents_ahead_and_survives_cycles():
    from morse_ai.otel_provider import _parents_first

    a = {"span_id": "a", "parent_span_id": "b"}
    b = {"span_id": "b", "parent_span_id": "a"}  # cycle
    orphan = {"span_id": "c", "parent_span_id": "missing"}
    out = _parents_first([a, b, orphan], None)
    assert {id(x) for x in out} == {id(a), id(b), id(orphan)}
    kid = {"span_id": "k", "parent_span_id": "p"}
    par = {"span_id": "p", "parent_span_id": None}
    assert _parents_first([kid, par], "p") == [par, kid]


# ---------------------------------------------------------------------------
# MHQ-1936: LangChain/LangGraph messages serialize as JSON, not repr
# ---------------------------------------------------------------------------


class _FakeMessage:
    """Pydantic-shaped stand-in for a LangChain ``BaseMessage``."""

    def __init__(self, type_: str, content, **extra):
        self.type = type_
        self.content = content
        self._extra = extra

    def model_dump(self):
        return {"type": self.type, "content": self.content, **self._extra}

    def __repr__(self):
        return f"FakeMessage(content={self.content!r})"


class TestJsonCappedMessages:
    def test_fake_message_serializes_as_structured_dict(self):
        msg = _FakeMessage(
            "ai",
            "hi",
            tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call_1"}],
            name=None,
            id="",
            additional_kwargs={},
            invalid_tool_calls=[],
            example=False,
        )
        decoded = json.loads(json_capped({"messages": [msg]}, 10_000))
        (dumped,) = decoded["messages"]
        assert dumped["type"] == "ai"
        assert dumped["content"] == "hi"
        assert dumped["tool_calls"][0]["name"] == "search"
        # Empty fields dropped, falsy-but-meaningful values kept.
        for key in ("name", "id", "additional_kwargs", "invalid_tool_calls"):
            assert key not in dumped
        assert dumped["example"] is False

    def test_fake_message_no_repr_string(self):
        encoded = json_capped([_FakeMessage("human", "hello")], 10_000)
        assert "FakeMessage(content=" not in encoded
        assert json.loads(encoded) == [{"type": "human", "content": "hello"}]

    def test_non_message_model_keeps_empty_fields(self):
        class _Model:
            def model_dump(self):
                return {"a": None, "b": []}

        assert json.loads(json_capped(_Model(), 10_000)) == {"a": None, "b": []}

    def test_unknown_object_falls_back_to_str(self):
        class _Opaque:
            __slots__ = ()

            def __str__(self):
                return "opaque!"

        assert json.loads(json_capped({"x": _Opaque()}, 10_000)) == {"x": "opaque!"}

    def test_size_cap_still_applies(self):
        msg = _FakeMessage("human", "x" * 5_000)
        encoded = json_capped([msg], 200)
        assert len(encoded.encode("utf-8")) <= 200 + len(
            TRUNCATION_MARKER.encode("utf-8")
        )
        assert encoded.endswith(TRUNCATION_MARKER)

    def test_real_langchain_messages(self):
        messages = pytest.importorskip("langchain_core.messages")
        human = messages.HumanMessage(content="What is 2+2?")
        ai = messages.AIMessage(
            content="",
            tool_calls=[{"name": "calc", "args": {"expr": "2+2"}, "id": "call_1"}],
        )
        encoded = json_capped({"messages": [human, ai]}, 10_000)
        assert "HumanMessage(content=" not in encoded
        assert "AIMessage(content=" not in encoded
        h, a = json.loads(encoded)["messages"]
        assert h["type"] == "human"
        assert h["content"] == "What is 2+2?"
        assert "id" not in h and "name" not in h
        assert a["type"] == "ai"
        assert a["content"] == ""  # kept even when empty
        assert a["tool_calls"][0]["name"] == "calc"
        assert a["tool_calls"][0]["args"] == {"expr": "2+2"}
