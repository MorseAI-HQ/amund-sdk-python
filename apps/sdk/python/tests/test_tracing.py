"""Tests for SDK v2 tracing core — SpanData, TraceContext, trace_agent, run(), span()."""

# ruff: noqa: E501, SIM117
import time
from unittest.mock import MagicMock

import pytest

from morse_ai.spans import SpanData, SpanType


class TestSpanType:
    def test_has_all_six_values(self):
        assert SpanType.AGENT == "agent"
        assert SpanType.LLM == "llm"
        assert SpanType.TOOL == "tool"
        assert SpanType.HTTP == "http"
        assert SpanType.DB == "db"
        assert SpanType.CUSTOM == "custom"

    def test_enum_values_match_backend(self):
        # "memory" added in Phase 9 (Memory Profiler), "retriever" already
        # present in the backend (apps/api/apps/traces/models.py) — keep in
        # sync with the backend SpanType TextChoices there.
        values = {s.value for s in SpanType}
        assert values == {
            "agent",
            "llm",
            "tool",
            "http",
            "db",
            "custom",
            "memory",
            "retriever",
            "guardrail",
            "handoff",
        }


class TestSpanData:
    def test_creates_with_required_fields(self):
        span = SpanData(
            trace_id="trace123",
            type=SpanType.AGENT,
            name="my-agent",
        )
        assert span.trace_id == "trace123"
        assert span.type == SpanType.AGENT
        assert span.name == "my-agent"

    def test_auto_generates_span_id(self):
        s1 = SpanData(trace_id="t1", type=SpanType.AGENT, name="a")
        s2 = SpanData(trace_id="t1", type=SpanType.AGENT, name="a")
        assert s1.span_id  # not empty
        assert s1.span_id != s2.span_id  # unique

    def test_auto_sets_started_at(self):
        span = SpanData(trace_id="t1", type=SpanType.AGENT, name="a")
        assert span.started_at  # ISO string set by default

    def test_optional_fields_default_to_none(self):
        span = SpanData(trace_id="t1", type=SpanType.AGENT, name="a")
        assert span.parent_span_id is None
        assert span.ended_at is None
        assert span.duration_ms is None
        assert span.model is None
        assert span.input_tokens is None
        assert span.output_tokens is None
        assert span.total_tokens is None
        assert span.cost_usd is None
        assert span.input_data is None
        assert span.output_data is None
        assert span.error_message is None
        assert span.metadata is None

    def test_to_dict_drops_none_values(self):
        span = SpanData(trace_id="t1", type=SpanType.AGENT, name="my-agent")
        d = span.to_dict()
        assert "model" not in d
        assert "input_tokens" not in d
        assert "error_message" not in d

    def test_to_dict_includes_non_none_values(self):
        span = SpanData(
            trace_id="trace-abc",
            type=SpanType.LLM,
            name="llm-call",
            model="claude-sonnet-4-6",
            input_tokens=1000,
            output_tokens=200,
            cost_usd=0.025,
        )
        d = span.to_dict()
        assert d["trace_id"] == "trace-abc"
        assert d["type"] == "llm"
        assert d["name"] == "llm-call"
        assert d["model"] == "claude-sonnet-4-6"
        assert d["input_tokens"] == 1000
        assert d["output_tokens"] == 200
        assert d["cost_usd"] == 0.025

    def test_to_dict_converts_span_type_to_string(self):
        span = SpanData(trace_id="t1", type=SpanType.TOOL, name="calculator")
        d = span.to_dict()
        assert d["type"] == "tool"
        assert isinstance(d["type"], str)

    def test_to_dict_is_json_serializable(self):
        import json

        span = SpanData(
            trace_id="t1",
            type=SpanType.AGENT,
            name="my-agent",
            input_tokens=500,
        )
        # should not raise
        json.dumps(span.to_dict())


class TestTraceAgent:
    def test_decorator_calls_function(self):
        from morse_ai.tracing import trace_agent

        called = []

        @trace_agent
        def my_agent():
            called.append(True)
            return "done"

        result = my_agent()
        assert result == "done"
        assert called == [True]

    def test_decorator_with_name(self):
        from morse_ai.tracing import trace_agent

        called = []

        @trace_agent("custom-agent-name")
        def my_agent():
            called.append(True)
            return 42

        result = my_agent()
        assert result == 42
        assert called == [True]

    def test_decorator_preserves_function_name(self):
        from morse_ai.tracing import trace_agent

        @trace_agent
        def my_named_function():
            pass

        assert my_named_function.__name__ == "my_named_function"

    def test_decorator_creates_agent_root_span(self):
        from morse_ai.tracing import _get_last_trace, trace_agent

        @trace_agent
        def my_agent():
            return "result"

        my_agent()
        trace = _get_last_trace()
        assert trace is not None
        assert len(trace["spans"]) >= 1
        root_span = trace["spans"][0]
        assert root_span["type"] == "agent"

    def test_decorator_auto_times_span(self):
        from morse_ai.tracing import _get_last_trace, trace_agent

        @trace_agent
        def my_agent():
            time.sleep(0.01)
            return "done"

        my_agent()
        trace = _get_last_trace()
        root_span = trace["spans"][0]
        assert root_span.get("duration_ms") is not None
        assert root_span["duration_ms"] >= 10

    def test_decorator_sets_failed_status_on_exception(self):
        from morse_ai.tracing import _get_last_trace, trace_agent

        @trace_agent
        def bad_agent():
            raise ValueError("something went wrong")

        with pytest.raises(ValueError):
            bad_agent()

        trace = _get_last_trace()
        root_span = trace["spans"][0]
        assert root_span["status"] == "failed"
        assert "something went wrong" in root_span.get("error_message", "")

    def test_decorator_generates_unique_trace_ids(self):
        from morse_ai.tracing import _get_last_trace, trace_agent

        @trace_agent
        def my_agent():
            return "ok"

        my_agent()
        trace1 = _get_last_trace()
        my_agent()
        trace2 = _get_last_trace()

        assert trace1["trace"]["trace_id"] != trace2["trace"]["trace_id"]

    def test_decorator_emits_trace(self):
        """MHQ-668: trace_agent's primary emission path is real OTel spans
        (not BatchTransport.track()) — verify via the semantic surface,
        ``_get_last_trace()``, rather than the retired wire-transport mock.
        """
        from morse_ai.tracing import _get_last_trace, trace_agent

        @trace_agent
        def my_agent():
            return "ok"

        my_agent()
        assert _get_last_trace() is not None

    def test_decorator_legacy_transport_escape_hatch(self, monkeypatch):
        """MORSE_LEGACY_TRANSPORT=1 reverts trace_agent to the pre-MHQ-668
        JSON BatchTransport.track() path (rollback escape hatch)."""
        from morse_ai.tracing import _set_transport, trace_agent

        monkeypatch.setenv("MORSE_LEGACY_TRANSPORT", "1")
        mock_transport = MagicMock()
        _set_transport(mock_transport)

        try:

            @trace_agent
            def my_agent():
                return "ok"

            my_agent()
            assert mock_transport.track.called
        finally:
            _set_transport(None)

    @pytest.mark.req("MHQ-1157")
    def test_decorator_does_not_claim_an_outcome_on_clean_return(self):
        """MHQ-1157: @trace_agent has no outcome API, so it must not send
        ``success`` — the server reads any success as a developer-declared
        outcome and stamps provenance='developer' on a NULL outcome."""
        from morse_ai.tracing import _get_last_trace, trace_agent

        @trace_agent
        def good_agent():
            return "result"

        good_agent()
        trace = _get_last_trace()
        assert "success" not in trace["trace"]
        assert "outcome" not in trace["trace"]
        assert trace["trace"]["status"] == "completed"

    @pytest.mark.req("MHQ-1157")
    def test_decorator_raise_records_failure_same_as_run(self):
        from morse_ai.tracing import _get_last_trace, trace_agent

        @trace_agent
        def bad_agent():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            bad_agent()

        trace = _get_last_trace()
        assert trace["trace"]["success"] is False
        assert trace["trace"]["status"] == "failed"
        assert trace["spans"][0]["status"] == "failed"

    @pytest.mark.req("MHQ-1157")
    def test_decorator_emits_no_outcome_attribute_on_the_wire(self):
        """What the server keys provenance on: morse.outcome.success."""
        from morse_ai.otel_provider import _span_dict_to_otel_attributes
        from morse_ai.tracing import _get_last_trace, run, trace_agent

        @trace_agent
        def good_agent():
            return "result"

        good_agent()
        t = _get_last_trace()
        attrs = _span_dict_to_otel_attributes(
            t["spans"][0],
            t["trace"],
            t.get("resource") or {},
            is_root=True,
            max_field_bytes=65536,
        )
        assert "morse.outcome.success" not in attrs
        assert "morse.outcome.label" not in attrs

        # Contrast: an explicit outcome via morse_ai.run still declares it.
        with run("my-agent") as r:
            r.set_outcome(success=True, outcome="qualified")
        t = _get_last_trace()
        attrs = _span_dict_to_otel_attributes(
            t["spans"][0],
            t["trace"],
            t.get("resource") or {},
            is_root=True,
            max_field_bytes=65536,
        )
        assert attrs["morse.outcome.success"] is True
        assert attrs["morse.outcome.label"] == "qualified"


class TestRunContextManager:
    def test_run_as_context_manager(self):
        from morse_ai.tracing import run

        with run("my-agent") as r:
            assert r is not None

    def test_run_set_outcome(self):
        from morse_ai.tracing import _get_last_trace, run

        with run("my-agent") as r:
            r.set_outcome(success=True, outcome="qualified")

        trace = _get_last_trace()
        assert trace["trace"]["success"] is True
        assert trace["trace"]["outcome"] == "qualified"

    def test_run_set_error(self):
        from morse_ai.tracing import _get_last_trace, run

        with run("my-agent") as r:
            r.set_error("custom error message")

        trace = _get_last_trace()
        assert trace["trace"]["status"] == "failed"

    def test_run_sets_failed_on_exception(self):
        from morse_ai.tracing import _get_last_trace, run

        with pytest.raises(RuntimeError), run("my-agent"):
            raise RuntimeError("test error")

        trace = _get_last_trace()
        root_span = trace["spans"][0]
        assert root_span["status"] == "failed"
        assert "test error" in root_span.get("error_message", "")

    @pytest.mark.req("MHQ-1157")
    def test_run_never_guesses_an_outcome_on_clean_exit(self):
        from morse_ai.tracing import _get_last_trace, run

        with run("my-agent"):
            pass

        trace = _get_last_trace()
        assert "success" not in trace["trace"]
        assert trace["trace"]["status"] == "completed"

    @pytest.mark.req("MHQ-1157")
    def test_run_exception_records_failure(self):
        from morse_ai.tracing import _get_last_trace, run

        with pytest.raises(RuntimeError), run("my-agent"):
            raise RuntimeError("blew up")

        trace = _get_last_trace()
        assert trace["trace"]["success"] is False
        assert trace["trace"]["status"] == "failed"

    @pytest.mark.req("MHQ-1157")
    def test_module_set_outcome_works_in_run_and_trace_agent(self):
        import morse_ai
        from morse_ai.tracing import _get_last_trace, run, trace_agent

        with run("my-agent"):
            morse_ai.set_outcome(True, outcome="qualified")
        t = _get_last_trace()["trace"]
        assert (t["success"], t["outcome"]) == (True, "qualified")

        @trace_agent
        def agent():
            morse_ai.set_outcome(False, outcome="rejected")

        agent()
        t = _get_last_trace()["trace"]
        assert (t["success"], t["outcome"]) == (False, "rejected")

    def test_module_set_outcome_without_open_trace_is_noop(self):
        import morse_ai

        morse_ai.set_outcome(True)  # must not raise

    def test_run_explicit_set_outcome_overrides_inference(self):
        from morse_ai.tracing import _get_last_trace, run

        # An exception still propagates, but the caller declared success
        # via set_outcome before raising — inference must not clobber it.
        with pytest.raises(RuntimeError), run("my-agent") as r:
            r.set_outcome(success=True, outcome="partial")
            raise RuntimeError("but the work was useful")

        trace = _get_last_trace()
        assert trace["trace"]["success"] is True
        assert trace["trace"]["outcome"] == "partial"

    def test_run_creates_agent_root_span(self):
        from morse_ai.tracing import _get_last_trace, run

        with run("my-agent"):
            pass

        trace = _get_last_trace()
        assert trace["trace"]["agent"] == "my-agent"
        assert len(trace["spans"]) >= 1
        root_span = trace["spans"][0]
        assert root_span["type"] == "agent"
        assert root_span["name"] == "my-agent"

    def test_run_emits_trace(self):
        """MHQ-668: run()'s primary emission path is real OTel spans (not
        BatchTransport.track()) — verify via ``_get_last_trace()``."""
        from morse_ai.tracing import _get_last_trace, run

        with run("my-agent"):
            pass
        assert _get_last_trace() is not None

    def test_run_legacy_transport_escape_hatch(self, monkeypatch):
        """MORSE_LEGACY_TRANSPORT=1 reverts run() to the pre-MHQ-668 JSON
        BatchTransport.track() path (rollback escape hatch)."""
        from morse_ai.tracing import _set_transport, run

        monkeypatch.setenv("MORSE_LEGACY_TRANSPORT", "1")
        mock_transport = MagicMock()
        _set_transport(mock_transport)

        try:
            with run("my-agent"):
                pass
            assert mock_transport.track.called
        finally:
            _set_transport(None)

    def test_run_wire_format(self):
        """``_get_last_trace()``'s dict shape is unchanged by MHQ-668 —
        {"trace": {...}, "spans": [...]} — even though the wire mechanism
        underneath (real OTel/OTLP) is entirely new."""
        from morse_ai.tracing import _get_last_trace, run

        with run("my-agent") as r:
            r.set_outcome(success=True, outcome="ok")

        entry = _get_last_trace()
        assert entry is not None
        assert "trace" in entry
        assert "spans" in entry


class TestSpanContextManager:
    def test_span_inside_run(self):
        from morse_ai.tracing import _get_last_trace, run, span

        with run("my-agent"), span("tool-call", span_type="tool"):
            pass

        trace = _get_last_trace()
        span_names = [s["name"] for s in trace["spans"]]
        assert "tool-call" in span_names

    def test_span_type_is_set(self):
        from morse_ai.tracing import _get_last_trace, run, span

        with run("my-agent"), span("db-query", span_type="db"):
            pass

        trace = _get_last_trace()
        db_span = next(s for s in trace["spans"] if s["name"] == "db-query")
        assert db_span["type"] == "db"

    def test_span_default_type_is_custom(self):
        from morse_ai.tracing import _get_last_trace, run, span

        with run("my-agent"), span("my-span"):
            pass

        trace = _get_last_trace()
        my_span = next(s for s in trace["spans"] if s["name"] == "my-span")
        assert my_span["type"] == "custom"

    def test_span_has_parent_span_id(self):
        from morse_ai.tracing import _get_last_trace, run, span

        with run("my-agent"), span("child-span"):
            pass

        trace = _get_last_trace()
        root_span = trace["spans"][0]
        child_span = next(s for s in trace["spans"] if s["name"] == "child-span")
        assert child_span["parent_span_id"] == root_span["span_id"]

    def test_nested_spans_correct_parent_chain(self):
        from morse_ai.tracing import _get_last_trace, run, span

        with run("my-agent"), span("parent-span"), span("child-span"):
            with span("grandchild-span"):
                pass

        trace = _get_last_trace()
        spans_by_name = {s["name"]: s for s in trace["spans"]}

        root = spans_by_name["my-agent"]
        parent = spans_by_name["parent-span"]
        child = spans_by_name["child-span"]
        grandchild = spans_by_name["grandchild-span"]

        # Root span has no parent
        assert root.get("parent_span_id") is None or "parent_span_id" not in root

        # parent -> root
        assert parent["parent_span_id"] == root["span_id"]

        # child -> parent
        assert child["parent_span_id"] == parent["span_id"]

        # grandchild -> child
        assert grandchild["parent_span_id"] == child["span_id"]

    def test_span_auto_times(self):
        from morse_ai.tracing import _get_last_trace, run, span

        with run("my-agent"), span("slow-span"):
            time.sleep(0.01)

        trace = _get_last_trace()
        timed_span = next(s for s in trace["spans"] if s["name"] == "slow-span")
        assert timed_span.get("duration_ms") is not None
        assert timed_span["duration_ms"] >= 10

    def test_span_outside_trace_is_noop(self):
        from morse_ai.tracing import span

        # Should not raise — silent no-op when no active trace context
        with span("orphan-span"):
            pass  # No exception expected

    def test_span_inside_trace_agent(self):
        from morse_ai.tracing import _get_last_trace, span, trace_agent

        @trace_agent
        def my_agent():
            with span("tool-call", span_type="tool"):
                return "result"

        my_agent()
        trace = _get_last_trace()
        span_names = [s["name"] for s in trace["spans"]]
        assert "tool-call" in span_names
