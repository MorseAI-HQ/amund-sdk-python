"""Tests for the LangGraph adapter's _active_trace + W3C trace_id bridges.

Issue 2 / Phase 12.5 follow-up: verify the LangGraphCallbackHandler

  1. Eagerly sets ``_active_trace`` on root ``on_chain_start`` so log lines
     fired inside the chain (via the MorseHandler) can read the active
     trace_id.
  2. Resets ``_active_trace`` on root ``on_chain_end`` / ``on_chain_error``.
  3. Adopts the active W3C trace_id (from a live OpenTelemetry SpanContext)
     as the Morse trace_id when one is available.
  4. Falls back cleanly to ``uuid4()`` generation when no OTel context is
     active or OTel isn't installed.
  5. Nested chains (``parent_run_id is not None``) do NOT clobber the root
     chain's trace state — they inherit.
  6. End-to-end: a stdlib log line emitted while the chain is open carries
     the same trace_id as the chain.

These tests do NOT require langchain, langgraph, or opentelemetry to be
installed — the W3C path is exercised by stubbing
``amund.adapters.langgraph._build_otel_traceparent``.
"""

# ruff: noqa: E501, SIM117
from __future__ import annotations

import logging as stdlib_logging
import uuid
from typing import Any

import pytest

from amund import tracing
from amund.adapters import langgraph as langgraph_adapter
from amund.adapters.langgraph import (
    LangGraphCallbackHandler,
    _w3c_trace_id_to_uuid_hex,
)
from amund.logging import _current_trace_context

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


# Canonical W3C example trace_id from the spec
# (https://www.w3.org/TR/trace-context/#examples) — used throughout the tests
# as a stable, well-known 128-bit trace_id we can assert against. Not a real
# secret; the allowlist pragma silences detect-secrets' high-entropy heuristic.
W3C_EXAMPLE_TRACE_HEX = "4bf92f3577b34da6a3ce929d0e0e4736"  # pragma: allowlist secret
W3C_EXAMPLE_SPAN_HEX = "00f067aa0ba902b7"  # pragma: allowlist secret


def _chain_serialized(name: str = "CompiledStateGraph") -> dict[str, Any]:
    """Mimic the LangGraph-shaped on_chain_start serialized blob."""
    return {"id": ["langgraph", "graph", name], "name": name}


def _make_w3c_traceparent(trace_hex: str, span_hex: str = W3C_EXAMPLE_SPAN_HEX) -> str:
    """Format a synthetic W3C traceparent given a 32-hex trace_id."""
    return f"00-{trace_hex}-{span_hex}-01"


@pytest.fixture(autouse=True)
def _reset_active_trace():
    """Ensure every test starts with a clean _active_trace context.

    A leaking ContextVar across tests would mask the bridges' behavior
    (handler would think a trace is already open and short-circuit).
    """
    # Capture the current token so we can restore even if a prior test
    # left _active_trace populated.
    token = tracing._active_trace.set(None)
    try:
        yield
    finally:
        tracing._active_trace.reset(token)


# ---------------------------------------------------------------------------
# Bridge 1 — _active_trace lifecycle on root chain
# ---------------------------------------------------------------------------


@pytest.mark.req("SDK-09")
def test_root_on_chain_start_sets_active_trace():
    """Root chain start must populate _active_trace before any child span fires.

    The MorseHandler.emit() (Phase 12.5 Tier 1.1) reads _active_trace at
    log time. Without this bridge, log lines fired in the chain ship with
    trace_id=None and never correlate to the AI trace on the dashboard.
    """
    handler = LangGraphCallbackHandler("test-agent")
    run_id = uuid.uuid4()

    assert tracing._active_trace.get() is None

    handler.on_chain_start(
        _chain_serialized("MyGraph"),
        {"messages": []},
        run_id=run_id,
    )

    state = tracing._active_trace.get()
    assert state is not None, "root on_chain_start must set _active_trace"
    assert state.agent_name == "test-agent"
    # trace_id is 32 lowercase hex (uuid4 hex format)
    assert len(state.trace_id) == 32
    int(state.trace_id, 16)  # validates hex

    # Cleanup so we don't leak the trace into the autouse fixture's reset.
    handler.on_chain_end({"output": "done"}, run_id=run_id)


@pytest.mark.req("SDK-09")
def test_root_on_chain_end_resets_active_trace():
    """The auto-trace opened by root on_chain_start must close on chain end."""
    handler = LangGraphCallbackHandler("test-agent")
    run_id = uuid.uuid4()

    handler.on_chain_start(
        _chain_serialized("MyGraph"),
        {"messages": []},
        run_id=run_id,
    )
    assert tracing._active_trace.get() is not None

    handler.on_chain_end({"output": "done"}, run_id=run_id)

    assert tracing._active_trace.get() is None, (
        "auto-trace should close when root chain ends"
    )


@pytest.mark.req("SDK-09")
def test_root_on_chain_error_resets_active_trace():
    """Even on a chain error, _active_trace must be reset (no leak across runs)."""
    handler = LangGraphCallbackHandler("test-agent")
    run_id = uuid.uuid4()

    handler.on_chain_start(
        _chain_serialized("MyGraph"),
        {"messages": []},
        run_id=run_id,
    )
    assert tracing._active_trace.get() is not None

    handler.on_chain_error(RuntimeError("boom"), run_id=run_id)

    assert tracing._active_trace.get() is None


# ---------------------------------------------------------------------------
# Bridge 1 — nested chains must NOT clobber the root trace
# ---------------------------------------------------------------------------


@pytest.mark.req("SDK-09")
def test_nested_chain_does_not_mutate_active_trace():
    """A nested chain (parent_run_id != None) inherits the root trace state.

    LangGraph emits an on_chain_start per node; only the OUTER root chain
    has parent_run_id=None. The bridge must skip _ensure_trace for nested
    chains so we don't open a competing trace and lose correlation.
    """
    handler = LangGraphCallbackHandler("test-agent")
    root_run = uuid.uuid4()
    nested_run = uuid.uuid4()

    handler.on_chain_start(
        _chain_serialized("RootGraph"),
        {"messages": []},
        run_id=root_run,
    )
    root_state = tracing._active_trace.get()
    assert root_state is not None
    root_trace_id = root_state.trace_id

    # Now a nested chain fires — same trace_id must persist.
    handler.on_chain_start(
        _chain_serialized("NestedNode"),
        {"input": "x"},
        run_id=nested_run,
        parent_run_id=root_run,
    )
    nested_state = tracing._active_trace.get()
    assert nested_state is not None
    assert nested_state.trace_id == root_trace_id, (
        "nested chain must inherit root chain's trace_id, not open a new one"
    )

    # Close in reverse order.
    handler.on_chain_end({"out": "y"}, run_id=nested_run)
    # Root trace stays open because root chain hasn't ended yet AND
    # the nested chain's child span hasn't been popped yet... actually
    # the parent's _maybe_close_auto_trace only fires when span_stack is
    # empty. Since the root's span is still on the stack, the trace
    # stays open. Verify that.
    assert tracing._active_trace.get() is not None

    handler.on_chain_end({"output": "done"}, run_id=root_run)
    assert tracing._active_trace.get() is None


# ---------------------------------------------------------------------------
# Bridge 2 — W3C trace_id adoption
# ---------------------------------------------------------------------------


@pytest.mark.req("SDK-09")
def test_w3c_trace_id_helper_round_trips():
    """The 128-bit-int -> 32-hex conversion must be lossless and round-trippable."""
    # Pick a known W3C trace_id (this one is from the W3C spec example).
    w3c_int = int(W3C_EXAMPLE_TRACE_HEX, 16)
    hex_id = _w3c_trace_id_to_uuid_hex(w3c_int)
    assert hex_id == W3C_EXAMPLE_TRACE_HEX
    assert int(hex_id, 16) == w3c_int
    # Length and lowercase invariants the SDK relies on elsewhere.
    assert len(hex_id) == 32
    assert hex_id == hex_id.lower()


@pytest.mark.req("SDK-09")
def test_root_chain_adopts_w3c_trace_id_when_otel_context_active(
    monkeypatch: pytest.MonkeyPatch,
):
    """When _build_otel_traceparent returns a valid traceparent, the trace
    adopts the W3C trace_id (not a fresh uuid4)."""
    w3c_trace_hex = W3C_EXAMPLE_TRACE_HEX
    traceparent = _make_w3c_traceparent(w3c_trace_hex)

    monkeypatch.setattr(
        langgraph_adapter, "_build_otel_traceparent", lambda: traceparent
    )

    handler = LangGraphCallbackHandler("test-agent")
    run_id = uuid.uuid4()
    handler.on_chain_start(
        _chain_serialized("MyGraph"),
        {"messages": []},
        run_id=run_id,
    )

    state = tracing._active_trace.get()
    assert state is not None
    assert state.trace_id == w3c_trace_hex, (
        "Morse trace_id must equal the W3C trace_id when an OTel context is active"
    )

    handler.on_chain_end({"output": "done"}, run_id=run_id)

    # Verify the emitted trace payload also carries the W3C trace_id —
    # this is the field that controls Infra-tab correlation server-side.
    trace = tracing._get_last_trace()
    assert trace is not None
    assert trace["trace"]["trace_id"] == w3c_trace_hex


@pytest.mark.req("SDK-09")
def test_falls_back_to_uuid4_when_no_otel_context(
    monkeypatch: pytest.MonkeyPatch,
):
    """Without an active W3C context, the trace uses uuid4()-style trace_id."""
    monkeypatch.setattr(langgraph_adapter, "_build_otel_traceparent", lambda: None)

    handler = LangGraphCallbackHandler("test-agent")
    run_id = uuid.uuid4()
    handler.on_chain_start(
        _chain_serialized("MyGraph"),
        {"messages": []},
        run_id=run_id,
    )

    state = tracing._active_trace.get()
    assert state is not None
    # 32-char lowercase hex — what uuid4().hex looks like.
    assert len(state.trace_id) == 32
    int(state.trace_id, 16)
    # Must NOT be a fixed W3C string from the previous test.
    assert state.trace_id != W3C_EXAMPLE_TRACE_HEX

    handler.on_chain_end({"output": "done"}, run_id=run_id)


@pytest.mark.req("SDK-09")
def test_falls_back_when_otel_traceparent_malformed(
    monkeypatch: pytest.MonkeyPatch,
):
    """A garbage traceparent must not crash — extract_traceparent returns None
    and the trace falls back to uuid4()."""
    monkeypatch.setattr(
        langgraph_adapter,
        "_build_otel_traceparent",
        lambda: "not-a-real-traceparent",
    )

    handler = LangGraphCallbackHandler("test-agent")
    run_id = uuid.uuid4()
    handler.on_chain_start(
        _chain_serialized("MyGraph"),
        {"messages": []},
        run_id=run_id,
    )

    state = tracing._active_trace.get()
    assert state is not None
    # Uuid4-style fallback when the traceparent is malformed.
    assert len(state.trace_id) == 32
    int(state.trace_id, 16)

    handler.on_chain_end({"output": "done"}, run_id=run_id)


@pytest.mark.req("SDK-09")
def test_does_not_overwrite_existing_active_trace(monkeypatch: pytest.MonkeyPatch):
    """If the caller wrapped the chain in `with amund.run():`, the handler
    must NOT open a competing auto-trace — it inherits.

    This is the "user supplies their own trace context" path. Without this,
    every customer who manually opens a trace + uses LangGraph would see two
    traces emitted per request.
    """
    monkeypatch.setattr(
        langgraph_adapter,
        "_build_otel_traceparent",
        lambda: _make_w3c_traceparent(W3C_EXAMPLE_TRACE_HEX),
    )

    with tracing.run("outer-agent") as outer:
        outer_trace_id = outer.trace_id
        assert outer_trace_id is not None

        handler = LangGraphCallbackHandler("test-agent")
        run_id = uuid.uuid4()
        handler.on_chain_start(
            _chain_serialized("MyGraph"),
            {"messages": []},
            run_id=run_id,
        )

        state = tracing._active_trace.get()
        assert state is not None
        assert state.trace_id == outer_trace_id, (
            "explicit user-managed trace must not be replaced by the handler's auto-trace"
        )

        handler.on_chain_end({"output": "done"}, run_id=run_id)

        # The outer trace is still open (user manages it).
        assert tracing._active_trace.get() is not None


# ---------------------------------------------------------------------------
# End-to-end — log line emitted inside the chain inherits the chain trace_id
# ---------------------------------------------------------------------------


@pytest.mark.req("SDK-09")
def test_end_to_end_log_line_inherits_chain_trace_id(
    monkeypatch: pytest.MonkeyPatch,
):
    """A stdlib log call inside the chain must produce a payload with the
    same trace_id as the chain's Morse trace.

    This is what the Logs tab on the Copilot trace relies on:
    MorseHandler.emit calls _current_trace_context which reads
    _active_trace.get(). Without Bridge 1, trace_id is None.
    """
    w3c_trace_hex = W3C_EXAMPLE_TRACE_HEX
    monkeypatch.setattr(
        langgraph_adapter,
        "_build_otel_traceparent",
        lambda: _make_w3c_traceparent(w3c_trace_hex),
    )

    handler = LangGraphCallbackHandler("test-agent")
    run_id = uuid.uuid4()

    handler.on_chain_start(
        _chain_serialized("MyGraph"),
        {"messages": []},
        run_id=run_id,
    )

    # Simulate the MorseHandler's read path — same function the
    # handler uses inside emit().
    log_trace_id, log_span_id = _current_trace_context()
    assert log_trace_id == w3c_trace_hex, (
        "log line inside chain must carry the chain's (W3C-adopted) trace_id"
    )
    # span_id is the current top of the span stack — the root chain's span.
    assert log_span_id is not None
    assert len(log_span_id) == 32  # SDK span_id format

    handler.on_chain_end({"output": "done"}, run_id=run_id)

    # After chain end, no active trace — log lines after this go uncorrelated
    # (by design, matches the parent's auto-trace semantics).
    log_trace_id_after, _ = _current_trace_context()
    assert log_trace_id_after is None


# ---------------------------------------------------------------------------
# Bridge 2 — _build_otel_traceparent unit tests (without OTel installed)
# ---------------------------------------------------------------------------


@pytest.mark.req("SDK-09")
def test_build_otel_traceparent_returns_none_without_otel():
    """When opentelemetry isn't importable, the helper returns None.

    The SDK ships without an OTel dependency; vanilla LangChain scripts hit
    this path. The fallback to uuid4() generation is what makes this safe.
    """
    # If OTel is installed in the test env, we just assert the function
    # doesn't crash. If it's not, we assert None.
    # Either way the helper must be exception-safe.
    result = langgraph_adapter._build_otel_traceparent()
    # In the SDK test env, OTel is not a dependency. Confirm None.
    # (If a future change adds OTel as a dev dep, this test still passes
    # when no W3C context is active — get_current_span returns a
    # non-recording span with is_valid=False.)
    assert result is None or (
        isinstance(result, str) and result.startswith("00-") and len(result) > 30
    )


@pytest.mark.req("SDK-09")
def test_build_otel_traceparent_handles_invalid_context(
    monkeypatch: pytest.MonkeyPatch,
):
    """A stubbed OTel module with an invalid SpanContext must yield None."""

    class _FakeCtx:
        is_valid = False
        trace_id = 0
        span_id = 0

    class _FakeSpan:
        def get_span_context(self):  # noqa: D401
            return _FakeCtx()

    class _FakeOTelTrace:
        @staticmethod
        def get_current_span():
            return _FakeSpan()

    import sys

    fake_module = type(sys)("opentelemetry")
    fake_trace_module = type(sys)("opentelemetry.trace")
    fake_trace_module.get_current_span = _FakeOTelTrace.get_current_span  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opentelemetry", fake_module)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", fake_trace_module)
    monkeypatch.setattr(fake_module, "trace", fake_trace_module, raising=False)

    assert langgraph_adapter._build_otel_traceparent() is None


@pytest.mark.req("SDK-09")
def test_build_otel_traceparent_formats_valid_context(
    monkeypatch: pytest.MonkeyPatch,
):
    """A valid OTel SpanContext must serialize as a W3C traceparent string."""

    class _FakeCtx:
        is_valid = True
        trace_id = int(W3C_EXAMPLE_TRACE_HEX, 16)
        span_id = int(W3C_EXAMPLE_SPAN_HEX, 16)
        trace_flags = 1

    class _FakeSpan:
        def get_span_context(self):  # noqa: D401
            return _FakeCtx()

    import sys

    fake_module = type(sys)("opentelemetry")
    fake_trace_module = type(sys)("opentelemetry.trace")
    fake_trace_module.get_current_span = lambda: _FakeSpan()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opentelemetry", fake_module)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", fake_trace_module)
    monkeypatch.setattr(fake_module, "trace", fake_trace_module, raising=False)

    tp = langgraph_adapter._build_otel_traceparent()
    assert tp == f"00-{W3C_EXAMPLE_TRACE_HEX}-{W3C_EXAMPLE_SPAN_HEX}-01"


# ---------------------------------------------------------------------------
# Backward-compat — existing tests that wrap handler in `with run()` still pass
# ---------------------------------------------------------------------------


@pytest.mark.req("SDK-09")
def test_explicit_trace_wrap_still_works_no_bridge_interference():
    """Sanity: the pre-bridge call pattern (wrap in `with run():`) still emits
    a single trace and the handler's children are nested under it."""
    handler = LangGraphCallbackHandler("test-agent")
    run_id = uuid.uuid4()

    with tracing.run("test-agent"):
        handler.on_chain_start(
            _chain_serialized("MyGraph"),
            {"messages": []},
            run_id=run_id,
        )
        handler.on_chain_end({"output": "done"}, run_id=run_id)

    trace = tracing._get_last_trace()
    assert trace is not None
    # AID-04: the graph IS the root agent now; no second forked agent span.
    agent_spans = [s for s in trace["spans"] if s.get("type") == "agent"]
    assert [s["name"] for s in agent_spans] == ["MyGraph"]


# ---------------------------------------------------------------------------
# Smoke — the bridges don't break the LangGraph handler's other paths
# ---------------------------------------------------------------------------


@pytest.mark.req("SDK-09")
def test_bridge_does_not_break_logging_handler_attach_via_root_logger(
    monkeypatch: pytest.MonkeyPatch,
):
    """Attach the MorseHandler to a logger and verify a log call inside
    the open chain ships a record (in-memory; no real transport).

    Belt-and-suspenders for the end-to-end story — the failure mode this
    catches is the handler raising on missing trace context (it shouldn't).
    """
    from amund.logging import MorseHandler

    captured: list[dict[str, Any]] = []

    def _fake_enqueue(payload):
        captured.append(payload)

    # Bypass the transport — capture payloads in-process.
    monkeypatch.setattr(
        "amund.logging._enqueue_log",
        _fake_enqueue,
    )
    monkeypatch.setattr(
        langgraph_adapter,
        "_build_otel_traceparent",
        lambda: _make_w3c_traceparent(W3C_EXAMPLE_TRACE_HEX),
    )

    test_logger = stdlib_logging.getLogger("amund.tests.bridge")
    test_logger.setLevel(stdlib_logging.INFO)
    test_logger.addHandler(MorseHandler())

    handler = LangGraphCallbackHandler("test-agent")
    run_id = uuid.uuid4()
    handler.on_chain_start(
        _chain_serialized("MyGraph"),
        {"messages": []},
        run_id=run_id,
    )
    test_logger.info("hello from inside the chain")
    handler.on_chain_end({"output": "done"}, run_id=run_id)

    # Cleanup logger to keep other tests clean.
    for h in list(test_logger.handlers):
        test_logger.removeHandler(h)

    assert captured, "MorseHandler must have enqueued at least one record"
    # _enqueue_log receives the un-wrapped record payload (the {"logs": [...]}
    # envelope is added inside _enqueue_log itself before transport.track).
    record = captured[0]
    assert record["trace_id"] == W3C_EXAMPLE_TRACE_HEX
    assert record["body"] == "hello from inside the chain"
