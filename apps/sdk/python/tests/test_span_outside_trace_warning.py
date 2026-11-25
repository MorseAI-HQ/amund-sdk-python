"""MHQ-806: span() outside a run must be a no-op that says so.

The no-op itself is correct — library code may be instrumented but reached from
an untraced path, and raising there would be worse than recording nothing. The
defect was that it was *silent*: every method on the returned handle discarded
its argument, with no error, no spans, and nothing to distinguish it from
working instrumentation. Writing `amund.span(...)` on its own is a natural
first read of the API, and it cost real debugging time during the 0.4.0
release-candidate acceptance test.

These lock in three things: the warning fires, it names the fix, and it fires
at most once per process (this sits inside request and agent loops, so a
per-call warning would bury the signal it exists to provide).
"""

import logging

import pytest

import amund
from amund.tracing import _reset_span_outside_trace_warning


@pytest.fixture(autouse=True)
def _clear_latch():
    _reset_span_outside_trace_warning()
    yield
    _reset_span_outside_trace_warning()


def test_span_outside_trace_warns(caplog):
    with (
        caplog.at_level(logging.WARNING, logger="amund"),
        amund.span("orphan", span_type="llm"),
    ):
        pass

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    # Names the offending span so the user can find it...
    assert "orphan" in message
    # ...and names the fix, not just the symptom.
    assert "amund.run" in message


def test_span_outside_trace_warns_only_once_per_process(caplog):
    with caplog.at_level(logging.WARNING, logger="amund"):
        for i in range(50):
            with amund.span(f"orphan-{i}"):
                pass

    assert len(caplog.records) == 1, "warning must be latched, not per-call"


def test_span_outside_trace_is_still_a_no_op(caplog):
    """Warning must not change behaviour — no raise, setters still absorb."""
    with (
        caplog.at_level(logging.WARNING, logger="amund"),
        amund.span("orphan") as s,
    ):
        s.set_output("discarded")
        s.set_metadata("k", "v")
        s.mark_failed("still fine")

    # Reaching here at all is the assertion: nothing raised.
    assert len(caplog.records) == 1


def test_no_warning_inside_a_run(caplog, monkeypatch):
    """The common, correct usage must stay quiet."""
    monkeypatch.setenv("MORSE_DISABLED", "1")
    with (
        caplog.at_level(logging.WARNING, logger="amund"),
        amund.run("agent"),
        amund.span("child", span_type="llm"),
    ):
        pass

    warnings = [r for r in caplog.records if "no active trace" in r.getMessage()]
    assert warnings == []
