"""MHQ-980 — no reader may attach spans to an already-emitted trace.

MHQ-672 fixed one instance of this in the `openai_agents` adapter. The class of
bug survived: 50 of 51 `_active_trace.get()` readers took the raw ContextVar,
where a `_TraceState` whose trace has already been finalized is indistinguishable
from a live one. The fix is a single accessor, `tracing.get_active_trace()`,
which every reader now goes through.

The leak itself is unfixable and these tests do not pretend otherwise — they
reproduce it (`_leak_a_closed_trace`) and assert it is *harmless*.
`TracedRun.__exit__` clears the var with `_active_trace.reset(token)`; when
`__enter__` and `__exit__` land in different contexts that raises ValueError and
the fallback `_active_trace.set(None)` writes to the wrong context. Python has
no way to reset another context's ContextVar.

Companion assertions matter as much as the guard ones: a guard that answers
"no trace" too eagerly would orphan every span, which is a worse bug than the
one being fixed. Each guarded behaviour therefore has an open-trace twin.
"""

from __future__ import annotations

import logging

import pytest

from morse_ai import tracing
from morse_ai.tracing import _active_trace, get_active_trace
from morse_ai.tracing import run as av_run


@pytest.fixture(autouse=True)
def _clear_active_trace():
    _active_trace.set(None)
    tracing._reset_span_outside_trace_warning()
    yield
    _active_trace.set(None)
    tracing._reset_span_outside_trace_warning()


def _leak_a_closed_trace():
    """Reproduce the post-__exit__ state the ValueError branch leaves behind."""
    cm = av_run("FirstConversation")
    cm.__enter__()
    state = _active_trace.get()
    cm.__exit__(None, None, None)
    _active_trace.set(state)  # the leak: closed, but still the active trace
    assert state.is_closed
    return state


# ---------------------------------------------------------------------------
# The accessor itself
# ---------------------------------------------------------------------------


def test_get_active_trace_hides_a_closed_trace():
    state = _leak_a_closed_trace()

    assert _active_trace.get() is state, "precondition: the leak is present"
    assert get_active_trace() is None


def test_get_active_trace_returns_an_open_trace():
    with av_run("live"):
        assert get_active_trace() is not None


def test_get_active_trace_returns_none_when_there_is_no_trace():
    assert get_active_trace() is None


# ---------------------------------------------------------------------------
# span() — the site where the drop actually happened
# ---------------------------------------------------------------------------


def test_span_is_not_appended_to_a_closed_trace():
    state = _leak_a_closed_trace()
    before = len(state.completed_spans)

    with tracing.span("tool-call", span_type="tool"):
        pass

    # The span went nowhere, which is correct: its trace was already sent.
    # What must NOT happen is it landing on the dead trace's span list, where
    # it is accounted for and never transmitted.
    assert len(state.completed_spans) == before


def test_span_is_still_appended_to_an_open_trace():
    """The over-correction guard: the fix must not orphan live spans."""
    with av_run("live"), tracing.span("tool-call", span_type="tool"):
        pass
    trace = tracing._get_last_trace()
    assert [s["name"] for s in trace["spans"] if s["type"] == "tool"] == ["tool-call"]


def test_span_on_a_closed_trace_warns_once(caplog):
    """MHQ-672 went unnoticed for weeks because nothing said anything.

    A span landing on a closed trace always means a leaked `_TraceState` — an
    SDK bug, never something the customer wrote — so it is worth a warning,
    and worth exactly one per process because this sits inside agent loops.
    """
    _leak_a_closed_trace()

    with caplog.at_level(logging.WARNING, logger="morse_ai"):
        with tracing.span("first"):
            pass
        with tracing.span("second"):
            pass

    warnings = [r for r in caplog.records if "already been finalized" in r.message]
    assert len(warnings) == 1
    assert "first" in warnings[0].getMessage()


def test_span_with_no_trace_at_all_warns_about_run_not_about_a_bug(caplog):
    """The two warnings must stay distinguishable — one is the customer's
    missing `run()`, the other is our leak. Sharing a latch would let the
    first mute the second."""
    with caplog.at_level(logging.WARNING, logger="morse_ai"), tracing.span("orphan"):
        pass

    messages = [r.getMessage() for r in caplog.records]
    assert any("no active trace" in m for m in messages)
    assert not any("already been finalized" in m for m in messages)


# ---------------------------------------------------------------------------
# set_context / adopt_thread_id — writes that would silently vanish
# ---------------------------------------------------------------------------


def test_set_context_does_not_write_to_a_closed_trace():
    state = _leak_a_closed_trace()

    tracing.set_context(customer_id="acme")

    assert state.user_metadata is None


def test_set_context_writes_to_an_open_trace():
    with av_run("live"):
        tracing.set_context(customer_id="acme")
        assert get_active_trace().user_metadata == {"customer_id": "acme"}


def test_adopt_thread_id_does_not_write_to_a_closed_trace():
    state = _leak_a_closed_trace()

    tracing.adopt_thread_id("thread-123")

    assert state.thread_id is None


def test_adopt_thread_id_writes_to_an_open_trace():
    with av_run("live"):
        tracing.adopt_thread_id("thread-123")
        assert get_active_trace().thread_id == "thread-123"


# ---------------------------------------------------------------------------
# trace_agent had no is_closed at all — its leak was undetectable by any guard
# ---------------------------------------------------------------------------


def test_trace_agent_marks_its_state_closed_on_exit():
    """`TracedRun.__exit__` set `is_closed`; the decorator never did.

    A state leaked from `@trace_agent` therefore stayed indistinguishable from
    a live trace even for the readers that DID check the flag — including the
    MHQ-672 guard. Every guard in this file is only as good as this line.
    """
    captured: list = []

    @tracing.trace_agent("decorated")
    def work():
        captured.append(_active_trace.get())

    work()

    assert captured[0].is_closed is True


def test_trace_agent_state_leaked_into_the_var_is_treated_as_no_trace():
    captured: list = []

    @tracing.trace_agent("decorated")
    def work():
        captured.append(_active_trace.get())

    work()
    _active_trace.set(captured[0])  # the leak

    assert get_active_trace() is None


def test_trace_agent_still_records_spans_while_it_is_running():
    """Over-correction guard for the flag change above."""

    @tracing.trace_agent("decorated")
    def work():
        with tracing.span("inner", span_type="tool"):
            pass

    work()

    trace = tracing._get_last_trace()
    assert [s["name"] for s in trace["spans"] if s["type"] == "tool"] == ["inner"]
