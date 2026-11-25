"""MHQ-672 — the openai_agents adapter must never adopt a CLOSED trace.

Adapter file: `apps/sdk/python/src/amund/adapters/openai_agents.py`

`_prepare_call` reads the active trace from a ContextVar and attaches to it
rather than opening a new one. The ContextVar can still hold a trace that has
already been finalized and emitted: `TracedRun.__exit__` clears it with
`_active_trace.reset(token)`, but when `__enter__` and `__exit__` land in
different contexts that raises ValueError, and the fallback
`_active_trace.set(None)` writes to the wrong context. Python offers no way to
reset another context's var, so the stale state survives.

This adapter makes that reachable by construction — `_prepare_call` runs
synchronously in the caller's context while the matching `__exit__` runs inside
the coroutine returned by `_traced_runner_call`. Every span from the next
`Runner.run()` onward was then appended to an already-sent trace and silently
dropped: 28 of 33 conversations lost in the dogfood run that found this, with
no error at any log level, including `MORSE_SDK_DEBUG=1`.

The leak cannot be prevented. The fix makes it harmless, which is the same
guard `adapters/langchain.py` already carries — the adapter that dropped
nothing in the same dogfood session.

Deliberately NOT in `test_openai_agents_adapter.py`: that module is skipped
when the `openai-agents` package is absent, and this defect needs no `agents`
import to reproduce. The regression must stay covered wherever the suite runs.
"""

from __future__ import annotations

import pytest

from amund.adapters import openai_agents as adapter
from amund.tracing import _active_trace
from amund.tracing import run as av_run


class _FakeAgent:
    """Duck-typed stand-in — _prepare_call only reads ``.name``."""

    name = "TriageAgent"


@pytest.fixture(autouse=True)
def _clear_active_trace():
    _active_trace.set(None)
    yield
    _active_trace.set(None)


def _leak_a_closed_trace():
    """Reproduce the post-__exit__ state the ValueError branch leaves behind."""
    cm = av_run("FirstConversation")
    cm.__enter__()
    state = _active_trace.get()
    cm.__exit__(None, None, None)
    _active_trace.set(state)  # the leak: closed, but still the active trace
    assert state.is_closed
    return state


@pytest.mark.req("MHQ-672")
def test_closed_trace_is_not_adopted():
    leaked = _leak_a_closed_trace()

    state, _holder, auto_run_cm, root_span, _hooks = adapter._prepare_call(
        _FakeAgent(), None, {}
    )

    assert state is not leaked, "adopted an already-emitted trace — spans are dropped"
    assert auto_run_cm is not None, "must open a fresh trace instead"
    assert root_span.trace_id != leaked.trace_id
    assert not state.is_closed

    auto_run_cm.__exit__(None, None, None)


@pytest.mark.req("MHQ-672")
def test_each_repeated_call_gets_its_own_trace():
    """The reported shape: repeated Runner.run() in one long-lived process."""
    trace_ids = []
    for _ in range(5):
        state, _h, auto_run_cm, root_span, _hk = adapter._prepare_call(
            _FakeAgent(), None, {}
        )
        trace_ids.append(root_span.trace_id)
        assert auto_run_cm is not None
        auto_run_cm.__exit__(None, None, None)
        # Simulate the leak: __exit__ failed to clear this context.
        _active_trace.set(state)

    assert len(set(trace_ids)) == 5, f"traces collapsed into {len(set(trace_ids))}"


@pytest.mark.req("MHQ-672")
def test_an_open_caller_trace_is_still_adopted():
    """The guard must not break explicit `with amund.run(...)` nesting.

    Only *closed* traces are rejected; an open caller-supplied trace must still
    parent the run, or every adapter span becomes its own orphan trace.
    """
    with av_run("CallerSuppliedTrace"):
        caller_state = _active_trace.get()

        state, _h, auto_run_cm, root_span, _hk = adapter._prepare_call(
            _FakeAgent(), None, {}
        )

        assert state is caller_state, "an OPEN caller trace must still be adopted"
        assert auto_run_cm is None, "must not open a nested trace"
        assert root_span.trace_id == caller_state.trace_id
