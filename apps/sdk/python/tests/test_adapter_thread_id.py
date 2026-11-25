"""MHQ-1022 — adapters must carry conversation identity onto the trace.

Every trace from the openai_agents adapter landed with ``thread_id: null``,
so a five-turn ChatKit conversation showed up in the dashboard as five
unrelated traces with nothing linking them. No adapter populated the field;
the trace model and list schema had supported it all along.

Extraction is per-framework (a ``Session``, a run context, a LangGraph
``configurable.thread_id``); the write is shared — ``tracing.adopt_thread_id``.

No ``agents`` / ``langchain`` import here on purpose: this must stay covered
in environments where those optional packages are absent.
"""

from __future__ import annotations

import pytest

from amund.adapters import openai_agents as adapter
from amund.tracing import _active_trace, adopt_thread_id
from amund.tracing import run as av_run


class _FakeAgent:
    name = "TriageAgent"


class _Thread:
    """ChatKit ThreadMetadata stand-in — the id lives on ``.thread.id``."""

    def __init__(self, id: str) -> None:
        self.id = id


class _ChatContext:
    def __init__(self, thread_id: str) -> None:
        self.thread = _Thread(thread_id)


@pytest.fixture(autouse=True)
def _clear_active_trace():
    _active_trace.set(None)
    yield
    _active_trace.set(None)


def _prepare(**kwargs):
    return adapter._prepare_call(_FakeAgent(), None, kwargs)


@pytest.mark.req("MHQ-1022")
def test_chatkit_context_thread_id_lands_on_the_trace():
    state, _h, cm, _root, _hooks = _prepare(context=_ChatContext("thr_d2f4ef47"))
    assert state.thread_id == "thr_d2f4ef47"
    cm.__exit__(None, None, None)


@pytest.mark.req("MHQ-1022")
def test_five_turns_on_one_thread_share_a_thread_id():
    """The reported shape: 5 traces, 5 trace_ids, but ONE conversation."""
    context = _ChatContext("thr_d2f4ef47")
    threads, traces = set(), set()
    for _ in range(5):
        state, _h, cm, _root, _hooks = _prepare(context=context)
        threads.add(state.thread_id)
        traces.add(state.trace_id)
        cm.__exit__(None, None, None)
        _active_trace.set(None)

    assert threads == {"thr_d2f4ef47"}
    assert len(traces) == 5, "turns must stay separate traces, grouped by thread"


@pytest.mark.req("MHQ-1022")
def test_session_wins_over_context():
    class _Session:
        session_id = "sess-42"

    state, _h, cm, _root, _hooks = _prepare(
        session=_Session(), context=_ChatContext("thr_ignored")
    )
    assert state.thread_id == "sess-42"
    cm.__exit__(None, None, None)


@pytest.mark.req("MHQ-1022")
def test_conversation_id_kwarg_is_used():
    state, _h, cm, _root, _hooks = _prepare(conversation_id="conv_9")
    assert state.thread_id == "conv_9"
    cm.__exit__(None, None, None)


@pytest.mark.req("MHQ-1022")
def test_explicit_run_thread_id_is_never_overwritten():
    with av_run("chat-agent", thread_id="app-owned") as _r:
        state, _h, cm, _root, _hooks = _prepare(context=_ChatContext("thr_adapter"))
        assert cm is None, "should have joined the caller's trace"
        assert state.thread_id == "app-owned"


@pytest.mark.req("MHQ-1022")
def test_no_conversation_signal_leaves_it_null():
    """Absence is the signal — never invent an id for a one-shot run."""
    state, _h, cm, _root, _hooks = _prepare()
    assert state.thread_id is None
    cm.__exit__(None, None, None)


@pytest.mark.req("MHQ-1022")
def test_hostile_context_cannot_break_the_run():
    class _Exploding:
        @property
        def thread_id(self):
            raise RuntimeError("customer code")

    state, _h, cm, _root, _hooks = _prepare(context=_Exploding())
    assert state.thread_id is None
    cm.__exit__(None, None, None)


@pytest.mark.req("MHQ-1022")
def test_adopt_thread_id_outside_a_trace_is_a_noop():
    adopt_thread_id("thr_orphan")  # must not raise


@pytest.mark.req("MHQ-1022")
def test_adopt_thread_id_survives_a_raising_str():
    class _BadStr:
        def __str__(self):
            raise RuntimeError("boom")

    with av_run("chat-agent") as _r:
        adopt_thread_id(_BadStr())
        assert _active_trace.get().thread_id is None
