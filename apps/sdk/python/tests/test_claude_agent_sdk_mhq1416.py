"""MHQ-1416 — claude-agent-sdk adapter: tool error detail + sub-agent parenting.

Dogfood round 2026-10-18 (claude-agent-sdk 0.2.168). Measured live order of
events for one lead + ``reader`` sub-agent run reading a missing file:

    PreToolUse(Agent) -> SubagentStart -> PostToolUse(Agent)   # at LAUNCH
    -> PreToolUse(Read, agent_id) -> PostToolUseFailure(Read) -> PostToolUse(Read)
    -> ResultMessage

and every message the sub-agent produces carries ``parent_tool_use_id`` (the
spawning Agent tool_use_id). Mock classes are matched by ``__name__``.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from amund import tracing
from amund.adapters import claude_agent_sdk as adapter


class TextBlock:
    def __init__(self, text):
        self.text = text


class ToolUseBlock:
    def __init__(self, name, inp=None, id=None):
        self.id = id
        self.name = name
        self.input = inp


class ToolResultBlock:
    def __init__(self, tool_use_id, content=None, is_error=False):
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error


class AssistantMessage:
    def __init__(self, content, model="claude-haiku-4-5", parent_tool_use_id=None):
        self.content = content
        self.usage = {"input_tokens": 10, "output_tokens": 5}
        self.model = model
        self.parent_tool_use_id = parent_tool_use_id


class ResultMessage:
    is_error = False
    total_cost_usd = None


def _reset():
    adapter._pending_task_tids.clear()
    adapter._agent_to_task_tid.clear()
    adapter._task_span_ids.clear()
    adapter._open_agent_spans.clear()
    adapter._task_id_to_task_tid.clear()
    adapter._tool_timers.clear()
    adapter._agent_span_ids.clear()
    adapter._task_agent_span_ids.clear()
    adapter._tool_spans_by_tid.clear()


@pytest.fixture(autouse=True)
def _clean():
    _reset()
    yield
    _reset()


def _call(fn, payload, tid):
    asyncio.run(fn(payload, tid, None))


def _tools(trace):
    return [s for s in trace["spans"] if s.get("type") == "tool"]


def _sub_agents(trace):
    return [
        s for s in trace["spans"] if s["type"] == "agent" and s.get("parent_span_id")
    ]


# --- item 1: tool error detail ------------------------------------------------


def test_failure_hook_records_error_message_and_exception_type():
    err = "File does not exist. Note: your current working directory is /w."
    with tracing.run("t-err"):
        _call(
            adapter._av_pre_tool_use,
            {"tool_name": "Read", "tool_use_id": "t1", "tool_input": {}},
            "t1",
        )
        _call(
            adapter._av_post_tool_use_failure,
            {
                "tool_name": "Read",
                "tool_use_id": "t1",
                "tool_input": {"file_path": "/w/x"},
                "error": err,
            },
            "t1",
        )
    (span,) = _tools(tracing._get_last_trace())
    assert span["status"] == "failed"
    assert span["error_message"] == err
    assert span["exception_type"] == "ReadToolError"


def test_failure_error_message_is_bounded_to_500_chars():
    with tracing.run("t-bound"):
        _call(
            adapter._av_post_tool_use_failure,
            {"tool_name": "Bash", "tool_use_id": "t2", "error": "x" * 5000},
            "t2",
        )
    (span,) = _tools(tracing._get_last_trace())
    assert len(span["error_message"]) == 500


@pytest.mark.parametrize("failure_first", [True, False])
def test_failure_and_post_hook_for_one_call_yield_one_failed_span(failure_first):
    """The CLI fires BOTH PostToolUseFailure and PostToolUse for a failed call."""
    base = {"tool_name": "Read", "tool_use_id": "t3", "tool_input": {}}
    fail = (adapter._av_post_tool_use_failure, {**base, "error": "boom"})
    ok = (adapter._av_post_tool_use, {**base, "tool_response": "boom"})
    with tracing.run("t-dup"):
        for fn, payload in [fail, ok] if failure_first else [ok, fail]:
            _call(fn, payload, "t3")
    (span,) = _tools(tracing._get_last_trace())
    assert span["status"] == "failed"
    assert span["error_message"] == "boom"


def test_stream_fallback_marks_failed_from_later_user_message_result():
    """Fallback path: ToolResultBlock(is_error) arrives in a LATER UserMessage."""
    with tracing.run("t-fb"):
        state = tracing._active_trace.get()
        adapter._emit_tool_spans_from_stream(
            [ToolUseBlock("Read", {"file_path": "/w/x"}, id="tu-1")], state
        )
        adapter._apply_tool_results_from_stream(
            [ToolResultBlock("tu-1", "File does not exist.", is_error=True)]
        )
    (span,) = _tools(tracing._get_last_trace())
    assert span["status"] == "failed"
    assert span["error_message"] == "File does not exist."
    assert span["exception_type"] == "ReadToolError"


# --- item 2: sub-agent parenting ----------------------------------------------


def _launch_subagent(task_tid, agent_id):
    _call(
        adapter._av_pre_tool_use,
        {"tool_name": "Agent", "tool_use_id": task_tid, "tool_input": {}},
        task_tid,
    )
    asyncio.run(
        adapter._av_subagent_start(
            {"agent_id": agent_id, "agent_type": "reader"}, None, None
        )
    )
    _call(
        adapter._av_post_tool_use,
        {
            "tool_name": "Agent",
            "tool_use_id": task_tid,
            "tool_input": {},
            "tool_response": "launched",
        },
        task_tid,
    )


@pytest.mark.req("AID-05")  # Claude Agent SDK sub-agent parents (group J finding 2)
def test_subagent_tool_span_nests_under_agent_span_after_launch_hook():
    with tracing.run("t-parent"):
        _launch_subagent("task-1", "agent-1")
        sub = {"tool_name": "Read", "tool_input": {}, "agent_id": "agent-1"}
        _call(adapter._av_pre_tool_use, {**sub, "tool_use_id": "r1"}, "r1")
        _call(
            adapter._av_post_tool_use_failure,
            {**sub, "tool_use_id": "r1", "error": "nope"},
            "r1",
        )
        adapter._handle_task_lifecycle_message(
            ResultMessage(), "ResultMessage", tracing._active_trace.get()
        )
    trace = tracing._get_last_trace()
    (agent,) = _sub_agents(trace)
    (read,) = [s for s in _tools(trace) if s["name"] == "Read"]
    assert agent["name"] == "reader"
    assert read["parent_span_id"] == agent["span_id"]


def test_agent_span_survives_launch_hook_and_is_finalized_by_result_message():
    with tracing.run("t-final"):
        _launch_subagent("task-2", "agent-2")
        assert "agent-2" in adapter._open_agent_spans
        adapter._handle_task_lifecycle_message(
            ResultMessage(), "ResultMessage", tracing._active_trace.get()
        )
        assert not adapter._open_agent_spans
    assert len(_sub_agents(tracing._get_last_trace())) == 1


@pytest.mark.req("AID-05")  # Claude Agent SDK sub-agent parents (group J finding 2)
def test_subagent_llm_span_nests_under_agent_span():
    with tracing.run("t-llm"):
        _launch_subagent("task-3", "agent-3")
        adapter._emit_llm_span_from_assistant_msg(
            AssistantMessage([TextBlock("hi")], parent_tool_use_id="task-3"),
            time.monotonic(),
        )
        adapter._emit_llm_span_from_assistant_msg(
            AssistantMessage([TextBlock("lead")]), time.monotonic()
        )
        adapter._handle_task_lifecycle_message(
            ResultMessage(), "ResultMessage", tracing._active_trace.get()
        )
    spans = tracing._get_last_trace()["spans"]
    (agent,) = _sub_agents({"spans": spans})
    root = next(
        s for s in spans if s["type"] == "agent" and not s.get("parent_span_id")
    )
    llms = [s for s in spans if s["type"] == "llm"]
    assert [s["parent_span_id"] for s in llms] == [agent["span_id"], root["span_id"]]


def test_stream_fallback_parents_subagent_tool_under_given_agent_span():
    with tracing.run("t-fb-parent"):
        state = tracing._active_trace.get()
        adapter._emit_tool_spans_from_stream(
            [ToolUseBlock("Read", {}, id="tu-9")], state, "agent-span-9"
        )
    (span,) = _tools(tracing._get_last_trace())
    assert span["parent_span_id"] == "agent-span-9"


# --- ingest ordering: a parent must precede its children in the payload --------


def test_order_parents_first_moves_agent_span_ahead_of_its_children():
    """Ingest rewrites a forward-referenced parent to the sentinel root
    (measured on dev), so children must never precede their parent."""
    with tracing.run("t-order"):
        state = tracing._active_trace.get()
        root = state.span_stack[-1].span_id
        agent = adapter.SpanData(
            trace_id=state.trace_id,
            type=adapter.SpanType.AGENT,
            name="reader",
            parent_span_id=root,
        )
        read = adapter.SpanData(
            trace_id=state.trace_id,
            type=adapter.SpanType.TOOL,
            name="Read",
            parent_span_id=agent.span_id,
        )
        llm = adapter.SpanData(
            trace_id=state.trace_id,
            type=adapter.SpanType.LLM,
            name="llm",
            parent_span_id=root,
        )
        state.completed_spans.extend([read, llm, agent])  # completion order
        adapter._order_parents_first(state)
        names = [s.name for s in state.completed_spans]
        assert names.index("reader") < names.index("Read")
        assert names == ["llm", "reader", "Read"]  # otherwise stable


def test_order_parents_first_is_stable_and_cycle_safe():
    with tracing.run("t-order2"):
        state = tracing._active_trace.get()
        a = adapter.SpanData(
            trace_id=state.trace_id, type=adapter.SpanType.TOOL, name="a"
        )
        b = adapter.SpanData(
            trace_id=state.trace_id, type=adapter.SpanType.TOOL, name="b"
        )
        a.parent_span_id, b.parent_span_id = b.span_id, a.span_id  # cycle
        state.completed_spans.extend([a, b])
        adapter._order_parents_first(state)
        assert [s.name for s in state.completed_spans] == ["a", "b"]
