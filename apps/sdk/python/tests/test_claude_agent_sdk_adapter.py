"""Tests for the claude-agent-sdk adapter.

These tests do NOT require claude_agent_sdk installed — the adapter classifies
content blocks by ``type(x).__name__``, so lightweight mock classes named
TextBlock / ThinkingBlock / ToolUseBlock / AssistantMessage are sufficient.

Covers the input/output capture + extended-thinking dedupe behaviour added
2026-06-22 (the research-agent dogfood fix).
"""

from __future__ import annotations

import asyncio as _asyncio

import pytest

from morse_ai import tracing
from morse_ai.adapters import claude_agent_sdk as adapter

# ---------------------------------------------------------------------------
# Mock content blocks / messages (matched by class __name__)
# ---------------------------------------------------------------------------


class TextBlock:
    def __init__(self, text: str):
        self.text = text


class ThinkingBlock:
    def __init__(self, thinking: str = "internal reasoning"):
        self.thinking = thinking


class ToolUseBlock:
    def __init__(self, name: str, inp: dict | None = None, id: str | None = None):
        self.id = id
        self.name = name
        self.input = inp


class ToolResultBlock:
    def __init__(self, tool_use_id: str, content=None):
        self.tool_use_id = tool_use_id
        self.content = content


class AssistantMessage:
    def __init__(self, content: list, usage: dict | None = None, model: str = ""):
        self.content = content
        self.usage = usage or {}
        self.model = model


def _usage(input_tokens=10, output_tokens=5, cache_write=0, cache_read=0) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cache_read,
    }


# ---------------------------------------------------------------------------
# _extract_content_text
# ---------------------------------------------------------------------------


def test_extract_content_text_from_text_block():
    assert adapter._extract_content_text([TextBlock("hello world")]) == "hello world"


def test_extract_content_text_includes_tool_use():
    out = adapter._extract_content_text(
        [TextBlock("calling tool"), ToolUseBlock("WebSearch", {"q": "france"})]
    )
    assert "calling tool" in out
    assert "[tool:WebSearch]" in out
    assert "france" in out


def test_extract_content_text_skips_thinking_blocks():
    out = adapter._extract_content_text([ThinkingBlock(), TextBlock("answer")])
    assert out == "answer"


def test_extract_content_text_empty_returns_none():
    assert adapter._extract_content_text([]) is None
    assert adapter._extract_content_text(None) is None


# ---------------------------------------------------------------------------
# _is_thinking_only (extended-thinking dedupe guard)
# ---------------------------------------------------------------------------


def test_is_thinking_only_true_for_thinking_block_only():
    assert adapter._is_thinking_only([ThinkingBlock()]) is True


def test_is_thinking_only_false_when_text_present():
    assert adapter._is_thinking_only([ThinkingBlock(), TextBlock("hi")]) is False


def test_is_thinking_only_false_for_empty():
    assert adapter._is_thinking_only([]) is False
    assert adapter._is_thinking_only(None) is False


# ---------------------------------------------------------------------------
# LLM span emission — input/output capture + dict shaping
# ---------------------------------------------------------------------------


def test_emit_span_captures_input_and_output_as_dicts():
    msg = AssistantMessage(
        content=[TextBlock("Paris is the capital of France.")],
        usage=_usage(input_tokens=10, output_tokens=8, cache_write=20),
        model="claude-haiku-4-5-20251001",
    )
    token = adapter._pending_prompt.set("What is the capital of France?")
    try:
        with tracing.run("claude-agent-sdk"):
            adapter._emit_llm_span_from_assistant_msg(msg, 0.0)
        trace = tracing._get_last_trace()
    finally:
        adapter._pending_prompt.reset(token)

    assert trace is not None
    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    assert len(llm_spans) == 1
    span = llm_spans[0]

    # input_data must be a dict (server rejects bare strings with 422)
    assert isinstance(span["input_data"], dict)
    assert span["input_data"]["messages"][0]["content"] == (
        "What is the capital of France?"
    )
    # output_data must be a dict carrying the assistant text
    assert isinstance(span["output_data"], dict)
    assert "Paris" in span["output_data"]["content"]
    # tokens: input includes cache-write; output as reported
    assert span["input_tokens"] == 30  # 10 + 20 cache-write
    assert span["output_tokens"] == 8


def test_emit_span_without_pending_prompt_has_null_input():
    msg = AssistantMessage(
        content=[TextBlock("answer")],
        usage=_usage(),
        model="claude-haiku-4-5-20251001",
    )
    # ensure no leftover prompt from another test
    token = adapter._pending_prompt.set(None)
    try:
        with tracing.run("claude-agent-sdk"):
            adapter._emit_llm_span_from_assistant_msg(msg, 0.0)
        trace = tracing._get_last_trace()
    finally:
        adapter._pending_prompt.reset(token)

    llm_spans = [s for s in trace["spans"] if s.get("type") == "llm"]
    assert len(llm_spans) == 1
    assert llm_spans[0].get("input_data") is None
    assert llm_spans[0]["output_data"]["content"] == "answer"


def test_emit_span_no_active_trace_is_noop():
    msg = AssistantMessage(content=[TextBlock("x")], usage=_usage(), model="m")
    # Outside any tracing.run() — must not raise.
    adapter._emit_llm_span_from_assistant_msg(msg, 0.0)


# ---------------------------------------------------------------------------
# A3 — tool span emission via hooks
# ---------------------------------------------------------------------------


def _make_post_hook_input(
    tool_name="Bash", tool_use_id="uid-1", tool_input=None, tool_response="ok"
):
    return {
        "hook_event_name": "PostToolUse",
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
        "tool_input": tool_input or {"command": "echo hi"},
        "tool_response": tool_response,
    }


@pytest.mark.req("ASKL-10")
def test_pre_post_hook_emits_tool_span():
    pre = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_use_id": "uid-2",
        "tool_input": {"command": "ls"},
    }
    post = _make_post_hook_input(tool_use_id="uid-2", tool_response="file.txt")

    with tracing.run("test-agent"):
        _asyncio.run(adapter._av_pre_tool_use(pre))
        _asyncio.run(adapter._av_post_tool_use(post))
    trace = tracing._get_last_trace()

    tool_spans = [s for s in trace["spans"] if s.get("type") == "tool"]
    assert len(tool_spans) == 1
    s = tool_spans[0]
    assert s["name"] == "Bash"
    assert s["status"] == "completed"
    assert "file.txt" in s["output_data"]["response"]


@pytest.mark.req("ASKL-10")
def test_task_tool_classified_as_spawn():
    pre = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Agent",
        "tool_use_id": "uid-3",
        "tool_input": {"description": "search"},
    }
    post = _make_post_hook_input(
        tool_name="Agent", tool_use_id="uid-3", tool_response="done"
    )

    with tracing.run("test-agent"):
        _asyncio.run(adapter._av_pre_tool_use(pre))
        _asyncio.run(adapter._av_post_tool_use(post))
    trace = tracing._get_last_trace()

    tool_spans = [s for s in trace["spans"] if s.get("type") == "tool"]
    assert len(tool_spans) == 1
    assert tool_spans[0]["metadata"]["askl_is_spawn"] is True


@pytest.mark.req("ASKL-10")
def test_failure_hook_emits_failed_status():
    pre = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_use_id": "uid-4",
        "tool_input": {},
    }
    failure = {
        "hook_event_name": "PostToolUseFailure",
        "tool_name": "Bash",
        "tool_use_id": "uid-4",
        "tool_input": {},
        "error": "permission denied",
    }

    with tracing.run("test-agent"):
        _asyncio.run(adapter._av_pre_tool_use(pre))
        _asyncio.run(adapter._av_post_tool_use_failure(failure))
    trace = tracing._get_last_trace()

    tool_spans = [s for s in trace["spans"] if s.get("type") == "tool"]
    assert len(tool_spans) == 1
    assert tool_spans[0]["status"] == "failed"


# ---------------------------------------------------------------------------
# C.3 — stream-parse fallback
# ---------------------------------------------------------------------------


@pytest.mark.req("ASKL-10")
def test_stream_fallback_emits_tool_span_with_paired_result():
    content = [
        ToolUseBlock("Bash", {"command": "ls"}, id="id-a"),
        ToolResultBlock("id-a", "README.md"),
    ]
    with tracing.run("test-agent"):
        state = tracing._active_trace.get()
        adapter._emit_tool_spans_from_stream(content, state)
    trace = tracing._get_last_trace()

    tool_spans = [s for s in trace["spans"] if s.get("type") == "tool"]
    assert len(tool_spans) == 1
    s = tool_spans[0]
    assert s["name"] == "Bash"
    assert s["input_data"]["input"]["command"] == "ls"
    assert "README.md" in s["output_data"]["response"]


@pytest.mark.req("ASKL-10")
def test_stream_fallback_emits_unpaired_tool_span_without_output():
    content = [ToolUseBlock("Read", {"file_path": "/tmp/x"}, id="id-b")]
    with tracing.run("test-agent"):
        state = tracing._active_trace.get()
        adapter._emit_tool_spans_from_stream(content, state)
    trace = tracing._get_last_trace()

    tool_spans = [s for s in trace["spans"] if s.get("type") == "tool"]
    assert len(tool_spans) == 1
    assert tool_spans[0].get("output_data") is None


@pytest.mark.req("ASKL-10")
def test_stream_fallback_classifies_task_as_spawn():
    content = [ToolUseBlock("Task", {"description": "do stuff"}, id="id-c")]
    with tracing.run("test-agent"):
        state = tracing._active_trace.get()
        adapter._emit_tool_spans_from_stream(content, state)
    trace = tracing._get_last_trace()

    tool_spans = [s for s in trace["spans"] if s.get("type") == "tool"]
    assert tool_spans[0]["metadata"]["askl_is_spawn"] is True


@pytest.mark.req("ASKL-10")
def test_stream_fallback_ignores_non_tool_blocks():
    content = [TextBlock("just text"), ThinkingBlock()]
    with tracing.run("test-agent"):
        state = tracing._active_trace.get()
        adapter._emit_tool_spans_from_stream(content, state)
    trace = tracing._get_last_trace()

    tool_spans = [s for s in trace["spans"] if s.get("type") == "tool"]
    assert len(tool_spans) == 0


@pytest.mark.req("ASKL-10")
def test_inject_av_hooks_merges_not_clobbers():
    """_inject_av_hooks must append AV matchers without removing customer hooks."""
    import sys
    import types as _types
    from unittest.mock import patch

    class FakeMatcher:
        def __init__(self, hooks=None):
            self.hooks = hooks or []
            self.matcher = None

    class FakeOptions:
        _av_hooks_injected = False

        def __init__(self):
            self.hooks = {"PreToolUse": [FakeMatcher(hooks=[lambda x: None])]}

    options = FakeOptions()

    fake_types_mod = _types.ModuleType("claude_agent_sdk.types")
    fake_types_mod.HookMatcher = FakeMatcher

    with patch.dict(sys.modules, {"claude_agent_sdk.types": fake_types_mod}):
        adapter._inject_av_hooks(options)

    assert len(options.hooks["PreToolUse"]) == 2
    assert "PostToolUse" in options.hooks
    assert "PostToolUseFailure" in options.hooks


# ---------------------------------------------------------------------------
# A4: sub-agent context propagation
# ---------------------------------------------------------------------------


@pytest.mark.req("ASKL-09")
def test_subagent_start_hook_maps_agent_to_task():
    """SubagentStart hook should map agent_id → pending task tool_use_id."""
    adapter._pending_task_tids.clear()
    adapter._agent_to_task_tid.clear()
    adapter._task_span_ids.clear()

    adapter._pending_task_tids.append("task-tid-1")
    result = _asyncio.run(
        adapter._av_subagent_start({"agent_id": "agent-42"}, None, None)
    )
    assert result == {}
    assert adapter._agent_to_task_tid.get("agent-42") == "task-tid-1"
    assert len(adapter._pending_task_tids) == 0

    adapter._agent_to_task_tid.clear()


@pytest.mark.req("ASKL-09")
def test_subagent_tool_span_parented_under_task_span():
    """Tool spans from a sub-agent should have parent_span_id = Task span_id."""
    adapter._pending_task_tids.clear()
    adapter._agent_to_task_tid.clear()
    adapter._task_span_ids.clear()

    # Simulate PreToolUse for Task
    task_tid = "task-tid-x"
    _asyncio.run(
        adapter._av_pre_tool_use(
            {"tool_use_id": task_tid, "tool_name": "Agent", "tool_input": {}},
            task_tid,
            None,
        )
    )
    task_span_id = adapter._task_span_ids.get(task_tid)
    assert task_span_id is not None, "pre-hook should allocate span_id for Agent spawn"

    # Simulate SubagentStart
    agent_id = "agent-77"
    _asyncio.run(adapter._av_subagent_start({"agent_id": agent_id}, None, None))

    # Simulate sub-agent PostToolUse (tool inside the sub-agent)
    sub_tid = "sub-tool-tid-1"
    _asyncio.run(
        adapter._av_pre_tool_use(
            {"tool_use_id": sub_tid, "tool_name": "Bash", "tool_input": {"cmd": "ls"}},
            sub_tid,
            None,
        )
    )
    with tracing.run("test-a4-agent"):
        _asyncio.run(
            adapter._av_post_tool_use(
                {
                    "tool_use_id": sub_tid,
                    "tool_name": "Bash",
                    "tool_input": {"cmd": "ls"},
                    "tool_response": "ok",
                    "agent_id": agent_id,
                },
                sub_tid,
                None,
            )
        )
    trace = tracing._get_last_trace()
    tool_spans = [s for s in trace["spans"] if s.get("type") == "tool"]
    assert len(tool_spans) == 1
    assert tool_spans[0]["parent_span_id"] == task_span_id, (
        f"sub-agent tool should be parented under Task span {task_span_id}, "
        f"got {tool_spans[0]['parent_span_id']}"
    )

    adapter._pending_task_tids.clear()
    adapter._agent_to_task_tid.clear()
    adapter._task_span_ids.clear()


@pytest.mark.req("ASKL-09", "ASKL-10")
def test_subagent_start_hook_wired_in_inject():
    """_inject_av_hooks must register SubagentStart alongside PreToolUse/PostToolUse."""
    import sys
    import types as _types
    from unittest.mock import patch

    class FakeMatcher:
        def __init__(self, hooks=None):
            self.hooks = hooks or []
            self.matcher = None

    class FakeOptions:
        _av_hooks_injected = False

        def __init__(self):
            self.hooks = None

    options = FakeOptions()
    fake_types_mod = _types.ModuleType("claude_agent_sdk.types")
    fake_types_mod.HookMatcher = FakeMatcher

    with patch.dict(sys.modules, {"claude_agent_sdk.types": fake_types_mod}):
        adapter._inject_av_hooks(options)

    assert "SubagentStart" in options.hooks
    assert "SubagentStop" in options.hooks


# ---------------------------------------------------------------------------
# C.1: named AGENT spans on SubagentStart
# ---------------------------------------------------------------------------


@pytest.mark.req("ASKL-09")
def test_subagent_start_emits_named_agent_span_nested_under_task():
    """One declared agent that actually spawns → one named AGENT span,
    parented under the Task span; sub-agent tool calls nest under the
    AGENT span, not directly under Task."""
    adapter._pending_task_tids.clear()
    adapter._agent_to_task_tid.clear()
    adapter._task_span_ids.clear()
    adapter._open_agent_spans.clear()

    with tracing.run("test-c1-agent"):
        task_tid = "task-tid-c1"
        _asyncio.run(
            adapter._av_pre_tool_use(
                {"tool_use_id": task_tid, "tool_name": "Task", "tool_input": {}},
                task_tid,
                None,
            )
        )
        task_span_id = adapter._task_span_ids.get(task_tid)

        agent_id = "agent-c1"
        _asyncio.run(
            adapter._av_subagent_start(
                {"agent_id": agent_id, "agent_type": "researcher"}, None, None
            )
        )
        assert agent_id in adapter._open_agent_spans
        agent_span = adapter._open_agent_spans[agent_id][0]
        assert agent_span.name == "researcher"
        assert agent_span.parent_span_id == task_span_id
        assert agent_span.cost_usd is None
        assert agent_span.total_tokens is None

        # Sub-agent's own tool call should nest under the AGENT span.
        sub_tid = "sub-tool-c1"
        _asyncio.run(
            adapter._av_pre_tool_use(
                {"tool_use_id": sub_tid, "tool_name": "WebSearch", "tool_input": {}},
                sub_tid,
                None,
            )
        )
        _asyncio.run(
            adapter._av_post_tool_use(
                {
                    "tool_use_id": sub_tid,
                    "tool_name": "WebSearch",
                    "tool_input": {},
                    "tool_response": "ok",
                    "agent_id": agent_id,
                },
                sub_tid,
                None,
            )
        )

        # Task completes — this must flush (finalize + append) the AGENT span.
        _asyncio.run(
            adapter._av_post_tool_use(
                {
                    "tool_use_id": task_tid,
                    "tool_name": "Task",
                    "tool_input": {},
                    "tool_response": "done",
                },
                task_tid,
                None,
            )
        )

    trace = tracing._get_last_trace()
    # Exclude the trace's own root "agent" span (parent_span_id is None) —
    # only sub-agent AGENT spans are under test here.
    agent_spans = [
        s
        for s in trace["spans"]
        if s.get("type") == "agent" and s.get("parent_span_id")
    ]
    assert len(agent_spans) == 1
    assert agent_spans[0]["name"] == "researcher"
    assert agent_spans[0]["parent_span_id"] == task_span_id

    tool_spans = {s["name"]: s for s in trace["spans"] if s.get("type") == "tool"}
    assert tool_spans["WebSearch"]["parent_span_id"] == agent_spans[0]["span_id"]

    assert agent_id not in adapter._open_agent_spans

    adapter._pending_task_tids.clear()
    adapter._agent_to_task_tid.clear()
    adapter._task_span_ids.clear()
    adapter._open_agent_spans.clear()


@pytest.mark.req("ASKL-09")
def test_subagent_start_no_span_for_declared_but_unspawned_agent():
    """Approach (b): a declared sub-agent that never spawns gets no span —
    only _av_subagent_start firing (a real spawn) opens one."""
    adapter._pending_task_tids.clear()
    adapter._agent_to_task_tid.clear()
    adapter._task_span_ids.clear()
    adapter._open_agent_spans.clear()

    with tracing.run("test-c1-no-spawn"):
        task_tid = "task-tid-c1b"
        _asyncio.run(
            adapter._av_pre_tool_use(
                {"tool_use_id": task_tid, "tool_name": "Task", "tool_input": {}},
                task_tid,
                None,
            )
        )
        # No _av_subagent_start call — declared agent never actually spawned.
        _asyncio.run(
            adapter._av_post_tool_use(
                {
                    "tool_use_id": task_tid,
                    "tool_name": "Task",
                    "tool_input": {},
                    "tool_response": "done",
                },
                task_tid,
                None,
            )
        )

    trace = tracing._get_last_trace()
    agent_spans = [
        s
        for s in trace["spans"]
        if s.get("type") == "agent" and s.get("parent_span_id")
    ]
    assert len(agent_spans) == 0

    adapter._pending_task_tids.clear()
    adapter._agent_to_task_tid.clear()
    adapter._task_span_ids.clear()
    adapter._open_agent_spans.clear()


@pytest.mark.req("ASKL-09")
def test_sanitize_agent_name_truncates_and_strips_control_chars():
    long_name = "x" * 200
    assert len(adapter._sanitize_agent_name(long_name)) == adapter._MAX_AGENT_NAME_LEN

    dirty = "resear\x00cher\x01"
    assert adapter._sanitize_agent_name(dirty) == "researcher"

    assert adapter._sanitize_agent_name("") == "agent"


# ---------------------------------------------------------------------------
# MHQ-762: _hook_trace_ref cleared before in-flight spawn hooks settle
# ---------------------------------------------------------------------------


@pytest.mark.req("ASKL-09")
def test_drain_pending_spawns_waits_for_in_flight_task_span(monkeypatch):
    """Should not return while a spawn's PostToolUse hook hasn't fired yet
    (i.e. its tool_use_id is still in _task_span_ids)."""
    monkeypatch.setattr(adapter, "_SPAWN_DRAIN_TIMEOUT_S", 1.0)
    monkeypatch.setattr(adapter, "_SPAWN_DRAIN_POLL_S", 0.01)
    adapter._task_span_ids.clear()
    adapter._task_span_ids["in-flight-tid"] = "span-1"

    async def _clear_after_delay():
        await _asyncio.sleep(0.05)
        adapter._task_span_ids.pop("in-flight-tid", None)

    async def _run():
        start = _asyncio.get_event_loop().time()
        clearer = _asyncio.ensure_future(_clear_after_delay())
        await adapter._drain_pending_spawns()
        elapsed = _asyncio.get_event_loop().time() - start
        await clearer
        return elapsed

    elapsed = _asyncio.run(_run())
    # Drained as soon as the span cleared, not held for the full timeout.
    assert 0.04 <= elapsed < 1.0
    assert adapter._task_span_ids == {}


@pytest.mark.req("ASKL-09")
def test_drain_pending_spawns_times_out_instead_of_hanging(monkeypatch):
    """A spawn hook that never fires (crash, unsupported SDK version) must not
    hang the response stream forever — bounded by _SPAWN_DRAIN_TIMEOUT_S."""
    monkeypatch.setattr(adapter, "_SPAWN_DRAIN_TIMEOUT_S", 0.1)
    monkeypatch.setattr(adapter, "_SPAWN_DRAIN_POLL_S", 0.01)
    adapter._task_span_ids.clear()
    adapter._task_span_ids["stuck-tid"] = "span-1"

    _asyncio.run(adapter._drain_pending_spawns())

    # Timed out — the stale entry is still there, but we didn't hang on it.
    assert "stuck-tid" in adapter._task_span_ids
    adapter._task_span_ids.clear()


@pytest.mark.req("ASKL-09")
def test_drain_pending_spawns_noop_when_nothing_in_flight(monkeypatch):
    monkeypatch.setattr(adapter, "_SPAWN_DRAIN_TIMEOUT_S", 1.0)
    monkeypatch.setattr(adapter, "_SPAWN_DRAIN_POLL_S", 0.01)
    adapter._task_span_ids.clear()

    async def _run():
        start = _asyncio.get_event_loop().time()
        await adapter._drain_pending_spawns()
        return _asyncio.get_event_loop().time() - start

    elapsed = _asyncio.run(_run())
    assert elapsed < 0.05


# ---------------------------------------------------------------------------
# MHQ-762 part 2: SubagentStop finalizes the wrapper span + named AGENT span
# ---------------------------------------------------------------------------


@pytest.mark.req("ASKL-09")
def test_subagent_stop_finalizes_wrapper_and_agent_spans():
    """PostToolUse never fires for the wrapper Task/Agent tool in current
    claude-agent-sdk versions (confirmed via live debug logging) — SubagentStop
    must finalize both the wrapper span and the named AGENT span itself."""
    adapter._pending_task_tids.clear()
    adapter._agent_to_task_tid.clear()
    adapter._task_span_ids.clear()
    adapter._open_agent_spans.clear()
    adapter._tool_timers.clear()

    task_tid = "task-tid-stop"
    agent_id = "agent-stop-1"

    with tracing.run("test-c1-subagent-stop"):
        _asyncio.run(
            adapter._av_pre_tool_use(
                {"tool_use_id": task_tid, "tool_name": "Agent", "tool_input": {}},
                task_tid,
                None,
            )
        )
        _asyncio.run(
            adapter._av_subagent_start(
                {"agent_id": agent_id, "agent_type": "researcher"}, None, None
            )
        )
        assert agent_id in adapter._open_agent_spans

        result = _asyncio.run(
            adapter._av_subagent_stop(
                {"agent_id": agent_id, "agent_type": "researcher"}, None, None
            )
        )
        assert result == {}

    assert agent_id not in adapter._open_agent_spans
    assert agent_id not in adapter._agent_to_task_tid
    assert task_tid not in adapter._task_span_ids

    trace = tracing._get_last_trace()
    span_names = [s.get("name") for s in trace["spans"]]
    span_types = {s.get("name"): s.get("type") for s in trace["spans"]}
    assert "Agent" in span_names
    assert span_types["Agent"] == "tool"
    assert span_types["researcher"] == "agent"

    adapter._pending_task_tids.clear()
    adapter._agent_to_task_tid.clear()
    adapter._task_span_ids.clear()
    adapter._open_agent_spans.clear()


@pytest.mark.req("ASKL-09")
def test_subagent_stop_noop_without_agent_id():
    adapter._open_agent_spans.clear()
    result = _asyncio.run(adapter._av_subagent_stop({}, None, None))
    assert result == {}


@pytest.mark.req("ASKL-09")
def test_subagent_stop_idempotent_if_already_finalized():
    """Calling SubagentStop twice (or after PostToolUse already cleaned up via
    the dead-code fallback) must not double-append spans or raise."""
    adapter._pending_task_tids.clear()
    adapter._agent_to_task_tid.clear()
    adapter._task_span_ids.clear()
    adapter._open_agent_spans.clear()

    agent_id = "agent-double-stop"
    with tracing.run("test-c1-subagent-stop-idempotent"):
        _asyncio.run(adapter._av_subagent_stop({"agent_id": agent_id}, None, None))
        # Second call — nothing left to finalize, must still no-op cleanly.
        result = _asyncio.run(
            adapter._av_subagent_stop({"agent_id": agent_id}, None, None)
        )
        assert result == {}


# ---------------------------------------------------------------------------
# MHQ-770: TaskNotificationMessage/TaskUpdatedMessage finalize spawn spans
# (the real completion signal — confirmed live that PostToolUse and
# SubagentStop never fire for the wrapper Task/Agent tool call)
# ---------------------------------------------------------------------------


class FakeTaskStartedMessage:
    def __init__(self, task_id, tool_use_id):
        self.task_id = task_id
        self.tool_use_id = tool_use_id


class FakeTaskNotificationMessage:
    def __init__(self, task_id, tool_use_id, status):
        self.task_id = task_id
        self.tool_use_id = tool_use_id
        self.status = status


class FakeTaskUpdatedMessage:
    def __init__(self, task_id, patch):
        self.task_id = task_id
        self.patch = patch


def _reset_spawn_state():
    adapter._pending_task_tids.clear()
    adapter._agent_to_task_tid.clear()
    adapter._task_span_ids.clear()
    adapter._open_agent_spans.clear()
    adapter._task_id_to_task_tid.clear()
    adapter._tool_timers.clear()


@pytest.mark.req("ASKL-09")
def test_task_notification_completed_finalizes_wrapper_and_agent_spans():
    _reset_spawn_state()
    task_tid = "task-tid-notif"
    agent_id = "agent-notif-1"

    with tracing.run("test-mhq770-notification"):
        _asyncio.run(
            adapter._av_pre_tool_use(
                {"tool_use_id": task_tid, "tool_name": "Agent", "tool_input": {}},
                task_tid,
                None,
            )
        )
        _asyncio.run(
            adapter._av_subagent_start(
                {"agent_id": agent_id, "agent_type": "researcher"}, None, None
            )
        )
        state = tracing._active_trace.get()
        adapter._handle_task_lifecycle_message(
            FakeTaskStartedMessage("task-123", task_tid), "TaskStartedMessage", state
        )
        assert adapter._task_id_to_task_tid["task-123"] == task_tid

        adapter._handle_task_lifecycle_message(
            FakeTaskNotificationMessage("task-123", task_tid, "completed"),
            "TaskNotificationMessage",
            state,
        )

    assert agent_id not in adapter._open_agent_spans
    assert task_tid not in adapter._task_span_ids
    assert "task-123" not in adapter._task_id_to_task_tid

    trace = tracing._get_last_trace()
    span_types = {s.get("name"): s.get("type") for s in trace["spans"]}
    assert span_types.get("Agent") == "tool"
    assert span_types.get("researcher") == "agent"

    _reset_spawn_state()


@pytest.mark.req("ASKL-09")
def test_task_updated_terminal_status_resolves_via_task_id_map():
    """Covers background tasks that skip TaskNotificationMessage entirely and
    report completion only via TaskUpdatedMessage (per the SDK's own docs)."""
    _reset_spawn_state()
    task_tid = "task-tid-updated"
    agent_id = "agent-updated-1"

    with tracing.run("test-mhq770-updated"):
        _asyncio.run(
            adapter._av_pre_tool_use(
                {"tool_use_id": task_tid, "tool_name": "Task", "tool_input": {}},
                task_tid,
                None,
            )
        )
        _asyncio.run(
            adapter._av_subagent_start(
                {"agent_id": agent_id, "agent_type": "data-analyst"}, None, None
            )
        )
        state = tracing._active_trace.get()
        adapter._handle_task_lifecycle_message(
            FakeTaskStartedMessage("task-456", task_tid), "TaskStartedMessage", state
        )
        # No TaskNotificationMessage this time — only TaskUpdatedMessage,
        # with the raw "killed" status a stopped background task reports.
        adapter._handle_task_lifecycle_message(
            FakeTaskUpdatedMessage("task-456", {"status": "killed"}),
            "TaskUpdatedMessage",
            state,
        )

    assert agent_id not in adapter._open_agent_spans
    assert task_tid not in adapter._task_span_ids

    trace = tracing._get_last_trace()
    span_types = {s.get("name"): s.get("type") for s in trace["spans"]}
    assert span_types.get("Agent") == "tool"
    assert span_types.get("data-analyst") == "agent"

    _reset_spawn_state()


@pytest.mark.req("ASKL-09")
def test_task_updated_non_terminal_status_does_not_finalize():
    _reset_spawn_state()
    task_tid = "task-tid-running"
    agent_id = "agent-running-1"

    with tracing.run("test-mhq770-running"):
        _asyncio.run(
            adapter._av_pre_tool_use(
                {"tool_use_id": task_tid, "tool_name": "Agent", "tool_input": {}},
                task_tid,
                None,
            )
        )
        _asyncio.run(
            adapter._av_subagent_start(
                {"agent_id": agent_id, "agent_type": "researcher"}, None, None
            )
        )
        state = tracing._active_trace.get()
        adapter._handle_task_lifecycle_message(
            FakeTaskStartedMessage("task-789", task_tid), "TaskStartedMessage", state
        )
        adapter._handle_task_lifecycle_message(
            FakeTaskUpdatedMessage("task-789", {"status": "running"}),
            "TaskUpdatedMessage",
            state,
        )

    # Still open — "running" is not terminal.
    assert agent_id in adapter._open_agent_spans
    assert task_tid in adapter._task_span_ids

    _reset_spawn_state()


@pytest.mark.req("ASKL-09")
def test_task_notification_no_state_is_noop():
    _reset_spawn_state()
    adapter._handle_task_lifecycle_message(
        FakeTaskNotificationMessage("task-x", "tid-x", "completed"),
        "TaskNotificationMessage",
        None,
    )
    # No exception, nothing to clean up either way.
    _reset_spawn_state()


@pytest.mark.req("ASKL-09")
def test_finalize_by_task_tid_idempotent_across_hook_and_stream_paths():
    """Both the SubagentStop hook (defensive fallback) and the stream-based
    TaskNotificationMessage path may fire for the same spawn — finalization
    must be a no-op the second time, not double-append spans."""
    _reset_spawn_state()
    task_tid = "task-tid-both"
    agent_id = "agent-both-1"

    with tracing.run("test-mhq770-idempotent"):
        _asyncio.run(
            adapter._av_pre_tool_use(
                {"tool_use_id": task_tid, "tool_name": "Agent", "tool_input": {}},
                task_tid,
                None,
            )
        )
        _asyncio.run(
            adapter._av_subagent_start(
                {"agent_id": agent_id, "agent_type": "researcher"}, None, None
            )
        )
        state = tracing._active_trace.get()
        # Stream path fires first...
        adapter._handle_task_lifecycle_message(
            FakeTaskNotificationMessage("task-both", task_tid, "completed"),
            "TaskNotificationMessage",
            state,
        )
        # ...then the hook fires too (both registered — real behavior may
        # vary by SDK version).
        _asyncio.run(adapter._av_subagent_stop({"agent_id": agent_id}, None, None))

    trace = tracing._get_last_trace()
    agent_spans = [s for s in trace["spans"] if s.get("name") == "Agent"]
    named_agent_spans = [s for s in trace["spans"] if s.get("name") == "researcher"]
    assert len(agent_spans) == 1
    assert len(named_agent_spans) == 1

    _reset_spawn_state()


# ---------------------------------------------------------------------------
# MHQ-783: ResultMessage fallback finalizes spawns that never got a terminal
# TaskNotificationMessage/TaskUpdatedMessage — confirmed live 2026-07-30 that
# claude-agent-sdk==0.2.106 can go a whole run without emitting either.
# ---------------------------------------------------------------------------


class FakeResultMessage:
    def __init__(self, is_error=False, total_cost_usd=None):
        self.is_error = is_error
        self.total_cost_usd = total_cost_usd


@pytest.mark.req("ASKL-09")
def test_result_message_finalizes_spawn_with_no_terminal_task_message():
    """MHQ-783 repro: TaskStartedMessage fires (spawn tracked) but neither
    TaskNotificationMessage nor TaskUpdatedMessage ever arrives — the spawn
    must still finalize when ResultMessage (end of turn) arrives, not stay
    open forever."""
    _reset_spawn_state()
    task_tid = "task-tid-result-fallback"
    agent_id = "agent-result-fallback-1"

    with tracing.run("test-mhq783-result-fallback"):
        _asyncio.run(
            adapter._av_pre_tool_use(
                {"tool_use_id": task_tid, "tool_name": "Agent", "tool_input": {}},
                task_tid,
                None,
            )
        )
        _asyncio.run(
            adapter._av_subagent_start(
                {"agent_id": agent_id, "agent_type": "researcher"}, None, None
            )
        )
        state = tracing._active_trace.get()
        adapter._handle_task_lifecycle_message(
            FakeTaskStartedMessage("task-fallback", task_tid),
            "TaskStartedMessage",
            state,
        )
        assert task_tid in adapter._task_span_ids  # still open — no terminal msg yet

        # No TaskNotificationMessage/TaskUpdatedMessage this time — only
        # ResultMessage, exactly the observed live regression.
        adapter._handle_task_lifecycle_message(
            FakeResultMessage(), "ResultMessage", state
        )

    assert agent_id not in adapter._open_agent_spans
    assert task_tid not in adapter._task_span_ids

    trace = tracing._get_last_trace()
    span_types = {s.get("name"): s.get("type") for s in trace["spans"]}
    assert span_types.get("Agent") == "tool"
    assert span_types.get("researcher") == "agent"

    _reset_spawn_state()


@pytest.mark.req("ASKL-09")
def test_result_message_noop_when_no_pending_spawns():
    """A run with no spawns (or where every spawn already finalized via a
    real terminal message) must not error or double-finalize on ResultMessage."""
    _reset_spawn_state()

    with tracing.run("test-mhq783-result-noop"):
        state = tracing._active_trace.get()
        adapter._handle_task_lifecycle_message(
            FakeResultMessage(), "ResultMessage", state
        )

    trace = tracing._get_last_trace()
    agent_spans = [s for s in trace["spans"] if s.get("name") == "Agent"]
    assert agent_spans == []

    _reset_spawn_state()


@pytest.mark.req("ASKL-09")
def test_result_message_does_not_reopen_already_finalized_spawn():
    """A spawn that already finalized via a real TaskNotificationMessage
    must not be touched again when ResultMessage arrives afterward."""
    _reset_spawn_state()
    task_tid = "task-tid-already-done"
    agent_id = "agent-already-done-1"

    with tracing.run("test-mhq783-already-finalized"):
        _asyncio.run(
            adapter._av_pre_tool_use(
                {"tool_use_id": task_tid, "tool_name": "Agent", "tool_input": {}},
                task_tid,
                None,
            )
        )
        _asyncio.run(
            adapter._av_subagent_start(
                {"agent_id": agent_id, "agent_type": "researcher"}, None, None
            )
        )
        state = tracing._active_trace.get()
        adapter._handle_task_lifecycle_message(
            FakeTaskNotificationMessage("task-done", task_tid, "completed"),
            "TaskNotificationMessage",
            state,
        )
        adapter._handle_task_lifecycle_message(
            FakeResultMessage(), "ResultMessage", state
        )

    trace = tracing._get_last_trace()
    agent_spans = [s for s in trace["spans"] if s.get("name") == "Agent"]
    named_agent_spans = [s for s in trace["spans"] if s.get("name") == "researcher"]
    assert len(agent_spans) == 1
    assert len(named_agent_spans) == 1

    _reset_spawn_state()
