"""claude-agent-sdk adapter — intercepts message stream for telemetry.

The Anthropic claude-agent-sdk (``claude-agent-sdk`` on PyPI) shells out to the
``claude`` CLI binary as a subprocess rather than calling
``anthropic.Anthropic().messages.create()`` inside the Python process. This means
the generic Anthropic adapter's ``install()`` never fires — no LLM spans, no
token counts, no cost.

This adapter wraps:
1. ``ClaudeSDKClient.receive_response()`` — captures LLM spans with token/cost.
2. ``ClaudeSDKClient.query()`` — captures the input prompt before it is
   serialised to the CLI subprocess stdin.
3. The module-level ``claude_agent_sdk.query()`` function — captures the prompt
   as (2) does AND emits LLM/tool spans off its own message stream, since it is
   a self-contained entry point that never reaches ``receive_response()``
   (MHQ-1053: it previously logged "adapter installed" and emitted nothing).
4. ``ClaudeSDKClient.__init__`` and the module-level ``query()`` — injects
   PreToolUse/PostToolUse hook callbacks so every tool invocation emits a
   first-class ``tool`` span (A3 — fixes flat orchestration tree).

Prompt interception works by storing the prompt string in a
``contextvars.ContextVar`` (``_pending_prompt``) when ``query()`` is called.
The ``receive_response()`` wrapper reads it when constructing the LLM span,
producing non-null ``input_data``.

``AssistantMessage.content`` blocks (TextBlock + ToolUseBlock) are extracted as
``output_data``, so both input and output are visible in the AV trace viewer.

Auto-detect integration: ``install()`` is registered in ``auto_detect.py`` so
the adapter fires automatically when ``claude_agent_sdk`` is first imported,
matching the zero-code-change setup story: ``MORSE_API_KEY=... pip install
morse_ai`` is a complete installation.

Zero-customer-code-change guarantee: all patches are at the class/module level.
Customer code is never modified.

Auto-trace creation: when a traced entry point is reached outside an active
``morse_ai.run()`` context, the adapter opens a ``TracedRun("claude-agent-sdk")``
for the duration of the response stream. When called inside an existing trace
(the customer already called ``with morse_ai.run(...)``), the adapter
piggybacks on that trace without opening a new one. Both entry points share one
consumer, ``_consume_traced`` — see its docstring for why that matters.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import time
from datetime import UTC, datetime
from typing import Any

from morse_ai._internal.active_framework import (
    clear_active_framework,
    set_active_framework,
)
from morse_ai._internal.errors import absorb_errors
from morse_ai.pricing import calculate_cost
from morse_ai.spans import SpanData, SpanType
from morse_ai.tracing import (
    TracedRun,
    _finalize_span,
    adopt_thread_id,
    get_active_trace,
)

logger = logging.getLogger("morse_ai")

_INSTALL_FLAG = "_morse_ai_installed"
_ORIGINAL_RECEIVE_RESPONSE = "_morse_ai_original_receive_response"
_ORIGINAL_CLIENT_QUERY = "_morse_ai_original_client_query"
_ORIGINAL_MODULE_QUERY = "_morse_ai_original_module_query"
_ORIGINAL_INIT = "_morse_ai_original_init"

# ponytail: global dict keyed by tool_use_id (unique per call) — no collision risk
_tool_timers: dict[str, tuple[float, str]] = {}

# A4: sub-agent context propagation.
# Maps task_tool_use_id → pre-allocated span_id so Task span is identifiable
# before PostToolUse fires (sub-agent tools need the parent span_id upfront).
_task_span_ids: dict[str, str] = {}
# task_tool_use_id of the most-recently-started Task tool (cleared on SubagentStart).
_pending_task_tids: list[str] = []  # stack for nested spawns
# agent_id (from SubagentStart) → task_tool_use_id that spawned it
_agent_to_task_tid: dict[str, str] = {}

# C.1: agent_id → (SpanData, start_mono) for the named AGENT span opened
# lazily on SubagentStart. Finalized when the spawning Task's PostToolUse
# fires (the sub-agent has necessarily finished by then). Sub-agent tool
# calls nest under this span instead of directly under the Task span.
_open_agent_spans: dict[str, tuple[Any, float]] = {}

# MHQ-770: task_id (from TaskStartedMessage, distinct from the tool_use_id/
# task_tid used everywhere else) → task_tid. TaskUpdatedMessage only carries
# task_id, so this is needed to resolve it back to the tool_use_id keying
# _task_span_ids/_agent_to_task_tid.
_task_id_to_task_tid: dict[str, str] = {}

# Terminal statuses across both TaskNotificationMessage.status and
# TaskUpdatedMessage.patch["status"] vocabularies (the SDK maps a killed
# background task to "stopped" for the notification but reports "killed"
# in the raw patch — treat both as terminal).
_TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "stopped", "killed"})

_MAX_AGENT_NAME_LEN = 128

# MHQ-1416: durable ids that outlive ``_open_agent_spans``. Confirmed live
# (2026-10-18, claude-agent-sdk 0.2.168) that the wrapper Agent tool's
# PostToolUse fires almost immediately (the sub-agent runs in the background),
# long before the sub-agent's own tool hooks and messages. Looking the parent
# up in the *open* maps therefore found nothing and the sub-agent's tool spans
# were detached. These maps are never popped on finalization (FIFO-capped).
_agent_span_ids: dict[str, str] = {}  # agent_id -> AGENT span_id
_task_agent_span_ids: dict[str, str] = {}  # task tool_use_id -> AGENT span_id
# tool_use_id -> emitted tool span, so the PostToolUseFailure / PostToolUse pair
# (both fire for one failed call) yields ONE span and the failure wins.
_tool_spans_by_tid: dict[str, Any] = {}
_BOUNDED_MAP_CAP = 2048

# Tool error text is bounded; redaction happens server-side.
_MAX_TOOL_ERROR_LEN = 500


def _remember(mapping: dict[str, Any], key: str, value: Any) -> None:
    """Insert into a FIFO-capped dict so a long-lived process cannot leak."""
    mapping[key] = value
    while len(mapping) > _BOUNDED_MAP_CAP:
        mapping.pop(next(iter(mapping)))


def _tool_error_text(error: Any) -> str | None:
    """Normalise a tool error (str or content-block list) to <=500 chars."""
    if error is None:
        return None
    if isinstance(error, list):
        parts = []
        for item in error:
            text = item.get("text") if isinstance(item, dict) else None
            parts.append(text if isinstance(text, str) else str(item))
        error = "\n".join(parts)
    text = error if isinstance(error, str) else str(error)
    text = text.strip()
    return text[:_MAX_TOOL_ERROR_LEN] or None


def _tool_error_type(tool_name: str) -> str:
    """Synthetic exception class name for a tool failure (no Python exc exists)."""
    cleaned = "".join(ch for ch in tool_name if ch.isalnum()) or "Tool"
    return f"{cleaned[:64]}ToolError"


def _mark_tool_span_failed(span: Any, tool_name: str, error: Any) -> None:
    span.status = "failed"
    message = _tool_error_text(error)
    if message:
        span.error_message = message
    span.exception_type = _tool_error_type(tool_name)


def _subagent_parent_span_id(msg: Any) -> str | None:
    """Parent span for a message emitted by a sub-agent.

    The CLI stamps ``parent_tool_use_id`` (the spawning Agent/Task tool call)
    on every message a sub-agent produces.
    """
    tid = getattr(msg, "parent_tool_use_id", None)
    if not tid:
        return None
    return _task_agent_span_ids.get(tid) or _task_span_ids.get(tid)


def _sanitize_agent_name(name: str) -> str:
    """Strip control chars and cap length — agent_type is developer-controlled
    (options.agents keys) and flows into fingerprint JSONB + orchestration stats."""
    cleaned = "".join(ch for ch in name if ch.isprintable())
    return cleaned[:_MAX_AGENT_NAME_LEN] or "agent"


# Hooks run in spawn_detached asyncio tasks that don't inherit ContextVars.
# This module-level reference lets hooks find the active trace without ContextVar.
# ponytail: single ref is fine — research-agent runs one query at a time
_hook_trace_ref: list[Any] = []  # [_TraceState] when active, [] otherwise

# LLM call start time set by traced_client_query before await original(), consumed
# by traced_receive_response to get accurate wall-clock start for the LLM span.
# ponytail: list-as-optional-cell, same pattern as _hook_trace_ref
_llm_call_start_ref: list[
    tuple[float, str]
] = []  # [(monotonic, iso)] when set, [] otherwise

# ContextVar that carries the captured prompt from the query() call into the
# receive_response() generator. Each asyncio Task gets its own value, so
# concurrent sub-agent calls don't bleed into each other.
_pending_prompt: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_av_pending_prompt", default=None
)


def _safe_serialize(value: Any) -> Any:
    """Return value if JSON-serializable primitive, else str()."""
    if isinstance(value, (dict, list, str, int, float, bool, type(None))):
        return value
    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return None


# MHQ-762: bounded window to let in-flight Task/Agent spawn hooks (running in
# detached asyncio tasks) settle before clearing _hook_trace_ref. A spawned
# sub-agent's own multi-turn conversation can still be finishing its PostToolUse
# hook after the parent response stream has already yielded its last message —
# clearing the ref before that hook fires silently drops the spawn span (and
# any SubagentStart-created AGENT span) since _get_hook_state() returns None.
_SPAWN_DRAIN_TIMEOUT_S = 5.0
_SPAWN_DRAIN_POLL_S = 0.05


async def _drain_pending_spawns() -> None:
    """Wait (bounded) for in-flight Task/Agent spawn tool calls to complete.

    ``_task_span_ids`` entries are set in ``_av_pre_tool_use`` when a spawn
    starts and popped in ``_av_post_tool_use`` when it completes — a non-empty
    dict means at least one spawn's PostToolUse hook hasn't fired yet. Times
    out rather than waiting forever in case a spawn's hook never fires (crash,
    SDK version without the hook).
    """
    deadline = time.monotonic() + _SPAWN_DRAIN_TIMEOUT_S
    while _task_span_ids and time.monotonic() < deadline:
        await asyncio.sleep(_SPAWN_DRAIN_POLL_S)


def _finalize_all_pending_spawns(state: Any) -> None:
    """MHQ-783: fallback finalizer for spawns that never got a terminal
    Task lifecycle message.

    Confirmed live (2026-07-30, ``claude-agent-sdk==0.2.106``) that
    ``TaskNotificationMessage``/``TaskUpdatedMessage`` — the two message
    types ``_handle_task_lifecycle_message`` treats as completion signals —
    can go missing entirely for a whole run; only ``TaskStartedMessage`` and
    the (non-terminal) ``TaskProgressMessage`` arrive. Without this fallback,
    sub-agent spawns silently never finalize and ``/agents`` shows only the
    root agent, regardless of how many real spawns happened.

    ``ResultMessage`` is the right trigger: the CLI only emits it once the
    entire turn — including every spawned sub-agent task — has actually
    finished, so any task_tid still open in ``_task_span_ids`` at that point
    is safe to finalize. End-time precision is coarser than a real per-task
    terminal signal (all pending spawns get ``ResultMessage``'s arrival time
    rather than their own actual completion time), but that's a strictly
    better trade than the spawn being invisible altogether.

    Idempotent — ``_finalize_spawn_by_task_tid`` pops as it goes, so calling
    this after some spawns already finalized via a real terminal message is
    a no-op for those.
    """
    pending = list(_task_span_ids)
    # MHQ-1416: a spawn whose wrapper span already closed (the Agent tool's
    # PostToolUse fires at launch) can still have an open AGENT span.
    pending += [t for t in _agent_to_task_tid.values() if t not in _task_span_ids]
    for task_tid in pending:
        _finalize_spawn_by_task_tid(task_tid, state)


def _handle_task_lifecycle_message(msg: Any, msg_type: str, state: Any) -> None:
    """MHQ-770/783: track/finalize Task spawns from the main response stream.

    Confirmed live (2026-07-30) that neither PostToolUse nor SubagentStop
    fire for the wrapper Task/Agent tool call in the currently-installed
    claude-agent-sdk — but TaskStartedMessage/TaskNotificationMessage/
    TaskUpdatedMessage all arrive in the main ``receive_response()`` stream
    regardless, independent of the hook-dispatch mechanism entirely.

    - TaskStartedMessage: record task_id -> tool_use_id (TaskUpdatedMessage
      only carries task_id, so this mapping is needed to resolve it later).
    - TaskNotificationMessage: carries tool_use_id directly — finalize if
      status is terminal. Not guaranteed for every task (background tasks
      may report only via TaskUpdatedMessage), so this alone isn't sufficient.
    - TaskUpdatedMessage: resolve task_id -> tool_use_id via the map above,
      finalize if patch.status is terminal. Covers tasks that skip the
      notification message.
    - ResultMessage (MHQ-783): fallback — finalize any spawn that still
      hasn't received either message above, since a future SDK version may
      stop emitting them altogether (confirmed happening on 0.2.106) rather
      than merely delaying them.

    Finalization is idempotent (see _finalize_spawn_by_task_tid), so a task
    that fires both messages is finalized once, harmlessly, on the second.
    """
    if msg_type == "TaskStartedMessage":
        task_id = getattr(msg, "task_id", None)
        tool_use_id = getattr(msg, "tool_use_id", None)
        if task_id and tool_use_id:
            _task_id_to_task_tid[task_id] = tool_use_id
        return

    if msg_type == "TaskNotificationMessage":
        status = getattr(msg, "status", None)
        tool_use_id = getattr(msg, "tool_use_id", None)
        task_id = getattr(msg, "task_id", None)
        if task_id:
            _task_id_to_task_tid.pop(task_id, None)
        if status in _TERMINAL_TASK_STATUSES and tool_use_id and state is not None:
            _finalize_spawn_by_task_tid(tool_use_id, state)
        return

    if msg_type == "TaskUpdatedMessage":
        patch = getattr(msg, "patch", None) or {}
        status = patch.get("status")
        task_id = getattr(msg, "task_id", None)
        if status in _TERMINAL_TASK_STATUSES and task_id:
            tool_use_id = _task_id_to_task_tid.pop(task_id, None)
            if tool_use_id and state is not None:
                _finalize_spawn_by_task_tid(tool_use_id, state)
        return

    if msg_type == "ResultMessage" and state is not None:
        _finalize_all_pending_spawns(state)


def _get_hook_state() -> Any:
    """Return the active trace state visible to hook callbacks.

    Hooks run in spawn_detached asyncio tasks that don't inherit ContextVars,
    so we fall back to the module-level _hook_trace_ref set by traced_receive_response.
    """
    return get_active_trace() or (_hook_trace_ref[0] if _hook_trace_ref else None)


async def _av_pre_tool_use(
    hook_input: Any, tool_use_id: str | None = None, context: Any = None
) -> dict:
    """PreToolUse hook — record wall-clock start time keyed by tool_use_id."""
    import uuid

    tid = (
        hook_input.get("tool_use_id", "") if isinstance(hook_input, dict) else ""
    ) or (tool_use_id or "")
    if tid:
        # Store (monotonic, wall_clock_iso) so the post-hook can set span.started_at.
        # Server derives duration_ms from (started_at, ended_at) wall-clock pair,
        # so started_at must be the actual tool-start wall-clock time.
        _tool_timers[tid] = (time.monotonic(), datetime.now(UTC).isoformat())
        # A4: pre-allocate span_id for Task (sub-agent spawn) tools so we can
        # set parent_span_id on the sub-agent's tools before PostToolUse fires.
        tool_name = (
            hook_input.get("tool_name", "") if isinstance(hook_input, dict) else ""
        ) or ""
        if tool_name in ("Task", "Agent"):
            _task_span_ids[tid] = uuid.uuid4().hex
            _pending_task_tids.append(tid)
    return {}


async def _av_post_tool_use(
    hook_input: Any,
    tool_use_id: str | None = None,
    context: Any = None,
    *,
    failed: bool = False,
) -> dict:
    """PostToolUse / PostToolUseFailure hook — emit a tool span."""
    if not isinstance(hook_input, dict):
        return {}
    tid = hook_input.get("tool_use_id", "") or (tool_use_id or "")
    tool_name = hook_input.get("tool_name", "") or "tool"
    tool_input = hook_input.get("tool_input") or {}
    tool_response = (
        hook_input.get("tool_response") if not failed else hook_input.get("error")
    )

    # Both PostToolUseFailure and PostToolUse fire for a failed call (measured
    # live, MHQ-1416). One span per tool_use_id; a failure always wins.
    existing = _tool_spans_by_tid.get(tid) if tid else None
    if existing is not None:
        if failed and existing.status != "failed":
            _mark_tool_span_failed(existing, tool_name, tool_response)
        return {}

    timer_entry = _tool_timers.pop(tid, None)
    if timer_entry:
        call_start, started_at_iso = timer_entry
    else:
        call_start, started_at_iso = time.monotonic(), None

    state = _get_hook_state()
    if state is None:
        return {}

    is_spawn = tool_name in ("Task", "Agent")
    # A4: determine parent and span_id.
    # Sub-agent tools (agent_id present) nest under their named AGENT span
    # (C.1) when one was opened at SubagentStart, else fall back to the
    # spawning Task span directly (e.g. SDK versions without SubagentStart).
    # Task spans themselves nest under the root agent span.
    agent_id = hook_input.get("agent_id", "") if isinstance(hook_input, dict) else ""
    if agent_id:
        open_agent = _open_agent_spans.get(agent_id)
        if open_agent:
            parent_span_id = open_agent[0].span_id
        elif agent_id in _agent_span_ids:
            # MHQ-1416: AGENT span already finalized (or its spawn already
            # settled) but this hook is still arriving.
            parent_span_id = _agent_span_ids[agent_id]
        else:
            task_tid = _agent_to_task_tid.get(agent_id)
            parent_span_id = _task_span_ids.get(task_tid) if task_tid else None
    else:
        parent_span_id = state.span_stack[-1].span_id if state.span_stack else None

    # Use pre-allocated span_id for Task spans so sub-agent tools already
    # reference it correctly (PostToolUse fires after the sub-agent finishes).
    pre_span_id = _task_span_ids.pop(tid, None) if is_spawn else None

    span = SpanData(
        trace_id=state.trace_id,
        type=SpanType.TOOL,
        name=tool_name,
        parent_span_id=parent_span_id,
        status="failed" if failed else "completed",
        input_data={"input": tool_input} if tool_input else None,
        output_data={"response": _safe_serialize(tool_response)}
        if tool_response is not None
        else None,
        metadata={"askl_is_spawn": True} if is_spawn else None,
    )
    if pre_span_id:
        span.span_id = pre_span_id
    if started_at_iso:
        span.started_at = started_at_iso
    _finalize_span(span, call_start, status="failed" if failed else "completed")
    if failed:
        _mark_tool_span_failed(span, tool_name, tool_response)
    state.completed_spans.append(span)
    if tid:
        _remember(_tool_spans_by_tid, tid, span)
    # MHQ-1416: the wrapper Agent tool's PostToolUse fires as soon as the
    # sub-agent is *launched*, not when it finishes, so it must NOT finalize
    # the AGENT span or drop the agent->task mapping: the sub-agent's tool and
    # LLM activity is still to come. The AGENT span is finalized by the
    # completion signals (TaskNotification / TaskUpdated / SubagentStop) or,
    # failing those, the ResultMessage fallback.
    return {}


async def _av_post_tool_use_failure(
    hook_input: Any, tool_use_id: str | None = None, context: Any = None
) -> dict:
    return await _av_post_tool_use(hook_input, tool_use_id, context, failed=True)


async def _av_subagent_start(
    hook_input: Any, tool_use_id: str | None = None, context: Any = None
) -> dict:
    """SubagentStart hook — map new agent_id to the Task tool that spawned it,
    and emit a named AGENT span (C.1) so the orchestration tree shows the
    declared sub-agent instead of an anonymous Task spawn.

    Lazy by design (approach b): only agents that actually spawn get a span —
    no phantom agents for declared-but-unused entries in options.agents.
    """
    if not isinstance(hook_input, dict):
        return {}
    agent_id = hook_input.get("agent_id", "")
    if not agent_id or not _pending_task_tids:
        return {}

    task_tid = _pending_task_tids.pop()
    _agent_to_task_tid[agent_id] = task_tid

    state = _get_hook_state()
    if state is None:
        return {}

    agent_type = hook_input.get("agent_type", "") or "agent"
    parent_span_id = _task_span_ids.get(task_tid)

    span = SpanData(
        trace_id=state.trace_id,
        type=SpanType.AGENT,
        name=_sanitize_agent_name(agent_type),
        parent_span_id=parent_span_id,
        metadata={"agent_id": agent_id},
    )
    _open_agent_spans[agent_id] = (span, time.monotonic())
    _remember(_agent_span_ids, agent_id, span.span_id)
    _remember(_task_agent_span_ids, task_tid, span.span_id)
    return {}


def _finalize_spawn_by_task_tid(task_tid: str, state: Any) -> None:
    """Finalize the Task/Agent wrapper span and the named AGENT span (if any)
    for a completed spawn, given its task tool_use_id.

    Shared by ``_av_subagent_stop`` (hook path — kept as a defensive fallback,
    see MHQ-762/MHQ-770) and the ``TaskNotificationMessage``/``TaskUpdatedMessage``
    stream-based path in ``traced_receive_response`` (MHQ-770 — the real
    completion signal for the currently-installed claude-agent-sdk, since
    neither PostToolUse nor SubagentStop fire for the wrapper tool). Pops are
    idempotent, so calling this twice for the same task_tid (e.g. both the
    hook and the stream message fire) is safe — the second call is a no-op.
    """
    pre_span_id = _task_span_ids.pop(task_tid, None)
    if pre_span_id:
        timer_entry = _tool_timers.pop(task_tid, None)
        call_start, started_at_iso = timer_entry or (time.monotonic(), None)
        parent_span_id = state.span_stack[-1].span_id if state.span_stack else None
        span = SpanData(
            trace_id=state.trace_id,
            type=SpanType.TOOL,
            name="Agent",
            parent_span_id=parent_span_id,
            metadata={"askl_is_spawn": True},
        )
        span.span_id = pre_span_id
        if started_at_iso:
            span.started_at = started_at_iso
        _finalize_span(span, call_start)
        state.completed_spans.append(span)

    agent_id = next(
        (aid for aid, tid in _agent_to_task_tid.items() if tid == task_tid), None
    )
    if agent_id is not None:
        _agent_to_task_tid.pop(agent_id, None)
        agent_span_entry = _open_agent_spans.pop(agent_id, None)
        if agent_span_entry:
            agent_span, agent_start_mono = agent_span_entry
            _finalize_span(agent_span, agent_start_mono)
            state.completed_spans.append(agent_span)


async def _av_subagent_stop(
    hook_input: Any, tool_use_id: str | None = None, context: Any = None
) -> dict:
    """SubagentStop hook — finalize the Task/Agent wrapper span and the named
    AGENT span opened at SubagentStart.

    MHQ-762 part 2 / MHQ-770: confirmed via live debug logging (2026-07-30)
    that neither PostToolUse nor SubagentStop actually fire for the wrapper
    Task/Agent tool call in the currently-installed claude-agent-sdk — kept
    registered as a defensive fallback in case a future CLI version does
    dispatch it. The real completion signal is ``TaskNotificationMessage``/
    ``TaskUpdatedMessage`` in the main response stream, handled in
    ``traced_receive_response``.
    """
    if not isinstance(hook_input, dict):
        return {}
    agent_id = hook_input.get("agent_id", "")
    if not agent_id:
        return {}

    state = _get_hook_state()
    if state is None:
        return {}

    task_tid = _agent_to_task_tid.get(agent_id)
    if task_tid:
        _finalize_spawn_by_task_tid(task_tid, state)
    return {}


def _inject_av_hooks(options: Any) -> None:
    """Merge AV tool-lifecycle hooks into options without clobbering existing ones.

    Idempotent — checks for the sentinel attribute before injecting.
    """
    if getattr(options, "_av_hooks_injected", False):
        return
    try:
        from claude_agent_sdk.types import HookMatcher  # type: ignore[import-not-found]
    except ImportError:
        return

    if options.hooks is None:
        options.hooks = {}

    for event, cb in (
        ("PreToolUse", _av_pre_tool_use),
        ("PostToolUse", _av_post_tool_use),
        ("PostToolUseFailure", _av_post_tool_use_failure),
        ("SubagentStart", _av_subagent_start),
        ("SubagentStop", _av_subagent_stop),
    ):
        matcher = HookMatcher(hooks=[cb])
        options.hooks.setdefault(event, []).append(matcher)

    object.__setattr__(options, "_av_hooks_injected", True) if hasattr(
        type(options), "__dataclass_fields__"
    ) else setattr(options, "_av_hooks_injected", True)


def _is_thinking_only(content_blocks: Any) -> bool:
    """Return True if all content blocks are ThinkingBlock (extended-thinking phase).

    The claude-agent-sdk emits two AssistantMessage events per LLM call when
    extended thinking is enabled: one with ThinkingBlock (internal reasoning)
    and one with TextBlock/ToolUseBlock (the actual response). Both carry
    identical usage data, so emitting a span for the thinking-only message
    would double-count tokens and cost.
    """
    if not content_blocks:
        return False
    return all(type(b).__name__ == "ThinkingBlock" for b in content_blocks)


def _extract_content_text(content_blocks: Any) -> str | None:
    """Extract human-readable text from AssistantMessage.content blocks.

    Handles TextBlock (.text), ToolUseBlock (.name + .input), and any
    unknown block type gracefully. ThinkingBlocks are skipped.
    """
    if not content_blocks:
        return None
    parts: list[str] = []
    for block in content_blocks:
        block_type = type(block).__name__
        if block_type == "TextBlock":
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        elif block_type == "ToolUseBlock":
            name = getattr(block, "name", "tool")
            inp = getattr(block, "input", None)
            parts.append(f"[tool:{name}] {inp}" if inp else f"[tool:{name}]")
    return "\n".join(parts) if parts else None


def _emit_llm_span_from_assistant_msg(
    msg: Any, start_mono: float, started_at_iso: str | None = None
) -> None:
    """Emit an AV ``llm`` span from a claude-agent-sdk ``AssistantMessage``.

    Called inline inside the wrapped ``receive_response()`` generator, where
    ``_active_trace`` is guaranteed to be set (either by the customer's
    ``morse_ai.run()`` or by the adapter's auto-created ``TracedRun``).
    """
    state = get_active_trace()
    if state is None:
        return
    try:
        usage = getattr(msg, "usage", None) or {}
        model = getattr(msg, "model", None) or ""

        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
        cache_read = usage.get("cache_read_input_tokens", 0) or 0
        cache_write = usage.get("cache_creation_input_tokens", 0) or 0

        total_input = input_tokens + cache_read + cache_write
        total_tokens = total_input + output_tokens

        cost: float | None = None
        if model:
            cost = calculate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                cached_input_tokens=cache_read,
                cache_write_tokens=cache_write,
            )

        # Capture prompt (set by the query() wrapper) and output content blocks.
        prompt_text = _pending_prompt.get()
        output_text = _extract_content_text(getattr(msg, "content", None))

        input_data = (
            {"messages": [{"role": "user", "content": prompt_text}]}
            if prompt_text
            else None
        )
        output_data = {"content": output_text} if output_text else None

        span = SpanData(
            trace_id=state.trace_id,
            type=SpanType.LLM,
            name="claude-agent-sdk.messages.create",
            parent_span_id=_subagent_parent_span_id(msg)
            or (state.span_stack[-1].span_id if state.span_stack else None),
            model=model or None,
            input_tokens=total_input if total_input else None,
            output_tokens=output_tokens if output_tokens else None,
            total_tokens=total_tokens if total_tokens else None,
            # MHQ-1236: 0 when usage was reported without cache activity.
            cached_input_tokens=cache_read if usage else None,
            cache_write_tokens=cache_write if usage else None,
            cost_usd=round(cost, 6) if cost is not None else None,
            input_data=input_data,
            output_data=output_data,
        )
        if started_at_iso:
            span.started_at = started_at_iso
        _finalize_span(span, start_mono)
        state.completed_spans.append(span)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "morse_ai: claude_agent_sdk llm span recording swallowed error: %s", exc
        )


def _apply_tool_results_from_stream(content: Any) -> None:
    """MHQ-1416: complete fallback tool spans from ToolResultBlocks.

    The CLI delivers ``ToolResultBlock`` in the *following* ``UserMessage``, not
    in the AssistantMessage that carried the ``ToolUseBlock``, so pairing inside
    one message never matched. Here a result is applied to the span already
    emitted for its ``tool_use_id``: output on success, ``failed`` status +
    bounded ``error_message`` + synthetic exception type when ``is_error``.
    """
    if not isinstance(content, list):
        return
    for block in content:
        if type(block).__name__ != "ToolResultBlock":
            continue
        span = _tool_spans_by_tid.get(getattr(block, "tool_use_id", None) or "")
        if span is None:
            continue
        result = getattr(block, "content", None)
        if getattr(block, "is_error", False):
            _mark_tool_span_failed(span, span.name, result)
        if result is not None and span.output_data is None:
            span.output_data = {"response": _safe_serialize(result)}


def _emit_tool_spans_from_stream(
    content: Any, state: Any, parent_override: str | None = None
) -> None:
    """C.3 fallback: parse ToolUseBlock/ToolResultBlock from AssistantMessage content.

    Called only when AV hooks were NOT injected (older SDK or injection failure).
    Pairs ToolUseBlock → ToolResultBlock by id within the same message; unpaired
    ToolUseBlocks get a span that a later UserMessage's ToolResultBlock completes
    (``_apply_tool_results_from_stream``). ``parent_override`` parents the spans
    under a sub-agent's AGENT span (MHQ-1416).
    """
    if not content:
        return
    now = time.monotonic()
    # First pass: collect ToolResultBlocks for pairing.
    result_blocks: dict[str, Any] = {}
    for block in content:
        if type(block).__name__ == "ToolResultBlock":
            tid = getattr(block, "tool_use_id", None)
            if tid:
                result_blocks[tid] = block

    parent = parent_override or (
        state.span_stack[-1].span_id if state.span_stack else None
    )
    for block in content:
        if type(block).__name__ != "ToolUseBlock":
            continue
        tool_id = getattr(block, "id", None)
        tool_name = getattr(block, "name", None) or "tool"
        tool_input = getattr(block, "input", {}) or {}
        result_block = result_blocks.get(tool_id) if tool_id else None
        result = getattr(result_block, "content", None)
        is_spawn = tool_name in ("Task", "Agent")
        span = SpanData(
            trace_id=state.trace_id,
            type=SpanType.TOOL,
            name=tool_name,
            parent_span_id=parent,
            status="completed",
            input_data={"input": tool_input} if tool_input else None,
            output_data={"response": _safe_serialize(result)}
            if result is not None
            else None,
            metadata={"askl_is_spawn": True} if is_spawn else None,
        )
        _finalize_span(span, now)
        if result_block is not None and getattr(result_block, "is_error", False):
            _mark_tool_span_failed(span, tool_name, result)
        state.completed_spans.append(span)
        if tool_id:
            _remember(_tool_spans_by_tid, tool_id, span)


def _wrap_client_query(original):
    """Wrap ClaudeSDKClient.query() to capture the prompt string before CLI dispatch."""

    async def traced_client_query(self_client, prompt, *args, **kwargs):
        if isinstance(prompt, str):
            _pending_prompt.set(prompt)
        _llm_call_start_ref.clear()
        _llm_call_start_ref.append((time.monotonic(), datetime.now(UTC).isoformat()))
        await original(self_client, prompt, *args, **kwargs)

    traced_client_query.__wrapped__ = original
    return traced_client_query


_LIFECYCLE_MESSAGES = (
    "TaskStartedMessage",
    "TaskNotificationMessage",
    "TaskUpdatedMessage",
)


def _apply_result_message(traced_run: TracedRun, msg: Any) -> None:
    """Stamp aggregate cost from a ResultMessage onto OUR trace.

    Only called for a trace this adapter opened. MHQ-1914: ``is_error`` is the CLI's
    exit status, not whether the task worked, so it is never written as a
    developer outcome (that label outranks later business records). Outcomes come
    from ``set_outcome()`` in the caller or from a connected business source.
    """
    total_cost = getattr(msg, "total_cost_usd", None)
    if total_cost is None:
        return
    # Expose aggregate cost as trace-level metadata for the dashboard even when
    # individual LLM spans are sparse.
    root = traced_run._root_span
    if root is None:
        return
    if root.metadata is None:
        root.metadata = {}
    root.metadata["sdk_total_cost_usd"] = round(float(total_cost), 6)


def _order_parents_first(state: Any) -> None:
    """Reorder ``state.completed_spans`` so every parent precedes its children.

    MHQ-1416: the ingest resolves ``parent_span_id`` only against spans that
    appear EARLIER in the payload; a forward reference is silently rewritten to
    the sentinel root (measured on dev 2026-10-18 with a no-LLM probe: child
    before parent -> parent ``...0001``; parent before child -> correct). The
    adapter appends spans in completion order, and a sub-agent's AGENT span
    completes last, so its tool/LLM children always preceded it. Stable: spans
    keep their relative order unless a parent must move ahead of a child.
    """
    spans = getattr(state, "completed_spans", None)
    if not spans or len(spans) < 2:
        return
    present = {sp.span_id for sp in spans}
    emitted: set[str] = set()
    ordered: list[Any] = []
    pending = list(spans)
    while pending:
        deferred = []
        for sp in pending:
            parent = sp.parent_span_id
            if parent is None or parent not in present or parent in emitted:
                ordered.append(sp)
                emitted.add(sp.span_id)
            else:
                deferred.append(sp)
        if len(deferred) == len(pending):  # cycle: give up, keep as-is
            ordered.extend(deferred)
            break
        pending = deferred
    spans[:] = ordered


async def _consume_traced(
    stream: Any, *, use_fallback: bool, call_start: float, call_start_iso: str
):
    """Yield every message from ``stream``, emitting spans as they pass.

    MHQ-1053: this is the single consumer behind BOTH entry points. It used to
    exist only inside ``receive_response()``'s wrapper — as two near-identical
    copies, one for "the caller already opened a trace" and one for "open our
    own" — while the module-level ``query()`` wrapper merely re-yielded. That
    is why ``query()`` logged "adapter installed" and then emitted nothing:
    a public, documented entry point with no observer attached.

    Sharing the loop is the fix AND the guard against the next one: a third
    entry point gets telemetry by construction rather than by remembering.
    """
    set_active_framework("claude-agent-sdk")
    state = get_active_trace()
    # A trace we open is a trace we must close, set the outcome on, and stamp
    # the ResultMessage cost onto. A caller's trace gets none of that.
    traced_run: TracedRun | None = None
    if state is None:
        traced_run = TracedRun("claude-agent-sdk")
        traced_run.__enter__()
        state = get_active_trace()

    # Expose state to hook callbacks in detached asyncio tasks, which do not
    # inherit ContextVars from this generator's context.
    if state is not None:
        _hook_trace_ref.clear()
        _hook_trace_ref.append(state)

    try:
        async for msg in stream:
            msg_type = type(msg).__name__
            # MHQ-1022: every CLI message carries the session id; it is stable
            # across turns of one client, which is exactly the grouping the
            # dashboard's thread_id expresses.
            adopt_thread_id(getattr(msg, "session_id", None))
            current = get_active_trace() or state

            if msg_type == "AssistantMessage" and getattr(msg, "usage", None):
                content = getattr(msg, "content", None)
                if not _is_thinking_only(content):
                    _emit_llm_span_from_assistant_msg(msg, call_start, call_start_iso)
                    if use_fallback and current is not None:
                        _emit_tool_spans_from_stream(
                            content, current, _subagent_parent_span_id(msg)
                        )
            elif msg_type == "UserMessage" and use_fallback:
                _apply_tool_results_from_stream(getattr(msg, "content", None))
            elif msg_type in _LIFECYCLE_MESSAGES:
                _handle_task_lifecycle_message(msg, msg_type, current)
            elif msg_type == "ResultMessage":
                _handle_task_lifecycle_message(msg, msg_type, current)
                if traced_run is not None:
                    _apply_result_message(traced_run, msg)

            yield msg
            # Reset per-turn start so each LLM span measures one inference.
            call_start = time.monotonic()
            call_start_iso = datetime.now(UTC).isoformat()
    except BaseException as exc:
        # MHQ-762: drain in-flight Task/Agent spawn hooks BEFORE finalizing —
        # __exit__ sends completed_spans immediately, so a spawn span that
        # appends after that point is already too late.
        await _drain_pending_spawns()
        if state is not None:
            _order_parents_first(state)
        if traced_run is not None:
            traced_run.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        await _drain_pending_spawns()
        if state is not None:
            _order_parents_first(state)
        if traced_run is not None:
            traced_run.__exit__(None, None, None)
    finally:
        _hook_trace_ref.clear()
        clear_active_framework()


def _wrap_module_query(original):
    """Wrap the top-level claude_agent_sdk.query() async generator."""

    async def traced_module_query(*args, **kwargs):
        prompt = kwargs.get("prompt") or (args[0] if args else None)
        if isinstance(prompt, str):
            _pending_prompt.set(prompt)
        # Inject AV hooks into options so tool spans appear in the tree.
        options = kwargs.get("options")
        if options is not None:
            _inject_av_hooks(options)
        # No options object means nowhere to hang the hooks, so tool spans have
        # to come off the message stream instead.
        hooks_injected = bool(getattr(options, "_av_hooks_injected", False))
        # Unlike the class-based path there is no separate query() call to time
        # against: the CLI is dispatched when this generator is first iterated,
        # so now IS the start of the first inference.
        async for msg in _consume_traced(
            original(*args, **kwargs),
            use_fallback=not hooks_injected,
            call_start=time.monotonic(),
            call_start_iso=datetime.now(UTC).isoformat(),
        ):
            yield msg

    traced_module_query.__wrapped__ = original
    return traced_module_query


def _wrap_init(original):
    """Wrap ClaudeSDKClient.__init__ to inject AV hooks into options at construction."""

    def traced_init(self_client, options=None, transport=None):
        if options is not None:
            _inject_av_hooks(options)
        original(self_client, options, transport)
        # Mark the client itself so traced_receive_response can detect injection
        # regardless of whether the SDK stores options as self.options.
        with contextlib.suppress(Exception):
            self_client._av_hooks_injected = True

    traced_init.__wrapped__ = original
    return traced_init


def _wrap_receive_response(original):
    """Build the async-generator replacement for ClaudeSDKClient.receive_response."""

    async def traced_receive_response(self_client, *args, **kwargs):
        # Use start time captured in traced_client_query (before CLI dispatch) when
        # available — receive_response() only reads buffered output, so measuring here
        # would give near-zero LLM durations.
        if _llm_call_start_ref:
            call_start, call_start_iso = _llm_call_start_ref.pop()
        else:
            call_start = time.monotonic()
            call_start_iso = datetime.now(UTC).isoformat()

        # AGE-535 / AGE-518: _consume_traced sets the active framework, without
        # which real claude-agent-sdk traces carry no ``morse_ai.framework``
        # resource attribute and the ASKL profile (Task-spawn classification,
        # LLM dedup) silently never applies to real traffic.
        async for msg in _consume_traced(
            original(self_client, *args, **kwargs),
            # C.3: use stream-parse fallback only when hooks weren't injected.
            use_fallback=not getattr(self_client, "_av_hooks_injected", False),
            call_start=call_start,
            call_start_iso=call_start_iso,
        ):
            yield msg

    traced_receive_response.__wrapped__ = original
    return traced_receive_response


@absorb_errors(fingerprint="morse_ai.adapters.claude_agent_sdk.install")
def install() -> bool:
    """Monkey-patch ClaudeSDKClient.receive_response for transparent telemetry.

    Idempotent — safe to call repeatedly. Returns True if installed (newly or
    already), False if ``claude_agent_sdk`` is unavailable or patching failed.
    """
    try:
        import claude_agent_sdk  # type: ignore[import-not-found]
    except ImportError:
        return False

    if getattr(claude_agent_sdk, _INSTALL_FLAG, False):
        return True

    try:
        from claude_agent_sdk.client import (  # type: ignore[import-not-found]
            ClaudeSDKClient,
        )

        original_rr = ClaudeSDKClient.receive_response
        ClaudeSDKClient.receive_response = _wrap_receive_response(original_rr)  # type: ignore[method-assign]
        setattr(ClaudeSDKClient, _ORIGINAL_RECEIVE_RESPONSE, original_rr)

        # Wrap __init__ to inject AV hooks into options at construction time (A3).
        original_init = ClaudeSDKClient.__init__
        ClaudeSDKClient.__init__ = _wrap_init(original_init)  # type: ignore[method-assign]
        setattr(ClaudeSDKClient, _ORIGINAL_INIT, original_init)

        # Wrap ClaudeSDKClient.query() to capture prompt for class-based callers.
        if hasattr(ClaudeSDKClient, "query"):
            original_cq = ClaudeSDKClient.query
            ClaudeSDKClient.query = _wrap_client_query(original_cq)  # type: ignore[method-assign]
            setattr(ClaudeSDKClient, _ORIGINAL_CLIENT_QUERY, original_cq)
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "morse_ai: claude_agent_sdk adapter could not patch "
            "ClaudeSDKClient: %s — SDK calls will run untraced",
            exc,
        )
        return False

    # Wrap the top-level query() async-generator used by functional-API callers
    # (e.g. research-agent uses `from claude_agent_sdk import query`).
    try:
        import claude_agent_sdk as _sdk_mod  # noqa: PLC0415

        if hasattr(_sdk_mod, "query") and not getattr(
            _sdk_mod.query, "__wrapped__", None
        ):
            original_mq = _sdk_mod.query
            _sdk_mod.query = _wrap_module_query(original_mq)
            setattr(_sdk_mod, _ORIGINAL_MODULE_QUERY, original_mq)
    except Exception as exc:  # noqa: BLE001
        logger.debug("morse_ai: could not wrap claude_agent_sdk.query: %s", exc)

    setattr(claude_agent_sdk, _INSTALL_FLAG, True)
    logger.info("morse_ai: claude_agent_sdk adapter installed")

    # Fire the dependency scan so "Detected stack" populates even when the
    # caller uses the adapter path instead of morse_ai.init().
    try:
        from morse_ai import _client as _av_client
        from morse_ai import _flush_dependencies_async

        if _av_client is not None:
            _flush_dependencies_async(_av_client)
    except Exception as exc:  # noqa: BLE001
        logger.debug("morse_ai: adapter dependency flush swallowed: %s", exc)

    return True


@absorb_errors(fingerprint="morse_ai.adapters.claude_agent_sdk.uninstall")
def uninstall() -> None:
    """Restore the original ClaudeSDKClient.receive_response. Idempotent."""
    try:
        import claude_agent_sdk  # type: ignore[import-not-found]
    except ImportError:
        return

    if not getattr(claude_agent_sdk, _INSTALL_FLAG, False):
        return

    try:
        from claude_agent_sdk.client import (  # type: ignore[import-not-found]
            ClaudeSDKClient,
        )

        original_rr = getattr(ClaudeSDKClient, _ORIGINAL_RECEIVE_RESPONSE, None)
        if original_rr is not None:
            ClaudeSDKClient.receive_response = original_rr  # type: ignore[method-assign]

        original_init = getattr(ClaudeSDKClient, _ORIGINAL_INIT, None)
        if original_init is not None:
            ClaudeSDKClient.__init__ = original_init  # type: ignore[method-assign]

        original_cq = getattr(ClaudeSDKClient, _ORIGINAL_CLIENT_QUERY, None)
        if original_cq is not None:
            ClaudeSDKClient.query = original_cq  # type: ignore[method-assign]
    except (ImportError, AttributeError):
        pass

    try:
        import claude_agent_sdk as _sdk_mod  # noqa: PLC0415

        original_mq = getattr(_sdk_mod, _ORIGINAL_MODULE_QUERY, None)
        if original_mq is not None:
            _sdk_mod.query = original_mq
    except Exception:  # noqa: BLE001
        pass

    setattr(claude_agent_sdk, _INSTALL_FLAG, False)
