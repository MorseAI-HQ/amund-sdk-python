"""OpenAI Agents SDK adapter — emits Morse spans for `Runner.run()` calls.

The package's import name is `agents` (see `openai-agents` on PyPI). The adapter
hooks the three `Runner` entry points — `run`, `run_sync`, `run_streamed` — by
monkey-patching them to inject an Morse `RunHooks` instance that observes
the run lifecycle without altering its semantics.

Spans emitted per `Runner.run(...)`:

- One root `agent` span (the run itself), with metadata:
    - `agent_name` (the starting agent's name)
    - `session_id` (when `session=` is passed; same id across runs sharing a Session)
    - `structured_output` (when `Agent(output_type=...)` is set)
- One child `llm` span per LLM round trip (via `on_llm_start` / `on_llm_end`)
- One child `tool` span per function tool invocation (via `on_tool_start`
  and `on_tool_end`)
- One child `guardrail` span per input/output guardrail check (MHQ-1024), via
  patched `InputGuardrail.run` / `OutputGuardrail.run`. `RunHooks` has no
  guardrail surface, so the guardrail classes themselves are patched. The span
  carries `metadata.guardrail_name`, `metadata.guardrail_kind`
  (`input`/`output`) and `metadata.tripwire_triggered`; `output_data` holds the
  guardrail's `output_info` (its reasoning), and a tripped guardrail closes
  `failed`. LLM calls the guardrail itself makes nest underneath it.
- One `handoff` span per routing decision (MHQ-1025), parented to the agent span
  that made it, with `from_agent` / `to_agent` metadata. It replaces the old
  `handoffs: [{from, to}, ...]` array on the root span — a flat list had no
  timestamp, no position in the tree, and could not be clicked or filtered.
  The target agent gets its own child `agent` span; subsequent LLM spans carry
  it as their `agent_name` metadata.

Idempotency: install() sets `_morse_ai_installed = True` on the `agents`
module so repeated calls (e.g., from `morse_ai.init()`) are safe no-ops.

Wrapped under `@absorb_errors` — installation/runtime failures are reported
out-of-band and never propagate to the customer's code.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from contextvars import ContextVar
from typing import Any

from morse_ai._internal.active_framework import (
    clear_active_framework,
    set_active_framework,
)
from morse_ai._internal.errors import absorb_errors
from morse_ai.pricing import calculate_cost
from morse_ai.spans import SpanData, SpanType
from morse_ai.tracing import (
    _finalize_span,
    _TraceState,
    adopt_thread_id,
    get_active_trace,
)
from morse_ai.tracing import (
    run as _av_run,
)

# AGE-240: framework auto-discovery tag emitted via the thread-local set
# around every intercepted ``Runner.run*``. Read at trace emission time by
# ``TracedRun`` to stamp ``resource["morse_ai.framework"]``.
_FRAMEWORK_NAME = "openai-agents"

logger = logging.getLogger("morse_ai")

_INSTALL_FLAG = "_morse_ai_installed"

# Cached references to the originals so uninstall() can restore them.
_original_run = None
_original_run_sync = None
_original_run_streamed = None
_original_input_guardrail_run = None
_original_output_guardrail_run = None

# The guardrail span currently being executed in this context, if any. Set by
# the patched `InputGuardrail.run` / `OutputGuardrail.run` for the duration of
# the check, and read by `_prepare_call` so the `Runner.run()` a guardrail
# makes internally nests under the guardrail span instead of the run root.
#
# A ContextVar (not an instance attribute) because the guardrail objects are
# shared across concurrent runs: input guardrails are dispatched with
# `asyncio.gather`, which copies the context per task, so each concurrent check
# sees only its own span.
_guardrail_span: ContextVar[SpanData | None] = ContextVar(
    "morse_ai_guardrail_span", default=None
)

# Internal mapping of Session id() -> stable session id string. The OpenAI
# Agents SDK doesn't expose a public session_id attribute, so we synthesize
# one based on the object's identity.
_session_ids: dict[int, str] = {}


def _resolve_session_id(session: Any) -> str | None:
    """Return a stable session id for a given Session object, or None."""
    if session is None:
        return None
    sid = getattr(session, "session_id", None)
    if sid:
        return str(sid)
    obj_id = id(session)
    cached = _session_ids.get(obj_id)
    if cached is not None:
        return cached
    # Fall back to a stable hex of the object's id. Stays the same for the
    # life of the Session object in this process.
    synth = f"av-session-{obj_id:x}"
    _session_ids[obj_id] = synth
    return synth


def _resolve_thread_id(kwargs: dict[str, Any]) -> str | None:
    """Find the conversation identity for one ``Runner.run*`` call, or None.

    MHQ-1022. Checked in order of how explicit the signal is:

    1. ``session=`` — the SDK's own multi-turn primitive.
    2. ``conversation_id=`` — the OpenAI Conversations API thread.
    3. ``context=`` — the app's run context. ChatKit's ``AgentContext`` hangs
       the thread off ``.thread.id`` (``thr_…``); apps that roll their own
       context usually name the field ``thread_id``/``session_id``/
       ``conversation_id``. This is the case the ticket was filed for: the
       demo app's only integration is ``morse_ai.init()``, so the id has to
       come off the context or not at all.

    Every read here can run customer property code, hence the blanket guard.
    """
    try:
        session_id = _resolve_session_id(kwargs.get("session"))
        if session_id:
            return session_id
        conversation_id = kwargs.get("conversation_id")
        if conversation_id:
            return str(conversation_id)
        context = kwargs.get("context")
        if context is None:
            return None
        thread_pk = getattr(getattr(context, "thread", None), "id", None)
        if thread_pk:
            return str(thread_pk)
        for attr in ("thread_id", "session_id", "conversation_id"):
            value = getattr(context, attr, None)
            if value:
                return str(value)
    except Exception as exc:  # noqa: BLE001
        logger.debug("morse_ai: thread_id resolution swallowed: %s", exc)
    return None


def _safe_json(value: Any, *, max_bytes: int = 4096) -> dict[str, Any] | None:
    """Return a JSON-serialisable dict, truncated for safety.

    The server schema requires input_data / output_data to be dicts.
    Dicts pass through directly; strings are wrapped as {"text": ...};
    lists and other JSON-able values are wrapped as {"value": ...}.
    """
    if value is None:
        return None
    try:
        if isinstance(value, dict):
            # Truncate large string leaves but keep the dict shape.
            text = json.dumps(value, default=str)
            if len(text) > max_bytes:
                return {"text": text[:max_bytes] + "...[truncated]"}
            return value
        if isinstance(value, str):
            if len(value) > max_bytes:
                value = value[:max_bytes] + "...[truncated]"
            return {"text": value}
        # Lists, numbers, booleans — serialise then wrap.
        text = json.dumps(value, default=str)
        if len(text) > max_bytes:
            text = text[:max_bytes] + "...[truncated]"
        return {"value": json.loads(text)}
    except Exception:  # noqa: BLE001
        try:
            return {"text": str(value)[:max_bytes]}
        except Exception:  # noqa: BLE001
            return None


def _extract_usage(response: Any) -> dict[str, Any]:
    """Pull token + cost from an `agents.items.ModelResponse`-shape object."""
    out: dict[str, Any] = {}
    usage = getattr(response, "usage", None)
    if usage is None:
        return out

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", None) or (
        input_tokens + output_tokens
    )
    cached_tokens = 0
    details = getattr(usage, "input_tokens_details", None)
    if details is not None:
        cached_tokens = getattr(details, "cached_tokens", 0) or 0

    out["input_tokens"] = input_tokens
    out["output_tokens"] = output_tokens
    out["total_tokens"] = total_tokens
    if cached_tokens:
        out["cached_tokens"] = cached_tokens
    return out


def _guardrail_name(guardrail: Any) -> str:
    """Best-effort display name for an Input/OutputGuardrail instance."""
    try:
        name = guardrail.get_name()
        if name:
            return str(name)
    except Exception:  # noqa: BLE001
        pass
    return getattr(guardrail, "name", None) or type(guardrail).__name__


def _open_guardrail_span(
    state: _TraceState, guardrail: Any, kind: str, guarded_input: Any
) -> tuple[SpanData, float, Any]:
    """Start a guardrail span and make it the parent for nested runs."""
    name = _guardrail_name(guardrail)
    parent = state.span_stack[-1] if state.span_stack else None
    span = SpanData(
        trace_id=state.trace_id,
        type=SpanType.GUARDRAIL,
        name=f"guardrail.{name}",
        parent_span_id=parent.span_id if parent is not None else None,
        metadata={"guardrail_name": name, "guardrail_kind": kind},
    )
    if guarded_input is not None:
        span.input_data = _safe_json(guarded_input)
    token = _guardrail_span.set(span)
    return span, time.monotonic(), token


def _close_guardrail_span(
    state: _TraceState,
    span: SpanData,
    start_mono: float,
    token: Any,
    result: Any,
    exc: BaseException | None,
) -> None:
    """Record the guardrail verdict and finalize the span."""
    with contextlib.suppress(Exception):
        _guardrail_span.reset(token)

    meta = span.metadata or {}
    status, error_message = "completed", None
    output = getattr(result, "output", None)
    triggered = bool(getattr(output, "tripwire_triggered", False))
    meta["tripwire_triggered"] = triggered
    if output is not None:
        info = getattr(output, "output_info", None)
        if info is not None:
            # Pydantic verdict models are the common shape here (the demo's
            # guardrails return one with a `reasoning` field).
            with contextlib.suppress(Exception):
                if hasattr(info, "model_dump"):
                    info = info.model_dump()
            span.output_data = _safe_json(info)
    if triggered:
        status = "failed"
        error_message = f"guardrail tripwire triggered: {meta['guardrail_name']}"
    if exc is not None:
        status = "failed"
        error_message = str(exc)
    span.metadata = meta
    _finalize_span(span, start_mono, status=status, error_message=error_message)
    state.completed_spans.append(span)


def _wrap_guardrail_run(original, kind: str, input_arg: tuple[str, int]):
    """Wrap `InputGuardrail.run` / `OutputGuardrail.run` to emit a span.

    ``input_arg`` is the (keyword, positional index) of the value being
    checked — the two classes take it in different positions
    (``run(agent, input, context)`` vs ``run(context, agent, agent_output)``).
    """
    kw_name, pos = input_arg

    async def traced_guardrail_run(self, *args, **kwargs):
        state = get_active_trace()
        if state is None or state.is_closed:
            return await original(self, *args, **kwargs)
        try:
            guarded = kwargs.get(kw_name, args[pos] if len(args) > pos else None)
            span, start_mono, token = _open_guardrail_span(state, self, kind, guarded)
        except Exception as exc:  # noqa: BLE001
            logger.debug("morse_ai: guardrail span open swallowed: %s", exc)
            return await original(self, *args, **kwargs)
        try:
            result = await original(self, *args, **kwargs)
        except BaseException as exc:
            with contextlib.suppress(Exception):
                _close_guardrail_span(state, span, start_mono, token, None, exc)
            raise
        try:
            _close_guardrail_span(state, span, start_mono, token, result, None)
        except Exception as exc:  # noqa: BLE001
            logger.debug("morse_ai: guardrail span close swallowed: %s", exc)
        return result

    traced_guardrail_run.__wrapped__ = original
    return traced_guardrail_run


def _build_av_run_hooks() -> Any:
    """Construct an Morse `RunHooks` instance lazily.

    Imported here (not at module load) so the adapter module itself imports
    cleanly when `agents` isn't installed.
    """
    from agents.lifecycle import RunHooks

    class _MorseRunHooks(RunHooks):
        """Lifecycle hooks that emit Morse spans for OpenAI Agents runs.

        One instance is created per `Runner.run()` call. The instance keeps
        per-run state (current LLM/tool span timers, handoff trail) so we
        avoid concurrency issues between concurrent runs.

        Wrapped methods are bound to a single trace's `_TraceState`; callers
        must NOT install one instance globally.
        """

        def __init__(
            self,
            state: _TraceState,
            root_span: SpanData,
            current_agent_holder: dict[str, str | None],
            user_hooks: Any | None,
        ) -> None:
            self._state = state
            self._root = root_span
            self._current_agent = current_agent_holder
            self._user_hooks = user_hooks
            self._llm_starts: dict[int, tuple[SpanData, float, str | None]] = {}
            self._tool_starts: dict[int, tuple[SpanData, float, Any | None]] = {}
            # The active agent span — LLM/tool spans parent here so they nest
            # under the agent that actually ran them. Starts as the root
            # (starting agent); each handoff emits a new child AGENT span and
            # switches this pointer. open child agent spans are finalized when
            # superseded by the next handoff, or at run end.
            self._current_agent_span: SpanData = root_span
            self._open_agent_spans: list[tuple[SpanData, float]] = []

        # ---- handoff agent-span lifecycle ----
        def _supersede_current_agent(self) -> None:
            """Finalize the most-recent open child agent span (if any)."""
            if self._open_agent_spans:
                span, start_mono = self._open_agent_spans[-1]
                if span.ended_at is None:
                    _finalize_span(span, start_mono)

        def finalize_open_agents(self) -> None:
            """Close any child agent spans still open at run end."""
            for span, start_mono in self._open_agent_spans:
                if span.ended_at is None:
                    _finalize_span(span, start_mono)

        # ---- lifecycle dispatch helpers ----
        async def _maybe_call_user(self, name: str, *args: Any) -> None:
            fn = getattr(self._user_hooks, name, None) if self._user_hooks else None
            if fn is None:
                return
            try:
                result = fn(*args)
                if hasattr(result, "__await__"):
                    await result
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "morse_ai: user RunHooks.%s raised: %s — swallowed", name, exc
                )

        async def on_agent_start(self, context: Any, agent: Any) -> None:
            try:  # noqa: SIM105
                self._current_agent["name"] = getattr(agent, "name", None)
            except Exception:  # noqa: BLE001
                pass
            await self._maybe_call_user("on_agent_start", context, agent)

        async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
            await self._maybe_call_user("on_agent_end", context, agent, output)

        async def on_handoff(
            self, context: Any, from_agent: Any, to_agent: Any
        ) -> None:
            try:
                from_name = getattr(from_agent, "name", None)
                to_name = getattr(to_agent, "name", None)
                self._current_agent["name"] = to_name
                # MHQ-1025: the routing decision itself is a span, parented to
                # the agent that made it (so it sits between the two agent spans
                # it connects) and closed immediately — a handoff is an instant,
                # not an interval. The target's work is the child agent span
                # below plus its own llm/tool spans.
                handoff_mono = time.monotonic()
                handoff_span = SpanData(
                    trace_id=self._state.trace_id,
                    type=SpanType.HANDOFF,
                    name=f"{from_name} → {to_name}",
                    parent_span_id=self._current_agent_span.span_id,
                    metadata={"from_agent": from_name, "to_agent": to_name},
                )
                _finalize_span(handoff_span, handoff_mono)
                self._state.completed_spans.append(handoff_span)
                # Emit a first-class child AGENT span for the handoff target so
                # the specialist agent shows as a node in the orchestration tree
                # (the server classifies SpanType.AGENT → CanonicalRole.AGENT and
                # counts it as a distinct sub-agent). Without this, handoff
                # targets only appear as agent_name labels on LLM spans.
                self._supersede_current_agent()
                start_mono = time.monotonic()
                agent_span = SpanData(
                    trace_id=self._state.trace_id,
                    type=SpanType.AGENT,
                    name=str(to_name) if to_name else "agent",
                    parent_span_id=self._root.span_id,
                    metadata={"agent_name": to_name, "askl_handoff_from": from_name},
                )
                self._state.completed_spans.append(agent_span)
                self._open_agent_spans.append((agent_span, start_mono))
                self._current_agent_span = agent_span
            except Exception as exc:  # noqa: BLE001
                logger.debug("morse_ai: on_handoff swallowed: %s", exc)
            await self._maybe_call_user("on_handoff", context, from_agent, to_agent)

        async def on_llm_start(
            self,
            context: Any,
            agent: Any,
            system_prompt: Any,
            input_items: Any,
        ) -> None:
            try:
                start_mono = time.monotonic()
                model = None
                try:
                    model = getattr(agent, "model", None)
                    if model is not None and not isinstance(model, str):
                        model = getattr(model, "model", None) or str(model)
                except Exception:  # noqa: BLE001
                    model = None

                span = SpanData(
                    trace_id=self._state.trace_id,
                    type=SpanType.LLM,
                    name="openai_agents.llm",
                    parent_span_id=self._current_agent_span.span_id,
                    model=model if isinstance(model, str) else None,
                )
                meta: dict[str, Any] = {}
                ag_name = self._current_agent.get("name") or getattr(
                    agent, "name", None
                )
                if ag_name:
                    meta["agent_name"] = ag_name
                if meta:
                    span.metadata = meta
                # Capture the prompt sent to the model as input_data.
                try:
                    messages = []
                    if system_prompt:
                        messages.append(
                            {"role": "system", "content": str(system_prompt)}
                        )
                    for item in input_items or []:
                        try:
                            messages.append(
                                item.model_dump()
                                if hasattr(item, "model_dump")
                                else dict(item)
                            )
                        except Exception:  # noqa: BLE001
                            messages.append({"raw": str(item)})
                    if messages:
                        span.input_data = {"messages": messages}
                except Exception:  # noqa: BLE001
                    pass
                # Track this LLM span by the agent's id so we can pair with
                # on_llm_end. RunHooks doesn't pass a token, so we key on
                # agent identity (a single agent typically only has one
                # in-flight LLM call in this hook surface).
                self._llm_starts[id(agent)] = (span, start_mono, model)
            except Exception as exc:  # noqa: BLE001
                logger.debug("morse_ai: on_llm_start swallowed: %s", exc)
            await self._maybe_call_user(
                "on_llm_start", context, agent, system_prompt, input_items
            )

        async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
            try:
                key = id(agent)
                started = self._llm_starts.pop(key, None)
                if started is None:
                    return
                span, start_mono, model_at_start = started
                # Prefer the model the API actually served over the agent's
                # configured model captured at on_llm_start — with model routing
                # they can differ, and pricing must follow what actually ran.
                resp_model = getattr(response, "model", None)
                if isinstance(resp_model, str) and resp_model:
                    span.model = resp_model
                usage = _extract_usage(response)
                if usage:
                    span.input_tokens = usage.get("input_tokens")
                    span.output_tokens = usage.get("output_tokens")
                    span.total_tokens = usage.get("total_tokens")
                    if span.model:
                        cost = calculate_cost(
                            input_tokens=usage.get("input_tokens", 0) or 0,
                            output_tokens=usage.get("output_tokens", 0) or 0,
                            model=span.model,
                            cached_input_tokens=usage.get("cached_tokens", 0) or 0,
                        )
                        if cost is not None:
                            span.cost_usd = round(cost, 6)
                # Capture the model's response as output_data.
                # Handles two shapes: text message (content[].text) and
                # function/tool call (arguments field, no content).
                try:
                    output_items = getattr(response, "output", None) or []
                    texts, tool_calls = [], []
                    for item in output_items:
                        for part in getattr(item, "content", None) or []:
                            t = getattr(part, "text", None)
                            if t:
                                texts.append(t)
                        name = getattr(item, "name", None)
                        args = getattr(item, "arguments", None)
                        if args is not None:
                            with contextlib.suppress(Exception):
                                args = json.loads(args)
                            tool_calls.append({"name": name, "arguments": args})
                    if texts:
                        span.output_data = {"text": " ".join(texts)}
                    elif tool_calls:
                        span.output_data = {"tool_calls": tool_calls}
                except Exception:  # noqa: BLE001
                    pass
                _finalize_span(span, start_mono)
                self._state.completed_spans.append(span)
            except Exception as exc:  # noqa: BLE001
                logger.debug("morse_ai: on_llm_end swallowed: %s", exc)
            await self._maybe_call_user("on_llm_end", context, agent, response)

        async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
            try:
                start_mono = time.monotonic()
                tool_name = getattr(tool, "name", None) or "tool"
                span = SpanData(
                    trace_id=self._state.trace_id,
                    type=SpanType.TOOL,
                    name=str(tool_name),
                    parent_span_id=self._current_agent_span.span_id,
                )
                input_args = getattr(context, "tool_arguments", None)
                if input_args is not None:
                    try:
                        parsed = (
                            json.loads(input_args)
                            if isinstance(input_args, str)
                            else input_args
                        )
                        # Skip empty dicts — no-arg tools have nothing to show
                        if parsed:
                            span.input_data = _safe_json(parsed)
                    except Exception:  # noqa: BLE001
                        span.input_data = {"arguments": str(input_args)}
                self._tool_starts[id(tool)] = (span, start_mono, input_args)
            except Exception as exc:  # noqa: BLE001
                logger.debug("morse_ai: on_tool_start swallowed: %s", exc)
            await self._maybe_call_user("on_tool_start", context, agent, tool)

        async def on_tool_end(
            self, context: Any, agent: Any, tool: Any, result: Any
        ) -> None:
            try:
                key = id(tool)
                started = self._tool_starts.pop(key, None)
                if started is None:
                    return
                span, start_mono, _input = started
                if result is not None:
                    span.output_data = _safe_json(result)
                _finalize_span(span, start_mono)
                self._state.completed_spans.append(span)
            except Exception as exc:  # noqa: BLE001
                logger.debug("morse_ai: on_tool_end swallowed: %s", exc)
            await self._maybe_call_user("on_tool_end", context, agent, tool, result)

    return _MorseRunHooks


def _merge_hooks(
    state: _TraceState,
    root_span: SpanData,
    current_agent_holder: dict[str, str | None],
    user_hooks: Any | None,
) -> Any:
    """Build the Morse RunHooks instance for one Runner.run() call."""
    HooksCls = _build_av_run_hooks()
    return HooksCls(state, root_span, current_agent_holder, user_hooks)


def _attach_run_metadata(
    root_span: SpanData,
    starting_agent: Any,
    input: Any,
    session: Any,
) -> None:
    """Populate root agent span with name, session, input, and output-type metadata.

    MHQ-1023: never overwrite what a previous run already attributed. Guardrails
    (and any other nested ``Runner.run``) reuse the outer trace, so ``root_span``
    here is the OUTER run's root — last-write-wins put the last guardrail's name
    in ``metadata.agent_name`` of every root agent span, mis-attributing the whole
    run in the agent registry, cost/latency rollups and topology.
    """
    meta: dict[str, Any] = root_span.metadata or {}
    name = getattr(starting_agent, "name", None)
    if name:
        meta.setdefault("agent_name", name)
    session_id = _resolve_session_id(session)
    if session_id:
        meta.setdefault("session_id", session_id)
    output_type = getattr(starting_agent, "output_type", None)
    if output_type is not None:
        try:
            schema_name = getattr(output_type, "__name__", None) or str(output_type)
            meta.setdefault("structured_output_type", schema_name)
        except Exception:  # noqa: BLE001
            pass
    if meta:
        root_span.metadata = meta
    if input is not None and root_span.input_data is None:
        root_span.input_data = _safe_json(input)


def _finalize_root_with_output(
    root_span: SpanData, starting_agent: Any, result: Any
) -> None:
    """Attach the parsed structured output to the root span if applicable."""
    try:
        if getattr(starting_agent, "output_type", None) is None:
            return
        final_output = getattr(result, "final_output", None)
        if final_output is None:
            return
        meta: dict[str, Any] = root_span.metadata or {}
        # MHQ-1023: same last-write-wins hazard — a nested guardrail run also
        # has an output_type and would replace the outer run's output.
        meta.setdefault("structured_output", _safe_json(final_output))
        root_span.metadata = meta
    except Exception as exc:  # noqa: BLE001
        logger.debug("morse_ai: structured output capture swallowed: %s", exc)


def _wrap_async_run(original):
    async def traced_run(starting_agent, input, *args, **kwargs):
        return await _traced_runner_call(
            original,
            "agents.Runner.run",
            starting_agent,
            input,
            args,
            kwargs,
            awaitable=True,
        )

    traced_run.__wrapped__ = original
    return traced_run


def _wrap_sync_run(original):
    def traced_run_sync(starting_agent, input, *args, **kwargs):
        return _traced_runner_call(
            original,
            "agents.Runner.run_sync",
            starting_agent,
            input,
            args,
            kwargs,
            awaitable=False,
        )

    traced_run_sync.__wrapped__ = original
    return traced_run_sync


def _wrap_streamed_run(original):
    def traced_run_streamed(starting_agent, input, *args, **kwargs):
        return _traced_runner_call(
            original,
            "agents.Runner.run_streamed",
            starting_agent,
            input,
            args,
            kwargs,
            awaitable=False,
            streamed=True,
        )

    traced_run_streamed.__wrapped__ = original
    return traced_run_streamed


def _prepare_call(
    starting_agent: Any, input: Any, kwargs: dict[str, Any]
) -> tuple[Any, dict[str, str | None], Any | None, Any, Any]:
    """Set up trace state, root span, Morse hooks merged with user hooks.

    Returns (state, current_agent_holder, auto_run_cm, root_span, av_hooks).
    """
    # MHQ-672: this adapter makes the stale-trace leak reachable by design —
    # _prepare_call runs synchronously in the caller's context while the
    # matching __exit__ runs inside the returned coroutine. Adopting a state
    # whose trace was already emitted appended every later span to a dead trace
    # and dropped it with no error at any log level (28 of 33 conversations
    # lost). ``get_active_trace()`` is what makes that unreachable; MHQ-980
    # moved the guard there so all 51 readers inherit it instead of this one.
    state = get_active_trace()
    auto_run_cm = None
    if state is None:
        # No active trace — create one rooted at the agent name.
        agent_name = getattr(starting_agent, "name", None)
        auto_run_cm = _av_run(agent_name)
        auto_run_cm.__enter__()
        state = get_active_trace()
    assert state is not None
    # MHQ-1024: a guardrail's own Runner.run() belongs under the guardrail
    # span, not next to the run root — otherwise its LLM calls surface as
    # anonymous siblings of the agent's own calls, which is the reason
    # guardrail work was invisible in the first place.
    guardrail_parent = _guardrail_span.get()
    if guardrail_parent is not None and guardrail_parent.trace_id == state.trace_id:
        root_span = guardrail_parent
    else:
        # The active trace's top-of-stack span is the run's root agent span.
        root_span = state.span_stack[-1] if state.span_stack else None
    if root_span is None:
        # Edge case — fabricate a root agent span if the stack is empty.
        root_span = SpanData(
            trace_id=state.trace_id,
            type=SpanType.AGENT,
            name=getattr(starting_agent, "name", None) or "openai_agents.run",
        )
        state.span_stack.append(root_span)

    _attach_run_metadata(root_span, starting_agent, input, kwargs.get("session"))
    adopt_thread_id(_resolve_thread_id(kwargs))

    current_agent_holder: dict[str, str | None] = {
        "name": getattr(starting_agent, "name", None)
    }
    user_hooks = kwargs.get("hooks")
    av_hooks = _merge_hooks(state, root_span, current_agent_holder, user_hooks)
    kwargs["hooks"] = av_hooks
    return state, current_agent_holder, auto_run_cm, root_span, av_hooks


def _wrap_stream_events_for_close(
    result: Any,
    root_span: SpanData,
    starting_agent: Any,
    auto_run_cm: Any,
    av_hooks: Any,
) -> None:
    """Monkey-patch result.stream_events() to close the AV trace on completion.

    StreamedRunResult is consumed asynchronously after Runner.run_streamed()
    returns. Closing the trace at return time would flush before any
    on_llm_end/on_tool_end hooks fire. Instead we wrap stream_events() so
    the trace closes exactly when the stream is exhausted or errors out.
    """
    original_stream_events = getattr(result, "stream_events", None)
    if original_stream_events is None:
        # Fallback: close immediately (better than leaking the trace).
        av_hooks.finalize_open_agents()
        _finalize_root_with_output(root_span, starting_agent, result)
        if auto_run_cm is not None:
            try:  # noqa: SIM105
                auto_run_cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        return

    def _close() -> None:
        av_hooks.finalize_open_agents()
        _finalize_root_with_output(root_span, starting_agent, result)
        if auto_run_cm is not None:
            try:  # noqa: SIM105
                auto_run_cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass

    async def _wrapped_stream_events():
        try:
            async for event in original_stream_events():
                yield event
        finally:
            _close()

    result.stream_events = _wrapped_stream_events


def _traced_runner_call(
    original,
    span_name: str,
    starting_agent: Any,
    input: Any,
    args: tuple,
    kwargs: dict[str, Any],
    awaitable: bool,
    streamed: bool = False,
):
    """Common wrapper body for Runner.run / run_sync / run_streamed."""
    # AGE-240: mark openai-agents as the active framework for this call.
    # Set BEFORE _prepare_call so the auto-opened trace (when no caller-supplied
    # ``with morse_ai.run(...)`` is active) captures the framework at enter.
    set_active_framework(_FRAMEWORK_NAME)
    try:
        state, _agent_holder, auto_run_cm, root_span, av_hooks = _prepare_call(
            starting_agent, input, kwargs
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("morse_ai: openai_agents prepare failed: %s", exc)
        # Fall back to untraced original call.
        try:
            return original(starting_agent, input, *args, **kwargs)
        finally:
            clear_active_framework()

    if awaitable:

        async def _run_async():
            try:
                result = await original(starting_agent, input, *args, **kwargs)
                av_hooks.finalize_open_agents()
                _finalize_root_with_output(root_span, starting_agent, result)
                return result
            finally:
                if auto_run_cm is not None:
                    try:  # noqa: SIM105
                        auto_run_cm.__exit__(None, None, None)
                    except Exception:  # noqa: BLE001
                        pass
                clear_active_framework()

        return _run_async()

    # Sync path (run_sync OR run_streamed — both return synchronously).
    try:
        result = original(starting_agent, input, *args, **kwargs)
    except BaseException:
        if auto_run_cm is not None:
            try:  # noqa: SIM105
                auto_run_cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        clear_active_framework()
        raise

    if streamed:
        # The stream is consumed asynchronously by the caller — closing the
        # trace here would flush before any on_llm_end / on_tool_end hooks
        # fire, losing all child spans and cost data. Instead, wrap
        # stream_events() so the trace closes when iteration completes.
        _wrap_stream_events_for_close(
            result, root_span, starting_agent, auto_run_cm, av_hooks
        )
        clear_active_framework()
        return result

    av_hooks.finalize_open_agents()
    _finalize_root_with_output(root_span, starting_agent, result)
    if auto_run_cm is not None:
        try:  # noqa: SIM105
            auto_run_cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
    clear_active_framework()
    return result


def _install_guardrail_patches() -> None:
    """Patch the guardrail classes so checks emit `guardrail` spans (MHQ-1024).

    Optional — `RunHooks` has no guardrail surface, and a version of
    openai-agents that moves these classes should still get run/tool/LLM spans.
    """
    global _original_input_guardrail_run, _original_output_guardrail_run

    try:
        from agents.guardrail import (  # type: ignore[import-not-found]
            InputGuardrail,
            OutputGuardrail,
        )
    except (ImportError, AttributeError) as exc:
        logger.debug("morse_ai: guardrail classes unavailable: %s", exc)
        return

    try:
        _original_input_guardrail_run = InputGuardrail.run
        InputGuardrail.run = _wrap_guardrail_run(  # type: ignore[method-assign]
            _original_input_guardrail_run, "input", ("input", 1)
        )
        _original_output_guardrail_run = OutputGuardrail.run
        OutputGuardrail.run = _wrap_guardrail_run(  # type: ignore[method-assign]
            _original_output_guardrail_run, "output", ("agent_output", 2)
        )
    except (AttributeError, TypeError) as exc:
        logger.warning("morse_ai: failed to patch guardrail run: %s", exc)


def _uninstall_guardrail_patches() -> None:
    """Restore the original guardrail `run` methods. Idempotent."""
    global _original_input_guardrail_run, _original_output_guardrail_run

    try:
        from agents.guardrail import (  # type: ignore[import-not-found]
            InputGuardrail,
            OutputGuardrail,
        )
    except (ImportError, AttributeError):
        return

    if _original_input_guardrail_run is not None:
        InputGuardrail.run = _original_input_guardrail_run  # type: ignore[method-assign]
        _original_input_guardrail_run = None
    if _original_output_guardrail_run is not None:
        OutputGuardrail.run = _original_output_guardrail_run  # type: ignore[method-assign]
        _original_output_guardrail_run = None


def is_installed() -> bool:
    """Return True if the OpenAI Agents adapter has installed its monkey-patches."""
    try:
        import agents  # type: ignore[import-not-found]
    except ImportError:
        return False
    return getattr(agents, _INSTALL_FLAG, False)


@absorb_errors(fingerprint="morse_ai.adapters.openai_agents.install")
def install() -> bool:
    """Monkey-patch the OpenAI Agents `Runner` class to emit Morse spans.

    Idempotent — safe to call repeatedly.
    """
    global _original_run, _original_run_sync, _original_run_streamed

    try:
        import agents  # type: ignore[import-not-found]
    except ImportError:
        return False

    if getattr(agents, _INSTALL_FLAG, False):
        return True

    try:
        from agents.run import Runner  # type: ignore[import-not-found]
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "morse_ai: OpenAI Agents adapter could not import Runner: %s — "
            "openai-agents calls will run untraced",
            exc,
        )
        return False

    try:
        _original_run = Runner.run
        Runner.run = _wrap_async_run(_original_run)  # type: ignore[method-assign]
    except (AttributeError, TypeError) as exc:
        logger.warning(
            "morse_ai: failed to patch Runner.run: %s",
            exc,
        )
        return False

    try:
        _original_run_sync = Runner.run_sync
        Runner.run_sync = _wrap_sync_run(_original_run_sync)  # type: ignore[method-assign]
    except (AttributeError, TypeError):
        # run_sync is optional — older versions may not have it.
        _original_run_sync = None

    try:
        _original_run_streamed = Runner.run_streamed
        Runner.run_streamed = _wrap_streamed_run(  # type: ignore[method-assign]
            _original_run_streamed
        )
    except (AttributeError, TypeError):
        _original_run_streamed = None

    _install_guardrail_patches()

    setattr(agents, _INSTALL_FLAG, True)
    logger.info("morse_ai: OpenAI Agents adapter installed")

    # `agents` is now in sys.modules — flush a dependency snapshot so
    # "Detected stack" populates without relying on the late-import watcher.
    try:
        from morse_ai import _client as _av_client
        from morse_ai import _flush_dependencies_async

        if _av_client is not None:
            _flush_dependencies_async(_av_client)
    except Exception as exc:  # noqa: BLE001
        logger.debug("morse_ai: adapter dependency flush swallowed: %s", exc)

    return True


@absorb_errors(fingerprint="morse_ai.adapters.openai_agents.uninstall")
def uninstall() -> None:
    """Restore original Runner methods. Idempotent."""
    global _original_run, _original_run_sync, _original_run_streamed

    try:
        import agents  # type: ignore[import-not-found]
    except ImportError:
        return

    if not getattr(agents, _INSTALL_FLAG, False):
        return

    try:
        from agents.run import Runner  # type: ignore[import-not-found]
    except (ImportError, AttributeError):
        return

    if _original_run is not None:
        Runner.run = _original_run  # type: ignore[method-assign]
        _original_run = None
    if _original_run_sync is not None:
        Runner.run_sync = _original_run_sync  # type: ignore[method-assign]
        _original_run_sync = None
    if _original_run_streamed is not None:
        Runner.run_streamed = _original_run_streamed  # type: ignore[method-assign]
        _original_run_streamed = None

    _uninstall_guardrail_patches()

    setattr(agents, _INSTALL_FLAG, False)
