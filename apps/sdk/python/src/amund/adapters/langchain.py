"""LangChain callback handler adapter — instruments chains, LLMs, and tools.

Drop-in instrumentation: add MorseCallbackHandler to any LangChain chain or agent
and spans are produced automatically without any manual SDK calls.

Usage:
    from amund.adapters.langchain import MorseCallbackHandler

    handler = MorseCallbackHandler(agent_name="my-agent")
    chain = MyChain(callbacks=[handler])
    chain.run("What is 2+2?")

If langchain_core is not installed, the class is still importable — it simply
inherits from a stub base class and produces no output (silent no-op).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("amund")

# AGE-260 (Phase 12.5 Issue 3): cap on the JSON-serialized size of any single
# input_data / output_data dict captured from a chain or node. LangGraph state
# can grow large over many turns (full message histories, retrieved docs, tool
# outputs); shipping a multi-MB Pydantic dump as input_data balloons the
# transport queue. Truncation falls back to a ``{"_truncated": true, ...}``
# marker so the dashboard still shows a meaningful preview rather than null.
_MAX_STATE_PAYLOAD_BYTES: int = 64 * 1024  # 64 KB JSON

# ---------------------------------------------------------------------------
# Optional dependency: langchain_core
# ---------------------------------------------------------------------------

try:
    from langchain_core.callbacks import BaseCallbackHandler as _LCBaseCallbackHandler

    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover
    # Define a stub so the class can be defined and imported without langchain.
    class _LCBaseCallbackHandler:  # type: ignore[no-redef]
        """Stub BaseCallbackHandler for environments without langchain_core."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    _LANGCHAIN_AVAILABLE = False


# ---------------------------------------------------------------------------
# MorseCallbackHandler
# ---------------------------------------------------------------------------

from amund._internal.active_framework import (  # noqa: E402
    clear_active_framework,
    set_active_framework,
)
from amund._internal.agent_name import (  # noqa: E402
    identity_legacy,
    resolve_agent_identity,
)
from amund.adapters._retrieval import build_documents  # noqa: E402
from amund.spans import SpanType  # noqa: E402
from amund.tracing import (  # noqa: E402
    TracedRun,
    TracedSpan,
    _active_trace,
    adopt_native_root,
    adopt_thread_id,
    get_active_trace,
    run,
    span,
)

if TYPE_CHECKING:
    from uuid import UUID

# AGE-240: framework auto-discovery tag. The plain LangChain handler emits
# ``langchain``; the LangGraph subclass overrides ``_framework_name`` to
# emit ``langgraph`` so dashboards can distinguish the two.
_FRAMEWORK_NAME_LANGCHAIN = "langchain"

__all__ = ["MorseCallbackHandler"]

# AID-01: compiled-graph / runnable names that are framework defaults, not an
# app's identity. Never a "framework-native root name".
_GENERIC_ROOT_NAMES = frozenset(
    {
        "langgraph",
        "langchain-agent",
        "chain",
        "agentexecutor",
        "runnable",
        "runnablesequence",
        "runnablelambda",
        "runnableparallel",
    }
)

# AID-02: nodes that are framework plumbing even when they call a model
# (prebuilt ReAct ``agent``/``tools``). They stay spans; their LLM call belongs
# to the nearest real node, else the root agent.
_NO_PROMOTE_NODES = frozenset(
    {"__start__", "__end__", "chain", "tools", "agent", "model", "route"}
    | _GENERIC_ROOT_NAMES
)


def _native_root_name(serialized: Any, kwargs: dict[str, Any]) -> str | None:
    """Name the framework gave the root runnable, unless it is a default."""
    for cand in (kwargs.get("name"), (serialized or {}).get("name")):
        if (
            isinstance(cand, str)
            and cand.strip()
            and cand.strip().lower() not in _GENERIC_ROOT_NAMES
        ):
            return cand.strip()
    return None


def _extract_token_usage(
    response: Any,
) -> tuple[int | None, int | None, int | None, int | None]:
    """Return (input_tokens, output_tokens, total_tokens, cached_input_tokens).

    AGE-240: LangChain provider integrations expose usage in three different
    places. Try each in order and return the first non-empty hit:

    1. ``generations[0][0].message.usage_metadata`` — LangChain v0.3 standard
       (Anthropic, OpenAI v0.3+). Keys: ``input_tokens``, ``output_tokens``,
       ``total_tokens``, optionally ``input_token_details.cache_read``.
    2. ``generations[0][0].message.response_metadata["token_usage"]`` — some
       chat integrations. Keys: ``prompt_tokens``, ``completion_tokens``,
       ``total_tokens``.
    3. ``llm_output["token_usage"]`` — OpenAI legacy. Keys:
       ``prompt_tokens``, ``completion_tokens``, ``total_tokens``.
    4. ``llm_output["usage"]`` — Anthropic legacy. Keys: ``input_tokens``,
       ``output_tokens``.

    Returns (None, None, None, None) when nothing matches.
    """
    # 1) generations[].message.usage_metadata (LC v0.3 standard)
    try:
        gens = getattr(response, "generations", None) or []
        if gens and gens[0]:
            msg = getattr(gens[0][0], "message", None)
            if msg is not None:
                um = getattr(msg, "usage_metadata", None) or {}
                if um:
                    in_t = um.get("input_tokens")
                    out_t = um.get("output_tokens")
                    total = um.get("total_tokens")
                    details = um.get("input_token_details") or {}
                    cached = details.get("cache_read") or details.get("cached")
                    if in_t is not None or out_t is not None:
                        return in_t, out_t, total, cached
                meta = getattr(msg, "response_metadata", None) or {}
                token_usage = meta.get("token_usage") or {}
                if token_usage:
                    in_t = token_usage.get("prompt_tokens") or token_usage.get(
                        "input_tokens"
                    )
                    out_t = token_usage.get("completion_tokens") or token_usage.get(
                        "output_tokens"
                    )
                    total = token_usage.get("total_tokens")
                    return in_t, out_t, total, None
    except Exception:  # noqa: BLE001 — never break the host
        pass

    # 2) llm_output.token_usage (OpenAI legacy shape)
    llm_output: dict[str, Any] = getattr(response, "llm_output", {}) or {}
    token_usage: dict[str, Any] = llm_output.get("token_usage") or {}
    if token_usage:
        in_t = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
        out_t = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
        total = token_usage.get("total_tokens")
        return in_t, out_t, total, None

    # 3) llm_output.usage (Anthropic legacy)
    usage = llm_output.get("usage") or {}
    if usage:
        return (
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("total_tokens"),
            usage.get("cache_read_input_tokens"),
        )

    return None, None, None, None


def _extract_cache_write_tokens(response: Any) -> int | None:
    """Prompt-cache WRITE tokens from LangChain's standard ``usage_metadata``.

    MHQ-1414: langchain-anthropic reports ``input_token_details.cache_creation``
    alongside ``cache_read``; both are included in ``input_tokens``. Without
    this the cache-write portion of a call was priced at the plain input rate.
    """
    try:
        gens = getattr(response, "generations", None) or []
        if gens and gens[0]:
            msg = getattr(gens[0][0], "message", None)
            um = getattr(msg, "usage_metadata", None) or {}
            details = um.get("input_token_details") or {}
            value = details.get("cache_creation")
            if isinstance(value, int) and value > 0:
                return value
    except Exception:  # noqa: BLE001 — never break the host
        pass
    llm_output: dict[str, Any] = getattr(response, "llm_output", {}) or {}
    usage = llm_output.get("usage") or {}
    if not isinstance(usage, dict):
        return None
    value = usage.get("cache_creation_input_tokens")
    return value if isinstance(value, int) and value > 0 else None


def _extract_model(response: Any) -> str | None:
    """Best-effort model-name extraction from an LLMResult.

    AGE-240: tries ``generations[0][0].message.response_metadata.model_name``
    (LC v0.3), then ``llm_output["model_name"]`` / ``llm_output["model"]``.
    """
    try:
        gens = getattr(response, "generations", None) or []
        if gens and gens[0]:
            msg = getattr(gens[0][0], "message", None)
            if msg is not None:
                meta = getattr(msg, "response_metadata", None) or {}
                model = meta.get("model_name") or meta.get("model")
                if model:
                    return str(model)
    except Exception:  # noqa: BLE001
        pass
    llm_output: dict[str, Any] = getattr(response, "llm_output", {}) or {}
    model = llm_output.get("model_name") or llm_output.get("model")
    return str(model) if model else None


def _extract_response_id(response: Any) -> str | None:
    """The provider's id for this call, from an LLMResult (MHQ-1880).

    ``response_metadata["id"]`` is the provider's message id (Anthropic
    ``msg_…``, OpenAI ``chatcmpl-…``). ``message.id`` is not used: LangChain
    may fill it with its own run id.
    """
    try:
        gens = getattr(response, "generations", None) or []
        if gens and gens[0]:
            msg = getattr(gens[0][0], "message", None)
            meta = getattr(msg, "response_metadata", None) or {}
            rid = meta.get("id")
            if isinstance(rid, str) and rid:
                return rid
    except Exception:  # noqa: BLE001
        pass
    llm_output: dict[str, Any] = getattr(response, "llm_output", {}) or {}
    rid = llm_output.get("id")
    return rid if isinstance(rid, str) and rid else None


def _documents_with_preview(documents: Any) -> list[dict[str, Any]]:
    """``build_documents`` plus the legacy 200-char ``content_preview`` key."""
    docs = build_documents(documents)
    for d in docs:
        d["content_preview"] = d["content"][:200]
    return docs


class MorseCallbackHandler(_LCBaseCallbackHandler):
    """LangChain callback handler that produces Amund spans automatically.

    Attach to any chain or agent via the ``callbacks`` argument:

        handler = MorseCallbackHandler("my-agent")
        chain = SomeChain(..., callbacks=[handler])

    Span mapping:
        - on_chain_start / on_chain_end  → span_type="agent"
        - on_llm_start / on_llm_end      → span_type="llm"
        - on_tool_start / on_tool_end     → span_type="tool"

    When called inside an existing ``amund.run()`` context the handler
    appends child spans to that trace.  When called without any enclosing
    context (e.g. in a standalone LangChain script) the handler auto-creates
    a trace that is flushed when the outermost chain completes.
    """

    # AGE-240: subclasses (LangGraphCallbackHandler) override to "langgraph".
    _framework_name: str = _FRAMEWORK_NAME_LANGCHAIN

    def __init__(self, agent_name: str | None = None) -> None:
        super().__init__()
        # MHQ-1100: None = not explicit; resolved per trace (see
        # amund._internal.agent_name.resolve_agent_name).
        # MHQ-1412 (documented behaviour, not a bug): ``agent_name`` only names
        # a trace this handler creates itself. Inside an enclosing
        # ``amund.run("name")`` the run's name wins and this argument is
        # ignored, because the handler never opens a trace of its own there.
        self.agent_name = agent_name
        # Framework-native name learned from callbacks (e.g. graph name).
        self._native_name: str | None = None
        # Map str(run_id) -> TracedSpan for every in-flight LangChain run
        self._span_stack: dict[str, TracedSpan] = {}
        # AGE-240: tracks whether on_chain_start populated the thread-local so
        # only the matching on_chain_end/error clears it (handles nested chains).
        self._fw_set_depth: int = 0
        # The auto-created TracedRun (set only when handler starts the trace)
        self._run_context: TracedRun | None = None
        self._auto_trace: bool = False
        # If the handler is constructed inside an explicit amund.run()
        # context, remember that parent state. Some frameworks invoke callbacks
        # after switching asyncio/task contexts where ContextVar propagation can
        # be lost; restoring this state keeps LLM/tool spans inside the intended
        # trace without app code manually threading trace ids.
        # Guarded: a state that is already closed at construction time is never
        # worth snapshotting. `_restore_inherited_trace` re-checks `is_closed`
        # anyway, because the parent can finish between here and the callback.
        self._inherited_trace_state = get_active_trace()
        # LangGraph fires a separate on_chain_start(parent_run_id=None) for
        # each execution step. Track all root run IDs so we don't close the
        # auto-trace until every root step has finished.
        self._root_run_ids: set[str] = set()
        # AID-05: span ownership by LangChain run id. Parents come from the
        # callback's ``parent_run_id`` tree, never from the shared stack top,
        # which parallel branches and async tasks interleave.
        self._span_ids: dict[str, str] = {}
        self._claimed_roots: set[str] = set()
        self._run_parents: dict[str, str | None] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_run(self, **kwargs: Any) -> TracedRun:
        """Build the auto-trace ``run()``, recording which ladder tier named it."""
        name, source = resolve_agent_identity(self.agent_name, self._native_name)
        traced_run = run(name, **kwargs)
        traced_run._identity_source = source
        return traced_run

    def _root_native_name(self, serialized: Any, kwargs: dict[str, Any]) -> str | None:
        return _native_root_name(serialized, kwargs)

    def _parent_span_for(self, parent_run_id: UUID | None) -> str | None:
        """Span id of the nearest tracked ancestor run, else None (stack top)."""
        if identity_legacy():
            return None
        key = str(parent_run_id) if parent_run_id is not None else None
        for _ in range(64):
            if key is None:
                return None
            if key in self._span_ids:
                return self._span_ids[key]
            key = self._run_parents.get(key)
        return None

    def _claim_root(
        self,
        run_id: UUID,
        serialized: Any,
        kwargs: dict[str, Any],
        name: str,
        inputs: Any = None,
    ) -> bool:
        """AID-01/04: the first root run IS the trace's root agent, not a child.

        The framework's own root name (graph name, ``run_name``) names the agent
        (or becomes the agent while an explicit ``run("app")`` name turns into
        the ``morse.app`` label). No extra agent span is emitted for it.
        """
        if identity_legacy() or self._chain_kind(name, kwargs) == "node":
            return False
        self._ensure_trace()
        state = get_active_trace()
        if state is None or state.root_adopted or not state.span_stack:
            return False
        # A name the customer gave this handler for the trace it opened itself
        # is explicit and is not re-derived from the graph.
        own_explicit = self._auto_trace and bool(self.agent_name)
        adopt_native_root(
            state, None if own_explicit else self._root_native_name(serialized, kwargs)
        )
        root = state.span_stack[0]
        self._span_ids[str(run_id)] = root.span_id
        self._run_parents[str(run_id)] = None
        self._claimed_roots.add(str(run_id))
        if root.input_data is None:
            wrapped = self._ensure_dict(inputs)
            if wrapped is not None:
                root.input_data = wrapped
        return True

    def _finish_claimed_root(
        self, run_id: UUID, outputs: Any = None, error: BaseException | None = None
    ) -> None:
        """The claimed root has no span of its own: record its IO/error on the
        trace's root span instead."""
        if str(run_id) not in self._claimed_roots:
            return
        self._claimed_roots.discard(str(run_id))
        state = get_active_trace() or self._inherited_trace_state
        root = state.span_stack[0] if state is not None and state.span_stack else None
        if root is None:
            return
        if error is not None:
            from amund.tracing import exception_type_name

            root.exception_type = root.exception_type or exception_type_name(error)
        elif root.output_data is None:
            wrapped = self._ensure_dict(outputs)
            if wrapped is not None:
                root.output_data = wrapped

    def _chain_kind(self, name: str, kwargs: dict[str, Any]) -> str:  # noqa: ARG002
        """AID-02: how a non-root chain is recorded (``chain`` or ``node``)."""
        return "chain"

    def _promote_owner(self, parent_run_id: UUID | None) -> None:
        """AID-02: a node that owns an LLM call is a real role, so it is an agent.

        Walks up from the LLM call to the nearest node that is not framework
        plumbing and retypes it. Nodes that never call a model stay spans.
        """
        key = str(parent_run_id) if parent_run_id is not None else None
        for _ in range(64):
            if key is None:
                return
            traced = self._span_stack.get(key)
            sp = traced._span if traced is not None else None
            md = (sp.metadata or {}) if sp is not None else {}
            if (
                sp is not None
                and md.get("node_kind") == "node"
                and sp.name.lower() not in _NO_PROMOTE_NODES
            ):
                sp.type = SpanType.AGENT
                return
            key = self._run_parents.get(key)

    def _ensure_trace(self) -> None:
        """Auto-create a trace if no active trace exists."""
        # MHQ-980: this read was unguarded even though the module below it
        # carries the comment explaining the hazard — a closed state in the var
        # made this return early, so no trace was created and every subsequent
        # span went nowhere. The failure mode is the opposite of MHQ-672's
        # (nothing adopted rather than the wrong thing adopted) with the same
        # result, which is why the guard belongs in one accessor, not per site.
        if get_active_trace() is not None:
            return
        if self._restore_inherited_trace():
            return
        if self._run_context is None:
            self._run_context = self._new_run()
            self._run_context.__enter__()
            self._auto_trace = True
        else:
            # ContextVar was cleared by an async context switch but the trace
            # is still open. Re-set the ContextVar so subsequent spans attach
            # to the same trace instead of spawning a brand-new one.
            _active_trace.set(self._run_context._state)

    def _restore_inherited_trace(self) -> bool:
        """Restore a parent trace captured at handler construction.

        Checks both that the parent state has open spans AND that it has not
        been marked closed. The ``is_closed`` flag is set by TracedRun.__exit__
        so handlers reused after the parent run exits do not attach new spans
        to a finished trace (the span_stack check alone is insufficient because
        the snapshot was taken at construction time, not at call time).
        """
        state = self._inherited_trace_state
        if state is None or not state.span_stack or state.is_closed:
            return False
        _active_trace.set(state)
        self._auto_trace = False
        return True

    def _maybe_close_auto_trace(self, error: BaseException | None = None) -> None:
        """Close the auto-created trace when all spans are done."""
        if (
            self._auto_trace
            and not self._span_stack
            and not self._root_run_ids
            and self._run_context is not None
        ):
            if error is not None:
                self._run_context.set_error(str(error))
            self._run_context.__exit__(None, None, None)
            self._run_context = None
            self._auto_trace = False
        if not self._span_stack and not self._root_run_ids:
            self._span_ids.clear()
            self._run_parents.clear()

    @staticmethod
    def _ensure_dict(data: Any) -> dict | None:
        """Coerce LangChain / LangGraph chain inputs/outputs into a dict payload.

        Server schema requires span ``input_data`` / ``output_data`` to be a
        dict. LangChain passes a wide variety of shapes through chain
        callbacks; this helper normalizes them so the SDK never silently
        drops state to ``None``:

        - ``None``                             → ``None`` (legitimately null)
        - ``dict``                              → unchanged
          (also covers ``TypedDict``, which is ``dict`` at runtime)
        - Pydantic v2 ``BaseModel``            → ``model.model_dump()``
        - Pydantic v1 ``BaseModel``            → ``model.dict()``
        - ``dataclass`` instance               → ``dataclasses.asdict(...)``
        - other (str / int / list / object)    → ``{"value": str(data)}``

        AGE-260 (Phase 12.5 Issue 3): replaces the prior strict isinstance
        check in ``on_chain_start`` / ``on_chain_end`` which dropped
        Pydantic + dataclass LangGraph state to ``None`` and made many node
        spans appear input/output-less on the dashboard.

        Oversized payloads (JSON > ``_MAX_STATE_PAYLOAD_BYTES``) are
        replaced with a truncation marker so the transport queue is
        bounded but the dashboard still shows a usable preview.
        """
        if data is None:
            return None

        coerced: dict | None = None

        # 1. dict / TypedDict — pass through unchanged.
        if isinstance(data, dict):
            coerced = data
        else:
            # 2. Pydantic v2 — has model_dump().
            model_dump = getattr(data, "model_dump", None)
            if callable(model_dump):
                try:
                    result = model_dump()
                    if isinstance(result, dict):
                        coerced = result
                except Exception:  # noqa: BLE001 — fall through to next coercion
                    coerced = None

            # 3. Pydantic v1 — has .dict() (but exclude generic .dict bound
            # methods on unrelated objects: only use it when also a BaseModel-
            # ish object: has __fields__ attribute too).
            if coerced is None:
                pydantic_v1_dict = getattr(data, "dict", None)
                if callable(pydantic_v1_dict) and hasattr(data, "__fields__"):
                    try:
                        result = pydantic_v1_dict()
                        if isinstance(result, dict):
                            coerced = result
                    except Exception:  # noqa: BLE001
                        coerced = None

            # 4. dataclass instance — asdict() handles nested dataclasses.
            if (
                coerced is None
                and dataclasses.is_dataclass(data)
                and not isinstance(data, type)
            ):
                try:
                    coerced = dataclasses.asdict(data)
                except Exception:  # noqa: BLE001
                    coerced = None

            # 5. Last-resort fallback — wrap str representation. NEVER
            # return None for a non-None input (the whole point of the
            # AGE-260 fix is that legitimate state never silently drops).
            if coerced is None:
                coerced = {"value": str(data)}

        # Enforce a soft payload cap so a 10MB Pydantic dump doesn't balloon
        # the transport queue. We JSON-encode with default=str so unhashable
        # / non-JSON-native values (datetimes, UUIDs, etc.) measure correctly.
        try:
            encoded = json.dumps(coerced, default=str, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            # Fallback: stringify the whole thing. Should be rare.
            return {
                "_truncated": True,
                "_reason": "unserializable",
                "value": str(coerced)[:1024],
            }

        if len(encoded.encode("utf-8")) > _MAX_STATE_PAYLOAD_BYTES:
            # Truncate by re-encoding the str view and capping bytes.
            preview = encoded[: _MAX_STATE_PAYLOAD_BYTES // 2]
            return {
                "_truncated": True,
                "_reason": "payload_too_large",
                "_original_size_bytes": len(encoded.encode("utf-8")),
                "_max_bytes": _MAX_STATE_PAYLOAD_BYTES,
                "preview": preview,
            }

        return coerced

    def _open_span(
        self,
        run_id: UUID,
        name: str,
        span_type: str,
        input_data: Any = None,
        parent_run_id: UUID | None = None,
        kind: str | None = None,
    ) -> TracedSpan | None:
        """Create and enter a new child span, store in _span_stack.

        ``kind`` marks an AID-02 chain/node span (framework bookkeeping, not an
        agent); it is recorded as span metadata ``node_kind``.
        """
        # MHQ-1412: framework used to be set only in on_chain_start, so traces
        # made of llm/tool/retriever callbacks alone had framework NULL. Chains
        # keep their own accounting (on_chain_start/end); other starts are
        # counted here and released in _close_span. Set before AND after
        # _ensure_trace: the first is read by a trace the handler creates, the
        # second stamps an already-open trace's state.
        marks_framework = span_type != "agent" and kind is None
        if marks_framework:
            set_active_framework(self._framework_name)
        self._ensure_trace()
        if marks_framework:
            set_active_framework(self._framework_name)
        traced = span(
            name,
            span_type=span_type,
            parent_span_id=self._parent_span_for(parent_run_id),
        )
        traced.__enter__()
        if traced._active and traced._span is not None:
            traced._marks_framework = marks_framework  # type: ignore[attr-defined]
            if marks_framework:
                self._fw_set_depth += 1
            self._span_ids[str(run_id)] = traced._span.span_id
            self._run_parents[str(run_id)] = (
                str(parent_run_id) if parent_run_id is not None else None
            )
            if kind is not None:
                traced.set_metadata("node_kind", kind)
            if span_type == "llm":
                self._promote_owner(parent_run_id)
            wrapped = self._ensure_dict(input_data)
            if wrapped is not None:
                traced._span.input_data = wrapped
            self._span_stack[str(run_id)] = traced
            return traced
        # No active trace (shouldn't happen after _ensure_trace, but be safe)
        return None

    def _close_span(
        self, run_id: UUID, output_data: Any = None, error: BaseException | None = None
    ) -> None:
        """Pop and close the span for run_id."""
        key = str(run_id)
        traced = self._span_stack.pop(key, None)
        if traced is None:
            return
        if getattr(traced, "_marks_framework", False):
            self._fw_set_depth = max(0, self._fw_set_depth - 1)
            if self._fw_set_depth == 0:
                clear_active_framework()
        if traced._active and traced._span is not None and output_data is not None:
            wrapped = self._ensure_dict(output_data)
            if wrapped is not None:
                traced._span.output_data = wrapped
        if error is not None:
            traced.__exit__(type(error), error, None)
        else:
            traced.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # Chain callbacks
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # AGE-240: SDK contract is silent no-op. ANY internal failure must
        # NOT propagate to the host (LangChain logs it as a warning, which
        # is noisy + scary in customer logs). The dogfood Copilot exposed
        # a NoneType.get error that the prior `(serialized or {}).get(...)`
        # fix didn't fully cover — root-cause some unknown LC v0.3 chain
        # shape. Defensive wrapper hides any future regressions too.
        try:
            set_active_framework(self._framework_name)
            self._fw_set_depth += 1
            # LangChain passes ``serialized=None`` for chains built outside
            # the registered class hierarchy.
            s = serialized if isinstance(serialized, dict) else {}
            id_path = s.get("id")
            if isinstance(id_path, list) and id_path:
                chain_name = id_path[-1]
            else:
                chain_name = s.get("name", "chain")
            # LangGraph fires on_chain_start(parent_run_id=None) for each
            # execution step, not just the outermost one. Track every root
            # run so _maybe_close_auto_trace doesn't close the trace between
            # steps (which would create a new trace for each subsequent step).
            if parent_run_id is None:
                self._root_run_ids.add(str(run_id))
            # AGE-260: route through _ensure_dict so Pydantic / dataclass /
            # TypedDict LangGraph state surfaces on the span instead of being
            # silently dropped to None (the prior strict-isinstance check was
            # the root cause of "many LangGraph nodes show input/output null").
            claimed = False
            if parent_run_id is None:
                self._native_name = self._root_native_name(serialized, kwargs)
                claimed = self._claim_root(
                    run_id, serialized, kwargs, str(chain_name), inputs
                )
            if not claimed:
                legacy = identity_legacy()
                self._open_span(
                    run_id,
                    str(chain_name),
                    span_type="agent" if legacy else "custom",
                    input_data=inputs,
                    parent_run_id=parent_run_id,
                    kind=None if legacy else self._chain_kind(str(chain_name), kwargs),
                )
            # MHQ-1022: LangGraph copies ``configurable.thread_id`` onto the
            # run metadata, which is the only place a checkpointed
            # conversation's identity reaches a callback handler.
            metadata = kwargs.get("metadata")
            if isinstance(metadata, dict):
                adopt_thread_id(metadata.get("thread_id"))
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "amund: MorseCallbackHandler.on_chain_start swallowed: %s",
                exc,
            )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            self._finish_claimed_root(run_id, outputs=outputs)
            self._close_span(run_id, output_data=outputs)
            self._root_run_ids.discard(str(run_id))
            self._maybe_close_auto_trace()
            self._fw_set_depth = max(0, self._fw_set_depth - 1)
            if self._fw_set_depth == 0:
                clear_active_framework()
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: MorseCallbackHandler.on_chain_end swallowed: %s", exc)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            self._finish_claimed_root(run_id, error=error)
            self._close_span(run_id, error=error)
            self._root_run_ids.discard(str(run_id))
            self._maybe_close_auto_trace(error=error)
            self._fw_set_depth = max(0, self._fw_set_depth - 1)
            if self._fw_set_depth == 0:
                clear_active_framework()
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "amund: MorseCallbackHandler.on_chain_error swallowed: %s",
                exc,
            )

    # ------------------------------------------------------------------
    # LLM callbacks
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            s = serialized if isinstance(serialized, dict) else {}
            id_path = s.get("id")
            if isinstance(id_path, list) and id_path:
                llm_name = id_path[-1]
            else:
                llm_name = s.get("name", "llm")
            model_name: str | None = None
            if isinstance(invocation_params, dict):
                model_name = invocation_params.get(
                    "model_name"
                ) or invocation_params.get("model")
            traced = self._open_span(
                run_id,
                str(llm_name),
                span_type="llm",
                input_data={"prompts": prompts} if prompts else None,
                parent_run_id=parent_run_id,
            )
            if (
                traced is not None
                and traced._active
                and traced._span is not None
                and model_name
            ):
                traced._span.model = model_name
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: MorseCallbackHandler.on_llm_start swallowed: %s", exc)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            self._on_llm_end_impl(response, run_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: MorseCallbackHandler.on_llm_end swallowed: %s", exc)
            _report(exc, "morse_ai.adapters.langchain.00")
            # Try to close the span anyway so it doesn't dangle.
            try:
                self._close_span(run_id)
                self._maybe_close_auto_trace()
            except Exception:  # noqa: BLE001
                pass

    def _on_llm_end_impl(self, response: Any, run_id: UUID) -> None:
        key = str(run_id)
        traced = self._span_stack.get(key)

        # Extract response text
        output_text = ""
        if hasattr(response, "generations") and response.generations:
            first = response.generations[0]
            if first and hasattr(first[0], "text"):
                output_text = first[0].text
            if not output_text and first:
                msg = getattr(first[0], "message", None)
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    output_text = content
                elif isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, str):
                            parts.append(block)
                        elif isinstance(block, dict) and isinstance(
                            block.get("text"), str
                        ):
                            parts.append(block["text"])
                    output_text = "\n".join(parts)
            # MHQ-1410: a tool-calling turn has empty text, so the span read
            # {"text": ""}. Append the requested calls, matching the raw
            # OpenAI/Anthropic adapters' "[tool_call:name] args" form.
            msg = getattr(first[0], "message", None) if first else None
            for tool_call in getattr(msg, "tool_calls", None) or []:
                if not isinstance(tool_call, dict):
                    continue
                name = tool_call.get("name") or "tool"
                args = tool_call.get("args")
                label = f"[tool_call:{name}]"
                if args:
                    try:
                        label = f"{label} {json.dumps(args, default=str)}"
                    except (TypeError, ValueError):
                        label = f"{label} {args}"
                output_text = f"{output_text}\n{label}" if output_text else label

        # AGE-240: extract usage + cost across provider shapes.
        # LangChain integrations stash usage in different places:
        #   - OpenAI legacy:   response.llm_output["token_usage"] with
        #                      prompt_tokens / completion_tokens / total_tokens
        #   - Anthropic / LC >= 0.3 standard: generations[0][0].message.usage_metadata
        #                      with input_tokens / output_tokens / total_tokens
        #   - Anthropic legacy: response.llm_output["usage"] with
        #                       input_tokens / output_tokens
        if traced is not None and traced._active and traced._span is not None:
            in_tok, out_tok, total_tok, cached_in = _extract_token_usage(response)
            if in_tok is not None or out_tok is not None:
                traced._span.input_tokens = in_tok
                traced._span.output_tokens = out_tok
                traced._span.total_tokens = (
                    total_tok
                    if total_tok is not None
                    else ((in_tok or 0) + (out_tok or 0)) or None
                )
                if cached_in:
                    traced._span.cached_input_tokens = cached_in
            cache_write = _extract_cache_write_tokens(response)
            if cache_write:
                traced._span.cache_write_tokens = cache_write
            # Response model wins over the invocation-param model set in
            # on_llm_start: the response carries the fully-resolved version
            # string (e.g. gpt-5-nano-2025-08-07 vs the requested gpt-5-nano).
            resolved_model = _extract_model(response)
            if resolved_model:
                traced._span.model = resolved_model
            model = traced._span.model
            # Compute cost if we have tokens + a known model.
            if model and (in_tok or out_tok):
                from amund.pricing import calculate_cost

                # calculate_cost adds cache writes on top of input_tokens, and
                # LangChain's input_tokens already includes them.
                cost = calculate_cost(
                    input_tokens=max((in_tok or 0) - (cache_write or 0), 0),
                    output_tokens=out_tok or 0,
                    model=model,
                    cached_input_tokens=cached_in or 0,
                    cache_write_tokens=cache_write or 0,
                )
                if cost is not None:
                    traced._span.cost_usd = cost

            # Cost-intelligence D3/D11: OpenRouter/LiteLLM (reached through the
            # LangChain LLM path) surface a real per-generation billed cost in
            # response.llm_output extras. Write it onto the canonical metadata
            # contract so the server prefers it over catalog computation. Does
            # not overwrite other keys in an existing cost block; no-op when
            # absent.
            from amund.adapters.base import merge_provider_reported_cost

            merged_md = merge_provider_reported_cost(traced._span.metadata, response)
            if merged_md is not None:
                traced._span.metadata = merged_md

            # MHQ-1880: lets the server merge this span with a gateway's record
            # of the same call instead of counting it twice.
            response_id = _extract_response_id(response)
            if response_id:
                traced._span.metadata = {
                    **(traced._span.metadata or {}),
                    "response_id": response_id,
                }

        self._close_span(run_id, output_data={"text": output_text})
        self._maybe_close_auto_trace()

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            self._close_span(run_id, error=error)
            self._maybe_close_auto_trace(error=error)
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: MorseCallbackHandler.on_llm_error swallowed: %s", exc)

    # ------------------------------------------------------------------
    # Tool callbacks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            s = serialized if isinstance(serialized, dict) else {}
            id_path = s.get("id")
            tool_name = s.get("name")
            if not tool_name and isinstance(id_path, list) and id_path:
                tool_name = id_path[-1]
            if not tool_name:
                tool_name = "tool"
            self._open_span_tool_input(
                run_id, str(tool_name), input_str, parent_run_id=parent_run_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "amund: MorseCallbackHandler.on_tool_start swallowed: %s",
                exc,
            )

    def _open_span_tool_input(
        self,
        run_id: UUID,
        tool_name: str,
        input_str: str,
        parent_run_id: UUID | None = None,
    ) -> None:
        self._open_span(
            run_id,
            tool_name,
            span_type="tool",
            input_data={"input": input_str},
            parent_run_id=parent_run_id,
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            self._close_span(run_id, output_data={"output": str(output)})
            self._maybe_close_auto_trace()
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: MorseCallbackHandler.on_tool_end swallowed: %s", exc)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            self._close_span(run_id, error=error)
            self._maybe_close_auto_trace(error=error)
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: MorseCallbackHandler.on_tool_error swallowed: %s", exc)

    # ------------------------------------------------------------------
    # Retriever callbacks (@beta — subject to change before SDK v1.0)
    # ------------------------------------------------------------------

    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """@beta — retriever span type is subject to change before SDK v1.0."""
        try:
            s = serialized if isinstance(serialized, dict) else {}
            retriever_name = s.get("id", ["retriever"])[-1] or "retriever"
            self._open_span(
                run_id,
                str(retriever_name),
                span_type="retriever",
                input_data={"query": query},
                parent_run_id=parent_run_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: on_retriever_start swallowed: %s", exc)
            _report(exc, "morse_ai.adapters.langchain.01")

    def on_retriever_end(
        self,
        documents: list[Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """@beta — retriever span type is subject to change before SDK v1.0."""
        try:
            # Extract doc metadata — be defensive; documents may be LangChain Document
            # objects or plain dicts.
            docs_out = []
            for doc in documents:
                if hasattr(doc, "page_content"):
                    entry: dict[str, Any] = {
                        "content_preview": str(doc.page_content)[:200]
                    }
                    if hasattr(doc, "metadata"):
                        entry["source"] = doc.metadata.get("source", "")
                        score = doc.metadata.get("score") or doc.metadata.get(
                            "relevance_score"
                        )
                        if score is not None:
                            entry["score"] = float(score)
                    docs_out.append(entry)
                elif isinstance(doc, dict):
                    docs_out.append(doc)

            self._close_span(
                run_id,
                output_data={
                    "docs_count": len(documents),
                    "top_score": docs_out[0].get("score") if docs_out else None,
                    "sources": [
                        d.get("source", "") for d in docs_out if d.get("source")
                    ],
                    # RAGQ-01 (@beta): ranked chunks {rank, id, content<=2KB,
                    # score, source} (max 10) + the legacy content_preview.
                    "documents": _documents_with_preview(documents),
                },
            )
            self._maybe_close_auto_trace()
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: on_retriever_end swallowed: %s", exc)
            _report(exc, "morse_ai.adapters.langchain.02")

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """@beta — retriever span type is subject to change before SDK v1.0."""
        try:
            self._close_span(run_id, error=error)
            self._maybe_close_auto_trace(error=error)
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: on_retriever_error swallowed: %s", exc)
            _report(exc, "morse_ai.adapters.langchain.03")


def _report(exc: BaseException, fingerprint: str) -> None:
    """Route a swallowed internal error to the diagnostics channel (SDKDIAG-07).

    Control flow is unchanged — the site still swallows, still logs locally,
    still returns the same value. It is simply no longer invisible to us.
    """
    try:
        from amund._internal import diagnostics

        diagnostics.report(exc, fingerprint=fingerprint)
    except Exception:  # noqa: BLE001 — reporting must never break the swallow
        pass
