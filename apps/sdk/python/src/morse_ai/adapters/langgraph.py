"""LangGraph-aware adapter — extends the generic LangChain callback with graph metadata.

LangGraph's `StateGraph` runs on top of LangChain's callback manager, so the
existing `MorseCallbackHandler` already produces correct agent/llm/tool
spans for LangGraph invocations. This adapter adds graph-aware metadata on
top:

- Span metadata `adapter: "langgraph"` so downstream tooling can filter
- Captures LangGraph node name on each chain start (when present in
  `serialized.id`) so the span tree mirrors the graph structure
- (TODO Phase 7+) True parallel-branch rendering — currently each parallel
  node still emits a sequential span, which is correct in content but not
  in visual time-overlap. The renderer side handles this acceptably for v1.
- (TODO Phase 7+) Conditional-edge metadata under `edge_type: "conditional"`
  with the selected target — requires a hook into LangGraph's edge dispatch.

**Phase 9 (MEM-10) — adapter v2 — memory-op surfaces:**

In addition to the LangChain-callback-driven retriever span (CTX-09), this
adapter monkey-patches LangGraph's `BaseCheckpointSaver` and `BaseStore`
classes at `install()` time to emit `memory`-type spans for:

| Surface                          | `memory_op_type` |
|----------------------------------|------------------|
| `BaseCheckpointSaver.put()`      | `extract`        |
| `BaseCheckpointSaver.get()`      | `retrieve`       |
| `BaseStore.put()`                | `extract`        |
| `BaseStore.search()`             | `retrieve`       |
| `BaseStore.delete()`             | `prune`          |
| Summarization node LLM call      | `consolidate`    |

Summarization-node detection: LangChain `on_chain_start` callbacks whose
node name matches `r"^(summarize|consolidate)"` (case-insensitive) flag
the active trace state. While that node is on the chain stack, every LLM
call emits a sibling `memory` span with `memory_op_type="consolidate"`,
and the next post-node LLM call carries `metadata.context_remediation`
with `strategy="summarize"`.

All memory-op hooks are no-ops outside an active `morse_ai.run` trace
context, and respect a 50ms dedup window keyed on
`(start_time_bucket, store_name, op_type)` to avoid double-emission when
a future explicit `morse_ai.memory_op()` API coexists.

Install pattern:

    from morse_ai.adapters.langgraph import install, uninstall
    install()    # adds LangGraphCallbackHandler globally + patches Saver/Store
    uninstall()  # restores

Auto-detection: the auto_detect layer prefers this adapter over the generic
LangChain callback when both `langchain` and `langgraph` are importable.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from typing import TYPE_CHECKING, Any

from morse_ai._internal.agent_name import resolve_agent_name
from morse_ai._internal.errors import absorb_errors
from morse_ai.adapters.langchain import MorseCallbackHandler
from morse_ai.spans import SpanData, SpanType
from morse_ai.tracing import _finalize_span, get_active_trace

if TYPE_CHECKING:
    from uuid import UUID

logger = logging.getLogger("morse_ai")

# Sentinel attribute on the langgraph module to mark instrumented.
_INSTALL_FLAG = "_morse_ai_installed"

# Default agent name when the adapter is auto-installed without explicit naming.
# MHQ-1100: "LangGraph" is the compiled-graph default name, i.e. not a real one.
_GENERIC_GRAPH_NAMES = frozenset({"langgraph", "langchain-agent"})

# Truncate retrieved-document content stored on the wire payload. The full
# pre-truncation hash is preserved on the segment so the server can de-dup
# across calls even when content is truncated.
_SEGMENT_CONTENT_MAX_BYTES = 4096
# Output-data preview cap for the memory span itself — separate from segment
# content (which carries the full unmangled hash).
_DOCUMENT_PREVIEW_MAX_CHARS = 500

# ----- Phase 9 (MEM-10) memory-op constants -----

# Dedup window for same-(store, op_type) memory-op emissions. Per design.md
# §3.2 and MEM-10 spec: 50ms keyed on (bucket, store_name, op_type).
_MEMORY_OP_DEDUP_WINDOW_MS = 50

# Summarization-node name pattern (MEM-10 spec — conservative match to avoid
# false positives from agents whose nodes happen to mention "summarize").
_SUMMARIZATION_NODE_PATTERN = re.compile(r"^(summarize|consolidate)", re.IGNORECASE)

# Trace-state attributes (set lazily on _TraceState instances via setattr).
# We DON'T modify _TraceState directly (out of file scope) — instead we attach
# dynamic attrs on the state object. These survive for the trace's lifetime
# and are discarded when the trace ends.
_AV_DEDUP_ATTR = "av_lg_memory_dedup"  # dict[(bucket, store, op_type), float]
_AV_SUMMARIZATION_STACK_ATTR = "av_lg_summarization_stack"  # list[str] of node names
_AV_REMEDIATION_PENDING_ATTR = "av_lg_remediation_pending"  # dict | None

# Patched-method storage for uninstall() restoration. Keys are dotted names
# like "BaseCheckpointSaver.put" so the matrix is self-documenting.
_patched_originals: dict[str, Any] = {}


def _classify_store_name(serialized: dict[str, Any] | None) -> str:
    """Best-effort vector-store class name from a LangChain serialized blob.

    Examples:
        {"id": ["langchain_community", "vectorstores", "FAISS"], "name": "FAISS"}
        → "FAISS"
        {"id": ["langchain", "schema", "retriever", "BaseRetriever"]}
        → "BaseRetriever"
    """
    if not serialized:
        return "unknown"
    name = serialized.get("name")
    if isinstance(name, str) and name:
        return name
    id_path = serialized.get("id")
    if isinstance(id_path, list) and id_path:
        last = id_path[-1]
        if isinstance(last, str) and last:
            return last
    return "unknown"


def _doc_content(doc: Any) -> str:
    """Return the textual content of a (LangChain or duck-typed) Document."""
    content = getattr(doc, "page_content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, bytes):
        try:
            return content.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""
    return str(content) if content is not None else ""


def _doc_metadata(doc: Any) -> dict[str, Any]:
    """Return the metadata dict of a Document — empty dict on missing."""
    md = getattr(doc, "metadata", None)
    return md if isinstance(md, dict) else {}


def _truncate_for_segment(content: str) -> tuple[str, bool]:
    """Truncate content to <= _SEGMENT_CONTENT_MAX_BYTES UTF-8 bytes.

    Returns (truncated_content, was_truncated). Truncation is byte-aware to
    avoid splitting a multi-byte codepoint at the cap.
    """
    encoded = content.encode("utf-8")
    if len(encoded) <= _SEGMENT_CONTENT_MAX_BYTES:
        return content, False
    cut = encoded[:_SEGMENT_CONTENT_MAX_BYTES]
    # Walk back to a valid UTF-8 boundary if we landed mid-codepoint.
    return cut.decode("utf-8", errors="ignore"), True


def _sha256_hex(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _next_segment_position(span: Any) -> int:
    """Return the next available position index for a fresh segment list.

    Memory-vector segments start at 0 in this adapter — they precede whatever
    the server-side classifier might emit later. We keep a per-list counter
    for clarity in case the existing segments list is non-empty (e.g. another
    adapter pre-populated it).
    """
    existing = getattr(span, "context_segments", None) or []
    return len(existing)


# ---------------------------------------------------------------------------
# Phase 9 (MEM-10) memory-op emission helpers
# ---------------------------------------------------------------------------


def _state_attr(state: Any, name: str, default_factory: Any) -> Any:
    """Lazily attach an adapter-private attribute on a _TraceState instance.

    We can't add fields to _TraceState (tracing.py is out of file scope for
    this change) so the adapter stashes its own state on the dataclass
    instance via setattr. The attr lives only for the trace's lifetime —
    the _TraceState is discarded on trace exit.
    """
    value = getattr(state, name, None)
    if value is None:
        value = default_factory()
        setattr(state, name, value)
    return value


def _dedup_should_skip(state: Any, store_name: str, op_type: str) -> bool:
    """Return True if a same-key memory-op was emitted within the dedup window.

    Bucket = monotonic-time floor-divided by `_MEMORY_OP_DEDUP_WINDOW_MS`.
    Two ops with the same (bucket, store, op_type) within 50ms collide; the
    second one is dropped per MEM-10 spec to prevent double-emission when a
    future explicit `morse_ai.memory_op()` API coexists with auto-detection.

    A small side-effect: this function also writes the current key into the
    state's dedup map so the next equivalent op short-circuits.
    """
    now_ms = time.monotonic() * 1000.0
    bucket = int(now_ms // _MEMORY_OP_DEDUP_WINDOW_MS)
    key = (bucket, store_name, op_type)
    dedup: dict[tuple[int, str, str], float] = _state_attr(state, _AV_DEDUP_ATTR, dict)

    if key in dedup:
        return True

    # Best-effort prune: drop entries older than one window so the dict
    # doesn't grow unbounded over long traces. O(N) on each emit but N is
    # the number of distinct (store, op_type) tuples per 50ms window — tiny.
    cutoff = now_ms - _MEMORY_OP_DEDUP_WINDOW_MS * 2
    stale = [k for k, t in dedup.items() if t < cutoff]
    for k in stale:
        del dedup[k]

    dedup[key] = now_ms
    return False


def _scope_for_op(memory_op_type: str, is_store_search: bool = False) -> str:
    """Map memory_op_type to morse.memory.scope value for ADR-023."""
    if memory_op_type == "consolidate":
        return "agent"
    if is_store_search or memory_op_type == "retrieve":
        return "retrieval"
    return "working"


def _emit_memory_op_span(
    *,
    name: str,
    memory_op_type: str,
    store_name: str,
    metadata_extra: dict[str, Any] | None = None,
    input_data: Any = None,
    output_data: Any = None,
    producer_span_id: str | None = None,
    is_store_search: bool = False,
) -> str | None:
    """Emit a synthetic memory-typed span on the current trace.

    Unlike the LangChain-callback-driven retriever span (which is open/closed
    via on_retriever_start/end and runs through the parent's _open_span /
    _close_span lifecycle), this helper emits a fully-formed, already-closed
    span by appending directly to state.completed_spans. We use this for
    surfaces that don't have a natural "start/end" pair we can hook into:
    checkpointer/store ops fire synchronously inside the user's code.

    Returns the span_id of the emitted span, or None if no active trace or
    the dedup window suppressed the emission.

    No-op when no active trace context (per MEM-10 spec).
    """
    state = get_active_trace()
    if state is None:
        return None

    if _dedup_should_skip(state, store_name, memory_op_type):
        return None

    # Resolve parent span_id from the current stack.
    parent_span_id: str | None = None
    if state.span_stack:
        parent_span_id = state.span_stack[-1].span_id

    metadata: dict[str, Any] = {
        "memory_op_type": memory_op_type,
        "store_name": store_name,
        "adapter": "langgraph",
        "memory_scope": _scope_for_op(memory_op_type, is_store_search),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    if producer_span_id is not None:
        metadata["producer_span_id"] = producer_span_id

    span_data = SpanData(
        trace_id=state.trace_id,
        type=SpanType.MEMORY,
        name=name,
        parent_span_id=parent_span_id,
        input_data=input_data,
        output_data=output_data,
        metadata=metadata,
    )
    # Spans emitted via this helper represent already-completed ops. We
    # finalize with a near-zero duration since the wrapped call (e.g.,
    # checkpointer.put) is observed at the boundary, not from within. The
    # span's started_at reflects when we observed the op fire.
    _finalize_span(span_data, time.monotonic(), status="completed")
    state.completed_spans.append(span_data)
    return span_data.span_id


def _identify_store(target: Any) -> str:
    """Best-effort store-name extraction for a checkpointer / store instance.

    LangGraph's `BaseStore` and `BaseCheckpointSaver` are abstract; concrete
    classes carry useful identity in `type(target).__name__` (e.g.,
    "InMemoryStore", "PostgresSaver", "SqliteSaver"). We use the bare class
    name as the store identifier — matches what users see in their imports.
    """
    try:
        return type(target).__name__
    except Exception:  # noqa: BLE001
        return "unknown"


def _peek_summarization_node(state: Any) -> str | None:
    """Return the topmost summarization-node name on the trace's stack, if any.

    The summarization stack is populated by on_chain_start when a node name
    matches `_SUMMARIZATION_NODE_PATTERN`. It's popped in on_chain_end / error.
    A non-empty stack means we're currently executing inside a summarization
    node, so LLM calls fired from here are summarization producers.
    """
    stack: list[str] = getattr(state, _AV_SUMMARIZATION_STACK_ATTR, None) or []
    return stack[-1] if stack else None


def _graph_name(
    serialized: dict[str, Any] | None, kwargs: dict[str, Any]
) -> str | None:
    """Compiled-graph name from a root on_chain_start, ignoring generic defaults."""
    candidates = [kwargs.get("name"), (serialized or {}).get("name")]
    for cand in candidates:
        if (
            isinstance(cand, str)
            and cand.strip()
            and cand.strip().lower() not in _GENERIC_GRAPH_NAMES
        ):
            return cand.strip()
    return None


def _extract_node_name(
    serialized: dict[str, Any] | None,
    kwargs: dict[str, Any],
) -> str | None:
    """Pull a LangGraph node name from on_chain_start arguments.

    LangGraph carries the node identity in two places depending on version:
    1. `kwargs["metadata"]["langgraph_node"]` (LangGraph >= 0.2)
    2. `serialized["name"]` or last element of `serialized["id"]` (fallback)

    Returns None if no usable name surfaced.
    """
    # Preferred path — LangGraph-provided node name in callback metadata.
    md = kwargs.get("metadata")
    if isinstance(md, dict):
        node = md.get("langgraph_node")
        if isinstance(node, str) and node:
            return node

    # Fallback — serialized blob.
    if not serialized:
        return None
    name = serialized.get("name")
    if isinstance(name, str) and name:
        return name
    id_path = serialized.get("id")
    if isinstance(id_path, list) and id_path:
        last = id_path[-1]
        if isinstance(last, str) and last:
            return last
    return None


# ---------------------------------------------------------------------------
# Issue 2 (Phase 12.5 follow-up) — W3C trace_id ↔ Morse trace_id bridge
# ---------------------------------------------------------------------------
#
# Two bridges are needed so the Correlation panel's Logs and Infra tabs
# populate on Copilot-style LangGraph traces:
#
# Bridge 1 — _active_trace must be set BEFORE any MorseHandler log
#   emission. The parent's _ensure_trace() already does this lazily on the
#   first _open_span call, but the first log line inside a chain can fire
#   between the LangChain callback dispatch and our _open_span. We satisfy
#   this by opening the trace eagerly on root on_chain_start.
#
# Bridge 2 — The Morse trace_id MUST equal the W3C trace_id active when
#   the chain starts. Without this, OTel auto-instrumented http/db spans
#   (which use the W3C trace_id from the incoming Django request) land on
#   a DIFFERENT Morse trace than the LangGraph callback emits, so the
#   Infra tab on the Copilot trace shows zero rows.
#
# The TracedRun already supports W3C trace_id adoption via incoming_traceparent.
# We harvest the active OTel SpanContext, format it as a synthetic traceparent
# string, and feed it to run() — reusing the existing extract_traceparent path.
#
# When OTel isn't installed or no W3C context is active, _build_otel_traceparent
# returns None and TracedRun falls back to its existing uuid4() generation.
# This means the bridge is zero-cost when OTel is absent.


def _build_otel_traceparent() -> str | None:
    """Synthesize a W3C traceparent from the current OTel SpanContext.

    Reads `opentelemetry.trace.get_current_span().get_span_context()` and, if
    the context is valid (i.e. a W3C trace is active — propagated from an
    upstream OTel-instrumented service or set by OTel's Django auto-
    instrumentation on the current request), formats it as the standard
    ``00-{trace_id_32hex}-{span_id_16hex}-{flags_2hex}`` string.

    Returns ``None`` when:
      - opentelemetry isn't installed (SDK ships zero-dep core; OTel is opt-in)
      - no W3C trace context is active on this thread / async task
      - the context exists but is invalid (zero trace_id / span_id)
      - any unexpected error reading the context (non-invasive guarantee)

    Why route through traceparent rather than ContextVar.set with a fabricated
    UUID directly: TracedRun's incoming_traceparent path is already battle-
    tested for cross-service propagation (Phase 6c). Reusing it keeps a single
    code path for "Morse adopts an upstream trace_id" semantics and means
    the validation rules (W3C version=00, 32hex trace_id, 16hex span_id,
    non-zero) are enforced by the existing extract_traceparent helper.

    The W3C trace_id is a 128-bit int; converting via ``uuid.UUID(int=...)``
    gives a 32-char lowercase hex string that the SDK already treats as a
    valid trace_id (uuid4().hex format, same length and alphabet).
    """
    try:
        # Lazy import — OTel is an optional dependency for the SDK. Customers
        # who don't run OTel (or who run the SDK in environments where OTel
        # isn't installed, like a vanilla LangChain script) take the
        # fallback path with zero import cost on first-trace.
        from opentelemetry import trace as otel_trace  # noqa: PLC0415
    except ImportError:
        return None

    try:
        span = otel_trace.get_current_span()
        if span is None:
            return None
        ctx = span.get_span_context()
        if ctx is None or not getattr(ctx, "is_valid", False):
            return None
        # 128-bit trace_id → 32-char lowercase hex. Pad on the left if OTel
        # returned a numerically smaller int (rare but legal per W3C spec).
        trace_hex = format(ctx.trace_id, "032x")
        # 64-bit span_id → 16-char lowercase hex.
        span_hex = format(ctx.span_id, "016x")
        # Trace flags are 8 bits; the W3C spec serializes them as 2 hex
        # chars. Default to "01" (sampled) when the OTel SDK didn't set a
        # specific value — we always sample our own emissions.
        flags = getattr(ctx, "trace_flags", None)
        if flags is None:
            flags_hex = "01"
        else:
            try:
                flags_hex = format(int(flags), "02x")
            except (TypeError, ValueError):
                flags_hex = "01"
        return f"00-{trace_hex}-{span_hex}-{flags_hex}"
    except Exception as exc:  # noqa: BLE001 — non-invasive
        logger.debug("morse_ai: OTel traceparent harvest swallowed: %s", exc)
        return None


def _w3c_trace_id_to_uuid_hex(w3c_trace_id_int: int) -> str:
    """Convert a W3C 128-bit trace_id integer to a 32-char lowercase hex string.

    Exposed for tests; internal callers use _build_otel_traceparent which
    already formats. This helper exists mainly so the test file can document
    the conversion contract precisely.

    The conversion is lossless and round-trippable:
        int -> uuid.UUID(int=...).hex -> int(..., 16) round-trips.
    """
    return uuid.UUID(int=w3c_trace_id_int).hex


class LangGraphCallbackHandler(MorseCallbackHandler):
    """Graph-aware extension of the generic LangChain callback handler.

    Inherits all chain/llm/tool callback semantics from the parent and adds:

    - Retriever spans (``type="memory"``) on ``on_retriever_start`` /
      ``on_retriever_end``. The span carries
      ``metadata={memory_type, store_name, top_k}`` and an
      ``output_data.documents[]`` preview list.
    - Cross-span linkage: each retrieved chunk is queued in
      ``_TraceState.pending_retrieved_docs``; the next ``on_chat_model_start``
      / ``on_llm_start`` callback consumes those chunks and emits them as
      ``source="memory_vector"`` ``context_segments`` on the LLM span, with
      ``producer_span_id`` pointing back to the originating memory span.
    - Trace-context bridges (Issue 2 / Phase 12.5 follow-up): on the root
      chain (``parent_run_id is None``), the handler eagerly opens an
      Morse trace and adopts the active W3C trace_id (when an
      OpenTelemetry SpanContext is active on the thread). This is what makes
      ``logger.info(...)`` lines inside the chain correlate with the AI
      trace AND makes OTel-emitted http/db spans land on the same trace_id
      as the LangGraph callback-emitted llm/tool spans. Nested chains
      (``parent_run_id is not None``) do NOT mutate ``_active_trace`` — they
      inherit the root chain's trace state.

    The pending-docs list is consumed (cleared) on the FIRST subsequent LLM
    call so a second LLM call in the same trace doesn't double-count the
    same retrieved chunks.
    """

    # AGE-240: framework auto-discovery tag — overrides the parent's default
    # so LangGraph traces are distinguishable from raw LangChain ones on the
    # dashboard.
    _framework_name: str = "langgraph"

    def __init__(
        self,
        agent_name: str | None = None,
        incoming_traceparent: str | None = None,
    ) -> None:
        """Create a handler.

        Args:
            agent_name: Agent name tag for the resulting trace.
            incoming_traceparent: Optional pre-resolved W3C traceparent string.
                When provided, overrides the lazy ``_build_otel_traceparent()``
                read inside ``_ensure_trace``. Use this when the caller knows
                the OTel context is available at handler-construction time
                but won't be at callback-fire time (e.g., LangGraph SSE
                streams that detach the on_chain_start callback from the
                Django request thread). Pass ``None`` for the standard lazy
                behavior.
        """
        super().__init__(agent_name=agent_name)
        self._explicit_traceparent = incoming_traceparent

    # ------------------------------------------------------------------
    # Issue 2 — _active_trace + W3C trace_id bridge.
    # ------------------------------------------------------------------

    def _ensure_trace(self) -> None:
        """Open an Morse trace adopting the active W3C trace_id if any.

        Overrides the parent's _ensure_trace to harvest the current OTel
        SpanContext and pass it through TracedRun's existing
        ``incoming_traceparent`` path. The result:

          - If an OTel W3C trace is active (e.g. OTel-Django auto-
            instrumentation opened one for the current request),
            TracedRun.__enter__ adopts that trace_id. OTel-emitted http/db
            spans during this request land on the SAME Morse trace as
            the LangGraph-emitted llm/tool spans.
          - If no OTel context is active (background task, test, vanilla
            LangChain script without OTel), TracedRun.__enter__ falls back
            to its existing ``uuid4().hex`` generation. Pre-existing
            behavior preserved exactly.

        Sub-agent correlation: if ``_active_trace`` is already set (a parent
        graph is running in this async context), this is a sub-agent
        invocation. We mark ``_auto_trace = False`` so the parent's trace
        state is inherited and this handler's ``_maybe_close_auto_trace``
        never exits the parent's trace. The sub-agent's spans attach to the
        parent's span stack as child spans — making them visible in the same
        trace waterfall as the supervisor.

        Idempotent: if we've already opened our own auto-trace, this is a
        no-op.
        """
        # If we already opened our own auto-trace, nothing to do.
        if self._run_context is not None:
            return

        # If this handler was constructed inside an explicit morse_ai.run()
        # context but LangGraph fires callbacks in a detached context, reattach
        # to that parent before considering auto-trace creation.
        self._restore_inherited_trace()

        # Sub-agent correlation: if a parent trace is already active in this
        # async context (ContextVar inherited from the caller's asyncio task),
        # inherit it rather than creating a new root trace. The sub-agent's
        # spans will appear nested under the parent's agent span in the same
        # trace waterfall.
        if get_active_trace() is not None:
            # Mark as non-auto so _maybe_close_auto_trace never closes the
            # parent trace when this sub-agent's chain ends.
            self._auto_trace = False
            return

        # Local import — keeps `run` out of this module's module-level deps
        # mirroring the parent's pattern.
        from morse_ai.tracing import run  # noqa: PLC0415

        # Explicit traceparent (passed at handler-construction time) wins over
        # the lazy OTel-context read. This is the dogfood-Copilot fix path:
        # the LangGraph callback fires in an async/SSE-detached context where
        # `_build_otel_traceparent()` can't reach the OTel SERVER span, so the
        # caller resolves the traceparent synchronously at view entry and
        # passes it down.
        traceparent = self._explicit_traceparent or _build_otel_traceparent()
        self._run_context = run(
            resolve_agent_name(self.agent_name, self._native_name),
            incoming_traceparent=traceparent,
        )
        self._run_context.__enter__()
        self._auto_trace = True

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # Issue 2 Bridge 1+2: on the ROOT chain (parent_run_id is None),
        # eagerly open the Morse trace BEFORE the parent's _open_span
        # path runs. This both sets _active_trace (so MorseHandler-routed
        # log lines correlate) AND adopts the active W3C trace_id (so OTel-
        # auto-instrumented http/db spans share the trace).
        #
        # Sub-agent correlation (Bug 3 fix): when parent_run_id is None BUT
        # _active_trace is already set in this async context, we are a
        # sub-agent invocation running inside an existing trace (e.g., a
        # LangGraph supervisor routing to a compiled sub-graph). In this case,
        # DON'T open a new root trace — inherit the parent's trace state so
        # the sub-agent's spans land on the same trace as the supervisor.
        # Without this, every sub-agent invocation produces a separate orphaned
        # trace entry, making multi-agent traces impossible to reconstruct.
        #
        # When the sub-agent's chain is truly independent (a fresh async task
        # with no inherited ContextVar), get_active_trace() returns None and
        # _ensure_trace() creates a new root trace as before.
        #
        # Nested chains (parent_run_id is not None) skip this entirely: they
        # inherit the root chain's state via the parent handler's span stack.
        if parent_run_id is None:
            self._native_name = _graph_name(serialized, kwargs)
            try:
                self._ensure_trace()
            except Exception as exc:  # noqa: BLE001 — non-invasive
                logger.debug(
                    "morse_ai: LangGraphCallbackHandler._ensure_trace swallowed: %s",
                    exc,
                )

        # Defer to the parent's behavior. The langgraph node name lives in
        # serialized["id"] (e.g., ["langgraph", "graph", "CompiledStateGraph"]).
        # The parent already records the chain start as a span; we only need
        # to make sure it doesn't break when LangGraph passes its specific
        # serialized shape.
        try:
            super().on_chain_start(
                serialized=serialized,
                inputs=inputs,
                run_id=run_id,
                parent_run_id=parent_run_id,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 — non-invasive guarantee
            logger.debug(
                "morse_ai: LangGraphCallbackHandler.on_chain_start swallowed: %s", exc
            )

        # Signal 3: rename the span using the LangGraph node name + stamp it in
        # metadata so server-side topology detection can read it.
        # LangGraph fires on_chain_start with metadata["langgraph_node"] = "<name>"
        # for each graph node. The parent names the span from serialized["id"][-1],
        # which resolves to "RunnableLambda" (the generic Python-function wrapper) —
        # not the meaningful node name. We rename it here so spans like
        # "warren_buffett_agent" appear instead of "RunnableLambda", and stamp
        # metadata.langgraph_node so server topology can detect node-to-orchestrator
        # relations without SDK-format-specific knowledge.
        node_name: str | None = None
        try:
            node_name = _extract_node_name(serialized, kwargs)
            if node_name:
                traced = self._span_stack.get(str(run_id))
                if traced is not None and traced._active and traced._span is not None:
                    traced._span.name = node_name
                    md = dict(traced._span.metadata or {})
                    md["langgraph_node"] = node_name
                    traced._span.metadata = md
        except Exception as exc:  # noqa: BLE001 — non-invasive guarantee
            logger.debug("morse_ai: LangGraph node rename swallowed: %s", exc)

        # MEM-10: summarization-node detection. Push the node name onto the
        # per-trace summarization stack so any LLM call fired while this node
        # is active emits a sibling consolidate span + arms remediation for
        # the next post-node LLM call. Reuses node_name extracted above.
        try:
            state = get_active_trace()
            if state is None:
                return
            if node_name is None:
                node_name = _extract_node_name(serialized, kwargs)
            if node_name is None or not _SUMMARIZATION_NODE_PATTERN.match(node_name):
                return
            stack: list[str] = _state_attr(state, _AV_SUMMARIZATION_STACK_ATTR, list)
            stack.append(node_name)
        except Exception as exc:  # noqa: BLE001 — non-invasive guarantee
            logger.debug("morse_ai: summarization-node detection swallowed: %s", exc)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        # Pop the summarization stack if the chain that's ending was one we
        # pushed. We do this BEFORE delegating to the parent so the chain
        # close doesn't accidentally re-trigger emission logic.
        try:
            self._maybe_pop_summarization_stack(run_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("morse_ai: summarization-stack pop swallowed: %s", exc)
        super().on_chain_end(outputs, run_id=run_id, **kwargs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            self._maybe_pop_summarization_stack(run_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("morse_ai: summarization-stack pop (error) swallowed: %s", exc)
        super().on_chain_error(error, run_id=run_id, **kwargs)

    def _maybe_pop_summarization_stack(self, run_id: UUID) -> None:
        """Pop the topmost summarization node from the trace's stack.

        We track the stack by node-name only, not by run_id, since LangGraph
        nests chain runs and the node name is what gates emission. Popping
        the top is safe under normal nesting; double-pop is a no-op.
        """
        state = get_active_trace()
        if state is None:
            return
        stack = getattr(state, _AV_SUMMARIZATION_STACK_ATTR, None)
        if not stack:
            return
        # Determine whether this run was for a summarization node. The
        # parent's _span_stack carries the chain name we recorded in
        # on_chain_start (via the parent's _open_span name).
        traced = self._span_stack.get(str(run_id))
        if traced is None or traced._span is None:
            return
        # If the popped span's name matches the top of the stack, pop. We
        # match by SUFFIX since the parent records chain spans by
        # `serialized["id"][-1]` and our stack pushes by extracted node
        # name (often identical, sometimes a richer name).
        top = stack[-1]
        span_name = traced._span.name or ""
        if span_name == top or span_name.endswith(top) or top.endswith(span_name):
            stack.pop()
            # Arm remediation for the next LLM span (post-node).
            setattr(
                state,
                _AV_REMEDIATION_PENDING_ATTR,
                {
                    "strategy": "summarize",
                    "remediation_metadata": {
                        "source": "langgraph_node",
                        "node_name": top,
                    },
                },
            )

    # ------------------------------------------------------------------
    # Retriever callbacks — emit a `memory` span per retrieve invocation.
    # ------------------------------------------------------------------

    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            state = get_active_trace()
            if state is None:
                # Retriever fired outside any active trace — silently no-op.
                return

            store_name = _classify_store_name(serialized)
            traced = self._open_span(
                run_id,
                f"retriever.{store_name}",
                span_type=SpanType.MEMORY.value,
                input_data={"query": str(query)},
            )
            if traced is None or not traced._active or traced._span is None:
                return

            # Provisional metadata — top_k is unknown until on_retriever_end
            # fires; we backfill it there from the doc count.
            # MEM-10: set memory_op_type="retrieve" explicitly so server-side
            # classifier doesn't have to infer (CTX-09 spans only carried
            # memory_type/store_name/top_k — server defaulted to "retrieve",
            # so this is a no-op behavior change for the classifier).
            traced._span.metadata = {
                "memory_type": "vector",
                "store_name": store_name,
                "top_k": None,
                "memory_op_type": "retrieve",
            }

            state.retriever_runs[str(run_id)] = traced._span.span_id
        except Exception as exc:  # noqa: BLE001 — non-invasive guarantee
            logger.debug("morse_ai: on_retriever_start swallowed: %s", exc)

    def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            state = get_active_trace()
            if state is None:
                return
            run_key = str(run_id)
            memory_span_id = state.retriever_runs.pop(run_key, None)
            traced = self._span_stack.get(run_key)

            docs_seq = list(documents) if documents is not None else []
            top_k = len(docs_seq)

            if traced is not None and traced._active and traced._span is not None:
                # Backfill top_k now that we know how many docs came back.
                md = traced._span.metadata or {}
                md["top_k"] = top_k
                traced._span.metadata = md

                preview_docs: list[dict[str, Any]] = []
                for d in docs_seq:
                    content = _doc_content(d)
                    preview_docs.append(
                        {
                            "content": content[:_DOCUMENT_PREVIEW_MAX_CHARS],
                            "metadata": _doc_metadata(d),
                        }
                    )
                traced._span.output_data = {"documents": preview_docs}

            # Stash chunks for the next LLM call — even if `traced` was None
            # we still try to record producer_span_id (best-effort link).
            if memory_span_id is not None:
                for d in docs_seq:
                    full_content = _doc_content(d)
                    state.pending_retrieved_docs.append(
                        {
                            "producer_span_id": memory_span_id,
                            "content": full_content,
                            "doc_metadata": _doc_metadata(d),
                        }
                    )

            self._close_span(run_id)
            self._maybe_close_auto_trace()
        except Exception as exc:  # noqa: BLE001 — non-invasive guarantee
            logger.debug("morse_ai: on_retriever_end swallowed: %s", exc)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            state = get_active_trace()
            if state is not None:
                state.retriever_runs.pop(str(run_id), None)
            self._close_span(run_id, error=error)
            self._maybe_close_auto_trace(error=error)
        except Exception as exc:  # noqa: BLE001 — non-invasive guarantee
            logger.debug("morse_ai: on_retriever_error swallowed: %s", exc)

    # ------------------------------------------------------------------
    # LLM callbacks — wire pending retrieved chunks onto context_segments.
    # ------------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Open the LLM span via the parent's on_llm_start path so all the
        # existing token/model extraction in on_llm_end keeps working.
        try:
            super().on_llm_start(
                serialized=serialized or {},
                prompts=[],  # parent uses prompts only for input_data
                run_id=run_id,
                parent_run_id=parent_run_id,
                invocation_params=invocation_params,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 — non-invasive guarantee
            logger.debug("morse_ai: on_chat_model_start.super swallowed: %s", exc)
            return

        # Replace the input_data with the actual messages payload (parent
        # wrote prompts=[] above) and attach context_segments.
        try:
            traced = self._span_stack.get(str(run_id))
            if traced is not None and traced._active and traced._span is not None:
                traced._span.input_data = {"messages": _summarize_messages(messages)}
                # Decompose the messages into context segments so the dashboard
                # can show context composition without any extra customer config.
                try:
                    from morse_ai.context.decomposer import decompose_openai_messages

                    flat_msgs = _flatten_messages(messages)
                    tools = None
                    if isinstance(invocation_params, dict):
                        tools = invocation_params.get("tools")
                    decomposed = decompose_openai_messages(flat_msgs, tools)
                    if decomposed:
                        existing = list(traced._span.context_segments or [])
                        # Renumber positions to come after any pre-existing
                        # memory-vector segments that _attach_pending_segments
                        # will add (they start at 0 from the retriever side).
                        # Simple approach: append after; positions are
                        # re-sequenced by the server on ingest.
                        offset = len(existing)
                        for seg in decomposed:
                            seg["position"] = seg["position"] + offset
                        traced._span.context_segments = existing + decomposed
                except Exception as exc_inner:  # noqa: BLE001
                    logger.debug(
                        "morse_ai: on_chat_model_start.decompose swallowed: %s",
                        exc_inner,
                    )
            self._attach_pending_segments(run_id)
            self._maybe_apply_remediation_to_llm_span(run_id)
        except Exception as exc:  # noqa: BLE001 — non-invasive guarantee
            logger.debug("morse_ai: on_chat_model_start.attach swallowed: %s", exc)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().on_llm_start(
                serialized=serialized,
                prompts=prompts,
                run_id=run_id,
                parent_run_id=parent_run_id,
                invocation_params=invocation_params,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 — non-invasive guarantee
            logger.debug("morse_ai: on_llm_start.super swallowed: %s", exc)
            return
        try:
            self._attach_pending_segments(run_id)
            self._maybe_apply_remediation_to_llm_span(run_id)
        except Exception as exc:  # noqa: BLE001 — non-invasive guarantee
            logger.debug("morse_ai: on_llm_start.attach swallowed: %s", exc)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        # MEM-10: if this LLM call fired while inside a summarization node,
        # emit a sibling memory span with memory_op_type="consolidate" linked
        # via producer_span_id to this LLM span. We do this BEFORE calling
        # super().on_llm_end so we can still read the LLM span_id from the
        # parent's _span_stack.
        try:
            state = get_active_trace()
            if state is not None and _peek_summarization_node(state) is not None:
                traced = self._span_stack.get(str(run_id))
                if traced is not None and traced._span is not None:
                    llm_span_id = traced._span.span_id
                    node_name = _peek_summarization_node(state) or ""
                    # Try to extract summary token count from the response.
                    summary_tokens: int | None = None
                    llm_output = getattr(response, "llm_output", {}) or {}
                    token_usage = llm_output.get("token_usage", {}) or {}
                    if isinstance(token_usage, dict):
                        summary_tokens = token_usage.get(
                            "completion_tokens"
                        ) or token_usage.get("output_tokens")
                    _emit_memory_op_span(
                        name=f"summarization.{node_name}",
                        memory_op_type="consolidate",
                        store_name=f"langgraph_node:{node_name}",
                        producer_span_id=llm_span_id,
                        metadata_extra={
                            "summary_token_count": summary_tokens,
                            "node_name": node_name,
                        },
                    )
        except Exception as exc:  # noqa: BLE001 — non-invasive
            logger.debug("morse_ai: on_llm_end summarization-emit swallowed: %s", exc)

        # Delegate to the parent for the actual LLM span close + token
        # extraction. Note: parent's on_llm_end also calls
        # _maybe_close_auto_trace which we don't override.
        super().on_llm_end(response, run_id=run_id, **kwargs)

    def _maybe_apply_remediation_to_llm_span(self, run_id: UUID) -> None:
        """If a remediation event is pending, stamp it on this LLM span's metadata.

        Called from on_llm_start / on_chat_model_start AFTER the parent opened
        the span. The pending remediation is consumed (one-shot) — the next
        LLM call won't double-stamp.
        """
        state = get_active_trace()
        if state is None:
            return
        pending = getattr(state, _AV_REMEDIATION_PENDING_ATTR, None)
        if pending is None:
            return
        # Consume regardless of whether we successfully apply — avoids
        # accidentally stamping a much-later LLM call if span lookup fails.
        setattr(state, _AV_REMEDIATION_PENDING_ATTR, None)

        traced = self._span_stack.get(str(run_id))
        if traced is None or not traced._active or traced._span is None:
            return
        md = traced._span.metadata or {}
        md["context_remediation"] = pending
        traced._span.metadata = md

    def _attach_pending_segments(self, run_id: UUID) -> None:
        """Drain pending retrieved docs into context_segments on the LLM span.

        Pending docs are cleared after one drain so a second LLM call in the
        same trace doesn't reattach the same chunks.
        """
        state = get_active_trace()
        if state is None or not state.pending_retrieved_docs:
            return
        traced = self._span_stack.get(str(run_id))
        if traced is None or not traced._active or traced._span is None:
            # We can't attach, but we still need to clear pending so it
            # doesn't leak into a later LLM call.
            state.pending_retrieved_docs.clear()
            return

        existing = traced._span.context_segments or []
        position = _next_segment_position(traced._span)
        new_segments: list[dict[str, Any]] = []
        for doc in state.pending_retrieved_docs:
            full_content: str = doc.get("content", "") or ""
            content_hash = _sha256_hex(full_content)
            truncated_content, was_truncated = _truncate_for_segment(full_content)
            new_segments.append(
                {
                    "position": position,
                    "source": "memory_vector",
                    "content_hash": content_hash,
                    "content": truncated_content,
                    "content_truncated": was_truncated,
                    "tokens": None,  # server tokenizes
                    "tokens_estimated": False,
                    "cache_state": None,
                    "attrs": {
                        "detection_confidence": "exact",
                        "doc_metadata": doc.get("doc_metadata", {}) or {},
                    },
                    "producer_span_id": doc.get("producer_span_id"),
                }
            )
            position += 1

        traced._span.context_segments = list(existing) + new_segments
        # One-shot consumption — clear so subsequent LLM calls don't reattach.
        state.pending_retrieved_docs.clear()


def _flatten_messages(messages: Any) -> list[Any]:
    """Flatten the nested list structure LangChain passes to on_chat_model_start.

    LangChain passes ``[[BaseMessage, ...]]`` — a list of message batches.
    The decomposer expects a flat list of message objects.
    """
    flat: list[Any] = []
    if isinstance(messages, (list, tuple)):
        for item in messages:
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)
    else:
        flat = [messages]
    return flat


def _summarize_messages(messages: Any) -> list[dict[str, Any]]:
    """Build a JSON-safe preview of LangChain chat-message lists.

    LangChain typically passes `[[BaseMessage, ...]]` (a list-per-input).
    We flatten and capture role + content text only — adapters never persist
    arbitrary raw objects on input_data because the queue serializer is
    JSON-only.
    """
    flat: list[Any] = []
    if isinstance(messages, (list, tuple)):
        for item in messages:
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)
    else:
        flat = [messages]

    summarized: list[dict[str, Any]] = []
    for m in flat:
        role = getattr(m, "type", None) or getattr(m, "role", None) or "message"
        content = getattr(m, "content", None)
        summarized.append(
            {"role": str(role), "content": "" if content is None else str(content)}
        )
    return summarized


# ---------------------------------------------------------------------------
# MEM-10 — monkey-patches for BaseCheckpointSaver + BaseStore surfaces
# ---------------------------------------------------------------------------
#
# LangGraph's checkpointer + persistent-store abstractions don't go through
# LangChain's callback manager — they're called directly by graph runtime
# code (e.g., `compiled_graph.invoke(state, config)` triggers a
# `checkpointer.put({...})` synchronously). To observe these we wrap the
# methods at install() time.
#
# All wrappers:
#   1. Resolve the active trace state; no-op when none exists.
#   2. Apply the 50ms (bucket, store, op_type) dedup window.
#   3. Call the wrapped original — exceptions propagate AS-IS (we never
#      swallow customer-code errors — only our own bookkeeping errors).
#   4. After successful return, emit the memory-typed span.


def _wrap_checkpointer_put(method_name: str, original: Any) -> Any:
    """Wrap BaseCheckpointSaver.put — emit a memory span with op=extract."""

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        try:
            state = get_active_trace()
            if state is not None:
                store_name = _identify_store(self)
                _emit_memory_op_span(
                    name=f"checkpointer.{store_name}.put",
                    memory_op_type="extract",
                    store_name=store_name,
                    metadata_extra={"memory_type": "checkpoint"},
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("morse_ai: %s memory-op emit swallowed: %s", method_name, exc)
        return result

    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    return wrapped


def _wrap_checkpointer_aput(method_name: str, original: Any) -> Any:
    """Wrap BaseCheckpointSaver.aput — async version of put, op=extract.

    LangGraph calls aput (not put) when the graph runs in an async context
    (FastAPI, astream, ainvoke). Without this wrapper, InMemorySaver,
    PostgresSaver, and all other async-graph backends produce zero memory
    spans, because put is never called on the async path.
    """

    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = await original(self, *args, **kwargs)
        try:
            state = get_active_trace()
            if state is not None:
                store_name = _identify_store(self)
                _emit_memory_op_span(
                    name=f"checkpointer.{store_name}.aput",
                    memory_op_type="extract",
                    store_name=store_name,
                    metadata_extra={"memory_type": "checkpoint", "async": True},
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("morse_ai: %s memory-op emit swallowed: %s", method_name, exc)
        return result

    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    return wrapped


def _wrap_checkpointer_aput_writes(method_name: str, original: Any) -> Any:
    """Wrap BaseCheckpointSaver.aput_writes — async incremental write, op=extract."""

    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = await original(self, *args, **kwargs)
        try:
            state = get_active_trace()
            if state is not None:
                store_name = _identify_store(self)
                _emit_memory_op_span(
                    name=f"checkpointer.{store_name}.aput_writes",
                    memory_op_type="extract",
                    store_name=store_name,
                    metadata_extra={"memory_type": "checkpoint", "async": True},
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("morse_ai: %s memory-op emit swallowed: %s", method_name, exc)
        return result

    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    return wrapped


def _wrap_checkpointer_get(method_name: str, original: Any) -> Any:
    """Wrap BaseCheckpointSaver.get — emit a memory span with op=retrieve."""

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        try:
            state = get_active_trace()
            if state is not None:
                store_name = _identify_store(self)
                _emit_memory_op_span(
                    name=f"checkpointer.{store_name}.get",
                    memory_op_type="retrieve",
                    store_name=store_name,
                    metadata_extra={"memory_type": "checkpoint"},
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("morse_ai: %s memory-op emit swallowed: %s", method_name, exc)
        return result

    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    return wrapped


def _wrap_checkpointer_aget_tuple(method_name: str, original: Any) -> Any:
    """Wrap BaseCheckpointSaver.aget_tuple — async checkpoint read, op=retrieve.

    LangGraph uses aget_tuple (not get) on the async execution path. Without
    this, checkpoint reads are invisible for all async-mode graphs.
    """

    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = await original(self, *args, **kwargs)
        try:
            state = get_active_trace()
            if state is not None:
                store_name = _identify_store(self)
                _emit_memory_op_span(
                    name=f"checkpointer.{store_name}.aget",
                    memory_op_type="retrieve",
                    store_name=store_name,
                    metadata_extra={"memory_type": "checkpoint", "async": True},
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("morse_ai: %s memory-op emit swallowed: %s", method_name, exc)
        return result

    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    return wrapped


def _wrap_store_put(method_name: str, original: Any) -> Any:
    """Wrap BaseStore.put — emit a memory span with op=extract."""

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        try:
            state = get_active_trace()
            if state is not None:
                store_name = _identify_store(self)
                _emit_memory_op_span(
                    name=f"store.{store_name}.put",
                    memory_op_type="extract",
                    store_name=store_name,
                    metadata_extra={"memory_type": "store"},
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("morse_ai: %s memory-op emit swallowed: %s", method_name, exc)
        return result

    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    return wrapped


def _wrap_store_search(method_name: str, original: Any) -> Any:
    """Wrap BaseStore.search — emit a memory span with op=retrieve.

    Per MEM-10 spec table, this surface "enriches existing retriever span"
    when the search routes through LangChain's retriever callback (which
    fires on_retriever_start/end, handled separately). When it doesn't —
    a direct BaseStore.search() call outside any LC retriever wrapper —
    we still want observability, so emit a memory span here. The dedup
    window guards against double-emission within 50ms.
    """

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        try:
            state = get_active_trace()
            if state is not None:
                store_name = _identify_store(self)
                _emit_memory_op_span(
                    name=f"store.{store_name}.search",
                    memory_op_type="retrieve",
                    store_name=store_name,
                    metadata_extra={"memory_type": "store"},
                    is_store_search=True,
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("morse_ai: %s memory-op emit swallowed: %s", method_name, exc)
        return result

    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    return wrapped


def _wrap_store_delete(method_name: str, original: Any) -> Any:
    """Wrap BaseStore.delete — emit a memory span with op=prune."""

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        try:
            state = get_active_trace()
            if state is not None:
                store_name = _identify_store(self)
                _emit_memory_op_span(
                    name=f"store.{store_name}.delete",
                    memory_op_type="prune",
                    store_name=store_name,
                    metadata_extra={"memory_type": "store"},
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("morse_ai: %s memory-op emit swallowed: %s", method_name, exc)
        return result

    wrapped.__wrapped__ = original  # type: ignore[attr-defined]
    return wrapped


# Table maps "ClassName.method_name" -> (import_path, wrapper_factory).
# Each wrapper_factory takes (method_name, original) and returns the patched
# callable. Stored in _patched_originals at install for uninstall restore.
_MEMORY_OP_PATCH_TABLE: list[tuple[str, str, str, Any]] = [
    # (key, import_module, attr_name, wrapper_factory)
    #
    # Sync path — called when the graph runs synchronously (invoke / stream).
    (
        "BaseCheckpointSaver.put",
        "langgraph.checkpoint.base",
        "BaseCheckpointSaver",
        _wrap_checkpointer_put,
    ),
    (
        "BaseCheckpointSaver.get",
        "langgraph.checkpoint.base",
        "BaseCheckpointSaver",
        _wrap_checkpointer_get,
    ),
    #
    # Async path — called when the graph runs asynchronously (ainvoke / astream).
    # LangGraph calls aput / aget_tuple / aput_writes on the async execution path,
    # NOT the sync put / get.  Any LangGraph agent running inside FastAPI,
    # an async framework, or via astream() will ONLY hit these methods.
    # Without patching them, InMemorySaver and PostgresSaver both produce
    # zero memory spans for async-mode graphs.
    (
        "BaseCheckpointSaver.aput",
        "langgraph.checkpoint.base",
        "BaseCheckpointSaver",
        _wrap_checkpointer_aput,
    ),
    (
        "BaseCheckpointSaver.aput_writes",
        "langgraph.checkpoint.base",
        "BaseCheckpointSaver",
        _wrap_checkpointer_aput_writes,
    ),
    (
        "BaseCheckpointSaver.aget_tuple",
        "langgraph.checkpoint.base",
        "BaseCheckpointSaver",
        _wrap_checkpointer_aget_tuple,
    ),
    #
    # InMemorySaver (alias MemorySaver) — LangGraph 0.2.x defines put/aput/
    # aget_tuple directly on InMemorySaver without calling super(), so the
    # BaseCheckpointSaver patches above are never reached for this common class.
    # Patch InMemorySaver's own methods as well. Best-effort; skipped if the
    # class or method isn't present in the installed version.
    (
        "InMemorySaver.put",
        "langgraph.checkpoint.memory",
        "InMemorySaver",
        _wrap_checkpointer_put,
    ),
    (
        "InMemorySaver.aput",
        "langgraph.checkpoint.memory",
        "InMemorySaver",
        _wrap_checkpointer_aput,
    ),
    (
        "InMemorySaver.aput_writes",
        "langgraph.checkpoint.memory",
        "InMemorySaver",
        _wrap_checkpointer_aput_writes,
    ),
    (
        "InMemorySaver.aget_tuple",
        "langgraph.checkpoint.memory",
        "InMemorySaver",
        _wrap_checkpointer_aget_tuple,
    ),
    (
        "InMemorySaver.get",
        "langgraph.checkpoint.memory",
        "InMemorySaver",
        _wrap_checkpointer_get,
    ),
    #
    # BaseStore — sync + async (search/put/delete).
    (
        "BaseStore.put",
        "langgraph.store.base",
        "BaseStore",
        _wrap_store_put,
    ),
    (
        "BaseStore.search",
        "langgraph.store.base",
        "BaseStore",
        _wrap_store_search,
    ),
    (
        "BaseStore.delete",
        "langgraph.store.base",
        "BaseStore",
        _wrap_store_delete,
    ),
]


def _install_memory_op_patches() -> None:
    """Monkey-patch BaseCheckpointSaver + BaseStore methods. Best-effort.

    LangGraph version drift: if a method or class isn't present in the
    installed LangGraph version, the corresponding patch silently skips —
    instrumentation for that surface degrades gracefully. The other
    surfaces still work.
    """
    import importlib

    for key, mod_path, cls_name, factory in _MEMORY_OP_PATCH_TABLE:
        method_name = key.split(".", 1)[1]
        try:
            module = importlib.import_module(mod_path)
            cls = getattr(module, cls_name, None)
            if cls is None:
                logger.debug(
                    "morse_ai: LangGraph class %s.%s not found — skipping %s",
                    mod_path,
                    cls_name,
                    key,
                )
                continue
            original = getattr(cls, method_name, None)
            if original is None or not callable(original):
                logger.debug(
                    "morse_ai: %s.%s not callable — skipping",
                    cls_name,
                    method_name,
                )
                continue
            # Skip if already wrapped (idempotent install).
            if getattr(original, "__wrapped__", None) is not None:
                continue
            _patched_originals[key] = (cls, method_name, original)
            setattr(cls, method_name, factory(method_name, original))
        except ImportError:
            logger.debug(
                "morse_ai: %s not importable — skipping memory-op patch %s",
                mod_path,
                key,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("morse_ai: failed to install memory-op patch %s: %s", key, exc)


def _uninstall_memory_op_patches() -> None:
    """Restore originals for all successfully patched methods. Idempotent."""
    for key, entry in list(_patched_originals.items()):
        try:
            cls, method_name, original = entry
            setattr(cls, method_name, original)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "morse_ai: failed to uninstall memory-op patch %s: %s", key, exc
            )
        finally:
            _patched_originals.pop(key, None)


# ---------------------------------------------------------------------------
# Install / uninstall — globally attaches the handler via LangChain's callback
# manager. The customer doesn't have to thread `callbacks=[handler]` through
# their graph invocations.
# ---------------------------------------------------------------------------

_original_callback_manager_init = None


@absorb_errors(fingerprint="morse_ai.adapters.langgraph.install")
def install(agent_name: str | None = None) -> bool:
    """Globally register a LangGraphCallbackHandler with LangChain.

    Idempotent — safe to call repeatedly. Returns True if installed (newly or
    already), False if LangGraph isn't available or installation failed.
    """
    global _original_callback_manager_init

    try:
        import langgraph  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False

    already_installed = getattr(langgraph, _INSTALL_FLAG, False)
    # Allow explicit agent_name to override a previously auto-installed "langgraph"
    # default. This handles the hot-reload race where auto_detect fires first with
    # the default name before the caller's explicit install() runs.
    if already_installed and agent_name is None:
        return True

    try:
        from langchain_core.callbacks.manager import (  # type: ignore[import-not-found]
            BaseCallbackManager,
        )
    except ImportError:
        logger.warning(
            "morse_ai: LangGraph detected but langchain_core unavailable — "
            "cannot register callback. Skipping LangGraph instrumentation."
        )
        return False

    # Only capture the original on first install; on re-installs the global
    # already holds the true original (current __init__ is the patched version).
    if not already_installed:
        _original_callback_manager_init = BaseCallbackManager.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        _original_callback_manager_init(self, *args, **kwargs)
        try:
            already_attached = any(
                isinstance(h, LangGraphCallbackHandler) for h in self.handlers
            )
            if not already_attached:
                self.add_handler(LangGraphCallbackHandler(agent_name=agent_name))
        except Exception as exc:  # noqa: BLE001 — non-invasive
            logger.debug(
                "morse_ai: failed to attach LangGraph callback to manager: %s", exc
            )

    BaseCallbackManager.__init__ = patched_init  # type: ignore[method-assign]

    # MEM-10: monkey-patch BaseCheckpointSaver + BaseStore surfaces.
    # Best-effort — degrades gracefully when a class/method is absent on
    # this LangGraph version.
    _install_memory_op_patches()

    setattr(langgraph, _INSTALL_FLAG, True)
    logger.info("morse_ai: LangGraph adapter installed (agent=%s)", agent_name)
    return True


@absorb_errors(fingerprint="morse_ai.adapters.langgraph.uninstall")
def uninstall() -> None:
    """Restore the original LangChain callback manager. Idempotent."""
    global _original_callback_manager_init

    try:
        import langgraph  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return

    if not getattr(langgraph, _INSTALL_FLAG, False):
        return

    if _original_callback_manager_init is not None:
        try:
            from langchain_core.callbacks.manager import (  # type: ignore[import-not-found]
                BaseCallbackManager,
            )

            BaseCallbackManager.__init__ = _original_callback_manager_init  # type: ignore[method-assign]
        except ImportError:
            pass
        _original_callback_manager_init = None

    # MEM-10: restore checkpointer/store originals.
    _uninstall_memory_op_patches()

    setattr(langgraph, _INSTALL_FLAG, False)


def is_installed() -> bool:
    """Return True if the LangGraph adapter has installed itself."""
    try:
        import langgraph  # type: ignore[import-not-found]
    except ImportError:
        return False
    return getattr(langgraph, _INSTALL_FLAG, False)
