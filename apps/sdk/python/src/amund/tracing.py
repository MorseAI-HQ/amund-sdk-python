"""SDK v2 tracing core — TraceContext, trace_agent decorator, run()/span() CM.

Design:
- contextvars.ContextVar provides async-safe and thread-safe trace propagation
- trace_agent / run() create a root "agent" span and manage the trace lifecycle
- span() creates child spans with automatic parent_span_id linking
- Transport is injected via _set_transport() (called by _Client.__init__)
- If no transport is configured, traces are captured in _last_trace for testing

MHQ-668 — OTel transport rebuild
---------------------------------
Everything above is UNCHANGED: ``_TraceState``, ``SpanData`` accumulation via
``span_stack``/``completed_spans``, and ``_build_trace_dict()`` all still work
exactly as before, so the adapters that build ``SpanData`` directly and
append to ``state.completed_spans`` (``adapters/anthropic.py``,
``adapters/openai.py``, ``adapters/openai_agents.py``,
``adapters/claude_agent_sdk.py``) needed zero changes.

What changed is what happens to the finished trace dict:

- ``record()`` (in ``client.py``) still calls ``_send_trace()``, which still
  posts ``{"traces": [entry]}`` to the legacy ``BatchTransport`` JSON queue —
  unchanged, per MHQ-668 design (record()/log()/deps-scan stay on HTTP JSON).
- ``run()`` / ``span()`` / ``trace_agent`` now call ``_finish_trace()``,
  which hands the SAME dict shape to ``otel_provider.emit_trace_entry()`` —
  real OTel spans, batched via ``BatchSpanProcessor``, redacted once at
  export time, sent as OTLP protobuf+gzip. ``AMUND_LEGACY_TRANSPORT=1``
  reverts this call to the old JSON queue path (rollback escape hatch).

``_last_trace`` (the ``_get_last_trace()`` test helper) is populated by
BOTH paths from the same unredacted dict, via a single ``redact_value()``
call — the previous code computed redaction twice (once inside
``BatchTransport.track()``, once again for ``_last_trace``); that duplicate
pass is gone.
"""

import functools
import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from amund._internal.agent_name import (
    GUESSED_SOURCES,
    SOURCE_FRAMEWORK,
    SOURCE_RUN,
    identity_legacy,
    resolve_agent_identity,
)
from amund._internal.attribution import (
    add_prompt_ref,
    add_record_key,
    clean_agent_version,
    clean_prompt_ref,
)
from amund._internal.errors import absorb_errors
from amund._internal.redaction import DISABLED_CONFIG, RedactionConfig
from amund.spans import SpanData, SpanType

logger = logging.getLogger("amund")

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------


@dataclass
class _TraceState:
    """Thread/coroutine-local state for an active trace."""

    trace_id: str
    agent_name: str
    span_stack: list[SpanData] = field(default_factory=list)
    # Set to True when TracedRun.__exit__ completes so _restore_inherited_trace
    # can distinguish a live parent trace from a stale snapshot captured at
    # handler construction time in a different async context.
    is_closed: bool = False
    completed_spans: list[SpanData] = field(default_factory=list)
    # Adapter-private side-channels. Keyed by stringified run_id (UUID hex
    # form) so callers don't need to import the uuid module just to read.
    # The LangGraph adapter uses these to wire memory-span span_ids onto
    # the next LLM span's `context_segments`.
    retriever_runs: dict[str, str] = field(default_factory=dict)
    pending_retrieved_docs: list[dict[str, Any]] = field(default_factory=list)
    # Per-trace override of the resource-level ``morse_ai.source`` tag.
    # Wins over the process-level ``_source`` when set; used by Amund's
    # own API process where copilot and investigator share a Django process
    # but must tag traces with different emitter labels. ``None`` falls back
    # to the process-level value.
    source_override: str | None = None
    # Outcome declared via module-level ``set_outcome()`` — the one outcome
    # API that works under both ``run()`` and ``@trace_agent``. ``None`` means
    # nobody declared one, and then nothing goes on the wire: any ``success``
    # the server receives is stored as provenance='developer'.
    success: bool | None = None
    outcome: str | None = None
    # AGE-240: framework auto-discovery (Phase 2). Captured at __enter__ time
    # from the thread-local set by adapters, then re-read at emit time so
    # adapters that activate mid-trace still get their framework recorded.
    # ``None`` => no adapter ran during this trace; we omit the
    # ``morse_ai.framework`` resource attribute entirely (absence is the
    # signal — never invent a default).
    framework: str | None = None
    # Multi-turn attribution. thread_id groups traces in the same conversation;
    # user_id is the app-layer end-user identifier. Both are optional.
    thread_id: str | None = None
    user_id: str | None = None
    # Cost-attribution dimensions — set via set_context() or Trace(user_metadata=…).
    # Keys/values are always strings at the SDK boundary.
    user_metadata: dict[str, str] | None = None
    # INFRACAP-04: per-trace counters for default-on infra capture (db/http/
    # queue) and the single summary span per kind that absorbs calls past the
    # cap. Owned by ``infra_capture``; kept here so they live and die with the
    # trace and need no global map.
    infra_counts: dict[str, int] = field(default_factory=dict)
    infra_summaries: dict[str, SpanData] = field(default_factory=dict)
    # ATTR-01/04/06: attribution declared by the app. ``record_keys`` links the
    # trace to business records ({"zendesk_ticket": "4812"}); ``agent_version``
    # is the app's own version label (per-run value; the init() default fills
    # the gap at build time); ``prompts`` are references to named prompts, each
    # optionally pinned to the span that was open when set_prompt() was called.
    record_keys: dict[str, str] = field(default_factory=dict)
    agent_version: str | None = None
    prompts: list[dict[str, Any]] = field(default_factory=list)
    # AID-01/04: which ladder tier named the root agent, the ``run("app")`` label
    # when a discovered framework root took the agent name (AID-04), and whether
    # a framework root has already been adopted for this trace.
    identity_source: str | None = None
    app: str | None = None
    root_adopted: bool = False
    # MHQ-1410: tool calls a raw-client LLM response asked for, keyed by the
    # provider's call id, waiting for the request that carries their result.
    # The OpenAI adapter turns each matched pair into a ``tool`` span.
    pending_tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    # MHQ-1928: manual ``span(..., "tool")`` spans already standing in for an
    # inferred tool span, so one manual span never absorbs two tool calls.
    matched_tool_spans: set[str] = field(default_factory=set)


def claim_manual_tool_span(
    state: "_TraceState", name: str, call_id: str | None, asked_iso: str
) -> SpanData | None:
    """The manual tool span that already records this tool call, if any.

    MHQ-1928. A raw client wrapped with ``wrap()`` infers a ``tool`` span for
    every tool call it sees answered. An app that also wraps its tool in
    ``span(name, "tool")`` would get the same call twice. The manual span wins:
    it carries real timing. It matches when it set ``tool_call_id`` metadata to
    this call's id, or, failing that, when it has the tool's name and started
    after the model asked for the call. The matched span gains the
    ``tool_call_id`` so the two stay linked.
    """
    try:
        asked = datetime.fromisoformat(asked_iso)
    except (TypeError, ValueError):
        return None
    by_name: SpanData | None = None
    for done in state.completed_spans:
        if done.type != SpanType.TOOL or done.span_id in state.matched_tool_spans:
            continue
        meta = done.metadata or {}
        if meta.get("timing") == "inferred":
            continue
        if call_id and meta.get("tool_call_id") == call_id:
            by_name = done
            break
        if by_name is not None or done.name != name or meta.get("tool_call_id"):
            continue
        try:
            started = datetime.fromisoformat(done.started_at)
        except (TypeError, ValueError):
            started = None
        if started is not None and started >= asked:
            by_name = done
    if by_name is None:
        return None
    state.matched_tool_spans.add(by_name.span_id)
    if call_id:
        if by_name.metadata is None:
            by_name.metadata = {}
        by_name.metadata.setdefault("tool_call_id", call_id)
    return by_name


# ContextVar is the correct tool here: async-safe, no thread-local race
_active_trace: ContextVar[_TraceState | None] = ContextVar(
    "_active_trace", default=None
)


def get_active_trace() -> "_TraceState | None":
    """The live trace, or None. A closed trace is not a trace.

    MHQ-980. Read this instead of ``_active_trace.get()`` anywhere the answer
    decides where a span goes. The raw var can hold a ``_TraceState`` whose
    trace has already been finalized and emitted, and to every reader that
    stale state is indistinguishable from a live one.

    **The leak cannot be prevented, only rendered harmless here.**
    ``TracedRun.__exit__`` clears the var with ``_active_trace.reset(token)``,
    and when ``__enter__`` and ``__exit__`` land in different contexts that
    raises ``ValueError``; the fallback ``_active_trace.set(None)`` writes to
    the *current* context, not the one holding the state. Python offers no way
    to reset another context's ContextVar. So the fix has to live at the point
    of use, which is why it lives in one function rather than 51.

    MHQ-672 is what this costs when it is missed: ``openai_agents`` adopted a
    finalized trace and every later span was appended to it and silently
    discarded — 28 of 33 conversations lost, no error at any log level.

    ``_active_trace`` stays exported and unguarded for the handful of callers
    that legitimately want the raw var: ``TracedRun``/``trace_agent``, which own
    the flag, and the adapters that re-``set()`` a state they are holding
    deliberately.
    """
    state = _active_trace.get()
    return None if state is not None and state.is_closed else state


# Transport singleton (set by client.init or _set_transport for tests)
_transport: Any = None

# Active redaction config — mirrors the transport's config so trace payloads
# captured into _last_trace (for in-process tests with no live transport)
# observe the same redaction semantics as transported traces.
_redaction_config: RedactionConfig = DISABLED_CONFIG

# Active resource-level source label (AMUND_INTERNAL_SOURCE / init kwarg).
# When set, it's stamped onto every emitted trace under
# ``resource["morse_ai.source"]`` so dashboards can discriminate dogfood
# emitters. None on customer integrations — absence is the signal.
_source: str | None = None

# Git commit SHA detected at init time (env var or subprocess). Attached to
# every trace as ``resource["git.revision"]`` for deployment-aware filtering.
_git_revision: str | None = None

# Declared agent version from ``init(agent_version=)``: the default for every
# trace. A per-run ``run(..., agent_version=)`` overrides it.
_agent_version: str | None = None

# SETUP-10: ``init(capture_end_user=True)`` opt-in. Off by default: adapters
# never read the provider's ``user`` / ``metadata.user_id`` unless this is set.
_capture_end_user: bool = False

# ``service.version`` from OTEL_RESOURCE_ATTRIBUTES, read once at init and sent
# as ``resource["service.version"]`` (ATTR-06). No init parameter by design.
_service_version: str | None = None

# Last completed trace — stored for test assertions, replaced on each trace
_last_trace: dict[str, Any] | None = None


def _ensure_sdk_initialized() -> None:
    """Trigger the SDK's lazy env-based auto-init from inside tracing entrypoints.

    The tracing primitives (``run()``, ``span()``, ``trace_agent``) need the
    same first-use auto-init that ``track()`` / ``record()`` get — otherwise
    a process that ONLY uses the tracing API never builds a transport and
    every trace silently lands in ``_last_trace`` instead of being POSTed.

    This was the gunicorn-sync-worker silent-drop bug (AGE-240): refagents
    used ``with amund.run(...)`` exclusively, never called ``track()``,
    and so the transport was never instantiated under workers that hadn't
    independently exercised ``track()`` first.

    Import is deferred to avoid the ``__init__ -> tracing -> __init__``
    circular at module load time.
    """
    try:
        from amund import _ensure_initialized  # noqa: PLC0415

        _ensure_initialized()
    except Exception as exc:  # noqa: BLE001
        logger.debug("amund: tracing auto-init swallowed error: %s", exc)
        _report(exc, "morse_ai.tracing.auto_init")


def _note_run_if_unsendable() -> None:
    """MHQ-1787: count a run opened with no API key, so exit can say it was not sent.

    ``_ensure_initialized`` returned False for a missing key (not ``AMUND_DISABLED``,
    which is a deliberate opt-out) and the run keeps its trace id but goes nowhere.
    """
    try:
        import amund  # noqa: PLC0415
        from amund._internal import diagnostics, env  # noqa: PLC0415

        no_key = not env.get_api_key()
        if amund._client is None and no_key and not env.is_disabled():
            diagnostics.record_unsent_run_no_key()
    except Exception as exc:  # noqa: BLE001
        logger.debug("amund: unsent-run note swallowed error: %s", exc)


def _report(exc: BaseException, fingerprint: str) -> None:
    """Route a swallowed internal error to the diagnostics channel.

    SDKDIAG-07: these sites already logged at DEBUG and returned, which made
    them invisible to us and nearly invisible to the customer. Control flow is
    unchanged — the site still swallows, still logs, still returns the same
    value; the failure is now also reported.
    """
    try:
        from amund._internal import diagnostics

        diagnostics.report(exc, fingerprint=fingerprint)
    except Exception:  # noqa: BLE001 — reporting must never break the swallow
        pass


def coerce_dims(dims: dict[str, Any], fingerprint: str) -> dict[str, str]:
    """`str()` every value, dropping any that raises. Never raises. (MHQ-947)

    Cost-attribution dimensions are USER values, and the SDK is the one that
    chooses to stringify them. A bare ``str(v)`` therefore runs arbitrary
    customer code — ``__str__``/``__repr__`` — on the telemetry path, and an
    exception there propagated straight into the caller: ``set_context()``
    raised into host code, and ``run(user_metadata=...)`` failed at trace
    START, taking the whole ``with`` block down with it. That is the exact
    failure the non-invasive guarantee exists to prevent, caused by
    instrumentation.

    The trigger is not exotic. An ORM object whose ``__str__`` lazily hits a
    dead connection, or a Django model with a broken ``__repr__``, is a
    realistic non-malicious source.

    Coercion is guarded PER KEY, not per call: wrapping the whole entry point
    in ``absorb_errors`` would silently discard every dimension because one was
    bad, and losing a customer's `customer_id` because their `feature` object
    misbehaved is a worse answer than dropping only the offender. Each failure
    is reported once so we learn about it, and the good dimensions survive.
    """
    out: dict[str, str] = {}
    for key, value in dims.items():
        try:
            out[str(key)] = str(value)
        except Exception as exc:  # noqa: BLE001 — customer __str__ may raise
            _report(exc, fingerprint)
    return out


def _set_transport(t: Any) -> None:
    """Inject the BatchTransport. Called by `_Client.__init__`."""
    global _transport
    _transport = t


def _set_redaction_config(config: RedactionConfig | None) -> None:
    """Inject the active RedactionConfig. Called by `_Client.__init__`."""
    global _redaction_config
    _redaction_config = config or DISABLED_CONFIG


def _set_source(value: str | None) -> None:
    """Inject the active resource-level source label. Called by `_Client.__init__`."""
    global _source
    _source = value


def _get_source() -> str | None:
    """Return the active source label (for tests)."""
    return _source


def _set_git_revision(value: str | None) -> None:
    """Inject the git commit SHA detected at init time. Called by `_Client.__init__`."""
    global _git_revision
    _git_revision = value


def _set_agent_version(value: str | None) -> None:
    """Inject the init-time default agent version. Called by `_Client.__init__`."""
    global _agent_version
    _agent_version = clean_agent_version(value)


def _set_capture_end_user(value: bool) -> None:
    """Inject the init-time end-user capture opt-in. Called by `_Client.__init__`."""
    global _capture_end_user
    _capture_end_user = bool(value)


def hash_end_user(value: Any) -> str | None:
    """``"h:" + sha256(value)[:32]``, or None for an empty / non-string value."""
    if not isinstance(value, str) or not value:
        return None
    import hashlib  # noqa: PLC0415

    return "h:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def apply_end_user(state: Any, raw: Any) -> None:
    """SETUP-10: set the trace ``user_id`` from a provider end-user value.

    No-op unless ``init(capture_end_user=True)``. Only the hash is stored; the
    raw value never leaves this function. An app-set ``user_id`` always wins.
    ``customer_id`` is never touched.
    """
    if not _capture_end_user or state is None:
        return
    if getattr(state, "user_id", None) is not None:
        return
    hashed = hash_end_user(raw)
    if hashed is not None:
        state.user_id = hashed


def _set_service_version(value: str | None) -> None:
    """Inject ``service.version`` from the env. Called by `_Client.__init__`."""
    global _service_version
    _service_version = value


def _get_transport() -> Any | None:
    return _transport


def _get_last_trace() -> dict[str, Any] | None:
    """Return the most recently completed trace payload (for tests)."""
    return _last_trace


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _elapsed_ms(start_monotonic: float) -> int:
    return int((time.monotonic() - start_monotonic) * 1000)


def exception_type_name(exc: BaseException | None) -> str | None:
    """Class name of ``exc``; module-qualified except for builtins."""
    if exc is None:
        return None
    cls = type(exc)
    module = getattr(cls, "__module__", "")
    return (
        cls.__qualname__
        if module in ("builtins", "")
        else f"{module}.{cls.__qualname__}"
    )


def _remove_from_stack(stack: list[SpanData], span: SpanData) -> None:
    """Drop ``span`` from the shared stack by identity.

    ``stack.pop()`` removes whichever span is on top, which under parallel
    branches (LangGraph fan-out) is another branch's span: the stack then
    carries a closed span and ends wrong for everything that follows (AID-05).
    """
    for i in range(len(stack) - 1, -1, -1):
        if stack[i] is span:
            del stack[i]
            return


def adopt_native_root(state: _TraceState | None, native_name: str | None) -> None:
    """AID-01/AID-04: let a discovered framework root name the trace's agent.

    A guessed name (service/entry script/default) yields to the framework root.
    An explicit ``run("app")`` name is kept as the ``morse.app`` label and the
    discovered root keeps the agent identity, so no second agent is forked.
    Idempotent; a no-op under ``AMUND_AGENT_IDENTITY=legacy``.
    """
    if state is None or state.root_adopted or identity_legacy():
        return
    if not state.span_stack:
        return
    state.root_adopted = True
    native = (native_name or "").strip()
    if not native:
        return
    if native.lower() == (state.agent_name or "").lower():
        return
    source = state.identity_source or SOURCE_RUN
    if source == SOURCE_RUN:
        state.app = state.agent_name
    elif source not in GUESSED_SOURCES:
        return
    state.agent_name = native
    state.identity_source = SOURCE_FRAMEWORK
    state.span_stack[0].name = native


def _finalize_span(
    span: SpanData,
    start_mono: float,
    status: str = "completed",
    error_message: str | None = None,
    exc: BaseException | None = None,
) -> None:
    """Set timing and status on a span in-place."""
    span.ended_at = _now_iso()
    span.duration_ms = _elapsed_ms(start_mono)
    span.status = status
    if error_message:
        span.error_message = error_message
    if exc is not None:
        span.exception_type = exception_type_name(exc)


def _build_trace_dict(
    state: _TraceState,
    root_span: SpanData,
    success: bool | None = None,
    outcome: str | None = None,
) -> dict[str, Any]:
    """Build the wire-format trace payload."""
    trace_fields: dict[str, Any] = {
        "trace_id": state.trace_id,
        # Server-side `TracePayloadIn` (apps/api/apps/traces/schemas.py)
        # expects `agent`, not `agent_name`. Read-side `TraceWithSpansOut`
        # exposes `agent_name`. We keep `state.agent_name` internally for
        # clarity and rename only at the wire boundary.
        "agent": state.agent_name,
        "status": root_span.status,
        "started_at": root_span.started_at,
    }
    if root_span.ended_at:
        trace_fields["ended_at"] = root_span.ended_at
    if root_span.duration_ms is not None:
        trace_fields["duration_ms"] = root_span.duration_ms
    if root_span.error_message:
        trace_fields["error_message"] = root_span.error_message
    if success is not None:
        trace_fields["success"] = success
    if outcome is not None:
        trace_fields["outcome"] = outcome
    if getattr(state, "thread_id", None) is not None:
        trace_fields["thread_id"] = state.thread_id
    if getattr(state, "user_id", None) is not None:
        trace_fields["user_id"] = state.user_id
    if getattr(state, "user_metadata", None):
        trace_fields["user_metadata"] = state.user_metadata
    # ATTR-01/04: omitted when empty, so old servers and old traces see no change.
    if getattr(state, "record_keys", None):
        trace_fields["record_keys"] = dict(state.record_keys)
    effective_agent_version = getattr(state, "agent_version", None) or _agent_version
    if effective_agent_version:
        trace_fields["agent_version"] = effective_agent_version
    if getattr(state, "prompts", None):
        trace_fields["prompts"] = [dict(p) for p in state.prompts]
    # AID-01/04 wire keys the server reads from trace metadata.
    if not identity_legacy():
        identity_md = {
            k: v
            for k, v in (
                ("identity_source", getattr(state, "identity_source", None)),
                ("morse_app", getattr(state, "app", None)),
            )
            if v
        }
        if identity_md:
            trace_fields["metadata"] = identity_md

    all_spans = [root_span] + list(state.completed_spans)
    entry: dict[str, Any] = {
        "trace": trace_fields,
        "spans": [s.to_dict() for s in all_spans],
    }
    # Per-trace ``source_override`` wins over the process-level ``_source``.
    # The override exists so Amund's own API process — which hosts both
    # the Copilot graph and the Anomaly Investigator agent — can tag each
    # trace with the correct emitter, even though the SDK init is shared.
    # MHQ-1100: read the env lazily at emission — apps/api sets
    # AMUND_INTERNAL_SOURCE (setdefault) after the SDK may already have
    # initialised, so the init-time snapshot in ``_source`` can be stale.
    from amund._internal.env import get_internal_source

    effective_source = (
        getattr(state, "source_override", None) or _source or get_internal_source()
    )
    # AGE-240: framework was captured at __enter__ and may have been refreshed
    # at exit; re-read the thread-local once more in case the adapter ran late
    # (e.g. trace opened manually with ``run()`` and the very first wrapped
    # call set the framework after enter).
    from amund._internal.active_framework import get_active_framework

    effective_framework = getattr(state, "framework", None) or get_active_framework()
    if (
        effective_source is not None
        or effective_framework is not None
        or _git_revision is not None
        or _service_version is not None
    ):
        # Resource-level attributes are per-process by spec, but we lift
        # them to the trace envelope so co-resident emitters (copilot +
        # investigator in the same Django process) can carry different
        # source labels without per-span duplication on the wire.
        resource: dict[str, Any] = {}
        if effective_source is not None:
            resource["morse_ai.source"] = effective_source
        if effective_framework is not None:
            resource["morse_ai.framework"] = effective_framework
        if _git_revision is not None:
            resource["git.revision"] = _git_revision
        if _service_version is not None:
            resource["service.version"] = _service_version
        entry["resource"] = resource
    return entry


def _capture_last_trace(entry: dict[str, Any]) -> dict[str, Any]:
    """Apply the SDK's single redaction pass for test introspection and
    cache the result in ``_last_trace``. Returns the redacted dict.

    Shared by both emission paths (``_send_trace`` for record()'s legacy
    JSON path, ``_finish_trace`` for the OTel path) so there is exactly one
    redaction computation per trace for this purpose — the OTel path's
    ACTUAL wire payload is redacted separately (and only once) inside
    ``otel_provider._RedactingSpanExporter.export()``, off the hot path;
    this call is cheap test/debug bookkeeping, not the wire-redaction pass.
    """
    global _last_trace
    redacted = _redaction_config.redact_value(entry)
    _last_trace = redacted
    return redacted


def _send_trace(entry: dict[str, Any]) -> None:
    """record()'s emission path: legacy JSON batch transport.

    Per MHQ-668 design, ``record()`` / ``amund.log()`` / the dependency
    scan POST keep their pre-existing HTTP JSON paths — they are not part
    of the OTLP migration. This posts ``{"traces": [entry]}`` to the
    ``BatchTransport`` queue exactly as before (redaction happens inside
    ``BatchTransport.track()``); also updates ``_last_trace``.
    """
    _capture_last_trace(entry)
    if _transport is not None:
        try:
            _transport.track({"traces": [entry]})
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: _send_trace swallowed error: %s", exc)
            _report(exc, "morse_ai.tracing._send_trace")


def _finish_trace(entry: dict[str, Any]) -> None:
    """``run()`` / ``span()`` / ``trace_agent``'s emission path: real OTel
    spans over OTLP (MHQ-668). Updates ``_last_trace`` the same way
    ``_send_trace`` does, then hands the dict to
    ``otel_provider.emit_trace_entry()`` — unless ``AMUND_LEGACY_TRANSPORT=1``,
    in which case it falls back to the old JSON queue path instead (rollback
    escape hatch; NOT used together with the OTel path — never both, to
    avoid double-ingesting the same trace).
    """
    _capture_last_trace(entry)
    from amund._internal import env as _env  # noqa: PLC0415

    if _env.is_legacy_transport():
        if _transport is not None:
            try:
                _transport.track({"traces": [entry]})
            except Exception as exc:  # noqa: BLE001
                logger.debug("amund: _finish_trace legacy-path swallowed: %s", exc)
                _report(exc, "morse_ai.tracing._finish_trace.legacy")
        return

    try:
        from amund.otel_provider import emit_trace_entry  # noqa: PLC0415

        emit_trace_entry(entry)
    except Exception as exc:  # noqa: BLE001
        logger.debug("amund: _finish_trace OTel emission swallowed: %s", exc)
        _report(exc, "morse_ai.tracing._finish_trace.otel")


# ---------------------------------------------------------------------------
# trace_agent decorator
# ---------------------------------------------------------------------------


def set_outcome(success: bool, outcome: str | None = None) -> None:
    """Declare the outcome of the trace that is currently open.

    Works the same inside ``with amund.run(...)`` and ``@trace_agent``.
    Amund never guesses an outcome: a trace with no declared outcome is stored
    with ``success`` unset when it returned cleanly (a raise records
    ``success=False``), so eval, classifier or webhook signals can fill it
    in. No-op when no trace is open.
    """
    state = get_active_trace()
    if state is None:
        logger.debug("amund: set_outcome() called with no open trace; ignored")
        return
    state.success = success
    state.outcome = outcome


@absorb_errors(fingerprint="morse_ai.link_record")
def link_record(key_type: str, key_value: str | int) -> None:
    """Link the open trace to a business record, e.g. a ticket or an order.

        amund.link_record("zendesk_ticket", 4812)

    Record keys let Amund attach outcomes that arrive later (a ticket closed,
    a refund issued) to the trace that handled the record. Call it as many
    times as you like: keys accumulate, and a repeated ``key_type`` replaces
    its earlier value (the wire shape is ``{type: value}``).

    ``key_type`` is lower-cased and must match ``^[a-z0-9][a-z0-9_.-]{0,63}$``.
    ``key_value`` may be a string or an int; it is trimmed and must be 1-256
    characters. Invalid keys are dropped (debug log, never an exception), and
    at most 20 keys are kept per trace. Use opaque record ids, not personal
    data such as email addresses.

    Works the same inside ``with amund.run(...)`` and ``@trace_agent``.
    No-op (with a debug log) when no trace is open.
    """
    state = get_active_trace()
    if state is None:
        logger.debug("amund: link_record() called with no open trace; ignored")
        return
    add_record_key(state.record_keys, key_type, key_value)


@absorb_errors(fingerprint="morse_ai.set_prompt")
def set_prompt(name: str, version: int | str | None = None) -> None:
    """Record which named prompt (and optionally which version) this trace used.

        amund.set_prompt("triage", version=7)
        amund.set_prompt("triage")   # prompts kept in code: no version needed

    Called inside an open ``amund.span(...)``, the reference is pinned to
    that span; called outside one, it belongs to the trace (the root span).
    Teams that keep prompts in code need not invent version numbers: the
    declared ``agent_version`` or the git revision identifies the change.

    ``name`` must be 1-255 characters. ``version`` is an int >= 1 or a
    whole-number string (``"3"``, ``"v3"``); any other value is dropped with a
    one-time warning and the name is kept. Invalid input never raises. At most
    20 references are kept per trace.

    Wire behaviour. The JSON path carries every distinct reference. The OTLP
    path carries one reference per span (``morse.prompt.name`` /
    ``morse.prompt.version``): if several land on the same span, the LAST one
    wins on that span, and none of them overwrites a different span's.
    No-op (with a debug log) when no trace is open.
    """
    state = get_active_trace()
    if state is None:
        logger.debug("amund: set_prompt() called with no open trace; ignored")
        return
    span_id: str | None = None
    # span_stack[0] is the trace's root span; only a deeper entry is a span the
    # caller opened, and only then is the span id worth pinning.
    if len(state.span_stack) > 1:
        span_id = state.span_stack[-1].span_id
    ref = clean_prompt_ref(name, version, span_id)
    if ref is not None:
        add_prompt_ref(state.prompts, ref)


def link_fetched_prompt(name: Any, version: Any) -> bool:
    """Link the open run (if any) to a prompt fetched via ``get_prompt`` (UP-09).

    Reads the active-run contextvar at call time. Returns True when a ref was
    added or upgraded. Never raises.
    """
    try:
        state = get_active_trace()
        if state is None:
            return False
        span_id: str | None = None
        if len(state.span_stack) > 1:
            span_id = state.span_stack[-1].span_id
        ref = clean_prompt_ref(name, version, span_id, link_source="fetch")
        if ref is None:
            return False
        add_prompt_ref(state.prompts, ref)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("amund: prompt fetch-link skipped: %s", exc)
        return False


def trace_agent(name_or_func=None):
    """Decorator that wraps a function in a trace with a root agent span.

    Supports both bare @trace_agent and parameterised @trace_agent("name"):

        @trace_agent
        def my_agent(): ...

        @trace_agent("custom-name")
        def my_agent(): ...
    """

    def _make_wrapper(func, agent_name: str):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Lazy auto-init — see _ensure_sdk_initialized() for rationale.
            _ensure_sdk_initialized()
            _note_run_if_unsendable()
            trace_id = uuid.uuid4().hex
            state = _TraceState(
                trace_id=trace_id, agent_name=agent_name, identity_source=SOURCE_RUN
            )
            # AGE-240: capture framework at trace open. Re-read at emit time
            # picks up any adapter that activates after we enter.
            from amund._internal.active_framework import (
                get_active_framework as _get_av_fw,
            )

            state.framework = _get_av_fw()
            token = _active_trace.set(state)

            root_span = SpanData(
                trace_id=trace_id,
                type=SpanType.AGENT,
                name=agent_name,
            )
            start_mono = time.monotonic()
            state.span_stack.append(root_span)

            exc_val = None
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as exc:
                exc_val = exc
                raise
            finally:
                try:
                    _remove_from_stack(state.span_stack, root_span)
                    if exc_val is not None:
                        _finalize_span(
                            root_span,
                            start_mono,
                            status="failed",
                            error_message=str(exc_val),
                            exc=exc_val,
                        )
                    else:
                        _finalize_span(root_span, start_mono)

                    try:
                        _active_trace.reset(token)
                    except ValueError:
                        # Token from a different context — same unavoidable leak
                        # TracedRun.__exit__ documents. Fall back to clearing
                        # the current context.
                        _active_trace.set(None)
                    # MHQ-980: mark it closed BEFORE emitting, exactly as
                    # TracedRun.__exit__ does. Without this the decorator's
                    # state leaked into the ContextVar still looking live, so
                    # `get_active_trace()` — and every is_closed check written
                    # before it — could not tell it apart from a running trace.
                    state.is_closed = True
                    # MHQ-1157: same rule as run() — a declared outcome wins,
                    # a raise is a fact (success=False), and a clean return
                    # is not proof of success, so it sends nothing.
                    success = state.success
                    if success is None and exc_val is not None:
                        success = False
                    entry = _build_trace_dict(
                        state, root_span, success=success, outcome=state.outcome
                    )
                    _finish_trace(entry)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("amund: trace_agent cleanup swallowed error: %s", exc)

        return wrapper

    # Bare @trace_agent — name_or_func is the actual function
    if callable(name_or_func):
        return _make_wrapper(name_or_func, name_or_func.__name__)

    # Parameterised @trace_agent("name")
    custom_name = name_or_func

    def decorator(func):
        resolved_name = custom_name if custom_name else func.__name__
        return _make_wrapper(func, resolved_name)

    return decorator


# ---------------------------------------------------------------------------
# run() context manager
# ---------------------------------------------------------------------------


class TracedRun:
    """Context manager for explicit trace control.

    Usage:
        with amund.run("my-agent") as r:
            result = do_work()
            r.set_outcome(success=True, outcome="qualified")

    Cross-service distributed tracing:
        # In a downstream service, pass the incoming traceparent to continue
        # the same logical trace:
        with amund.run(  # noqa: E501
            "downstream-agent", incoming_traceparent=headers.get("traceparent")
        ):
            ...
    """

    def __init__(
        self,
        agent_name: str | None,
        incoming_traceparent: str | None = None,
        source: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        user_metadata: dict[str, str] | None = None,
        record_keys: dict[str, str | int] | None = None,
        agent_version: str | None = None,
    ) -> None:
        # MHQ-1100: None/empty resolves via convention at trace start.
        self._agent_name, self._identity_source = resolve_agent_identity(agent_name)
        self._incoming_traceparent = incoming_traceparent
        # Per-trace resource-attr override — wins over the process-level
        # ``_source``. Only honoured when in the accepted whitelist; an
        # invalid value falls through to whatever the process has set.
        self._source = source
        self._thread_id = thread_id
        self._user_id = user_id
        self._user_metadata = user_metadata
        self._record_keys = record_keys
        self._agent_version = agent_version
        self._success: bool | None = None
        self._outcome: str | None = None
        self._error: str | None = None
        self._state: _TraceState | None = None
        self._token = None
        self._root_span: SpanData | None = None
        self._start_mono: float = 0.0

    @property
    def trace_id(self) -> str | None:
        """Return the trace_id assigned when the context entered.

        ``None`` before ``__enter__`` or if init failed. Useful for callers
        that want to log the trace_id on their HTTP response so users can
        cross-reference the dashboard URL.
        """
        if self._state is not None:
            return self._state.trace_id
        return None

    def set_outcome(self, success: bool, outcome: str | None = None) -> None:
        """Signal success/failure and optional outcome label."""
        self._success = success
        self._outcome = outcome

    def set_input(self, data: Any) -> None:
        """Attach input data to the root agent span (e.g. the user prompt).

        `data` may be any JSON-serialisable value — a dict, a bare string, a
        list, a number. It is stored and returned verbatim, never coerced or
        wrapped, so what the dashboard renders is exactly what you passed.
        """
        if self._root_span is not None:
            self._root_span.input_data = data

    def set_output(self, data: Any) -> None:
        """Attach output data to the root agent span (e.g. the LLM answer).

        Same contract as :meth:`set_input` — any JSON-serialisable value,
        stored verbatim. A bare string is fine and common.
        """
        if self._root_span is not None:
            self._root_span.output_data = data

    def set_error(self, message: str) -> None:
        """Signal a failure with a custom error message."""
        self._error = message
        self._success = False

    def _apply_attribution(self, state: _TraceState) -> None:
        """Seed ``record_keys`` / ``agent_version`` from the run() arguments.

        Never raises: a malformed ``record_keys`` argument must not fail
        trace start (the MHQ-947 failure mode).
        """
        try:
            if self._record_keys:
                for key_type, key_value in dict(self._record_keys).items():
                    add_record_key(state.record_keys, key_type, key_value)
            state.agent_version = clean_agent_version(self._agent_version)
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: run() attribution swallowed error: %s", exc)
            _report(exc, "morse_ai.tracing.run.attribution")

    def __enter__(self) -> "TracedRun":
        # Lazy auto-init — see _ensure_sdk_initialized() for rationale.
        # Without this, processes that only use the tracing API (refagents,
        # any customer integration that adopts amund.run() before
        # amund.track()) never build a transport and traces silently
        # land in _last_trace instead of being POSTed.
        _ensure_sdk_initialized()
        _note_run_if_unsendable()
        # Attempt to continue an existing distributed trace via incoming traceparent.
        # Import here to avoid circular import (propagation imports from tracing).
        trace_id: str | None = None
        if self._incoming_traceparent:
            try:
                from amund.propagation import extract_traceparent

                result = extract_traceparent(
                    {"traceparent": self._incoming_traceparent}
                )
                if result is not None:
                    trace_id = result[0]
            except Exception as exc:  # noqa: BLE001
                logger.debug("amund: failed to parse incoming traceparent: %s", exc)
                _report(exc, "morse_ai.tracing.traceparent_parse")

        if trace_id is None:
            trace_id = uuid.uuid4().hex

        self._state = _TraceState(
            trace_id=trace_id,
            agent_name=self._agent_name,
            identity_source=self._identity_source,
        )
        # AGE-240: capture the framework active at trace start (adapters
        # generally activate later via __enter__ on the wrapped call, but
        # callers may pre-mark e.g. with a manual ``set_active_framework``
        # in tests). Re-read at emit time wins if a later adapter populates.
        from amund._internal.active_framework import get_active_framework

        self._state.framework = get_active_framework()
        # Validate against the same whitelist as the SDK init env-var path,
        # without importing the helper (avoids a hot-path import cycle).
        if self._source in {"copilot", "investigator", "refagent", "ui-bot"}:
            self._state.source_override = self._source
        self._state.thread_id = self._thread_id
        self._state.user_id = self._user_id
        if self._user_metadata:
            self._state.user_metadata = coerce_dims(
                self._user_metadata, "morse_ai.tracing.run.user_metadata"
            )
        self._apply_attribution(self._state)
        self._token = _active_trace.set(self._state)

        self._root_span = SpanData(
            trace_id=trace_id,
            type=SpanType.AGENT,
            name=self._agent_name,
        )
        self._start_mono = time.monotonic()
        self._state.span_stack.append(self._root_span)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        assert self._state is not None
        assert self._root_span is not None

        try:
            _remove_from_stack(self._state.span_stack, self._root_span)

            if exc_type is not None:
                error_msg = self._error or str(exc_val)
                _finalize_span(
                    self._root_span,
                    self._start_mono,
                    status="failed",
                    error_message=error_msg,
                    exc=exc_val,
                )
                # A crash is a fact, not a guess — failure metrics and alerts
                # filter on success=False. Only a clean exit stays unlabelled.
                if self._success is None and self._state.success is None:
                    self._success = False
            elif self._error is not None:
                _finalize_span(
                    self._root_span,
                    self._start_mono,
                    status="failed",
                    error_message=self._error,
                )
                # set_error() already flipped self._success to False.
            else:
                _finalize_span(self._root_span, self._start_mono)

            try:
                _active_trace.reset(self._token)
            except ValueError:
                # ContextVar token created in a different async context (e.g.
                # LangGraph runs callbacks in a different asyncio Task).
                # Fall back to setting the var to None in the current context.
                _active_trace.set(None)
            self._state.is_closed = True
            entry = _build_trace_dict(
                self._state,
                self._root_span,
                success=self._success
                if self._success is not None
                else self._state.success,
                outcome=self._outcome or self._state.outcome,
            )
            _finish_trace(entry)
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: run().__exit__ swallowed error: %s", exc)
            _report(exc, "morse_ai.tracing.run_exit")
        return False  # do not suppress exceptions


def run(
    agent_name: str | None,
    incoming_traceparent: str | None = None,
    source: str | None = None,
    thread_id: str | None = None,
    user_id: str | None = None,
    user_metadata: dict[str, str] | None = None,
    record_keys: dict[str, str | int] | None = None,
    agent_version: str | None = None,
) -> TracedRun:
    """Create a TracedRun context manager for explicit trace control.

    Args:
        agent_name: Name of the agent for this trace. ``None`` resolves by
            convention (env service name, entry script, ``default-agent``).
        incoming_traceparent: Optional W3C traceparent header value from an upstream
            service. When provided and valid, the trace will use the upstream trace_id
            so that all downstream spans appear in the same unified trace.
        source: Internal-use only. Per-trace override for the resource-level
            ``morse_ai.source`` tag. Amund's own API process uses this
            so co-resident emitters (copilot + investigator) tag traces with
            the right label even though they share an SDK init. Customer
            code should leave this unset. Accepted values:
            ``copilot|investigator|refagent|ui-bot``; anything else is
            silently ignored and the process-level source is used.
        thread_id: Application-layer conversation/thread identifier. Groups all
            traces from the same multi-turn session so the dashboard can show
            the full conversation history for a thread.
        user_id: Application-layer end-user identifier (e.g. a user primary key
            or email). Enables per-user filtering in the dashboard.
        user_metadata: Cost-attribution dimensions attached to this trace
            (e.g. ``customer_id``, ``feature``, ``env``, ``team``). All values
            are coerced to strings. Can also be set mid-trace via
            ``amund.set_context(**dims)``.
        record_keys: Business-record ids this trace handled, e.g.
            ``{"zendesk_ticket": 4812}``. Same rules as
            :func:`link_record` (invalid keys are dropped, max 20); more can be
            added mid-trace with ``amund.link_record(...)``. Use opaque ids,
            not personal data.
        agent_version: Your own label for the deployed agent version (max 128
            chars). Overrides the ``init(agent_version=)`` default for this
            trace.

    Example (multi-turn agent with thread attribution):
        with amund.run(
            "chat-agent",
            thread_id=session_id,
            user_id=current_user.id,
            user_metadata={"customer_id": tenant_id, "feature": "doc-qa"},
        ) as r:
            ...
    """
    return TracedRun(
        agent_name,
        incoming_traceparent=incoming_traceparent,
        source=source,
        thread_id=thread_id,
        user_id=user_id,
        user_metadata=user_metadata,
        record_keys=record_keys,
        agent_version=agent_version,
    )


# MHQ-947: `amund/__init__.py` states "Every public function in this module
# is wrapped with @absorb_errors" — and `set_context` was the one export where
# that was untrue, which is how a raising `__str__` reached host code. The
# per-key guard in `coerce_dims` runs INSIDE this, so a bad dimension still
# only costs that dimension; this outer net is for everything else.
@absorb_errors(fingerprint="morse_ai.set_context")
def set_context(**dims: str) -> None:
    """Attach cost-attribution dimensions to the active trace.

    Call once per request (e.g. in Django middleware) to propagate
    customer_id, feature, env, team, etc. to all child spans automatically.

        amund.set_context(customer_id=request.customer_id, feature="doc-qa")

    Values must be strings. Non-string values are coerced via str().
    No-op outside an active trace.
    """
    state = get_active_trace()
    if state is None:
        return
    if state.user_metadata is None:
        state.user_metadata = {}
    state.user_metadata.update(coerce_dims(dims, "morse_ai.tracing.set_context"))


def adopt_thread_id(value: Any) -> None:
    """Set ``thread_id`` on the active trace when nothing has claimed it yet.

    MHQ-1022: no adapter populated ``thread_id``, so every auto-opened trace
    landed with ``thread_id: null`` and a five-turn conversation showed up as
    five unrelated traces. Each framework carries conversation identity in its
    own place (an openai-agents ``Session``/run context, a LangGraph
    ``configurable.thread_id``), so extraction stays per-adapter — this is the
    one place that writes it, so precedence is decided once.

    An explicit ``run(thread_id=...)`` from app code always wins; adapters only
    fill the gap. Never raises: ``str()`` runs customer ``__str__`` code.
    """
    state = get_active_trace()
    if state is None or state.thread_id or not value:
        return
    try:
        state.thread_id = str(value)
    except Exception as exc:  # noqa: BLE001
        logger.debug("amund: adopt_thread_id swallowed error: %s", exc)
        _report(exc, "morse_ai.tracing.adopt_thread_id")


# ---------------------------------------------------------------------------
# span() context manager
# ---------------------------------------------------------------------------

# MHQ-806. Guards the "no active trace" warning so it fires at most once per
# process — see TracedSpan.__enter__ for why it warns at all, and why not more.
_warned_span_outside_trace = False


def _warn_span_outside_trace_once(span_name: str) -> None:
    global _warned_span_outside_trace
    if _warned_span_outside_trace:
        return
    _warned_span_outside_trace = True
    logger.warning(
        "amund: span(%r) was entered with no active trace, so nothing was "
        "recorded. Wrap the work in `with amund.run('<agent-name>'):` — "
        "run() opens the trace that span() attaches to (and initialises the "
        "SDK if needed). This warning is logged once per process.",
        span_name,
    )


# MHQ-980. Separate latch from the one above: "you forgot run()" is a customer
# mistake and "a finalized trace is still in the ContextVar" is ours, so one
# firing must not mute the other.
_warned_span_on_closed_trace = False


def _warn_span_on_closed_trace_once(span_name: str) -> None:
    global _warned_span_on_closed_trace
    if _warned_span_on_closed_trace:
        return
    _warned_span_on_closed_trace = True
    logger.warning(
        "amund: span(%r) was entered while the context still held a trace that "
        "had already been finalized and sent, so nothing was recorded. This is "
        "a bug in amund, not in your code — please report it with the "
        "framework and SDK version. This warning is logged once per process.",
        span_name,
    )


def _reset_span_outside_trace_warning() -> None:
    """Test hook — clears the once-per-process latches."""
    global _warned_span_outside_trace, _warned_span_on_closed_trace
    _warned_span_outside_trace = False
    _warned_span_on_closed_trace = False


class TracedSpan:
    """Context manager for creating a child span in the active trace.

    If there is no active trace context, __enter__/__exit__ are silent no-ops.

    Usage:
        with amund.span("tool-call", span_type="tool"):
            result = call_external_api()
    """

    def __init__(
        self,
        name: str,
        span_type: str = "custom",
        input_data: Any | None = None,
        parent_span_id: str | None = None,
    ) -> None:
        self._name = name
        # AID-05: adapters that know the real parent (a callback's parent run)
        # pass it; None falls back to the top of the shared stack.
        self._parent_span_id = parent_span_id
        self._span_type = span_type
        self._input_data = input_data
        self._span: SpanData | None = None
        self._state: _TraceState | None = None
        self._start_mono: float = 0.0
        self._active = False
        self._explicit_failed = False
        self._explicit_error: str | None = None

    def set_metadata(self, key: str, value: Any) -> None:
        """Attach a metadata key/value to the span (no-op outside a trace)."""
        if self._span is None:
            return
        if self._span.metadata is None:
            self._span.metadata = {}
        self._span.metadata[key] = value

    def set_output(self, data: Any) -> None:
        """Attach output data to the span (no-op outside a trace).

        `data` may be any JSON-serialisable value — dict, bare string, list,
        number — and is stored verbatim, never coerced or wrapped.
        """
        if self._span is not None:
            self._span.output_data = data

    def mark_failed(self, error_message: str | None = None) -> None:
        """Mark this span failed even when the body raised no exception (e.g. an
        HTTP call that returned a 5xx). A propagating exception still wins."""
        self._explicit_failed = True
        if error_message:
            self._explicit_error = error_message

    def __enter__(self) -> "TracedSpan":
        state = get_active_trace()
        if state is None and _active_trace.get() is not None:
            # MHQ-980: the var holds a trace that has already been emitted. The
            # span cannot be recorded — its trace is gone — but unlike "no trace
            # at all" this is never something the customer wrote. It always
            # means a leaked _TraceState, i.e. an SDK bug, and MHQ-672 went
            # unnoticed for weeks precisely because nothing said so at any log
            # level. Once per process: it sits inside agent loops.
            _warn_span_on_closed_trace_once(self._name)
        if state is None:
            # No active trace — no-op. Deliberately does NOT raise: library code
            # may be instrumented but called from an untraced path, and blowing
            # up there would be worse than emitting nothing.
            #
            # But it must not be *silent* (MHQ-806). Writing `amund.span(...)`
            # on its own is a natural first read of the API, and every method on
            # the returned handle then discards its argument — no error, no
            # spans, nothing to distinguish it from working instrumentation.
            #
            # Warn once per process, not per call: this sits inside request and
            # agent loops, so a per-call warning would bury the signal it exists
            # to provide.
            _warn_span_outside_trace_once(self._name)
            return self

        self._state = state
        self._active = True

        # Resolve parent: top of span_stack
        parent_span_id: str | None = self._parent_span_id
        if parent_span_id is None and state.span_stack:
            parent_span_id = state.span_stack[-1].span_id

        # Normalise span_type
        try:
            resolved_type = SpanType(self._span_type)
        except ValueError:
            resolved_type = SpanType.CUSTOM

        self._span = SpanData(
            trace_id=state.trace_id,
            type=resolved_type,
            name=self._name,
            parent_span_id=parent_span_id,
            input_data=self._input_data,
        )
        self._start_mono = time.monotonic()
        state.span_stack.append(self._span)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self._active or self._span is None or self._state is None:
            return False

        try:
            _remove_from_stack(self._state.span_stack, self._span)

            if exc_type is not None:
                _finalize_span(
                    self._span,
                    self._start_mono,
                    status="failed",
                    error_message=str(exc_val),
                    exc=exc_val,
                )
            elif self._explicit_failed:
                _finalize_span(
                    self._span,
                    self._start_mono,
                    status="failed",
                    error_message=self._explicit_error,
                )
            else:
                _finalize_span(self._span, self._start_mono)

            self._state.completed_spans.append(self._span)
        except Exception as exc:  # noqa: BLE001
            logger.debug("amund: span().__exit__ swallowed error: %s", exc)
            _report(exc, "morse_ai.tracing.span_exit")
        return False  # do not suppress exceptions


def span(
    name: str,
    span_type: str = "custom",
    input_data: Any | None = None,
    parent_span_id: str | None = None,
) -> TracedSpan:
    """Create a child span context manager under the current trace context.

    ``span_type`` is one of ``tool``, ``retriever``, ``db``, ``http``,
    ``queue``, ``memory``, ``guardrail``, ``handoff``, ``agent``, ``llm`` or
    ``custom`` (the default). ``input_data`` optionally attaches a payload to
    the span (the tool's arguments, a retriever's query, the parameterized SQL
    of a ``db`` span). The handle has ``set_output()``, ``set_metadata()`` and
    ``mark_failed()``. No-op, with a one-time warning, outside an active trace.

    A tool span replaces the one ``wrap()`` infers for the same tool call
    (MHQ-1928): it matches on the tool's name, or on ``tool_call_id`` metadata
    when the span has another name::

        with amund.span("search_docs", "tool", input_data=args) as s:
            s.set_output(search_docs(**args))
    """
    return TracedSpan(
        name, span_type, input_data=input_data, parent_span_id=parent_span_id
    )
