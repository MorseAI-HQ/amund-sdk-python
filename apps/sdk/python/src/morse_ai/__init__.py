"""Morse — Unified observability for AI agents and infrastructure.

V2 public API (module-level singleton):

    import morse_ai

    # Three-step zero-setup integration:
    #   1. Generate API key in the Morse dashboard
    #   2. pip install morse_ai
    #   3. Set MORSE_API_KEY in your environment
    #
    # That's it — no init() call required. The SDK auto-initializes on
    # first use from the env var.

    # Submit a single run with structured fields:
    morse_ai.track(agent_id="lead-qualifier", run_data={...})

    # Or submit a high-level outcome:
    morse_ai.record(agent="lead-qualifier", success=True, cost=0.043)

    # Or use the tracing primitives:
    @morse_ai.trace_agent
    def my_agent():
        with morse_ai.span("tool-call", span_type="tool"):
            ...

    # Optional explicit init (overrides env-var values):
    morse_ai.init(api_key="...", endpoint="https://...", flush_interval=2.0)

The internal `_Client` class backs the singleton but is NOT part of the
public API. Customers should never import or instantiate it directly.

Non-invasive guarantee
----------------------
Every public function in this module is wrapped with `@absorb_errors`.
Internal SDK exceptions are caught at the function boundary and reported
out-of-band to Morse's own diagnostics endpoint. The customer's call
returns None and their code keeps running. Telemetry must never break the
host application.
"""

import logging
import threading
from typing import Any

from morse_ai._internal import auto_detect, env
from morse_ai._internal.errors import absorb_errors
from morse_ai._internal.redaction import RedactionConfig
from morse_ai._version import __version__
from morse_ai.client import _Client
from morse_ai.propagation import extract_traceparent, inject_traceparent
from morse_ai.record import RunRecord
from morse_ai.tracing import (
    _set_redaction_config,
    _set_source,
    run,
    set_context,
    set_outcome,
    span,
    trace_agent,
)

logger = logging.getLogger("morse_ai")

__all__ = [
    # V2 module-level public API
    "init",
    "track",
    "batch_track",
    "record",
    "log",
    "flush",
    "shutdown",
    "quota_state",
    # v2 tracing API
    "trace_agent",
    "run",
    "span",
    "set_context",
    "set_outcome",
    # W3C trace context propagation
    "inject_traceparent",
    "extract_traceparent",
    # Public types (the class itself is private — only the data shape is exported)
    "RunRecord",
]

# Module-level singleton — private. Customers must NOT touch this directly.
_client: _Client | None = None

# Init guard — protects against concurrent first-use auto-init across threads.
_init_lock = threading.Lock()

# One-time-per-process flags
_no_api_key_warning_emitted = False
_last_logged_endpoint: str | None = None


def _log_destination_url(client: _Client) -> None:
    """Emit the destination-URL transparency log line.

    Fires exactly once per (process, endpoint) combination. If `init()` is
    called again with a different endpoint, the new endpoint is logged.
    """
    global _last_logged_endpoint
    endpoint = client.endpoint
    if endpoint == _last_logged_endpoint:
        return
    logger.info(
        "Morse SDK initialized — sending to %s (sdk version %s)",
        endpoint,
        __version__,
    )
    _last_logged_endpoint = endpoint


def _build_client(
    *,
    api_key: str,
    endpoint: str | None,
    flush_interval: float | None,
    redaction_enabled: bool = True,
    redaction_disabled_patterns: list[str] | None = None,
    redaction_extra_patterns: list[str] | None = None,
    source: str | None = None,
) -> _Client:
    """Construct a fresh `_Client` from explicit args + env-var fallbacks.

    Precedence for ``source``: explicit ``source`` kwarg wins; if None, fall
    back to ``MORSE_INTERNAL_SOURCE``. Invalid env values are silently
    dropped (see ``env.get_internal_source``).
    """
    resolved_endpoint = endpoint if endpoint is not None else env.get_endpoint()
    resolved_flush = (
        flush_interval if flush_interval is not None else env.get_flush_interval()
    )
    if source is not None:
        resolved_source = source if env.is_valid_internal_source(source) else None
    else:
        resolved_source = env.get_internal_source()
    common_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "flush_interval": resolved_flush,
        "redaction_enabled": redaction_enabled,
        "redaction_disabled_patterns": redaction_disabled_patterns,
        "redaction_extra_patterns": redaction_extra_patterns,
        "source": resolved_source,
        "git_revision": env.get_git_revision(),
    }
    if resolved_endpoint is None:
        return _Client(**common_kwargs)
    return _Client(endpoint=resolved_endpoint, **common_kwargs)


def _ensure_initialized() -> bool:
    """Lazy auto-init from env vars. Returns True if `_client` is usable.

    Called at the top of every public submission function. If `init()` has
    been called explicitly, this is a no-op cache check. Otherwise, the
    function attempts to initialize from `MORSE_API_KEY`. If the env
    var is unset, a one-time-per-process warning is logged and the function
    returns False — caller should silently no-op.
    """
    global _client, _no_api_key_warning_emitted

    if env.is_disabled():
        return False
    if _client is not None:
        return True

    with _init_lock:
        # Double-check: another thread may have initialized while we waited.
        if _client is not None:
            return True
        api_key = env.get_api_key()
        if not api_key:
            if not _no_api_key_warning_emitted:
                msg = (
                    "morse_ai not initialized — set MORSE_API_KEY "
                    "or call morse_ai.init() (this warning prints once "
                    "per process; SDK calls are silent no-ops)"
                )
                logger.warning(msg)
                _no_api_key_warning_emitted = True
            return False
        # MHQ-1054: same registration as init(), and this is the path that
        # matters most — the zero-code-change story means most installs never
        # call init() at all, so hooking only there would have left the common
        # case exactly as broken as it was.
        from morse_ai._internal import diagnostics  # noqa: PLC0415

        diagnostics.register_exit_drain()
        _client = _build_client(api_key=api_key, endpoint=None, flush_interval=None)
        _log_destination_url(_client)
        auto_detect.install_adapters()
        _maybe_enable_infra_capture()
        return True


def init(
    api_key: str | None = None,
    *,
    endpoint: str | None = None,
    flush_interval: float = 2.0,
    redaction_enabled: bool = True,
    redaction_disabled_patterns: list[str] | None = None,
    redaction_extra_patterns: list[str] | None = None,
    source: str | None = None,
) -> None:
    """Initialize the Morse SDK singleton.

    Call this once at app startup. After init(), all module-level functions
    (`track`, `record`, `trace_agent`, `run`, `span`, `flush`) are wired up.

    `endpoint` is optional — when omitted, the SDK resolves in this order:
        1. The `MORSE_ENDPOINT` env var (if set).
           Must be the FULL ingest URL including path, e.g.:
           ``https://api.morsehq.dev/api/v1/traces/batch``
           NOT just the base URL — the SDK posts directly to this value.
        2. The default prod URL:
           ``https://api.morsehq.dev/api/v1/traces/batch``

    `MORSE_DISABLED=1` overrides everything: init() is a no-op.

    Calling init() a second time replaces the singleton — the previous
    client is shut down cleanly and a new one is constructed with the new
    args.

    Args:
        api_key: Your Morse API key (starts with "mhq_"). Optional —
            if omitted, the SDK reads ``MORSE_API_KEY`` from the
            environment. Raises ``ValueError`` if neither is provided.
        endpoint: Optional ingestion URL override. None = env or default.
        flush_interval: Seconds between batch flushes (default: 2.0)
        redaction_enabled: When True (default), the SDK redacts PII from span
            attributes and event payloads BEFORE they leave the process.
            Setting False disables ALL built-in patterns at the SDK; the
            server-side mandatory floor (api_key_prefixed, jwt, credit_card,
            ssn_us) still applies on ingest, so opting out shifts WHERE
            redaction happens — it does not remove it.
        redaction_disabled_patterns: Pattern keys to skip at the SDK layer
            (e.g. ``["email"]``). Mandatory patterns are not disable-able and
            raise ``ValueError``. Known keys: ``api_key_prefixed``, ``jwt``,
            ``credit_card``, ``email``, ``phone_us``, ``phone_e164``,
            ``ssn_us``, ``ip_address``, ``secret_assignment``.
        redaction_extra_patterns: Customer-supplied regex strings (max 10
            patterns, each ≤ 256 chars). Each must compile via ``re.compile``
            or ``ValueError`` is raised. Matches are replaced with
            ``[REDACTED:custom]``.
        source: Internal-use resource-level tag. Morse's own dogfood
            emitters (``copilot``, ``investigator``, ``refagent``, ``ui-bot``)
            set this so dashboards can keep synthetic baselines from
            polluting the customer signal. **Customers should leave this
            unset.** The kwarg wins over the ``MORSE_INTERNAL_SOURCE``
            env var; values outside the accepted set are silently dropped.
    """
    # Resolve api_key: explicit arg wins, then env var, then error.
    resolved_key = api_key or env.get_api_key()
    if not resolved_key:
        raise ValueError(
            "api_key must be provided or MORSE_API_KEY env var must be set"
        )

    # Validate redaction config BEFORE the absorb_errors-wrapped body runs.
    # Misconfiguration is a programming error at app startup, not a runtime
    # ingest hiccup — it must surface loudly. Other errors inside _init_impl
    # remain absorbed by the @absorb_errors decorator.
    RedactionConfig.build(
        enabled=redaction_enabled,
        disabled_patterns=redaction_disabled_patterns,
        extra_patterns=redaction_extra_patterns,
    )
    _init_impl(
        api_key=resolved_key,
        endpoint=endpoint,
        flush_interval=flush_interval,
        redaction_enabled=redaction_enabled,
        redaction_disabled_patterns=redaction_disabled_patterns,
        redaction_extra_patterns=redaction_extra_patterns,
        source=source,
    )


@absorb_errors(fingerprint="morse_ai.init")
def _init_impl(
    *,
    api_key: str,  # always resolved before this is called
    endpoint: str | None,
    flush_interval: float,
    redaction_enabled: bool,
    redaction_disabled_patterns: list[str] | None,
    redaction_extra_patterns: list[str] | None,
    source: str | None = None,
) -> None:
    """Inner init body — wrapped with absorb_errors so non-config failures
    (network, transport startup) don't crash the host app at boot."""
    global _client
    if env.is_disabled():
        logger.info("morse_ai: MORSE_DISABLED=1 — SDK is a no-op")
        return
    # MHQ-1054: register BEFORE anything that builds a transport or configures
    # the OTel engine, so atexit's LIFO order puts the diagnostics drain last —
    # after the exporters that record the drops it has to report.
    # Deferred import to avoid a circular at module load, matching shutdown().
    from morse_ai._internal import diagnostics  # noqa: PLC0415

    diagnostics.register_exit_drain()
    with _init_lock:
        if _client is not None:
            _client.shutdown()
        _client = _build_client(
            api_key=api_key,
            endpoint=endpoint,
            flush_interval=flush_interval,
            redaction_enabled=redaction_enabled,
            redaction_disabled_patterns=redaction_disabled_patterns,
            redaction_extra_patterns=redaction_extra_patterns,
            source=source,
        )
        _log_destination_url(_client)
        auto_detect.install_adapters()
        _maybe_enable_infra_capture()
        _flush_dependencies_async(_client)


def _flush_dependencies_async(client: _Client) -> None:
    """Fire-and-forget POST of the current import scan to /ingest/dependencies/.

    Runs in a daemon thread so it never blocks init(). Errors are swallowed —
    dependency tracking is best-effort and must never break the host app.
    """
    try:
        from morse_ai.instrumentation.imports import scan_imported_libraries

        deps = scan_imported_libraries()
        if not deps:
            return

        def _post() -> None:
            try:
                import httpx

                from morse_ai._internal import env as _env

                traces_url = getattr(client, "_endpoint", None) or ""
                if not traces_url:
                    return
                from urllib.parse import urlsplit, urlunsplit

                parts = urlsplit(traces_url)
                deps_url = urlunsplit(
                    (parts.scheme, parts.netloc, "/api/v1/ingest/dependencies", "", "")
                )
                api_key = getattr(client, "_api_key", None) or _env.get_api_key() or ""
                httpx.post(
                    deps_url,
                    json={"libraries": deps},
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=5.0,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("morse_ai: dependency flush swallowed error: %s", exc)

        t = threading.Thread(target=_post, daemon=True)
        t.start()

        # Also install a watcher so libraries imported after init() are captured.
        from morse_ai.instrumentation.imports import install_import_watcher

        def _on_late_import(library: str, version: str) -> None:
            def _post_one() -> None:
                try:
                    from urllib.parse import urlsplit, urlunsplit

                    import httpx

                    from morse_ai._internal import env as _env

                    traces_url = getattr(client, "_endpoint", None) or ""
                    if not traces_url:
                        return
                    parts = urlsplit(traces_url)
                    deps_url = urlunsplit(
                        (
                            parts.scheme,
                            parts.netloc,
                            "/api/v1/ingest/dependencies",
                            "",
                            "",
                        )
                    )
                    api_key = (
                        getattr(client, "_api_key", None) or _env.get_api_key() or ""
                    )
                    httpx.post(
                        deps_url,
                        json={"libraries": {library: version}},
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=5.0,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "morse_ai: late-import dependency flush swallowed: %s", exc
                    )

            threading.Thread(target=_post_one, daemon=True).start()

        install_import_watcher(_on_late_import)
    except Exception as exc:  # noqa: BLE001
        logger.debug("morse_ai: _flush_dependencies_async swallowed error: %s", exc)


def _maybe_enable_infra_capture() -> None:
    """ADR-019 F2.1 — opt-in in-process DB capture (MORSE_CAPTURE_DB=1)."""
    try:
        from morse_ai.infra_capture import maybe_enable_from_env

        maybe_enable_from_env()
    except Exception as exc:  # pragma: no cover — never let capture setup break init
        logger.debug("morse_ai: infra capture setup skipped: %s", exc)


@absorb_errors(fingerprint="morse_ai.track")
def track(agent_id: str, run_data: dict[str, Any]) -> None:
    """Submit a single agent run to the ingestion endpoint.

    Primary V2 submission method. Non-blocking, <0.1ms. The `run_data` dict's
    shape mirrors RunRecord fields (agent, success, cost_usd, model, tokens,
    duration_ms, etc.); see `morse_ai.RunRecord`.

    Silently no-ops if the SDK isn't initialized (and `MORSE_API_KEY`
    isn't set). This is the non-invasive guarantee — telemetry must never
    break the host application.
    """
    if not _ensure_initialized():
        return
    assert _client is not None  # appease type checker
    _client.track(agent_id=agent_id, run_data=run_data)


@absorb_errors(fingerprint="morse_ai.batch_track")
def batch_track(runs: list[dict[str, Any]]) -> None:
    """Submit multiple agent runs in bulk. Silently no-ops if uninitialized."""
    if not _ensure_initialized():
        return
    assert _client is not None
    _client.batch_track(runs)


@absorb_errors(fingerprint="morse_ai.record")
def record(
    *,
    agent: str,
    success: bool | None = None,
    outcome: str | None = None,
    cost: float | None = None,
    model: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    duration_ms: int | None = None,
    error_message: str | None = None,
    metadata: dict[str, Any] | None = None,
    status: str = "completed",
) -> None:
    """High-level convenience: record an agent outcome with named fields.

    Builds a RunRecord internally and submits via `track()`. Silently no-ops
    if uninitialized.
    """
    if not _ensure_initialized():
        return
    assert _client is not None
    _client.record(
        agent=agent,
        success=success,
        outcome=outcome,
        cost=cost,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        duration_ms=duration_ms,
        error_message=error_message,
        metadata=metadata,
        status=status,
    )


@absorb_errors(fingerprint="morse_ai.log")
def log(
    level: str,
    message: str,
    *,
    service_name: str | None = None,
    attributes: dict[str, Any] | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
) -> None:
    """Emit a log record correlated to the current trace context (AGE-240).

    Args:
        level: Severity — one of DEBUG / INFO / WARN / ERROR / CRITICAL.
            Case-insensitive; normalised server-side.
        message: The log body. PII is redacted by the same allow-list the
            traces path uses.
        service_name: Logical service tag (defaults to the active agent
            name, then ``"unknown"``).
        attributes: Free-form structured fields persisted on the record.
        trace_id / span_id: Override the auto-detected correlation IDs.
            Use only when you're emitting a log on behalf of a known
            external trace (rare).

    Behavior:
        - Auto-correlates to the current ``morse_ai.run()`` /
          ``morse_ai.span()`` context. The dashboard's trace detail
          ``Logs`` tab surfaces these against the trace.
        - Synchronous HTTP POST (a single ~50ms call). Wrapped in
          ``@absorb_errors`` so a failure never breaks the host.
        - Silent no-op when the SDK isn't initialized.
    """
    if not _ensure_initialized():
        return
    assert _client is not None
    # Auto-pull trace + span IDs from the active context.
    from morse_ai.tracing import get_active_trace

    if trace_id is None or span_id is None:
        state = get_active_trace()
        if state is not None:
            if trace_id is None:
                trace_id = state.trace_id
            if span_id is None and state.span_stack:
                span_id = state.span_stack[-1].span_id
    _client.log(
        level=level,
        message=message,
        service_name=service_name,
        attributes=attributes,
        trace_id=trace_id,
        span_id=span_id,
    )


@absorb_errors(fingerprint="morse_ai.flush")
def flush() -> None:
    """Block until all queued events are sent. Silently no-ops if uninitialized."""
    if _client is not None:
        _client.flush()


@absorb_errors(fingerprint="morse_ai.shutdown")
def shutdown() -> None:
    """Flush and stop the background transport. Call on app exit."""
    global _client
    if _client is not None:
        _client.shutdown()
        _client = None
    # MHQ-946: diagnostics keeps its OWN buffer and daemon thread, deliberately
    # separate from the transport so a diagnostic storm can never evict
    # customer telemetry. That thread is a daemon on a 5s interval, so a
    # short-lived process exits with the buffer unsent and nothing else drains
    # it. Drained AFTER the transport above, which can itself report an
    # internal error or record delivery-loss counters on the way down.
    #
    # Outside the `_client is not None` guard on purpose: a failed init leaves
    # `_client` None while diagnostics almost certainly has the report that
    # explains why.
    #
    # Deferred import to avoid a circular at module load, matching
    # `_reset_for_testing` below.
    from morse_ai._internal import diagnostics

    diagnostics.drain_for_shutdown()


@absorb_errors(fingerprint="morse_ai.quota_state")
def quota_state() -> dict[str, Any] | None:
    """Latest ingest quota / rate-limit state observed by the SDK, or None.

    TQE-04 (MHQ-676): when the Morse backend answers a telemetry batch with
    HTTP 429 — the monthly event limit (``event_limit_exceeded``) or the
    per-second ingest rate limit (``RATE_LIMIT_INGEST``) — the transport
    parses the response and stores it. This function exposes that state so
    host applications can alert on quota exhaustion instead of discovering
    it from missing telemetry.

    Returns None when the SDK is uninitialized or no 429 has been observed.
    The dict includes at least ``kind`` (``"event_limit"``, ``"rate_limit"``,
    or ``"unknown"``), ``status``, and ``received_at`` (epoch seconds);
    event-limit states add ``current_count`` / ``limit_count`` / ``plan_id``
    / ``period_start`` / ``upgrade_url``.
    """
    if _client is None:
        return None
    return _client.transport.quota_state()


def _get_transport():
    """Return the current client's transport. Used by tracing module."""
    if _client is not None:
        return _client.transport
    return None


def _reset_for_testing() -> None:
    """Reset the module-level singleton for tests.

    Private — for Morse's own test suite. Resets the singleton, the
    one-time warning flag, the destination-URL log dedup state, and the
    diagnostics reporter state. Calling this from customer code is
    unsupported.
    """
    global _client, _no_api_key_warning_emitted, _last_logged_endpoint
    if _client is not None:
        _client._reset_for_testing()  # noqa: SLF001 — test-only fast teardown
        _client = None
    _no_api_key_warning_emitted = False
    _last_logged_endpoint = None
    # Reset the tracing module's active redaction config so tests that don't
    # call init() observe pass-through behaviour.
    _set_redaction_config(None)
    _set_source(None)
    # Defer import to avoid a circular at module load.
    from morse_ai._internal import diagnostics

    diagnostics._reset_for_testing()
    auto_detect._reset_for_testing()
