"""ADR-019 F2.1 — in-process DB auto-capture (zero-config).

When an Morse trace is active, DB queries made through a supported
PEP-249 driver are recorded as ``db`` spans on that trace — no OpenTelemetry
collector, no exporter, no customer setup. This is the "differentiated,
minimal-effort" wedge: the customer already uses the Morse SDK to trace
their agent; their database calls just *appear* on the same timeline.

Opt-in for now (``enable_db_capture()`` or ``MORSE_CAPTURE_DB=1``) while
the approach is proven; the intent is to make it default-on once it's been
validated safe across drivers.

Implementation note
-------------------
Driver cursor types (sqlite3.Cursor, psycopg2 cursor) are C-extension types
whose methods cannot be monkey-patched. We instead wrap the module-level
``connect()`` factory to return a thin proxy connection whose cursors emit a
span on ``execute``/``executemany``. The proxy delegates everything else to
the real driver object via ``__getattr__`` and explicitly forwards the
context-manager and iterator protocols so existing customer code keeps
working unchanged. When no trace is active the proxy is a transparent
pass-through (zero overhead beyond one attribute lookup).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from morse_ai.tracing import get_active_trace, span

logger = logging.getLogger("morse_ai")

# Original connect() callables, keyed by "module.connect", so enable is
# idempotent and disable can restore (tests rely on this).
_originals: dict[str, Any] = {}


# Max length of the captured SQL statement stored on a db span. A giant
# IN (...) clause shouldn't bloat the span; truncate with a marker.
_MAX_STATEMENT_LEN = 2048


def _sql_verb(statement: Any) -> str:
    """Best-effort operation name for the span (SELECT/INSERT/...). Never the
    full SQL — keeps cardinality + PII out of span names."""
    try:
        first = str(statement).strip().split(None, 1)[0].upper()
        return first or "QUERY"
    except Exception:
        return "QUERY"


def _safe_statement(statement: Any) -> str | None:
    """Return the PARAMETERIZED SQL for the span's input_data.

    In standard DB-API usage the ``statement`` carries only placeholders
    (``%s`` / ``?``); the bound values are passed separately as ``params`` and
    we deliberately never store those — so this is PII-safe by construction
    (an exception is code that string-formats values directly into the SQL,
    which is the caller's anti-pattern, not ours). Truncated to a cap.
    """
    try:
        text = str(statement).strip()
    except Exception:
        return None
    if not text:
        return None
    if len(text) > _MAX_STATEMENT_LEN:
        return text[:_MAX_STATEMENT_LEN] + " …[truncated]"
    return text


def _db_input_data(statement: Any) -> dict | None:
    """Span ``input_data`` payload for a db span. The backend SpanIn schema
    requires a dict (not a bare string), so wrap the statement under the
    OTel-style ``db.statement`` key."""
    stmt = _safe_statement(statement)
    return {"db.statement": stmt} if stmt else None


class _TracingCursor:
    """Proxy cursor that wraps execute/executemany in a ``db`` span when a
    trace is active, and delegates everything else to the real cursor."""

    __slots__ = ("_cur",)

    def __init__(self, real_cursor: Any) -> None:
        object.__setattr__(self, "_cur", real_cursor)

    def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
        if get_active_trace() is None:
            return self._cur.execute(statement, *args, **kwargs)
        with span(
            _sql_verb(statement), span_type="db", input_data=_db_input_data(statement)
        ):
            return self._cur.execute(statement, *args, **kwargs)

    def executemany(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
        if get_active_trace() is None:
            return self._cur.executemany(statement, *args, **kwargs)
        with span(
            _sql_verb(statement), span_type="db", input_data=_db_input_data(statement)
        ):
            return self._cur.executemany(statement, *args, **kwargs)

    # --- protocol + attribute delegation -------------------------------
    def __getattr__(self, name: str) -> Any:
        return getattr(self._cur, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._cur, name, value)

    def __iter__(self) -> Any:
        return iter(self._cur)

    def __enter__(self) -> _TracingCursor:
        self._cur.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._cur.__exit__(*exc)


class _TracingConnection:
    """Proxy connection whose ``cursor()`` returns a tracing cursor. Also wraps
    the optional ``Connection.execute`` shortcut some drivers expose."""

    __slots__ = ("_conn",)

    def __init__(self, real_conn: Any) -> None:
        object.__setattr__(self, "_conn", real_conn)

    def cursor(self, *args: Any, **kwargs: Any) -> _TracingCursor:
        return _TracingCursor(self._conn.cursor(*args, **kwargs))

    def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
        return self._shortcut("execute", statement, *args, **kwargs)

    def executemany(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
        return self._shortcut("executemany", statement, *args, **kwargs)

    def _shortcut(self, method: str, statement: Any, *args: Any, **kwargs: Any) -> Any:
        # sqlite3 (and some others) expose execute/executemany shortcuts on the
        # connection that create a cursor internally — wrap them too.
        real = getattr(self._conn, method, None)
        if real is None:
            raise AttributeError(method)
        if get_active_trace() is None:
            return real(statement, *args, **kwargs)
        with span(
            _sql_verb(statement), span_type="db", input_data=_db_input_data(statement)
        ):
            return real(statement, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._conn, name, value)

    def __enter__(self) -> Any:
        self._conn.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._conn.__exit__(*exc)


def _wrap_connect(module_name: str, connect_attr: str = "connect") -> bool:
    """Wrap ``<module>.connect`` to return a tracing connection. Returns True
    if patched (module importable + not already patched)."""
    key = f"{module_name}.{connect_attr}"
    if key in _originals:
        return True
    try:
        import importlib

        module = importlib.import_module(module_name)
    except Exception:
        return False  # driver not installed — fine
    orig = getattr(module, connect_attr, None)
    if orig is None:
        return False

    def _connect(*args: Any, **kwargs: Any) -> _TracingConnection:
        return _TracingConnection(orig(*args, **kwargs))

    _originals[key] = (module, connect_attr, orig)
    setattr(module, connect_attr, _connect)
    return True


# Drivers we attempt to wrap. sqlite3 is stdlib (always present); the rest are
# best-effort (skipped silently if not installed).
_DB_DRIVERS = ["sqlite3", "psycopg2", "psycopg"]


def enable_db_capture() -> list[str]:
    """Enable in-process DB span capture. Idempotent. Returns the list of
    drivers actually wrapped (importable ones)."""
    wrapped = [name for name in _DB_DRIVERS if _wrap_connect(name)]
    if wrapped:
        logger.info(
            "morse_ai: in-process DB capture enabled for %s", ", ".join(wrapped)
        )
    return wrapped


def disable_db_capture() -> None:
    """Restore original connect() factories. Primarily for tests."""
    for module, attr, orig in _originals.values():
        setattr(module, attr, orig)
    _originals.clear()


# ---------------------------------------------------------------------------
# Django DB capture — pooled/persistent-connection safe (the dogfood path)
# ---------------------------------------------------------------------------
#
# The connect()-factory wrap above only catches connections opened AFTER it's
# installed, so it misses Django's persistent/pooled connections (CONN_MAX_AGE)
# that are opened once at startup and reused. Django provides the official
# execute_wrapper hook for exactly this: a callable invoked for EVERY query on
# a connection regardless of when the connection was opened. We register it on
# every connection via the connection_created signal (and onto already-open
# connections), gated on the SDK's _active_trace ContextVar so it's a no-op
# outside an Morse trace and correlates correctly even through streaming
# (SSE) responses, since the ContextVar — not Django's request lifecycle —
# decides attribution.

_django_installed = False


def _av_execute_wrapper(execute, sql, params, many, context):  # type: ignore[no-untyped-def]
    if get_active_trace() is None:
        return execute(sql, params, many, context)
    with span(_sql_verb(sql), span_type="db", input_data=_db_input_data(sql)):
        return execute(sql, params, many, context)


def _attach_wrapper(connection) -> None:  # type: ignore[no-untyped-def]
    wrappers = getattr(connection, "execute_wrappers", None)
    if wrappers is not None and _av_execute_wrapper not in wrappers:
        wrappers.append(_av_execute_wrapper)


def enable_django_db_capture() -> bool:
    """Install a Django execute_wrapper that emits a ``db`` span for every
    query made while an Morse trace is active. Pooled-connection safe.

    Idempotent. Returns True if Django is importable and the hook is active.
    Intended to be called once at backend startup (Morse's own dogfood
    wires this in; Django customers call it from AppConfig.ready()).
    """
    global _django_installed
    if _django_installed:
        return True
    try:
        from django.db import connections
        from django.db.backends.signals import connection_created
    except Exception:
        return False

    def _on_connection_created(sender, connection, **kwargs):  # type: ignore[no-untyped-def]
        _attach_wrapper(connection)

    connection_created.connect(
        _on_connection_created, dispatch_uid="morse_ai_db_capture"
    )
    # Also attach to connections already open at install time.
    for conn in connections.all():
        _attach_wrapper(conn)

    _django_installed = True
    logger.info("morse_ai: Django DB capture enabled (execute_wrapper)")
    return True


def disable_django_db_capture() -> None:
    """Remove the Django execute_wrapper from all connections. For tests."""
    global _django_installed
    try:
        from django.db import connections
        from django.db.backends.signals import connection_created

        connection_created.disconnect(dispatch_uid="morse_ai_db_capture")
        for conn in connections.all():
            wrappers = getattr(conn, "execute_wrappers", None)
            if wrappers and _av_execute_wrapper in wrappers:
                wrappers.remove(_av_execute_wrapper)
    except Exception:
        pass
    _django_installed = False


# ---------------------------------------------------------------------------
# F2.2 — in-process HTTP auto-capture (httpx + requests)
# ---------------------------------------------------------------------------

# LLM provider + Morse hosts are excluded by default: the LLM calls are
# already recorded as `llm` spans by the adapters (recording them as http too
# would double-count), and the SDK's own telemetry POST shouldn't self-trace.
_DEFAULT_EXCLUDED_HOSTS = {
    "api.anthropic.com",
    "api.openai.com",
    "generativelanguage.googleapis.com",  # Gemini
    "api.cohere.ai",
    "api.mistral.ai",
    # Hugging Face Inference Providers / serverless router: these are reached via
    # the OpenAI-compatible adapter, so the call is already an `llm` span — also
    # recording it as http would double-count (and the http sibling would be an
    # unpriced `hf` span, dragging Cost Coverage). NOTE: *.endpoints.huggingface.cloud
    # (Dedicated Endpoints) is deliberately NOT excluded — those raw calls have no
    # adapter span, so capturing them gives honest `hf` (unpriced) visibility.
    "router.huggingface.co",
    "api-inference.huggingface.co",
}
_excluded_hosts: set[str] = set(_DEFAULT_EXCLUDED_HOSTS)
_http_originals: dict[str, Any] = {}


def _host_excluded(host: str | None) -> bool:
    if not host:
        return False
    host = host.lower()
    if host in _excluded_hosts:
        return True
    # Subdomain match so e.g. eu.api.anthropic.com / *.openai.azure.com are covered.
    return any(host == h or host.endswith("." + h) for h in _excluded_hosts)


def _http_span_name(method: str, host: str | None) -> str:
    # Method + host only — never the full path (keeps PII + cardinality out).
    return f"{method.upper()} {host}" if host else method.upper()


def _should_capture_http(host: str | None) -> bool:
    return get_active_trace() is not None and not _host_excluded(host)


def _is_http_error_status(code: int) -> bool:
    """A status code that signals the *dependency* failed (server error, request
    timeout, or rate-limit) — not an ordinary 4xx client error, which is the
    caller's bug, not a dependency outage."""
    return code >= 500 or code in (408, 429)


def _record_http_response(http_span, response) -> None:
    """Record the response status on the http span: always store
    `http.status_code` (for correlation), and mark the span failed for
    dependency-failure statuses so 5xx/429 aren't silently 'completed'.
    Best-effort — never disrupts the wrapped call."""
    try:
        code = getattr(response, "status_code", None)
        if not isinstance(code, int):
            return
        http_span.set_metadata("http.status_code", code)
        if _is_http_error_status(code):
            http_span.mark_failed(f"HTTP {code}")
    except Exception:  # noqa: BLE001 — capture must never break the request
        pass


def _wrap_httpx() -> bool:
    try:
        import httpx
    except Exception:
        return False
    if "httpx" in _http_originals:
        return True

    sync_send = httpx.Client.send
    async_send = httpx.AsyncClient.send

    def send(self, request, *args, **kwargs):  # type: ignore[no-untyped-def]
        host = getattr(getattr(request, "url", None), "host", None)
        if not _should_capture_http(host):
            return sync_send(self, request, *args, **kwargs)
        with span(_http_span_name(request.method, host), span_type="http") as http_span:
            response = sync_send(self, request, *args, **kwargs)
            _record_http_response(http_span, response)
            return response

    async def asend(self, request, *args, **kwargs):  # type: ignore[no-untyped-def]
        host = getattr(getattr(request, "url", None), "host", None)
        if not _should_capture_http(host):
            return await async_send(self, request, *args, **kwargs)
        with span(_http_span_name(request.method, host), span_type="http") as http_span:
            response = await async_send(self, request, *args, **kwargs)
            _record_http_response(http_span, response)
            return response

    _http_originals["httpx"] = (httpx, sync_send, async_send)
    httpx.Client.send = send
    httpx.AsyncClient.send = asend
    return True


def _wrap_requests() -> bool:
    try:
        import requests
    except Exception:
        return False
    if "requests" in _http_originals:
        return True
    from urllib.parse import urlparse

    orig_send = requests.sessions.Session.send

    def send(self, request, **kwargs):  # type: ignore[no-untyped-def]
        host = urlparse(getattr(request, "url", "") or "").hostname
        if not _should_capture_http(host):
            return orig_send(self, request, **kwargs)
        with span(
            _http_span_name(getattr(request, "method", "GET"), host), span_type="http"
        ) as http_span:
            response = orig_send(self, request, **kwargs)
            _record_http_response(http_span, response)
            return response

    _http_originals["requests"] = (requests.sessions.Session, orig_send)
    requests.sessions.Session.send = send
    return True


def _av_endpoint_host() -> str | None:
    """Host of the Morse ingest endpoint, so the SDK's own telemetry POST
    doesn't self-trace as an http span."""
    from urllib.parse import urlparse

    ep = os.environ.get("MORSE_ENDPOINT", "").strip()
    if not ep:
        return None
    try:
        return urlparse(ep).hostname
    except Exception:
        return None


def enable_http_capture(extra_excluded_hosts: list[str] | None = None) -> list[str]:
    """Enable in-process HTTP span capture (httpx + requests). Idempotent.

    LLM-provider hosts AND Morse's own ingest endpoint are excluded by
    default — the former to avoid double-counting the `llm` spans the adapters
    emit, the latter so the SDK's telemetry/log POST doesn't self-trace. Pass
    extra hosts to exclude your own high-volume internal endpoints.
    """
    av_host = _av_endpoint_host()
    if av_host:
        _excluded_hosts.add(av_host.lower())
    if extra_excluded_hosts:
        _excluded_hosts.update(
            h.strip().lower() for h in extra_excluded_hosts if h.strip()
        )
    wrapped = []
    if _wrap_httpx():
        wrapped.append("httpx")
    if _wrap_requests():
        wrapped.append("requests")
    if wrapped:
        logger.info(
            "morse_ai: in-process HTTP capture enabled for %s", ", ".join(wrapped)
        )
    return wrapped


def disable_http_capture() -> None:
    """Restore original send() methods + reset the excluded-host set. For tests."""
    httpx_entry = _http_originals.pop("httpx", None)
    if httpx_entry:
        httpx_mod, sync_send, async_send = httpx_entry
        httpx_mod.Client.send = sync_send
        httpx_mod.AsyncClient.send = async_send
    requests_entry = _http_originals.pop("requests", None)
    if requests_entry:
        session_cls, orig_send = requests_entry
        session_cls.send = orig_send
    _excluded_hosts.clear()
    _excluded_hosts.update(_DEFAULT_EXCLUDED_HOSTS)


def add_excluded_host(host: str) -> None:
    """Exclude an additional host from HTTP capture at runtime."""
    if host and host.strip():
        _excluded_hosts.add(host.strip().lower())


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


def maybe_enable_from_env() -> None:
    """Called during SDK init. Enables capture per env flags:

    - MORSE_CAPTURE_DB=1     → DB capture
    - MORSE_CAPTURE_HTTP=1   → HTTP capture
    - MORSE_CAPTURE_INFRA=1  → both (convenience)
    - MORSE_CAPTURE_HTTP_EXCLUDE_HOSTS=a.com,b.com → extra excluded hosts
    """
    both = _truthy(os.environ.get("MORSE_CAPTURE_INFRA", ""))
    if both or _truthy(os.environ.get("MORSE_CAPTURE_DB", "")):
        enable_db_capture()
    if both or _truthy(os.environ.get("MORSE_CAPTURE_HTTP", "")):
        extra = os.environ.get("MORSE_CAPTURE_HTTP_EXCLUDE_HOSTS", "")
        enable_http_capture(extra.split(",") if extra else None)
