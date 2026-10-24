"""ADR-019 F2.1 — in-process DB auto-capture (zero-config).

When an Morse trace is active, DB queries made through a supported
PEP-249 driver are recorded as ``db`` spans on that trace — no OpenTelemetry
collector, no exporter, no customer setup. This is the "differentiated,
minimal-effort" wedge: the customer already uses the Morse SDK to trace
their agent; their database calls just *appear* on the same timeline.

Default-on since 0.7.0 (INFRACAP-01): ``morse_ai.init()`` enables DB and HTTP
capture, and nothing is recorded unless a Morse trace is active. Opt out with
``MORSE_CAPTURE_DB=0``, ``MORSE_CAPTURE_HTTP=0`` or ``MORSE_CAPTURE_INFRA=0``.
Per trace, at most 200 db and 100 http spans are kept; the rest collapse into
one summary span (INFRACAP-04). Literal values written inline in SQL are
scrubbed (INFRACAP-05).

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

import contextvars
import functools
import logging
import os
import re
import sys
from datetime import UTC, datetime
from typing import Any

from morse_ai.spans import SpanData, SpanType
from morse_ai.tracing import get_active_trace, span

logger = logging.getLogger("morse_ai")

# Original connect() callables, keyed by "module.connect", so enable is
# idempotent and disable can restore (tests rely on this).
_originals: dict[str, Any] = {}


# Max length of the captured SQL statement stored on a db span. A giant
# IN (...) clause shouldn't bloat the span; truncate with a marker.
_MAX_STATEMENT_LEN = 2048

# INFRACAP-04 — spans kept per trace, per kind, before collapsing into one
# summary span. Module-level so tests can lower them.
_CAPS: dict[str, int] = {"db": 200, "http": 100, "queue": 100}
_KIND_TYPE = {"db": SpanType.DB, "http": SpanType.HTTP, "queue": SpanType.CUSTOM}

# True while a db span from this module is open, so a driver proxy nested
# inside Django's execute_wrapper (or the reverse) records the query once.
_in_db_span: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "morse_in_db_span", default=False
)


class _NoSpan:
    """Stand-in returned when a call is over the cap. Same surface the capture
    sites use, does nothing."""

    def __enter__(self) -> _NoSpan:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def set_metadata(self, key: str, value: Any) -> None:
        return None

    def mark_failed(self, error_message: str | None = None) -> None:
        return None


_NO_SPAN = _NoSpan()


def _over_cap(kind: str) -> bool:
    """Count one call of ``kind`` against the active trace. Returns True when it
    is past the cap and was folded into the trace's summary span."""
    state = get_active_trace()
    if state is None:
        return False
    cap = _CAPS[kind]
    counts = state.infra_counts
    n = counts.get(kind, 0)
    if n < cap:
        counts[kind] = n + 1
        return False
    summary = state.infra_summaries.get(kind)
    if summary is None:
        now = datetime.now(UTC).isoformat()
        summary = SpanData(
            trace_id=state.trace_id,
            type=_KIND_TYPE[kind],
            name=kind,
            parent_span_id=state.span_stack[-1].span_id if state.span_stack else None,
            ended_at=now,
            duration_ms=0,
            metadata={"infra.suppressed_count": 0, "infra.cap": cap},
        )
        state.infra_summaries[kind] = summary
        state.completed_spans.append(summary)
    meta = summary.metadata or {}
    meta["infra.suppressed_count"] = meta.get("infra.suppressed_count", 0) + 1
    summary.metadata = meta
    summary.name = (
        f"+{meta['infra.suppressed_count']} more {kind} calls (capped at {cap})"
    )
    return True


def _infra_span(kind: str, name: str, input_data: Any | None = None) -> Any:
    """A span of ``kind`` ("db" | "http" | "queue") under the active trace, or a
    no-op when over the per-trace cap. The kind is the wire span type: infra spans
    are non-billable and never set trace status (ADR-033 D1, MHQ-1404)."""
    if _over_cap(kind):
        return _NO_SPAN
    return span(name, span_type=kind, input_data=input_data)


# --- SQL literal scrubbing (INFRACAP-05) ----------------------------------
#
# One left-to-right pass so a quote inside a comment, or `--` inside a string,
# is read in the right context. Order of the alternatives matters.
_SQL_TOKEN = re.compile(
    r"""
    (?P<dollar>\$(?P<tag>[A-Za-z_]\w*|)\$.*?\$(?P=tag)\$)
  | (?P<string>(?<![A-Za-z0-9_])[EeNnXxBb]?'(?:[^'\\]|\\.|'')*')
  | (?P<lcomment>--[^\n]*)
  | (?P<bcomment>/\*.*?\*/)
  | (?P<hex>0[xX][0-9A-Fa-f]+)
  | (?P<num>(?<![\w.$:@#])\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?!\w))
    """,
    re.VERBOSE | re.DOTALL,
)


def _scrub_token(m: re.Match[str]) -> str:
    return "" if m.lastgroup in ("lcomment", "bcomment") else "?"


@functools.lru_cache(maxsize=512)
def _scrub_cached(text: str) -> str:
    out = _SQL_TOKEN.sub(_scrub_token, text)
    # A quote that survived is an unterminated string. Everything after it is
    # string content, so drop it rather than risk leaking it.
    cut = out.find("'")
    if cut != -1:
        out = out[:cut] + "?"
    return out


def scrub_sql(statement: str) -> str:
    """Replace literal values written inline in SQL (strings, numbers, hex,
    dollar-quoted bodies, comments) with ``?``. Placeholders are kept. Bound
    parameters are never seen here — they are not stored at all."""
    if len(statement) > 8 * _MAX_STATEMENT_LEN:
        statement = statement[: 8 * _MAX_STATEMENT_LEN]
    return _scrub_cached(statement)


def _sql_verb(statement: Any) -> str:
    """Best-effort operation name for the span (SELECT/INSERT/...). Never the
    full SQL — keeps cardinality + PII out of span names."""
    try:
        first = str(statement).strip().split(None, 1)[0].upper()
        return first or "QUERY"
    except Exception:
        return "QUERY"


def _safe_statement(statement: Any) -> str | None:
    """Return the scrubbed, parameterized SQL for the span's input_data.

    Bound values are passed separately as ``params`` and are never stored.
    Literals the caller formatted straight into the SQL are replaced with
    ``?`` by :func:`scrub_sql`. Truncated to a cap.
    """
    try:
        text = scrub_sql(str(statement).strip()).strip()
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


def _db_error_message(exc: BaseException) -> str:
    """Error text for a failed db span: the exception class plus a driver error
    code, never the message. Driver messages routinely carry row values
    (Postgres: ``DETAIL: Key (email)=(a@b.com) already exists``; MySQL:
    ``Duplicate entry 'a@b.com'``; ``invalid input syntax ... "a@b.com"``)."""
    parts = [type(exc).__name__]
    for attr in ("pgcode", "sqlite_errorname", "sqlstate", "errno"):
        value = getattr(exc, attr, None)
        if isinstance(value, (str, int)) and value != "":
            parts.append(str(value))
            break
    return " ".join(parts)


def _http_error_message(exc: BaseException) -> str:
    """Error text for a failed http span: the exception class only. ``requests``
    puts the full URL, path and query string into its connection errors."""
    return type(exc).__name__


class _SafeErrorSpan:
    """Wrap a span context manager so a propagating exception is recorded by
    class, not by message (see ``_db_error_message`` / ``_http_error_message``)."""

    __slots__ = ("_cm", "_fmt", "_sp")

    def __init__(self, cm: Any, fmt: Any) -> None:
        self._cm = cm
        self._fmt = fmt
        self._sp: Any = None

    def __enter__(self) -> Any:
        self._sp = self._cm.__enter__()
        return self._sp

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        suppressed = bool(self._cm.__exit__(exc_type, exc_val, exc_tb))
        if exc_val is not None:
            data = getattr(self._sp, "_span", None)
            if data is not None:
                try:
                    data.error_message = self._fmt(exc_val)
                except Exception:  # noqa: BLE001
                    data.error_message = type(exc_val).__name__
        return suppressed


class _GuardedDbSpan(_SafeErrorSpan):
    """db span that also marks "a db span is open" for the nested-capture guard."""

    __slots__ = ("_token",)

    def __init__(self, statement: Any) -> None:
        super().__init__(
            _infra_span("db", _sql_verb(statement), _db_input_data(statement)),
            _db_error_message,
        )
        self._token: contextvars.Token[bool] | None = None

    def __enter__(self) -> Any:
        res = super().__enter__()
        self._token = _in_db_span.set(True)
        return res

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self._token is not None:
            _in_db_span.reset(self._token)
        return super().__exit__(exc_type, exc_val, exc_tb)


def _http_span(name: str) -> Any:
    return _SafeErrorSpan(_infra_span("http", name), _http_error_message)


def _db_span(statement: Any) -> Any:
    """db span for ``statement``, or a no-op when nested inside another db span
    this module already opened (driver proxy under Django's execute_wrapper)."""
    if _in_db_span.get():
        return _NO_SPAN
    return _GuardedDbSpan(statement)


class _TracingCursor:
    """Proxy cursor that wraps execute/executemany in a ``db`` span when a
    trace is active, and delegates everything else to the real cursor."""

    __slots__ = ("_cur",)

    def __init__(self, real_cursor: Any) -> None:
        object.__setattr__(self, "_cur", real_cursor)

    def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
        if get_active_trace() is None:
            return self._cur.execute(statement, *args, **kwargs)
        with _db_span(statement):
            return self._cur.execute(statement, *args, **kwargs)

    def executemany(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
        if get_active_trace() is None:
            return self._cur.executemany(statement, *args, **kwargs)
        with _db_span(statement):
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
        with _db_span(statement):
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


@functools.lru_cache(maxsize=32)
def _traced_psycopg_cursor_class(base: type) -> type:
    """A subclass of psycopg 3's ``base`` cursor class that records a ``db`` span.

    MHQ-1357: psycopg 3 connections are returned as-is (so
    ``isinstance(conn, psycopg.Connection)`` holds -- LangGraph's
    ``PostgresSaver`` rejected the old proxy with "Invalid connection type")
    and capture goes through psycopg's own ``cursor_factory`` hook instead.
    ``Connection.execute`` builds its cursor from that factory too.
    """

    class _MorseTracedCursor(base):  # type: ignore[valid-type, misc]
        def execute(self, query: Any, *args: Any, **kwargs: Any) -> Any:
            if get_active_trace() is None:
                return super().execute(query, *args, **kwargs)
            with _db_span(query):
                return super().execute(query, *args, **kwargs)

        def executemany(self, query: Any, *args: Any, **kwargs: Any) -> Any:
            if get_active_trace() is None:
                return super().executemany(query, *args, **kwargs)
            with _db_span(query):
                return super().executemany(query, *args, **kwargs)

    _MorseTracedCursor.__name__ = base.__name__
    _MorseTracedCursor.__qualname__ = base.__qualname__
    _MorseTracedCursor._morse_traced = True  # type: ignore[attr-defined]
    return _MorseTracedCursor


def _trace_psycopg_connection(conn: Any) -> Any:
    """Point a real psycopg 3 connection's cursor_factory at a traced subclass."""
    try:
        factory = getattr(conn, "cursor_factory", None)
        if isinstance(factory, type) and not getattr(factory, "_morse_traced", False):
            conn.cursor_factory = _traced_psycopg_cursor_class(factory)
    except Exception as exc:  # noqa: BLE001 — never break the customer's connect
        logger.debug("morse_ai: psycopg cursor_factory wrap skipped: %s", exc)
    return conn


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

    if module_name == "psycopg":

        def _connect(*args: Any, **kwargs: Any) -> Any:
            return _trace_psycopg_connection(orig(*args, **kwargs))

        _originals[key] = (module, connect_attr, orig)
        setattr(module, connect_attr, _connect)
        return True

    def _connect(*args: Any, **kwargs: Any) -> _TracingConnection:  # type: ignore[misc]
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
    with _db_span(sql):
        return execute(sql, params, many, context)


def _attach_wrapper(connection) -> None:  # type: ignore[no-untyped-def]
    wrappers = getattr(connection, "execute_wrappers", None)
    if wrappers is not None and _av_execute_wrapper not in wrappers:
        # Django's temporary execute_wrapper contexts remove the last entry
        # on exit. Lazy SDK initialization can happen inside such a context
        # (e.g. pghistory's request middleware); appending would make it pop
        # our permanent hook and leave its own hook behind without state.
        wrappers.insert(0, _av_execute_wrapper)


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
    # Also attach to connections already open at install time. Settings may
    # not be configured yet (SDK init runs early); the signal covers that case.
    try:
        for conn in connections.all():
            _attach_wrapper(conn)
    except Exception:  # noqa: BLE001
        pass

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
    # Exact or subdomain match (eu.api.anthropic.com, *.openai.azure.com): try the
    # host and each parent domain. At most a handful of set lookups per call.
    while True:
        if host in _excluded_hosts:
            return True
        dot = host.find(".")
        if dot == -1:
            return False
        host = host[dot + 1 :]


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
        if get_active_trace() is None:
            return sync_send(self, request, *args, **kwargs)
        host = getattr(getattr(request, "url", None), "host", None)
        if _host_excluded(host):
            return sync_send(self, request, *args, **kwargs)
        with _http_span(_http_span_name(request.method, host)) as http_span:
            response = sync_send(self, request, *args, **kwargs)
            _record_http_response(http_span, response)
            return response

    async def asend(self, request, *args, **kwargs):  # type: ignore[no-untyped-def]
        if get_active_trace() is None:
            return await async_send(self, request, *args, **kwargs)
        host = getattr(getattr(request, "url", None), "host", None)
        if _host_excluded(host):
            return await async_send(self, request, *args, **kwargs)
        with _http_span(_http_span_name(request.method, host)) as http_span:
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
        if get_active_trace() is None:
            return orig_send(self, request, **kwargs)
        host = urlparse(getattr(request, "url", "") or "").hostname
        if _host_excluded(host):
            return orig_send(self, request, **kwargs)
        with _http_span(
            _http_span_name(getattr(request, "method", "GET"), host)
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


_FALSY = ("0", "false", "no", "off")
_TRUTHY = ("1", "true", "yes", "on")


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in _TRUTHY


def _flag(name: str) -> bool | None:
    """True/False for a recognised value, None for unset, empty or garbage."""
    raw = os.environ.get(name, "").strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    return None


def capture_enabled(specific: str) -> bool:
    """Default-on switch (INFRACAP-01/02). A specific flag (``MORSE_CAPTURE_DB``,
    ``MORSE_CAPTURE_HTTP``, ``MORSE_CAPTURE_QUEUE``) wins over
    ``MORSE_CAPTURE_INFRA``; with neither set, capture is on."""
    value = _flag(specific)
    if value is None:
        value = _flag("MORSE_CAPTURE_INFRA")
    return True if value is None else value


def maybe_enable_from_env() -> None:
    """Called during SDK init. Capture is ON unless switched off:

    - MORSE_CAPTURE_DB=0     → no DB capture
    - MORSE_CAPTURE_HTTP=0   → no HTTP capture
    - MORSE_CAPTURE_INFRA=0  → neither (a specific flag still wins)
    - MORSE_CAPTURE_HTTP_EXCLUDE_HOSTS=a.com,b.com → extra excluded hosts
    """
    if capture_enabled("MORSE_CAPTURE_DB"):
        enable_db_capture()
        if "django" in sys.modules:
            enable_django_db_capture()
    if capture_enabled("MORSE_CAPTURE_HTTP"):
        extra = os.environ.get("MORSE_CAPTURE_HTTP_EXCLUDE_HOSTS", "")
        enable_http_capture(extra.split(",") if extra else None)
