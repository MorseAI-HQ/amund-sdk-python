"""Morse log handler + structlog processor (Phase 12.5 Tier 1.1).

Closes the SDK side of Phase 12.5 Gap A for Python: customers can install one
handler (or one structlog processor) and have their existing logger
auto-correlate every log line with the active Morse trace. No more
per-line ``morse_ai.log()`` calls.

Why a new module instead of extending ``morse_ai.log()``:

- ``morse_ai.log()`` makes a synchronous HTTP POST per call. That's fine for
  an explicit "I want this one line correlated" gesture, but it is *not* fine
  as the default cost of every ``logger.info(...)``. The handler must reuse
  the SDK's existing batched ``BatchTransport`` (2s flush, 100-event batch,
  background thread) so installing the handler doesn't add blocking I/O to
  the customer's hot path.
- Stdlib ``logging`` and ``structlog`` are different ergonomic shapes — the
  handler subclasses ``logging.Handler``; the processor is a plain callable.
  Both compute the same correlated payload via ``_build_log_payload``.

Non-invasive guarantee:

- Internal exceptions from this module are absorbed at the boundary
  (``emit`` / ``structlog_processor``) and never propagate into the
  customer's logging call. This mirrors the pattern documented in
  ``openspec/specs/sdk/spec.md:318-360`` and used by every adapter under
  ``morse_ai.adapters.*``.

Public surface (re-exported from this module):

    from morse_ai.logging import MorseHandler, structlog_processor

Usage with stdlib ``logging``::

    import logging
    from morse_ai.logging import MorseHandler

    logging.basicConfig(handlers=[MorseHandler()], level=logging.INFO)

Usage with ``structlog``::

    import structlog
    from morse_ai.logging import structlog_processor

    structlog.configure(processors=[structlog_processor, ...])
"""

from __future__ import annotations

import contextlib

# Absolute-import the stdlib module under a non-clashing alias so this file's
# package name (``morse_ai.logging``) doesn't shadow it.
import logging as stdlib_logging
from datetime import UTC, datetime
from typing import Any

logger = stdlib_logging.getLogger("morse_ai")

# Map stdlib levelname → Morse severity. Stdlib's WARNING becomes WARN
# to match the LogRecord model's enum.
_LEVEL_TO_SEVERITY: dict[str, str] = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARN",
    "WARN": "WARN",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
    "FATAL": "CRITICAL",
}


def _normalize_severity(level: str | None) -> str:
    """Coerce any stdlib / structlog level string into the wire severity."""
    if not level:
        return "INFO"
    upper = str(level).upper()
    return _LEVEL_TO_SEVERITY.get(upper, upper)


def _current_trace_context() -> tuple[str | None, str | None]:
    """Read the active trace_id / span_id from the SDK context var.

    Mirrors the pattern in ``morse_ai.log()`` (``__init__.py:359-411``).
    Returns ``(None, None)`` when no trace is active — the handler still
    emits the record; the dashboard's Logs tab simply won't correlate it.
    """
    # Import lazily to avoid a circular at module load (tracing imports nothing
    # from us, but importing it eagerly here would force a dep on tracing.py
    # being importable at logging-module import time, which Python sometimes
    # does very early).
    try:
        from morse_ai.tracing import get_active_trace
    except Exception:  # noqa: BLE001
        return None, None

    try:
        state = get_active_trace()
    except Exception:  # noqa: BLE001
        return None, None
    if state is None:
        return None, None
    trace_id = state.trace_id
    span_id = state.span_stack[-1].span_id if state.span_stack else None
    return trace_id, span_id


def _build_log_payload(
    *,
    severity: str,
    body: str,
    service_name: str | None = None,
    attributes: dict[str, Any] | None = None,
    timestamp: str | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
) -> dict[str, Any]:
    """Build the wire-format log dict the ingest endpoint expects.

    Shape matches ``apps/api/apps/logs/controllers.py:81-93`` exactly.
    """
    return {
        "timestamp": timestamp or datetime.now(UTC).isoformat(),
        "severity": _normalize_severity(severity),
        "service_name": service_name or "morse_ai.sdk",
        "body": body,
        "attributes": attributes or {},
        "trace_id": trace_id,
        "span_id": span_id,
    }


def _enqueue_log(payload: dict[str, Any]) -> None:
    """Enqueue a built log payload via the active SDK transport.

    Wraps the payload in the ``{"logs": [...]}`` envelope so
    ``BatchTransport._send_batch`` routes it to ``/api/v1/logs/ingest``
    rather than the traces endpoint.

    Silently no-ops when the SDK isn't initialized (no transport yet).
    Internal exceptions are absorbed — never re-raised to the customer.
    """
    try:
        # Defer import to avoid a circular at module load.
        from morse_ai import _ensure_initialized, _get_transport

        if not _ensure_initialized():
            return
        transport = _get_transport()
        if transport is None:
            return
        transport.track({"logs": [payload]})
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except BaseException as exc:  # noqa: BLE001
        # Last-resort isolation: log to the SDK's stderr logger but never
        # propagate to the host application. Deliberately the ONE swallow site
        # NOT routed through `_internal.diagnostics` (SDKDIAG-07): this code
        # path runs inside the customer's logger, and `report()` logs locally
        # by design (SDKDIAG-10), so reporting from here re-enters the logging
        # stack. The reporter's re-entrancy fence would catch the recursion,
        # but relying on a guard inside a logging handler to stop a logging
        # loop is not a bet worth taking for one swallow site.
        with contextlib.suppress(Exception):
            logger.debug("morse_ai: log enqueue swallowed error: %s", exc)


class MorseHandler(stdlib_logging.Handler):
    """A ``logging.Handler`` that ships every record to Morse.

    Drop this into ``logging.basicConfig(handlers=[MorseHandler()])``
    or attach it to any specific logger. Every record emitted while an
    Morse trace is active (i.e. inside ``morse_ai.run()`` /
    ``@morse_ai.trace_agent`` / a ``with morse_ai.span(...)``) gets
    correlated with that trace via ``trace_id`` + ``span_id``.

    Records emitted outside a trace context still flow through, but with
    ``trace_id=None``. The dashboard's Logs tab will simply not match
    them to a trace (by design — fuzzy correlation is deferred per the
    Phase 12.5 plan §4 cross-cutting).

    Args:
        service_name: Logical service tag persisted on every record. Defaults
            to ``"morse_ai.sdk"`` to match ``morse_ai.log()``. Customers
            should set this to their own service name (e.g. ``"api-prod"``).
        level: Standard ``logging.Handler`` level argument. Default
            ``logging.NOTSET`` lets the parent logger's level decide.

    Non-invasive guarantee:
        ``emit()`` absorbs every exception — a malformed record, a
        transport hiccup, or a missing trace context never propagates
        back into the customer's ``logger.info(...)`` call.
    """

    def __init__(
        self,
        service_name: str | None = None,
        level: int = stdlib_logging.NOTSET,
    ) -> None:
        super().__init__(level=level)
        self._service_name = service_name

    def emit(self, record: stdlib_logging.LogRecord) -> None:
        """Convert a ``LogRecord`` into the Morse wire format + enqueue."""
        try:
            body = self.format(record)
            trace_id, span_id = _current_trace_context()
            # Carry over a couple of stdlib record fields as structured
            # attributes — useful for filtering in the Logs tab.
            attributes: dict[str, Any] = {
                "logger": record.name,
            }
            if record.funcName and record.funcName != "<module>":
                attributes["function"] = record.funcName
            if record.module:
                attributes["module"] = record.module
            payload = _build_log_payload(
                severity=record.levelname,
                body=body,
                service_name=self._service_name,
                attributes=attributes,
                timestamp=datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
                trace_id=trace_id,
                span_id=span_id,
            )
            _enqueue_log(payload)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except BaseException:  # noqa: BLE001
            # Non-invasive guarantee: customer's logger.info(...) must
            # return normally even if we failed to ship the record.
            # Mirror handleError's semantics but skip the stderr noise.
            with contextlib.suppress(Exception):
                self.handleError(record)


def structlog_processor(
    _logger: Any,
    _method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """A ``structlog`` processor that ships every event to Morse.

    Drop into ``structlog.configure(processors=[structlog_processor, ...])``.
    Pass-through semantics: the event_dict is returned unchanged (apart from
    a ``trace_id`` / ``span_id`` injection when an Morse trace is
    active — handy for downstream processors that want to render trace
    context in console output).

    Args:
        _logger: The structlog logger (unused — present for signature parity).
        _method_name: The level name structlog is dispatching on
            (``"info"``, ``"warning"``, ...). Used as the severity source.
        event_dict: The structured log payload.

    Returns:
        The event_dict, optionally enriched with ``trace_id`` / ``span_id``
        when a trace is active. NEVER replaces customer keys — only
        ``setdefault``-style insertion.

    Non-invasive guarantee:
        Any exception inside the processor is absorbed; the event_dict is
        always returned (potentially unmodified) so the structlog chain
        continues normally.
    """
    try:
        trace_id, span_id = _current_trace_context()
        # Extract event body — structlog convention is ``event`` key.
        body = str(event_dict.get("event", ""))
        # Non-id, non-event fields become structured attributes. Drop empty
        # values to keep the payload small.
        attributes: dict[str, Any] = {
            k: v
            for k, v in event_dict.items()
            if k not in {"event", "level", "logger", "timestamp"} and v is not None
        }
        timestamp = event_dict.get("timestamp")
        if isinstance(timestamp, datetime):
            timestamp_str = timestamp.astimezone(UTC).isoformat()
        elif isinstance(timestamp, str):
            timestamp_str = timestamp
        else:
            timestamp_str = None
        payload = _build_log_payload(
            severity=str(event_dict.get("level") or _method_name or "INFO"),
            body=body,
            service_name=str(event_dict.get("logger") or "") or None,
            attributes=attributes,
            timestamp=timestamp_str,
            trace_id=trace_id,
            span_id=span_id,
        )
        _enqueue_log(payload)
        # Side-effect-only by default — inject trace context into the event
        # dict so downstream console renderers can show it. Don't clobber
        # keys the customer set explicitly.
        if trace_id is not None:
            event_dict.setdefault("trace_id", trace_id)
        if span_id is not None:
            event_dict.setdefault("span_id", span_id)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except BaseException as exc:  # noqa: BLE001
        with contextlib.suppress(Exception):
            logger.debug("morse_ai: structlog_processor swallowed error: %s", exc)
    return event_dict


__all__ = ["MorseHandler", "structlog_processor"]
