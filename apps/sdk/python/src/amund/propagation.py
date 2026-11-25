"""W3C Trace Context propagation — inject and extract traceparent headers.

W3C traceparent format: {version}-{trace_id}-{span_id}-{flags}
  - version:  "00" (fixed)
  - trace_id: 32 lowercase hex chars (128-bit)
  - span_id:  16 lowercase hex chars (64-bit)
  - flags:    "01" (sampled) / "00" (not sampled) — MHQ-668: reflects the
              REAL sampling decision (``AMUND_SAMPLE_RATE``,
              ``TraceIdRatioBased`` keyed on trace_id) rather than a
              hardcoded always-sampled constant.

Example: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01

References:
  https://www.w3.org/TR/trace-context/
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from opentelemetry.trace import get_current_span
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from amund._internal import env as _env
from amund._internal.span_ids import otel_span_id_for
from amund.spans import SpanType
from amund.tracing import get_active_trace

logger = logging.getLogger("amund")

W3C_TRACEPARENT_HEADER = "traceparent"
W3C_VERSION = "00"
W3C_FLAGS_SAMPLED = "01"
W3C_FLAGS_NOT_SAMPLED = "00"

_TRACE_ID_LEN = 32  # 128-bit as hex
_SPAN_ID_LEN = 16  # 64-bit as hex

_propagator = TraceContextTextMapPropagator()
_init_propagate: bool | None = None


def _resolve_sampled_flag(trace_id_hex: str) -> str:
    """Return the W3C flags byte for ``trace_id_hex`` per the configured
    sampler (MHQ-668) — falls back to "always sampled" if the OTel engine
    hasn't been configured yet (e.g. no ``amund.init()``/API key), which
    matches ``AMUND_SAMPLE_RATE``'s own default of 1.0.
    """
    try:
        from amund.otel_provider import get_engine  # noqa: PLC0415

        sampled = get_engine().resolve_sampled(int(trace_id_hex, 16))
    except Exception as exc:  # noqa: BLE001
        logger.debug("amund: sampled-flag resolution swallowed error: %s", exc)
        sampled = True
    return W3C_FLAGS_SAMPLED if sampled else W3C_FLAGS_NOT_SAMPLED


def _build_traceparent(state, raw_span_id: str) -> str:
    trace_id = state.trace_id  # 32-char uuid4 hex
    span_id = otel_span_id_for(raw_span_id)
    flags = _resolve_sampled_flag(trace_id)
    return f"{W3C_VERSION}-{trace_id}-{span_id}-{flags}"


def inject_traceparent(headers: dict) -> dict:
    """Add W3C traceparent header from the current trace context.

    Format: 00-{trace_id}-{span_id}-{flags}
    - trace_id: 32 hex chars — taken from the active trace's trace_id (uuid4 hex)
    - span_id:  16 hex chars — the OTel span id of the current span
      (``otel_span_id_for(span.span_id)``), the SAME id the OTLP export gives
      that span, so a receiver can parent its own span to it (MHQ-1880)
    - flags:    reflects the real per-trace sampling decision (MHQ-668)

    Mutates and returns the headers dict for convenience.
    No-op (returns headers unchanged) if no trace context is active.
    """
    state = get_active_trace()
    if state is None:
        return headers

    # The top span on the stack is the current span.
    if state.span_stack:
        raw_span_id = state.span_stack[-1].span_id
    elif state.completed_spans:
        raw_span_id = state.completed_spans[-1].span_id
    else:
        # No spans yet — use the trace_id as the (hashed-to-16-hex) fallback
        raw_span_id = state.trace_id

    headers[W3C_TRACEPARENT_HEADER] = _build_traceparent(state, raw_span_id)
    return headers


def propagation_enabled() -> bool:
    """False when ``AMUND_PROPAGATE_TRACE_CONTEXT`` is ``false``/``0`` (case
    insensitive) or ``init(propagate_trace_context=False)`` was called."""
    raw = _env.get("PROPAGATE_TRACE_CONTEXT").strip().lower()
    if raw in ("false", "0", "no", "off"):
        return False
    return _init_propagate is not False


def set_propagate_default(value: bool | None) -> None:
    """``init(propagate_trace_context=...)`` hook; ``None`` leaves env decide."""
    global _init_propagate  # noqa: PLW0603
    _init_propagate = value


def _has_traceparent(headers: Any) -> bool:
    try:
        return any(str(k).lower() == W3C_TRACEPARENT_HEADER for k in headers)
    except TypeError:
        return False


def prepare_llm_call(
    kwargs: dict[str, Any], *, own_span: bool, propagate: bool = True
) -> tuple[dict[str, Any], str | None]:
    """Add a ``traceparent`` to a model-client call's ``extra_headers``.

    ``own_span=True``: the SDK will record this call's ``llm`` span after the
    response, so mint its span id now. The header points at it and the id is
    returned so the caller builds the ``SpanData`` with it.
    ``own_span=False``: a framework already owns an open ``llm`` span; the
    header points at it (top of the span stack) and no id is returned. If the
    top span is not an ``llm`` span (e.g. Strands traces on its own provider)
    no header is added.

    Returns ``(request_kwargs, minted_span_id)``. The caller's kwargs and
    headers are never mutated; a header the caller set is never overwritten.
    Never raises: any failure returns the inputs unchanged.
    """
    try:
        if not propagate or not propagation_enabled():
            return kwargs, None
        state = get_active_trace()
        if state is None:
            return kwargs, None
        existing = kwargs.get("extra_headers")
        if existing is not None and not hasattr(existing, "items"):
            return kwargs, None
        if existing and _has_traceparent(existing):
            return kwargs, None
        if own_span:
            span_id = uuid.uuid4().hex
        else:
            top = state.span_stack[-1] if state.span_stack else None
            if top is None or top.type != SpanType.LLM:
                return kwargs, None
            span_id = top.span_id
        headers = dict(existing or {})
        headers[W3C_TRACEPARENT_HEADER] = _build_traceparent(state, span_id)
        return {**kwargs, "extra_headers": headers}, (span_id if own_span else None)
    except Exception as exc:  # noqa: BLE001 — never break the customer's call
        logger.debug("amund: traceparent injection swallowed error: %s", exc)
        return kwargs, None


def extract_traceparent(headers: dict) -> tuple[str, str] | None:
    """Parse a W3C traceparent header via OTel's own
    ``TraceContextTextMapPropagator`` (MHQ-668 — was a hand-rolled
    regex/split parser; now delegates validation to the OTel SDK, which
    also feeds the sampled flag into the extracted context so a
    continued trace inherits the upstream sampling decision per
    ``ParentBased`` composition).

    Returns (trace_id, parent_span_id) as lowercase hex strings, or None if
    the header is missing or malformed. Same return shape as before:
    - trace_id is 32 hex chars (our full uuid4 hex format)
    - parent_span_id is 16 hex chars
    """
    raw = headers.get(W3C_TRACEPARENT_HEADER)
    if not raw:
        return None

    context = _propagator.extract({W3C_TRACEPARENT_HEADER: raw})
    span_context = get_current_span(context).get_span_context()
    if not span_context.is_valid:
        return None

    trace_id = format(span_context.trace_id, "032x")
    span_id = format(span_context.span_id, "016x")
    return (trace_id, span_id)


def get_current_traceparent() -> str | None:
    """Return the current W3C traceparent header value, or None.

    Convenience function for use without needing a headers dict.
    Returns None if no trace context is active.
    """
    state = get_active_trace()
    if state is None:
        return None

    headers: dict = {}
    inject_traceparent(headers)
    return headers.get(W3C_TRACEPARENT_HEADER)
