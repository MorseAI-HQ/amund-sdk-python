"""W3C Trace Context propagation — inject and extract traceparent headers.

W3C traceparent format: {version}-{trace_id}-{span_id}-{flags}
  - version:  "00" (fixed)
  - trace_id: 32 lowercase hex chars (128-bit)
  - span_id:  16 lowercase hex chars (64-bit)
  - flags:    "01" (sampled) / "00" (not sampled) — MHQ-668: reflects the
              REAL sampling decision (``MORSE_SAMPLE_RATE``,
              ``TraceIdRatioBased`` keyed on trace_id) rather than a
              hardcoded always-sampled constant.

Example: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01

References:
  https://www.w3.org/TR/trace-context/
"""

from __future__ import annotations

import logging

from opentelemetry.trace import get_current_span
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from morse_ai.tracing import get_active_trace

logger = logging.getLogger("morse_ai")

W3C_TRACEPARENT_HEADER = "traceparent"
W3C_VERSION = "00"
W3C_FLAGS_SAMPLED = "01"
W3C_FLAGS_NOT_SAMPLED = "00"

_TRACE_ID_LEN = 32  # 128-bit as hex
_SPAN_ID_LEN = 16  # 64-bit as hex

_propagator = TraceContextTextMapPropagator()


def _resolve_sampled_flag(trace_id_hex: str) -> str:
    """Return the W3C flags byte for ``trace_id_hex`` per the configured
    sampler (MHQ-668) — falls back to "always sampled" if the OTel engine
    hasn't been configured yet (e.g. no ``morse_ai.init()``/API key), which
    matches ``MORSE_SAMPLE_RATE``'s own default of 1.0.
    """
    try:
        from morse_ai.otel_provider import get_engine  # noqa: PLC0415

        sampled = get_engine().resolve_sampled(int(trace_id_hex, 16))
    except Exception as exc:  # noqa: BLE001
        logger.debug("morse_ai: sampled-flag resolution swallowed error: %s", exc)
        sampled = True
    return W3C_FLAGS_SAMPLED if sampled else W3C_FLAGS_NOT_SAMPLED


def inject_traceparent(headers: dict) -> dict:
    """Add W3C traceparent header from the current trace context.

    Format: 00-{trace_id}-{span_id}-{flags}
    - trace_id: 32 hex chars — taken from the active trace's trace_id (uuid4 hex)
    - span_id:  16 hex chars — first 16 chars of the current span's span_id
    - flags:    reflects the real per-trace sampling decision (MHQ-668)

    Mutates and returns the headers dict for convenience.
    No-op (returns headers unchanged) if no trace context is active.
    """
    state = get_active_trace()
    if state is None:
        return headers

    trace_id = state.trace_id  # 32-char uuid4 hex

    # Use the top span on the stack (current span) for the W3C span_id.
    # W3C span_id is 16 hex chars; our span_ids are 32-char uuid4 hex,
    # so we take the first 16 chars. (Real OTel span ids for this trace
    # don't exist yet at injection time — they're minted when the trace
    # closes and its spans are converted/exported; see otel_provider.py.)
    if state.span_stack:
        raw_span_id = state.span_stack[-1].span_id
    elif state.completed_spans:
        raw_span_id = state.completed_spans[-1].span_id
    else:
        # No spans yet — use first 16 chars of trace_id as a fallback
        raw_span_id = trace_id

    span_id = raw_span_id[:_SPAN_ID_LEN]
    flags = _resolve_sampled_flag(trace_id)

    headers[W3C_TRACEPARENT_HEADER] = f"{W3C_VERSION}-{trace_id}-{span_id}-{flags}"
    return headers


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
