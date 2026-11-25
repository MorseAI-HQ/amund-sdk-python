"""Amund span id -> OTel/W3C 64-bit span id (MHQ-1880).

Amund mints 128-bit uuid4-hex ``span_id`` values; OTLP and W3C ``traceparent``
carry 64-bit ids. The mapping is deterministic so the id placed in an outgoing
``traceparent`` header (before the span exists) equals the id the OTLP export
later gives the same span: both sides call :func:`otel_span_id_for`.
"""

from __future__ import annotations

import hashlib

_HEX = frozenset("0123456789abcdef")


def otel_span_id_for(span_id: str) -> str:
    """Return the 16 lowercase-hex, non-zero OTel span id for an Amund span id."""
    head = span_id[:16].lower()
    if len(head) == 16 and set(head) <= _HEX and int(head, 16) != 0:
        return head
    digest = hashlib.sha256(span_id.encode("utf-8")).hexdigest()[:16]
    return digest if int(digest, 16) != 0 else "0000000000000001"
