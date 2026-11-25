"""Failed spans carry the exception type (failure-groups, ERRGRP-05).

Additive: ``exception_type`` is only present on the wire for spans that failed
with a real exception, and the server stores it as ``metadata.exception.type``.
"""

from __future__ import annotations

import time

from amund import tracing
from amund.spans import SpanData, SpanType
from amund.tracing import _active_trace, _finalize_span, exception_type_name


class CustomBoom(Exception):
    pass


def test_builtin_exception_uses_bare_name():
    assert exception_type_name(ValueError("x")) == "ValueError"
    assert exception_type_name(None) is None


def test_custom_exception_is_module_qualified():
    assert exception_type_name(CustomBoom("x")) == f"{__name__}.CustomBoom"


def test_finalize_span_sets_exception_type_only_when_given():
    span = SpanData(trace_id="t", type=SpanType.TOOL, name="search")
    _finalize_span(
        span, time.monotonic(), status="failed", error_message="boom", exc=KeyError("k")
    )
    assert span.to_dict()["exception_type"] == "KeyError"

    ok = SpanData(trace_id="t", type=SpanType.TOOL, name="search")
    _finalize_span(ok, time.monotonic())
    assert "exception_type" not in ok.to_dict()

    explicit = SpanData(trace_id="t", type=SpanType.TOOL, name="search")
    _finalize_span(
        explicit, time.monotonic(), status="failed", error_message="mark_failed"
    )
    assert "exception_type" not in explicit.to_dict()


def test_span_context_manager_records_exception_type():
    _active_trace.set(None)
    try:
        try:
            with tracing.run("ExcAgent"), tracing.span("lookup", "tool"):
                raise TimeoutError("Timeout after 30ms")
        except TimeoutError:
            pass
        entry = tracing._get_last_trace()
        spans = {s["name"]: s for s in entry["spans"]}
        assert spans["lookup"]["exception_type"] == "TimeoutError"
        assert spans["lookup"]["error_message"] == "Timeout after 30ms"
    finally:
        _active_trace.set(None)
