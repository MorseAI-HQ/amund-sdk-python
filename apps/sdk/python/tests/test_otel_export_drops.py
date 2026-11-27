"""A failed OTLP export must be counted, not just logged (MHQ-1030).

The bug this pins: an ingest 429 made the OTLP exporter retry, give up, and
discard the batch, leaving one line in the host's stdout and nothing anywhere
else — no counter, no diagnostic, no dashboard signal. A rate-limited window
and a healthy one were indistinguishable from both sides.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext, TraceFlags

from amund import otel_provider
from amund._internal import diagnostics
from tests._otlp_stub import script_exporter


@pytest.fixture(autouse=True)
def _reset():
    diagnostics._reset_for_testing()
    yield
    diagnostics._reset_for_testing()


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    """Skip the MHQ-673 rate-limit backoff — this module is about drop REASONS.

    Without it the 429 case sits through 1+2+4s of real backoff before giving
    up, which is both dead suite time and a pollution window: `_counters` is
    process-global, so a background BatchSpanProcessor from another test can
    land its own drop in the map while this one waits, and `_drain_counters()`
    then returns two entries. The retry behaviour itself is covered in
    test_otel_429_retry.py.
    """
    monkeypatch.setattr(
        otel_provider._DropReportingOTLPSpanExporter,
        "_wait_before_retry",
        lambda self, delay: True,
    )


def _one_span() -> ReadableSpan:
    return ReadableSpan(
        name="test-span",
        context=SpanContext(
            trace_id=0x1234567890ABCDEF1234567890ABCDEF,
            span_id=0x1234567890ABCDEF,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        ),
        start_time=1,
        end_time=2,
    )


def _exporter(status: int | None):
    """An exporter whose every HTTP attempt answers ``status``.

    The stub is a transport adapter on the exporter's session (see
    ``tests/_otlp_stub.py``), below both the stock exporter and Amund's status
    recording, so this runs the real code on every supported otel version.

    ``timeout`` is deliberately tiny so the stock retry loop hits its deadline
    on the first backoff instead of sleeping through six attempts.
    """
    exporter = otel_provider._DropReportingOTLPSpanExporter(
        endpoint="http://localhost:1/api/v1/otlp/v1/traces",
        headers={"Authorization": "Bearer test"},
        timeout=0.001,
    )
    script_exporter(exporter, [None if status is None else (status, {})])
    return exporter


@pytest.mark.req("SDKDIAG-09")
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, diagnostics.REASON_QUOTA_REJECTED),
        (401, diagnostics.REASON_AUTH_REJECTED),
        (403, diagnostics.REASON_AUTH_REJECTED),
        (404, diagnostics.REASON_ENDPOINT_REJECTED),
        (413, diagnostics.REASON_ENDPOINT_REJECTED),
    ],
)
def test_failed_export_is_counted_with_the_reason_a_reader_can_act_on(status, expected):
    exporter = _exporter(status)

    result = exporter.export([_one_span()])

    assert result is SpanExportResult.FAILURE
    assert diagnostics._drain_counters() == [
        {"reason": expected, "category": "span", "quantity": 1}
    ]


@pytest.mark.req("SDKDIAG-09")
def test_network_fault_is_counted_too():
    """No status at all still costs the customer their spans."""
    exporter = _exporter(None)

    assert exporter.export([_one_span()]) is SpanExportResult.FAILURE
    assert diagnostics._drain_counters(), "a dropped batch must never be silent"


@pytest.mark.req("SDKDIAG-09")
def test_successful_export_counts_nothing():
    exporter = _exporter(200)

    assert exporter.export([_one_span()]) is SpanExportResult.SUCCESS
    assert diagnostics._drain_counters() == []


@pytest.mark.req("SDKDIAG-09")
def test_drop_is_warned_about_locally_with_a_count(caplog):
    """SDKDIAG-10: nothing we learn is private from the host."""
    exporter = _exporter(429)

    with caplog.at_level("WARNING", logger="amund._internal"):
        exporter.export([_one_span()])

    assert "dropped 1 span" in caplog.text
