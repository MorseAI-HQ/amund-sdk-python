"""MHQ-673 — a rate-limited batch must be delayed, never dropped.

`opentelemetry-exporter-otlp-proto-http` retries 408 and 5xx only. It does not
honor `Retry-After`, and **429 is not retried at all** — the batch is discarded
on its first response.

MHQ-660 then made the ingest limiter actually fire for Bearer-authenticated SDK
traffic, which it previously never did, so Morse created exactly the condition
under which its own client silently loses telemetry: an over-quota customer.
MHQ-660's acceptance criterion — "429 with Retry-After → backoff → successful
redelivery, no data loss" — was therefore unmet by the OTel rebuild.

The stub is a transport adapter on the exporter's session
(`tests/_otlp_stub.py`), so the stock exporter of the installed version and
Amund's `export` override both run for real.
"""

from __future__ import annotations

import time
from importlib.metadata import version

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext, TraceFlags

from amund import otel_provider
from amund._internal import diagnostics
from tests import _otlp_stub

# opentelemetry-exporter-otlp-proto-http 1.45 started retrying 429 itself.
_STOCK_RETRIES_429 = tuple(
    int(p) for p in version("opentelemetry-exporter-otlp-proto-http").split(".")[:2]
) >= (1, 45)


@pytest.fixture(autouse=True)
def _reset():
    diagnostics._reset_for_testing()
    yield
    diagnostics._reset_for_testing()


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    """Record what the exporter WOULD have waited instead of waiting.

    Patched on the class under test, not on `time.sleep`, so the assertions are
    about the delay the exporter chose — which is the contract — rather than
    about how it implements waiting.
    """
    waited: list[float] = []

    def _fake_wait(self, delay):
        waited.append(delay)
        return True  # not interrupted by shutdown

    monkeypatch.setattr(
        otel_provider._DropReportingOTLPSpanExporter, "_wait_before_retry", _fake_wait
    )
    return waited


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


def _exporter(responses):
    """Exporter whose HTTP attempts answer ``responses`` in order.

    Each entry is ``(status, retry_after)``. The last entry repeats once the
    list is exhausted, so a test can say "429 forever" with one element.
    ``exporter._calls`` records the attempts actually made.
    """
    exporter = otel_provider._DropReportingOTLPSpanExporter(
        endpoint="http://localhost:1/api/v1/otlp/v1/traces",
        headers={"Authorization": "Bearer test"},
        timeout=0.001,
    )
    calls: list[tuple[int, str | None]] = []
    queue = list(responses)

    def _next():
        status, retry_after = queue.pop(0) if len(queue) > 1 else queue[0]
        calls.append((status, retry_after))
        return status, ({} if retry_after is None else {"Retry-After": retry_after})

    adapter = _otlp_stub._ScriptedAdapter(_next)
    exporter._amund_session.mount("http://", adapter)
    exporter._calls = calls
    return exporter


# ---------------------------------------------------------------------------
# The acceptance criterion: 429 + Retry-After -> wait -> redeliver, no loss
# ---------------------------------------------------------------------------


def test_429_with_retry_after_is_redelivered_and_nothing_is_dropped(_no_real_sleeping):
    exporter = _exporter([(429, "1"), (200, None)])

    result = exporter.export([_one_span()])

    assert result is SpanExportResult.SUCCESS
    assert _no_real_sleeping == [1.0], "must wait the server's Retry-After"
    assert diagnostics._drain_counters() == [], "a delivered batch is not a drop"


def test_429_honors_the_servers_retry_after_over_its_own_backoff(_no_real_sleeping):
    exporter = _exporter([(429, "7"), (200, None)])

    exporter.export([_one_span()])

    assert _no_real_sleeping == [7.0]


def test_repeated_429s_keep_retrying_up_to_the_budget(_no_real_sleeping):
    exporter = _exporter([(429, "1"), (429, "1"), (429, "1"), (200, None)])

    result = exporter.export([_one_span()])

    assert result is SpanExportResult.SUCCESS
    assert len(_no_real_sleeping) == 3


def test_a_429_that_never_clears_is_eventually_counted_as_a_drop(_no_real_sleeping):
    """Bounded, not infinite: this blocks the BatchSpanProcessor's only worker,
    so waiting forever would lose spans upstream instead — a bigger loss."""
    exporter = _exporter([(429, "1")])

    result = exporter.export([_one_span()])

    assert result is SpanExportResult.FAILURE
    assert len(_no_real_sleeping) == otel_provider._RATE_LIMIT_MAX_RETRIES
    assert diagnostics._drain_counters() == [
        {"reason": diagnostics.REASON_QUOTA_REJECTED, "category": "span", "quantity": 1}
    ]


def test_missing_retry_after_falls_back_to_exponential_backoff(_no_real_sleeping):
    exporter = _exporter([(429, None)])

    exporter.export([_one_span()])

    assert _no_real_sleeping == [1.0, 2.0, 4.0]


def test_a_zero_retry_after_retries_rather_than_dropping(_no_real_sleeping):
    """``Retry-After: 0`` means "retry now", not "give up".

    The parser accepts 0 as a well-formed value, so it reached the loop as a
    zero delay and was indistinguishable from a spent budget — one attempt,
    batch dropped, which is precisely the pre-fix behaviour this file exists to
    forbid.
    """
    exporter = _exporter([(429, "0"), (200, None)])

    result = exporter.export([_one_span()])

    assert result is SpanExportResult.SUCCESS
    assert len(exporter._calls) == 2, "a 0 must not end the retry loop"
    # On otel 1.45+ the stock client may retry the 429 itself before its
    # (tiny) timeout runs out, so whether our loop waits at all is timing.
    # What must hold either way: if it waits, it backs off rather than
    # busy-looping on the 0.
    assert _no_real_sleeping in ([1.0], []), "falls back to backoff"
    if not _STOCK_RETRIES_429:
        assert _no_real_sleeping == [1.0]
    assert diagnostics._drain_counters() == []


def test_an_absurd_retry_after_is_not_obeyed(_no_real_sleeping):
    """A broken or hostile value is not an instruction to stall the exporter."""
    exporter = _exporter([(429, "99999"), (200, None)])

    exporter.export([_one_span()])

    assert _no_real_sleeping == [1.0], "out-of-range header ignored, backoff used"


def test_a_garbage_retry_after_falls_back_rather_than_raising(_no_real_sleeping):
    exporter = _exporter([(429, "Wed, 21 Oct 2026 07:28:00 GMT"), (200, None)])

    result = exporter.export([_one_span()])

    assert result is SpanExportResult.SUCCESS
    if _STOCK_RETRIES_429:
        # 1.45+ reads the HTTP-date itself (a past date means "now") and
        # retries inside the stock client, so our loop never has to wait.
        assert _no_real_sleeping in ([1.0], [])
    else:
        assert _no_real_sleeping == [1.0]


def test_a_shutdown_during_the_wait_stops_retrying(monkeypatch):
    """A process on its way out must not be held open by a rate-limit window."""
    monkeypatch.setattr(
        otel_provider._DropReportingOTLPSpanExporter,
        "_wait_before_retry",
        lambda self, delay: False,
    )
    exporter = _exporter([(429, "1")])

    result = exporter.export([_one_span()])

    assert result is SpanExportResult.FAILURE
    assert len(exporter._calls) == 1, "no further attempt after shutdown"


# ---------------------------------------------------------------------------
# Over-correction guards: only 429 gets this treatment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403, 404, 413])
def test_non_429_failures_are_not_retried(status, _no_real_sleeping):
    """A revoked key does not get better by waiting; retrying it would turn a
    clear, immediate signal into a slow one."""
    exporter = _exporter([(status, None)])

    result = exporter.export([_one_span()])

    assert result is SpanExportResult.FAILURE
    assert _no_real_sleeping == []


def test_a_successful_export_neither_waits_nor_counts(_no_real_sleeping):
    exporter = _exporter([(200, None)])

    assert exporter.export([_one_span()]) is SpanExportResult.SUCCESS
    assert _no_real_sleeping == []
    assert diagnostics._drain_counters() == []


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", 1.0),
        (" 30 ", 30.0),
        ("0", 0.0),
        (None, None),
        ("", None),
        ("soon", None),
        ("-5", None),
        ("999999", None),
    ],
)
def test_parse_retry_after(raw, expected):
    assert otel_provider._parse_retry_after(raw) == expected


# ---------------------------------------------------------------------------
# The ticket's stated acceptance, against a real socket
# ---------------------------------------------------------------------------


def test_integration_real_server_429_then_redelivery_loses_nothing(monkeypatch):
    """MHQ-673's acceptance criterion, end to end over HTTP.

    Everything above stubs the `requests` session, which cannot catch what
    happens between `requests` and the header: `Retry-After` arrives in a
    CaseInsensitiveDict from a real response, and a stub dict would pass a
    lookup that the real one changed. This is also the only test here that
    proves the exporter genuinely waits rather than merely computing a delay,
    so it uses the smallest honest value the server can send.
    """
    import http.server
    import threading as _threading

    # Undo the autouse no-sleep patch: proving the exporter actually waits is
    # this test's entire reason to exist, and a stubbed wait would let it pass
    # against an exporter that retried instantly.
    monkeypatch.undo()

    # The drop counter is process-wide, and this test waits a real second —
    # long enough for a BatchSpanProcessor leaked by an earlier test (pointed
    # at a dead port) to exhaust its retries and record a drop that is not
    # ours. export() runs synchronously on this thread, so count only drops
    # recorded here.
    me = _threading.current_thread()
    own_drops: list[tuple] = []
    real_record_drop = diagnostics.record_drop

    def _record_own_drop(*args, **kwargs):
        if _threading.current_thread() is me:
            own_drops.append(args)
            real_record_drop(*args, **kwargs)

    monkeypatch.setattr(diagnostics, "record_drop", _record_own_drop)

    attempts: list[str] = []
    barrier = {"served_429": False}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            attempts.append(self.path)
            if not barrier["served_429"]:
                barrier["served_429"] = True
                self.send_response(429)
                self.send_header("Retry-After", "1")
            else:
                self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = _threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        exporter = otel_provider._DropReportingOTLPSpanExporter(
            endpoint=f"http://127.0.0.1:{port}/api/v1/otlp/v1/traces",
            headers={"Authorization": "Bearer test"},
            timeout=5,
        )
        started = time.monotonic()
        result = exporter.export([_one_span()])
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
        server.server_close()

    assert result is SpanExportResult.SUCCESS
    assert len(attempts) == 2, "the batch was redelivered, not dropped"
    assert elapsed >= 1.0, "the server's Retry-After was actually waited out"
    assert own_drops == [], "zero spans lost"
