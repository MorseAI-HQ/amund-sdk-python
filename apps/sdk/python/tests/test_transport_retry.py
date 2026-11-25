"""TQE-04 (MHQ-676) — BatchTransport retry/backoff + quota-state surfacing.

Contract under test (ingestion-hotpath-hardening proposal, SDK section):
3 attempts, jittered exponential backoff, 4xx never retried except 429,
Retry-After honored, quota state surfaced before a 429 batch is dropped.
"""

import logging
import time

import httpx
import pytest
import respx

import amund.transport as transport_mod
from amund.transport import BatchTransport, _parse_retry_after

ENDPOINT = "https://test.api/api/v1/traces/batch"

EVENT_LIMIT_BODY = {
    "error": "event_limit_exceeded",
    "message": "Monthly event limit of 5,000 spans reached for plan 'free'.",
    "current_count": 5000,
    "limit_count": 5000,
    "plan_id": "free",
    "period_start": "2026-07-01T00:00:00+00:00",
    "top_sources_by_volume": [],
    "recent_spike_at": None,
    "upgrade_url": "https://app.morsehq.dev/billing",
    "sample_rate_url": "https://app.morsehq.dev/billing/usage/breakdown",
}

RATE_LIMIT_BODY = {
    "error": {
        "code": "RATE_LIMIT_INGEST",
        "message": "Ingestion rate limit exceeded.",
        "details": {"limit": 10, "plan": "free"},
    }
}


@pytest.fixture
def mock_endpoint():
    with respx.mock(assert_all_called=False) as mock:
        yield mock


@pytest.fixture
def fast_backoff(monkeypatch):
    """Shrink retry delays so retry tests run in milliseconds."""
    monkeypatch.setattr(transport_mod, "_BACKOFF_BASE_SECONDS", 0.01)
    monkeypatch.setattr(transport_mod, "_BACKOFF_MAX_SECONDS", 0.05)


def make_transport(**kwargs) -> BatchTransport:
    kwargs.setdefault("endpoint", ENDPOINT)
    kwargs.setdefault("api_key", "av_test")
    kwargs.setdefault("flush_interval", 0.05)
    return BatchTransport(**kwargs)


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


@pytest.mark.req("TQE-04")
class TestRetryPolicy:
    def test_429_retried_three_attempts_then_surfaced(
        self, mock_endpoint, fast_backoff, caplog
    ):
        route = mock_endpoint.post(ENDPOINT).mock(
            return_value=httpx.Response(
                429, json=EVENT_LIMIT_BODY, headers={"Retry-After": "0"}
            )
        )
        caplog.set_level(logging.DEBUG, logger="amund")
        transport = make_transport()
        try:
            transport.track({"traces": [{"trace": {"agent": "a"}, "spans": []}]})
            assert wait_for(lambda: route.call_count >= 3)
            assert route.call_count == 3  # 1 initial + 2 retries, then dropped
            state = transport.quota_state()
            assert state is not None
            assert state["kind"] == "event_limit"
            warnings = [
                r
                for r in caplog.records
                if r.levelno == logging.WARNING and "quota exceeded" in r.message
            ]
            assert len(warnings) == 1  # surfaced, loudly, before dropping
        finally:
            transport.shutdown()

    def test_429_recovers_when_retry_succeeds(
        self, mock_endpoint, fast_backoff, caplog
    ):
        route = mock_endpoint.post(ENDPOINT).mock(
            side_effect=[
                httpx.Response(429, json=RATE_LIMIT_BODY, headers={"Retry-After": "0"}),
                httpx.Response(200, json={"accepted": 1}),
            ]
        )
        caplog.set_level(logging.DEBUG, logger="amund")
        transport = make_transport()
        try:
            transport.track({"traces": [{"trace": {"agent": "a"}, "spans": []}]})
            assert wait_for(lambda: route.call_count >= 2)
            assert route.call_count == 2
            # Recovered — no drop warning for this batch. Scoped to `trace`
            # because caplog is process-wide: since MHQ-1030 a failed OTLP
            # *span* export on some other component's background thread also
            # logs a drop, and that has nothing to say about this retry.
            assert not [
                r
                for r in caplog.records
                if r.levelno == logging.WARNING
                and "dropped" in r.message
                and "trace" in r.getMessage()
            ]
        finally:
            transport.shutdown()

    def test_400_is_never_retried(self, mock_endpoint, fast_backoff, caplog):
        route = mock_endpoint.post(ENDPOINT).mock(
            return_value=httpx.Response(400, text="bad payload: missing trace_id")
        )
        caplog.set_level(logging.DEBUG, logger="amund")
        transport = make_transport()
        try:
            transport.track({"traces": [{"trace": {"agent": "a"}, "spans": []}]})
            assert wait_for(lambda: route.call_count >= 1)
            # Give the flush loop a couple more cycles to prove no retry lands.
            time.sleep(0.3)
            assert route.call_count == 1
            warnings = [
                r
                for r in caplog.records
                if r.levelno == logging.WARNING and "status=400" in r.message
            ]
            assert len(warnings) == 1
            assert "bad payload" in warnings[0].message  # body excerpt included
        finally:
            transport.shutdown()

    def test_5xx_is_retried_then_gives_up(self, mock_endpoint, fast_backoff, caplog):
        route = mock_endpoint.post(ENDPOINT).mock(
            return_value=httpx.Response(503, text="upstream sad")
        )
        caplog.set_level(logging.DEBUG, logger="amund")
        transport = make_transport()
        try:
            transport.track({"traces": [{"trace": {"agent": "a"}, "spans": []}]})
            assert wait_for(lambda: route.call_count >= 3)
            assert route.call_count == 3
            assert [
                r
                for r in caplog.records
                if r.levelno == logging.WARNING and "status=503" in r.message
            ]
        finally:
            transport.shutdown()

    def test_5xx_recovers_on_second_attempt(self, mock_endpoint, fast_backoff):
        route = mock_endpoint.post(ENDPOINT).mock(
            side_effect=[
                httpx.Response(500, text="flake"),
                httpx.Response(200, json={"accepted": 1}),
            ]
        )
        transport = make_transport()
        try:
            transport.track({"traces": [{"trace": {"agent": "a"}, "spans": []}]})
            assert wait_for(lambda: route.call_count >= 2)
            assert route.call_count == 2
        finally:
            transport.shutdown()

    def test_network_error_is_retried(self, mock_endpoint, fast_backoff):
        route = mock_endpoint.post(ENDPOINT).mock(
            side_effect=[
                httpx.ConnectError("connection refused"),
                httpx.Response(200, json={"accepted": 1}),
            ]
        )
        transport = make_transport()
        try:
            transport.track({"traces": [{"trace": {"agent": "a"}, "spans": []}]})
            assert wait_for(lambda: route.call_count >= 2)
            assert route.call_count == 2
        finally:
            transport.shutdown()

    def test_shutdown_during_retry_does_not_hang(self, mock_endpoint):
        # Real (uncapped-by-fixture) Retry-After of 30s — shutdown must
        # interrupt the retry sleep, not wait it out.
        route = mock_endpoint.post(ENDPOINT).mock(
            return_value=httpx.Response(
                429, json=EVENT_LIMIT_BODY, headers={"Retry-After": "30"}
            )
        )
        transport = make_transport()
        transport.track({"traces": [{"trace": {"agent": "a"}, "spans": []}]})
        assert wait_for(lambda: route.call_count >= 1)
        start = time.monotonic()
        transport.shutdown()
        assert time.monotonic() - start < 5.0


@pytest.mark.req("TQE-04")
class TestQuotaState:
    def test_quota_state_none_before_any_429(self):
        transport = make_transport(flush_interval=60.0)
        try:
            assert transport.quota_state() is None
        finally:
            transport.shutdown()

    def test_event_limit_body_parsed(self, mock_endpoint, fast_backoff):
        mock_endpoint.post(ENDPOINT).mock(
            return_value=httpx.Response(
                429,
                json=EVENT_LIMIT_BODY,
                headers={
                    "Retry-After": "1",
                    "X-Morse-Quota-Limit": "5000",
                    "X-Morse-Quota-Remaining": "0",
                    "X-Morse-Quota-Reset": "1754006400",
                },
            )
        )
        transport = make_transport()
        try:
            transport.track({"traces": [{"trace": {"agent": "a"}, "spans": []}]})
            assert wait_for(lambda: transport.quota_state() is not None)
            state = transport.quota_state()
            assert state["kind"] == "event_limit"
            assert state["current_count"] == 5000
            assert state["limit_count"] == 5000
            assert state["plan_id"] == "free"
            assert state["period_start"] == "2026-07-01T00:00:00+00:00"
            assert state["upgrade_url"] == "https://app.morsehq.dev/billing"
            assert state["quota_limit"] == 5000
            assert state["quota_remaining"] == 0
            assert state["quota_reset"] == 1754006400
            assert state["retry_after_seconds"] == 1.0
        finally:
            transport.shutdown()

    def test_rate_limit_middleware_body_parsed(self, mock_endpoint, fast_backoff):
        mock_endpoint.post(ENDPOINT).mock(
            return_value=httpx.Response(
                429, json=RATE_LIMIT_BODY, headers={"Retry-After": "1"}
            )
        )
        transport = make_transport()
        try:
            transport.track({"traces": [{"trace": {"agent": "a"}, "spans": []}]})
            assert wait_for(lambda: transport.quota_state() is not None)
            state = transport.quota_state()
            assert state["kind"] == "rate_limit"
            assert state["limit_count"] == 10
            assert state["plan_id"] == "free"
        finally:
            transport.shutdown()

    def test_unknown_429_body_is_still_recorded(self, mock_endpoint, fast_backoff):
        mock_endpoint.post(ENDPOINT).mock(
            return_value=httpx.Response(429, text="Too Many Requests")
        )
        transport = make_transport()
        try:
            transport.track({"traces": [{"trace": {"agent": "a"}, "spans": []}]})
            assert wait_for(lambda: transport.quota_state() is not None)
            state = transport.quota_state()
            assert state["kind"] == "unknown"
            assert state["status"] == 429
        finally:
            transport.shutdown()

    def test_warns_once_per_billing_period_then_debug(
        self, mock_endpoint, fast_backoff, caplog
    ):
        route = mock_endpoint.post(ENDPOINT).mock(
            return_value=httpx.Response(
                429, json=EVENT_LIMIT_BODY, headers={"Retry-After": "0"}
            )
        )
        caplog.set_level(logging.DEBUG, logger="amund")
        transport = make_transport()
        try:
            transport.track({"traces": [{"trace": {"agent": "a"}, "spans": []}]})
            assert wait_for(lambda: route.call_count >= 3)  # first batch dropped
            transport.track({"traces": [{"trace": {"agent": "b"}, "spans": []}]})
            assert wait_for(lambda: route.call_count >= 6)  # second batch dropped
            quota_logs = [r for r in caplog.records if "quota exceeded" in r.message]
            warning_logs = [r for r in quota_logs if r.levelno == logging.WARNING]
            debug_logs = [r for r in quota_logs if r.levelno == logging.DEBUG]
            assert len(warning_logs) == 1  # once per billing period
            assert len(debug_logs) >= 1  # subsequent drops demoted to DEBUG
            assert "upgrade" in warning_logs[0].message
        finally:
            transport.shutdown()

    def test_quota_state_returns_copy(self, mock_endpoint, fast_backoff):
        mock_endpoint.post(ENDPOINT).mock(
            return_value=httpx.Response(429, json=EVENT_LIMIT_BODY)
        )
        transport = make_transport()
        try:
            transport.track({"traces": [{"trace": {"agent": "a"}, "spans": []}]})
            assert wait_for(lambda: transport.quota_state() is not None)
            first = transport.quota_state()
            first["kind"] = "mutated"
            assert transport.quota_state()["kind"] == "event_limit"
        finally:
            transport.shutdown()


@pytest.mark.req("TQE-04")
class TestModuleLevelQuotaState:
    def test_uninitialized_returns_none(self):
        import amund

        amund._reset_for_testing()
        try:
            assert amund.quota_state() is None
        finally:
            amund._reset_for_testing()

    def test_initialized_exposes_transport_state(
        self, mock_endpoint, fast_backoff, monkeypatch
    ):
        import amund

        monkeypatch.delenv("MORSE_DISABLED", raising=False)
        amund._reset_for_testing()
        try:
            amund.init(
                api_key="av_test",  # pragma: allowlist secret
                endpoint=ENDPOINT,
                flush_interval=0.05,
            )
            mock_endpoint.post(ENDPOINT).mock(
                return_value=httpx.Response(429, json=EVENT_LIMIT_BODY)
            )
            amund.track("agent-x", {"agent": "agent-x", "success": True})
            assert wait_for(lambda: amund.quota_state() is not None)
            assert amund.quota_state()["kind"] == "event_limit"
        finally:
            amund._reset_for_testing()


@pytest.mark.req("TQE-04")
class TestRetryAfterParsing:
    def test_seconds_form(self):
        assert _parse_retry_after("2") == 2.0
        assert _parse_retry_after("0") == 0.0

    def test_capped_at_max(self):
        assert _parse_retry_after("3600") == transport_mod._RETRY_AFTER_CAP_SECONDS

    def test_invalid_values_fall_back(self):
        assert _parse_retry_after(None) is None
        assert _parse_retry_after("") is None
        assert _parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None
        assert _parse_retry_after("-5") is None


class TestDeliveryReasonCodes:
    """MHQ-971 — the counter must name the failure a reader would act on.

    Until 2026-08-10 every non-429 4xx reported ``encode_failed``, so a revoked
    key and a misconfigured endpoint both arrived looking like a serialisation
    bug — the counter pointed away from the fault it existed to reveal. Nothing
    pinned the status -> reason mapping, which is why it survived review.
    """

    @staticmethod
    def _drops(monkeypatch) -> list[tuple[str, str, int]]:
        seen: list[tuple[str, str, int]] = []
        monkeypatch.setattr(
            transport_mod.diagnostics,
            "record_drop",
            lambda reason, category, quantity=1: seen.append(
                (reason, category, quantity)
            ),
        )
        return seen

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (401, "auth_rejected"),
            (403, "auth_rejected"),
            (400, "endpoint_rejected"),
            (404, "endpoint_rejected"),
            (422, "endpoint_rejected"),
        ],
    )
    def test_4xx_reason_names_the_actual_fault(
        self, mock_endpoint, fast_backoff, monkeypatch, status, expected
    ):
        seen = self._drops(monkeypatch)
        mock_endpoint.post(ENDPOINT).mock(
            return_value=httpx.Response(status, text="no")
        )
        transport = make_transport()
        try:
            transport.track({"traces": [{"trace": {"agent": "a"}, "spans": []}]})
            assert wait_for(lambda: len(seen) >= 1)
        finally:
            transport.shutdown()
        assert seen[0][0] == expected, f"{status} reported as {seen[0][0]!r}"
        # The regression this class exists to prevent.
        assert seen[0][0] != "encode_failed"

    def test_encode_failure_is_the_only_encode_failed(
        self, mock_endpoint, fast_backoff, monkeypatch
    ):
        """`encode_failed` must mean json.dumps raised — nothing else."""
        seen = self._drops(monkeypatch)
        mock_endpoint.post(ENDPOINT).mock(return_value=httpx.Response(200))
        transport = make_transport()

        class Unserialisable:
            def __repr__(self) -> str:  # pragma: no cover - defensive
                raise RuntimeError("boom")

        monkeypatch.setattr(
            transport_mod.json,
            "dumps",
            lambda *a, **kw: (_ for _ in ()).throw(TypeError("not serialisable")),
        )
        try:
            transport.track({"traces": [{"trace": {"agent": "a"}, "spans": []}]})
            assert wait_for(lambda: len(seen) >= 1)
        finally:
            transport.shutdown()
        assert seen[0][0] == "encode_failed"

    def test_5xx_exhaustion_still_reports_retry_exhausted(
        self, mock_endpoint, fast_backoff, monkeypatch
    ):
        """Python DOES retry, so `retry_exhausted` is honest here."""
        seen = self._drops(monkeypatch)
        mock_endpoint.post(ENDPOINT).mock(return_value=httpx.Response(500, text="nope"))
        transport = make_transport()
        try:
            transport.track({"traces": [{"trace": {"agent": "a"}, "spans": []}]})
            assert wait_for(lambda: len(seen) >= 1)
        finally:
            transport.shutdown()
        assert seen[0][0] == "retry_exhausted"
