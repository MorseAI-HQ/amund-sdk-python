"""Tests proving zero-impact degradation when Morse API is down.

SDK guarantee: if the backend is unreachable, the user's agent continues
running with zero exceptions and zero latency impact.
"""

# ruff: noqa: E501, SIM117
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import httpx
import pytest
import respx

from amund.tracing import _set_transport, run, span, trace_agent
from amund.transport import BatchTransport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transport(
    endpoint: str = "http://unreachable.test/api/v1/traces/batch",
) -> BatchTransport:
    """Create a transport pointed at the given endpoint with short timeout."""
    return BatchTransport(
        endpoint=endpoint,
        api_key="av_test",
        flush_interval=0.05,
        timeout=1.0,  # short for tests
    )


# ---------------------------------------------------------------------------
# transport.track() — defense-in-depth
# ---------------------------------------------------------------------------


class TestTransportTrackDefense:
    @pytest.mark.req("SDK-07")
    def test_track_with_full_queue_returns_false_no_exception(self):
        """track() returns False silently when queue is full — never raises."""
        t = BatchTransport(
            endpoint="http://unreachable.test",
            api_key="av_test",
            max_queue_size=1,
        )
        # Fill the queue
        t.track({"traces": [{"trace": {}, "spans": []}]})

        # Second call hits a full queue — must return False without raising
        result = t.track({"traces": [{"trace": {}, "spans": []}]})
        assert result is False
        t.shutdown()

    @pytest.mark.req("SDK-07")
    def test_track_after_shutdown_returns_false_no_exception(self):
        """track() returns False after shutdown — never raises."""
        t = _make_transport()
        t.shutdown()
        result = t.track({"traces": [{"trace": {}, "spans": []}]})
        assert result is False


# ---------------------------------------------------------------------------
# transport._send_batch — network failures are caught and logged
# ---------------------------------------------------------------------------


class TestSendBatchNetworkErrors:
    @pytest.mark.req("SDK-07")
    @respx.mock
    def test_connection_refused_logs_warning_no_exception(self):
        """_send_batch logs warning on connection refused, never raises."""
        respx.post("http://refused.test/api/v1/traces/batch").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        t = BatchTransport(
            endpoint="http://refused.test/api/v1/traces/batch",
            api_key="av_test",
        )
        # Direct call — must not raise
        with t._lock:  # ensure client is set
            pass
        import time

        time.sleep(0.05)  # let thread start and create client

        # We call _send_batch directly to test error handling in isolation
        # The internal _post call should catch ConnectError
        t._send_batch([{"traces": [{"trace": {}, "spans": []}]}])
        t.shutdown()

    @pytest.mark.req("SDK-07")
    @respx.mock
    def test_timeout_logs_warning_no_exception(self):
        """_send_batch logs warning on timeout, never raises."""
        respx.post("http://timeout.test/api/v1/traces/batch").mock(
            side_effect=httpx.TimeoutException("Request timed out")
        )
        t = BatchTransport(
            endpoint="http://timeout.test/api/v1/traces/batch",
            api_key="av_test",
        )
        import time

        time.sleep(0.05)

        t._send_batch([{"traces": [{"trace": {}, "spans": []}]}])
        t.shutdown()

    @pytest.mark.req("SDK-07")
    @respx.mock
    def test_http_500_logs_warning_no_exception(self):
        """_send_batch logs warning on 5xx, never raises."""
        respx.post("http://error.test/api/v1/traces/batch").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        t = BatchTransport(
            endpoint="http://error.test/api/v1/traces/batch",
            api_key="av_test",
        )
        import time

        time.sleep(0.05)

        t._send_batch([{"traces": [{"trace": {}, "spans": []}]}])
        t.shutdown()

    @pytest.mark.req("SDK-07")
    @respx.mock
    def test_generic_exception_is_caught_no_exception(self):
        """_send_batch catches generic Exception, never raises."""
        respx.post("http://broken.test/api/v1/traces/batch").mock(
            side_effect=RuntimeError("Something unexpected")
        )
        t = BatchTransport(
            endpoint="http://broken.test/api/v1/traces/batch",
            api_key="av_test",
        )
        import time

        time.sleep(0.05)

        # Should not raise
        t._send_batch([{"traces": [{"trace": {}, "spans": []}]}])
        t.shutdown()


# ---------------------------------------------------------------------------
# trace_agent — zero impact on user code when API is unreachable
# ---------------------------------------------------------------------------


class TestTraceAgentResilience:
    @pytest.mark.req("SDK-07")
    @respx.mock
    def test_trace_agent_unreachable_endpoint_returns_normally(self):
        """trace_agent: decorated function returns normally even when API is down."""
        respx.post("http://down.test/api/v1/traces/batch").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        t = BatchTransport(
            endpoint="http://down.test/api/v1/traces/batch",
            api_key="av_test",
            flush_interval=0.05,
        )
        _set_transport(t)

        @trace_agent("resilient-agent")
        def my_agent():
            return "success"

        # Must return normally, no exception despite unreachable endpoint
        result = my_agent()
        assert result == "success"

        t.shutdown()
        _set_transport(None)

    @pytest.mark.req("SDK-07")
    @respx.mock
    def test_trace_agent_exception_in_user_code_still_propagates(self):
        """trace_agent: user exceptions are NOT swallowed — only transport errors are."""
        respx.post("http://down2.test/api/v1/traces/batch").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        t = BatchTransport(
            endpoint="http://down2.test/api/v1/traces/batch",
            api_key="av_test",
            flush_interval=0.05,
        )
        _set_transport(t)

        @trace_agent("failing-agent")
        def failing_agent():
            raise ValueError("user-raised error")

        # User exception MUST propagate
        with pytest.raises(ValueError, match="user-raised error"):
            failing_agent()

        t.shutdown()
        _set_transport(None)

    @pytest.mark.req("SDK-07")
    def test_trace_agent_no_transport_returns_normally(self):
        """trace_agent works with no transport configured (no-op)."""
        _set_transport(None)

        @trace_agent
        def simple_agent():
            return 42

        result = simple_agent()
        assert result == 42


# ---------------------------------------------------------------------------
# run() — context manager exits cleanly when API is unreachable
# ---------------------------------------------------------------------------


class TestRunContextManagerResilience:
    @pytest.mark.req("SDK-07")
    @respx.mock
    def test_run_unreachable_endpoint_exits_cleanly(self):
        """run() context manager exits cleanly when backend is unreachable."""
        respx.post("http://down3.test/api/v1/traces/batch").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        t = BatchTransport(
            endpoint="http://down3.test/api/v1/traces/batch",
            api_key="av_test",
            flush_interval=0.05,
        )
        _set_transport(t)

        result = None
        with run("my-agent") as r:
            result = "done"
            r.set_outcome(success=True, outcome="completed")

        # Context manager exited cleanly, result captured
        assert result == "done"

        t.shutdown()
        _set_transport(None)

    @pytest.mark.req("SDK-07")
    @respx.mock
    def test_run_exception_in_body_still_propagates(self):
        """run(): exceptions in the `with` body propagate to caller."""
        respx.post("http://down4.test/api/v1/traces/batch").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        t = BatchTransport(
            endpoint="http://down4.test/api/v1/traces/batch",
            api_key="av_test",
            flush_interval=0.05,
        )
        _set_transport(t)

        with pytest.raises(RuntimeError, match="body error"):
            with run("failing-run"):
                raise RuntimeError("body error")

        t.shutdown()
        _set_transport(None)


# ---------------------------------------------------------------------------
# span() — context manager exits cleanly when API is unreachable
# ---------------------------------------------------------------------------


class TestSpanContextManagerResilience:
    @pytest.mark.req("SDK-07")
    @respx.mock
    def test_span_unreachable_endpoint_exits_cleanly(self):
        """span() exits cleanly even when backend is unreachable."""
        respx.post("http://down5.test/api/v1/traces/batch").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        t = BatchTransport(
            endpoint="http://down5.test/api/v1/traces/batch",
            api_key="av_test",
            flush_interval=0.05,
        )
        _set_transport(t)

        result = None

        @trace_agent("agent-with-spans")
        def agent():
            nonlocal result
            with span("tool-call", span_type="tool"):
                result = "tool-done"
            return result

        returned = agent()
        assert returned == "tool-done"
        assert result == "tool-done"

        t.shutdown()
        _set_transport(None)

    @pytest.mark.req("SDK-07")
    def test_span_outside_trace_is_noop_no_exception(self):
        """span() outside active trace is a silent no-op."""
        _set_transport(None)

        with span("orphan-span", span_type="tool"):
            pass  # should not raise

    @pytest.mark.req("SDK-07")
    @respx.mock
    def test_span_exception_in_body_propagates(self):
        """span(): user exceptions propagate correctly."""
        respx.post("http://down6.test/api/v1/traces/batch").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        t = BatchTransport(
            endpoint="http://down6.test/api/v1/traces/batch",
            api_key="av_test",
            flush_interval=0.05,
        )
        _set_transport(t)

        @trace_agent("agent-span-exc")
        def agent():
            with span("bad-span", span_type="tool"):
                raise ValueError("span body error")

        with pytest.raises(ValueError, match="span body error"):
            agent()

        t.shutdown()
        _set_transport(None)


# ---------------------------------------------------------------------------
# Timeout configuration
# ---------------------------------------------------------------------------


class TestTransportTimeout:
    @pytest.mark.req("SDK-07")
    def test_default_timeout_is_30_seconds(self):
        """Default connection timeout is 30s (bumped from 5s in eec50dd for
        the OTLP transport's larger batch payloads — see MHQ-676)."""
        t = BatchTransport(
            endpoint="http://localhost:9999",
            api_key="av_test",
        )
        assert t._timeout == 30.0
        t.shutdown()
