"""Tests for BatchTransport."""

import time

import httpx
import pytest
import respx

from morse_ai.transport import BatchTransport


@pytest.fixture
def mock_endpoint():
    with respx.mock(assert_all_called=False) as mock:
        yield mock


class TestBatchTransport:
    def test_track_and_flush(self, mock_endpoint):
        route = mock_endpoint.post("https://test.api/v1/runs/batch").mock(
            return_value=httpx.Response(200, json={"accepted": 1, "rejected": 0})
        )
        transport = BatchTransport(
            endpoint="https://test.api/v1/runs/batch",
            api_key="av_test123",
            flush_interval=0.1,
        )
        try:
            transport.track({"agent": "test", "success": True})
            time.sleep(0.5)
            transport.flush()
            assert route.called
            request = route.calls.last.request
            assert b"test" in request.content
        finally:
            transport.shutdown()

    def test_sends_auth_header(self, mock_endpoint):
        route = mock_endpoint.post("https://test.api/v1/runs/batch").mock(
            return_value=httpx.Response(200)
        )
        transport = BatchTransport(
            endpoint="https://test.api/v1/runs/batch",
            api_key="av_mykey",
            flush_interval=0.1,
        )
        try:
            transport.track({"agent": "test"})
            time.sleep(0.5)
            transport.flush()
            assert route.called
            auth = route.calls.last.request.headers["authorization"]
            assert auth == "Bearer av_mykey"
        finally:
            transport.shutdown()

    def test_drops_on_full_queue(self):
        transport = BatchTransport(
            endpoint="https://test.api/v1/runs/batch",
            api_key="av_test",
            max_queue_size=1,
            flush_interval=60.0,  # don't flush during test
        )
        try:
            assert transport.track({"first": True}) is True
            assert transport.track({"second": True}) is False  # dropped
        finally:
            transport.shutdown()

    def test_closed_transport_rejects(self):
        transport = BatchTransport(
            endpoint="https://test.api/v1/runs/batch",
            api_key="av_test",
        )
        transport.shutdown()
        assert transport.track({"after_close": True}) is False

    def test_shutdown_is_idempotent(self):
        transport = BatchTransport(
            endpoint="https://test.api/v1/runs/batch",
            api_key="av_test",
        )
        transport.shutdown()
        transport.shutdown()  # should not raise

    def test_http_error_does_not_raise(self, mock_endpoint):
        mock_endpoint.post("https://test.api/v1/runs/batch").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        transport = BatchTransport(
            endpoint="https://test.api/v1/runs/batch",
            api_key="av_test",
            flush_interval=0.1,
        )
        try:
            transport.track({"agent": "test"})
            time.sleep(0.5)
            transport.flush()
            # No exception raised — graceful degradation
        finally:
            transport.shutdown()
