"""Phase 12.5 Tier 1.1 — Python logging.Handler + structlog processor tests.

Verifies the SDK side of Gap A from the Phase 12.5 plan:

- ``MorseHandler`` captures stdlib logger calls inside an
  ``amund.run()`` context and enqueues a log payload with the active
  ``trace_id`` + ``span_id``.
- The handler still emits (with ``trace_id=None``) when no trace is active.
- Internal transport errors are absorbed — the customer's
  ``logger.info(...)`` call returns normally.
- ``structlog_processor`` enriches the event_dict with trace context only
  when an active trace exists; otherwise passes through unchanged.

Mocking strategy: replace the SDK's transport with a fake that records every
call to ``track()``. This avoids the live HTTP path while still exercising
the full handler → ``_enqueue_log`` → ``transport.track()`` chain.
"""

from __future__ import annotations

import logging as stdlib_logging
from typing import Any

import pytest

import amund
from amund.logging import MorseHandler, structlog_processor


class _FakeTransport:
    """Drop-in replacement for ``BatchTransport`` that records enqueues.

    Implements just enough of the ``BatchTransport`` surface that
    ``_Client`` / ``amund._reset_for_testing`` won't blow up:
    ``track`` (the one we actually assert on), ``flush``, ``shutdown``,
    and the ``redaction_config`` property.
    """

    def __init__(self, *, raise_on_track: bool = False) -> None:
        self.events: list[dict[str, Any]] = []
        self.raise_on_track = raise_on_track
        from amund._internal.redaction import DISABLED_CONFIG

        self.redaction_config = DISABLED_CONFIG

    def track(self, event: dict[str, Any]) -> bool:
        if self.raise_on_track:
            raise RuntimeError("simulated transport failure")
        self.events.append(event)
        return True

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


@pytest.fixture
def fake_transport(monkeypatch: pytest.MonkeyPatch) -> _FakeTransport:
    """Initialize the SDK and swap in a FakeTransport.

    We have to init() first (so ``_ensure_initialized`` returns True), then
    overwrite the singleton's transport reference. Going via the public init
    keeps the rest of the SDK state (redaction config, source tag) consistent.

    We also register the fake with the tracing module so ``amund.run()``
    routes its trace emission through the fake instead of the real
    ``BatchTransport`` (which would try to POST to ``test.invalid`` and log a
    noisy warning).
    """
    from amund.tracing import _set_transport

    amund._reset_for_testing()
    amund.init(
        api_key="av_phase125_test",  # pragma: allowlist secret
        endpoint="http://test.invalid/api/v1/traces/batch",
    )
    fake = _FakeTransport()
    # Replace the client's transport with the fake. ``_get_transport`` reads
    # ``_client.transport`` so we patch the underlying attribute.
    assert amund._client is not None
    monkeypatch.setattr(amund._client, "_transport", fake)
    # Also route trace emissions through the fake (tracing reads its own ref).
    _set_transport(fake)
    yield fake
    amund._reset_for_testing()


@pytest.fixture
def isolated_logger() -> stdlib_logging.Logger:
    """A logger with NO inherited handlers, so each test asserts deterministically."""
    log = stdlib_logging.getLogger(f"amund.test.phase125.{id(object())}")
    log.handlers.clear()
    log.propagate = False
    log.setLevel(stdlib_logging.DEBUG)
    return log


class TestMorseHandlerInsideTrace:
    """Handler under an active ``amund.run()`` context."""

    def test_captures_log_inside_trace_with_matching_trace_id(
        self,
        fake_transport: _FakeTransport,
        isolated_logger: stdlib_logging.Logger,
    ) -> None:
        handler = MorseHandler()
        isolated_logger.addHandler(handler)

        with amund.run("test-agent") as ctx:
            isolated_logger.info("hi from inside a trace")
            active_trace_id = ctx.trace_id

        # Exactly one log envelope landed on the fake transport.
        log_events = [e for e in fake_transport.events if "logs" in e]
        assert len(log_events) == 1
        records = log_events[0]["logs"]
        assert len(records) == 1
        record = records[0]
        assert record["body"] == "hi from inside a trace"
        assert record["severity"] == "INFO"
        assert record["trace_id"] == active_trace_id
        # span_id is the agent root span — non-null while inside the context.
        assert record["span_id"] is not None

    def test_severity_levelname_mapping(
        self,
        fake_transport: _FakeTransport,
        isolated_logger: stdlib_logging.Logger,
    ) -> None:
        """WARNING maps to WARN; ERROR / CRITICAL pass through unchanged."""
        isolated_logger.addHandler(MorseHandler())

        with amund.run("severity-agent"):
            isolated_logger.warning("careful")
            isolated_logger.error("oops")
            isolated_logger.critical("boom")

        log_events = [e for e in fake_transport.events if "logs" in e]
        severities = [r["severity"] for env in log_events for r in env["logs"]]
        assert severities == ["WARN", "ERROR", "CRITICAL"]

    def test_attributes_include_logger_and_module(
        self,
        fake_transport: _FakeTransport,
        isolated_logger: stdlib_logging.Logger,
    ) -> None:
        isolated_logger.addHandler(MorseHandler())
        with amund.run("attr-agent"):
            isolated_logger.info("with attrs")

        log_events = [e for e in fake_transport.events if "logs" in e]
        record = log_events[-1]["logs"][0]
        assert record["attributes"]["logger"] == isolated_logger.name
        # Module is captured by stdlib LogRecord — should be present.
        assert "module" in record["attributes"]

    def test_custom_service_name_kwarg(
        self,
        fake_transport: _FakeTransport,
        isolated_logger: stdlib_logging.Logger,
    ) -> None:
        isolated_logger.addHandler(MorseHandler(service_name="api-prod"))
        with amund.run("svc-agent"):
            isolated_logger.info("service-tagged")

        log_events = [e for e in fake_transport.events if "logs" in e]
        record = log_events[-1]["logs"][0]
        assert record["service_name"] == "api-prod"


class TestMorseHandlerOutsideTrace:
    """Handler with no active trace context."""

    def test_emits_with_null_trace_id_no_crash(
        self,
        fake_transport: _FakeTransport,
        isolated_logger: stdlib_logging.Logger,
    ) -> None:
        isolated_logger.addHandler(MorseHandler())
        isolated_logger.info("outside any trace")

        log_events = [e for e in fake_transport.events if "logs" in e]
        assert len(log_events) == 1
        record = log_events[0]["logs"][0]
        assert record["body"] == "outside any trace"
        # No active trace ⇒ trace_id is None. The dashboard will simply not
        # correlate this record; it MUST NOT be dropped.
        assert record["trace_id"] is None
        assert record["span_id"] is None


class TestMorseHandlerErrorAbsorption:
    """Non-invasive guarantee: transport errors never propagate to the caller."""

    def test_transport_exception_does_not_propagate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        isolated_logger: stdlib_logging.Logger,
    ) -> None:
        from amund.tracing import _set_transport

        amund._reset_for_testing()
        amund.init(
            api_key="av_abs_test",  # pragma: allowlist secret
            endpoint="http://test.invalid/api/v1/traces/batch",
        )
        # Install a transport that raises on every track() call.
        explosive = _FakeTransport(raise_on_track=True)
        assert amund._client is not None
        monkeypatch.setattr(amund._client, "_transport", explosive)
        _set_transport(explosive)

        handler = MorseHandler()
        isolated_logger.addHandler(handler)

        try:
            # The customer's logger.info() must return normally even though
            # _enqueue_log raises internally. We assert by not catching.
            with amund.run("explode-agent"):
                isolated_logger.info("this would crash a naive handler")
        finally:
            amund._reset_for_testing()

    def test_uninitialized_sdk_is_silent_noop(
        self,
        isolated_logger: stdlib_logging.Logger,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Handler installed before init() must not raise on log calls.

        Also disables the env var so ``_ensure_initialized`` returns False
        without auto-init kicking in.
        """
        amund._reset_for_testing()
        monkeypatch.delenv("MORSE_API_KEY", raising=False)
        monkeypatch.setenv("MORSE_DISABLED", "1")
        try:
            isolated_logger.addHandler(MorseHandler())
            # No init() called. logger.info() must return normally.
            isolated_logger.info("uninitialized — no-op")
        finally:
            amund._reset_for_testing()


class TestStructlogProcessor:
    """Processor enriches event_dict only when a trace is active."""

    def test_inserts_trace_id_when_active(self, fake_transport: _FakeTransport) -> None:
        with amund.run("structlog-agent") as ctx:
            event_dict = structlog_processor(
                None,
                "info",
                {"event": "hi from structlog", "level": "info"},
            )
            assert event_dict["trace_id"] == ctx.trace_id
            assert event_dict["span_id"] is not None

        # And one log was enqueued.
        log_events = [e for e in fake_transport.events if "logs" in e]
        assert len(log_events) == 1
        record = log_events[0]["logs"][0]
        assert record["body"] == "hi from structlog"
        assert record["severity"] == "INFO"

    def test_passthrough_outside_trace(self, fake_transport: _FakeTransport) -> None:
        event_dict = structlog_processor(
            None,
            "warning",
            {"event": "no trace", "level": "warning", "user": 42},
        )
        # No trace_id / span_id injection when there's no active trace.
        assert "trace_id" not in event_dict
        assert "span_id" not in event_dict
        # The event_dict itself is returned unchanged (apart from no injection).
        assert event_dict["event"] == "no trace"
        assert event_dict["user"] == 42

        # The log was still enqueued for ingest — just with null IDs.
        log_events = [e for e in fake_transport.events if "logs" in e]
        assert len(log_events) == 1
        record = log_events[0]["logs"][0]
        assert record["trace_id"] is None
        assert record["severity"] == "WARN"

    def test_does_not_clobber_existing_trace_id(
        self, fake_transport: _FakeTransport
    ) -> None:
        with amund.run("clobber-agent"):
            event_dict = structlog_processor(
                None,
                "info",
                {
                    "event": "with explicit ids",
                    "trace_id": "custom-trace",
                    "span_id": "custom-span",
                },
            )
            # Customer-supplied trace_id wins.
            assert event_dict["trace_id"] == "custom-trace"
            assert event_dict["span_id"] == "custom-span"

    def test_returns_event_dict_even_on_internal_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The processor must NEVER raise — structlog chain integrity."""
        from amund.tracing import _set_transport

        amund._reset_for_testing()
        amund.init(
            api_key="av_struct_err",  # pragma: allowlist secret
            endpoint="http://test.invalid/api/v1/traces/batch",
        )
        explosive = _FakeTransport(raise_on_track=True)
        assert amund._client is not None
        monkeypatch.setattr(amund._client, "_transport", explosive)
        _set_transport(explosive)
        try:
            result = structlog_processor(None, "info", {"event": "boom"})
            # Returns the original event_dict; does NOT raise.
            assert result["event"] == "boom"
        finally:
            amund._reset_for_testing()


class TestTransportLogRouting:
    """The Tier-1.1 transport extension routes ``{"logs": [...]}`` correctly."""

    def test_logs_envelope_partitioned_separately_from_traces(self) -> None:
        """End-to-end via the real BatchTransport's _send_batch routing."""
        import time

        import httpx
        import respx

        from amund.transport import BatchTransport

        traces_url = "https://test.api/v1/traces/batch"
        logs_url = "https://test.api/api/v1/logs/ingest"
        with respx.mock(assert_all_called=False) as mock:
            traces_route = mock.post(traces_url).mock(
                return_value=httpx.Response(200, json={"accepted": 0})
            )
            logs_route = mock.post(logs_url).mock(
                return_value=httpx.Response(200, json={"accepted": 1})
            )
            transport = BatchTransport(
                endpoint=traces_url,
                api_key="av_route_test",  # pragma: allowlist secret
                flush_interval=0.1,
            )
            try:
                # A logs envelope should NOT hit the traces endpoint.
                transport.track(
                    {
                        "logs": [
                            {
                                "timestamp": "2026-05-27T00:00:00Z",
                                "severity": "INFO",
                                "service_name": "test",
                                "body": "routed correctly",
                                "attributes": {},
                                "trace_id": None,
                                "span_id": None,
                            }
                        ]
                    }
                )
                time.sleep(0.5)
                transport.flush()
                assert logs_route.called
                assert not traces_route.called
            finally:
                transport.shutdown()
