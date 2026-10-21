"""Morse internal client — backs the module-level singleton.

This module defines `_Client`, the private class that holds transport state
behind the module-level `morse_ai.init()` / `morse_ai.track()` / etc.
public API. Customers should NOT import or instantiate this class directly —
the public surface is `import morse_ai; morse_ai.init(...)`.

The class still exists because the singleton IS an instance — but it's a
single-leading-underscore private symbol, not part of `__all__`, and its
constructor signature may change without notice between SDK versions.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from morse_ai._internal import env as _env
from morse_ai._internal.redaction import RedactionConfig
from morse_ai.spans import SpanData, SpanType
from morse_ai.tracing import (
    _send_trace,
    _set_agent_version,
    _set_capture_end_user,
    _set_git_revision,
    _set_redaction_config,
    _set_service_version,
    _set_source,
    _set_transport,
)
from morse_ai.transport import BatchTransport

logger = logging.getLogger("morse_ai")

_DEFAULT_ENDPOINT = "https://api.morsehq.dev/api/v1/traces/batch"
_OTLP_TRACES_PATH = "/api/v1/otlp/v1/traces"


def _derive_otlp_endpoint(traces_endpoint: str) -> str:
    """Derive the OTLP ingest URL from the legacy JSON traces batch URL.

    Keeps host/scheme; replaces the path, mirroring
    ``transport._derive_logs_endpoint``.
    """
    parts = urlsplit(traces_endpoint)
    return urlunsplit((parts.scheme, parts.netloc, _OTLP_TRACES_PATH, "", ""))


class _Client:
    """Private SDK client. Backs the `morse_ai.*` module-level singleton.

    Not part of the public API. Construct via `morse_ai.init(...)`, never directly.
    """

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = _DEFAULT_ENDPOINT,
        flush_interval: float = 2.0,
        max_batch_size: int = 100,
        max_queue_size: int = 10_000,
        redaction_enabled: bool = True,
        redaction_disabled_patterns: list[str] | None = None,
        redaction_extra_patterns: list[str] | None = None,
        source: str | None = None,
        git_revision: str | None = None,
        agent_version: str | None = None,
        service_version: str | None = None,
        capture_end_user: bool = False,
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._source = source
        self._git_revision = git_revision
        # Build + validate the redaction config eagerly so misconfigurations
        # surface at Client construction (loud failure) rather than later when
        # a span tries to flush. Mandatory patterns cannot be disabled; extra
        # patterns must compile; both raise ``ValueError``.
        self._redaction_config = RedactionConfig.build(
            enabled=redaction_enabled,
            disabled_patterns=redaction_disabled_patterns,
            extra_patterns=redaction_extra_patterns,
        )
        self._transport = BatchTransport(
            endpoint=endpoint,
            api_key=api_key,
            flush_interval=flush_interval,
            max_batch_size=max_batch_size,
            max_queue_size=max_queue_size,
            redaction_config=self._redaction_config,
        )
        # Register transport + redaction with tracing module so trace_agent/
        # run/span can use them.
        _set_transport(self._transport)
        _set_redaction_config(self._redaction_config)
        _set_source(self._source)
        _set_git_revision(self._git_revision)
        _set_agent_version(agent_version)
        _set_capture_end_user(capture_end_user)
        _set_service_version(service_version)

        # MHQ-668: run()/span()/trace_agent's real emission pipeline — real
        # OTel spans over OTLP protobuf+gzip. Configured unconditionally
        # (even under MORSE_LEGACY_TRANSPORT=1) so the engine is ready if
        # the flag flips at runtime; the flag only affects which pipeline
        # tracing._finish_trace() feeds.
        from morse_ai.otel_provider import (
            get_engine as _get_otel_engine,  # noqa: PLC0415
        )

        self._otel_engine = _get_otel_engine()
        self._otel_engine.configure(
            endpoint=_derive_otlp_endpoint(endpoint),
            api_key=api_key,
            sample_rate=_env.get_sample_rate(),
            schedule_delay_millis=flush_interval * 1000,
            max_queue_size=_env.get_max_queue_size(),
            max_export_batch_size=_env.get_max_batch_size(),
            max_field_bytes=_env.get_max_field_bytes(),
            # 10s (not the exporter's un-overridden 30s ceiling): a genuine
            # network blip still gets several retries with backoff (see
            # opentelemetry.exporter.otlp.proto.http._is_retryable — 408 and
            # 5xx are retried, exponential backoff with jitter, no
            # Retry-After honored in the installed version); a fully dead
            # endpoint fails within a bounded, real-shutdown-friendly window
            # instead of the library's full 30s ceiling.
            timeout=10.0,
            redaction_config=self._redaction_config,
        )
        logger.debug(
            "morse_ai: client initialized endpoint=%s source=%s git_revision=%s",
            endpoint,
            self._source,
            self._git_revision,
        )

    @property
    def source(self) -> str | None:
        """Resource-level ``morse_ai.source`` (None on customer integrations)."""
        return self._source

    @property
    def transport(self) -> BatchTransport:
        """The underlying transport, exposed for tracing module access."""
        return self._transport

    @property
    def endpoint(self) -> str:
        """The resolved ingestion endpoint URL."""
        return self._endpoint

    @property
    def redaction_config(self) -> RedactionConfig:
        """The active redaction config (kwargs resolved at construction)."""
        return self._redaction_config

    def track(self, agent_id: str, run_data: dict[str, Any]) -> None:
        """Submit a single agent run to the ingestion endpoint.

        Primary V2 submission method. Non-blocking, <0.1ms. The `run_data`
        dict's shape mirrors RunRecord fields (agent, success, cost_usd,
        model, tokens, duration_ms, etc.).

        Args:
            agent_id: Name of the agent (e.g., "lead-qualifier"). Used to
                set the `agent` field on the wire payload.
            run_data: Dict of run fields. The `agent` key is overridden with
                `agent_id` if present.
        """
        payload = dict(run_data)
        payload["agent"] = agent_id
        self._transport.track(payload)

    def batch_track(self, runs: list[dict[str, Any]]) -> None:
        """Submit multiple agent runs in bulk.

        Each run dict is queued individually; the transport batches them on
        its normal flush schedule (2s interval, 100-event max batch). Spec
        conformance note: the "single HTTP request" semantic is best-effort
        — runs queued together will typically flush together, but there is
        no guarantee against an interleaved flush from another thread.
        """
        for run in runs:
            self._transport.track(dict(run))

    def record(
        self,
        *,
        agent: str,
        success: bool | None = None,
        outcome: str | None = None,
        cost: float | None = None,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        duration_ms: int | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "completed",
    ) -> None:
        """High-level convenience: record an agent outcome with named fields.

        Builds a single-span v2 trace and submits via `_send_trace()`. Use
        this when you have flat outcome data; use the `trace_agent` decorator
        or `run()` context manager for multi-span traces.
        """
        # Build a minimal v2 trace + single agent span so the payload lands on
        # POST /traces/batch ({"traces":[...]}) rather than the defunct v1 runs path.
        trace_id = uuid.uuid4().hex
        now = datetime.now(UTC).isoformat()
        resolved_status = "failed" if error_message else status

        span = SpanData(trace_id=trace_id, type=SpanType.AGENT, name=agent)
        span.ended_at = now
        span.status = resolved_status
        span.duration_ms = duration_ms
        span.model = model
        span.input_tokens = input_tokens
        span.output_tokens = output_tokens
        span.total_tokens = total_tokens
        span.cost_usd = cost
        span.error_message = error_message
        span.metadata = metadata

        trace_payload: dict[str, Any] = {
            "trace_id": trace_id,
            "agent": agent,
            "status": resolved_status,
            "started_at": span.started_at,
            "ended_at": now,
        }
        if success is not None:
            trace_payload["success"] = success
        if outcome is not None:
            trace_payload["outcome"] = outcome
        if duration_ms is not None:
            trace_payload["duration_ms"] = duration_ms
        if error_message is not None:
            trace_payload["error_message"] = error_message

        _send_trace({"trace": trace_payload, "spans": [span.to_dict()]})

    def log(
        self,
        *,
        level: str,
        message: str,
        service_name: str | None = None,
        attributes: dict[str, Any] | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        """Emit a log record correlated to a trace (AGE-240).

        Synchronous HTTP POST to the logs ingest endpoint. The caller's
        wrapper (``morse_ai.log``) is decorated with ``@absorb_errors``
        so any failure here is swallowed — host never sees an exception.

        The endpoint URL is derived from ``self._endpoint`` (the traces
        batch URL) by replacing the path with ``/api/v1/logs/ingest``.
        """
        import datetime
        from urllib.parse import urlsplit, urlunsplit

        import httpx

        parts = urlsplit(self._endpoint)
        logs_endpoint = urlunsplit(
            (parts.scheme, parts.netloc, "/api/v1/logs/ingest", "", "")
        )
        payload = {
            "logs": [
                {
                    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                    "severity": (level or "INFO").upper(),
                    "service_name": service_name or "morse_ai.sdk",
                    "body": str(message),
                    "attributes": attributes or {},
                    "trace_id": trace_id,
                    "span_id": span_id,
                }
            ]
        }
        # PII redaction reuses the trace path's allow-list config.
        redacted = self._redaction_config.redact_value(payload)
        with httpx.Client(timeout=3.0) as client:
            client.post(
                logs_endpoint,
                json=redacted,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )

    def flush(self) -> None:
        """Block until all queued events are sent.

        Flushes BOTH pipelines: the legacy JSON queue (record()/log()/deps
        scan) and the OTel provider's ``force_flush()`` (run()/span()/
        trace_agent) — the latter BLOCKS until in-flight OTLP exports
        complete (or its timeout elapses), fixing the old sentinel-drain
        race where ``flush()`` could return before a background POST that
        started just before it actually landed.
        """
        self._transport.flush()
        self._otel_engine.flush()

    def shutdown(self) -> None:
        """Flush and stop the background transport thread + OTel provider."""
        _set_transport(None)
        self._transport.shutdown()
        self._otel_engine.shutdown()

    def _reset_for_testing(self) -> None:
        """Test-only teardown: stops the legacy transport thread (fast —
        single non-retrying attempt, see ``transport.py``) but drops the
        OTel engine via its FAST reset rather than a real ``shutdown()``.
        See ``otel_provider._OtelEngine.reset_for_testing`` for why a real
        blocking shutdown against tests' fake endpoints is unusably slow.
        Not used at genuine app exit — that path is ``shutdown()`` above.
        """
        _set_transport(None)
        self._transport.shutdown()
        self._otel_engine.reset_for_testing()


# Backwards-compatible alias for any in-flight references during the V1→V2
# transition. Removed in a follow-up cleanup commit before Wave 2 begins.
MorseClient = _Client
