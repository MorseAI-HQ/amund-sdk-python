"""OTel-backed transport engine for the v2 tracing primitives (MHQ-668).

Architecture
------------
``run()`` / ``span()`` / ``@trace_agent`` still build the same in-memory
``SpanData`` / ``_TraceState`` structures they always have (see
``tracing.py`` — nothing there changed for adapters). What changed is what
happens to that data once a trace closes: instead of JSON-encoding it and
handing it to a hand-rolled queue + background-thread HTTP sender
(``transport.py``, now used only by ``record()`` / ``morse_ai.log()`` /
the dependency-scan POST — see their module docstrings), the trace's
already-built wire dict (the same shape ``_get_last_trace()`` returns) is
converted into real ``opentelemetry.sdk.trace.Span`` objects and hand
those to a standard OTel pipeline:

    tracer.start_span() / span.end()
        -> BatchSpanProcessor (stdlib OTel: 512 queue / 5s / 2048 batch
           defaults, overridable via env)
        -> _RedactingSpanExporter (this module — delegating wrapper)
        -> _DropReportingOTLPSpanExporter (this module — stock
           opentelemetry-exporter-otlp-proto-http exporter, plus a
           delivery-loss counter on terminal give-up; protobuf + gzip,
           POSTs to ``<endpoint>/api/v1/otlp/v1/traces``)

Why convert at trace-close time rather than incrementally as each
``TracedSpan``/``TracedRun`` enters and exits: several adapters
(``adapters/anthropic.py``, ``adapters/openai.py``,
``adapters/openai_agents.py``, ``adapters/claude_agent_sdk.py``) build
``SpanData`` objects directly and append them to
``_TraceState.completed_spans`` without ever going through the
``TracedSpan`` context manager (they intercept provider SDK calls and
reconstruct spans from callback data). Converting once, at the end, from
the same ``completed_spans`` list that already backs ``_get_last_trace()``
means every one of those call sites keeps working completely unmodified —
they never touch this module, transport.py, or queue.py.

Trace/span identifiers
-----------------------
Morse's own ``trace_id`` (``uuid.uuid4().hex``, 32 hex chars / 128 bits)
maps 1:1 onto OTel's 128-bit ``TraceId``. Morse's ``span_id`` is ALSO a
32-hex uuid4 (128 bits) — but OTel's ``SpanId`` is only 64 bits. There is
no lossless mapping, and none is needed: span ids are wire-protocol-local
identifiers, not semantic content, and the server already round-trips
OTel's 64-bit span ids by zero-extending them into the 128-bit UUID column
(``adapters/otlp.py::_hex_span_id_to_uuid``) — built for exactly this.
Span ids are derived from Morse's (``_internal/span_ids.py``: first 64 bits)
and staged into OTel's ``IdGenerator`` so a ``traceparent`` header sent mid-call
matches the exported span (MHQ-1880). The *trace* id is forced via the
standard "synthetic remote parent" technique also used for W3C traceparent
continuation (see ``propagation.py``): a ``NonRecordingSpan`` carrying a ``SpanContext``
with the desired ``trace_id`` and a pre-computed ``sampled`` flag is set
as the parent context for ``tracer.start_span()``, so every span in the
trace (root included) inherits the same trace id and the same sampling
decision from ``ParentBased``.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import threading
import time
from datetime import UTC, datetime
from typing import Any

from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.sdk.trace.sampling import ParentBased, Sampler, TraceIdRatioBased
from opentelemetry.trace import (
    NonRecordingSpan,
    Span,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    TraceFlags,
    set_span_in_context,
)

from morse_ai._internal.agent_name import env_service_name
from morse_ai._internal.redaction import DISABLED_CONFIG, RedactionConfig
from morse_ai._internal.span_ids import otel_span_id_for
from morse_ai._version import __version__

logger = logging.getLogger("morse_ai")

_INSTRUMENTATION_NAME = "morse_ai"
DEFAULT_MAX_FIELD_BYTES = 65536
TRUNCATION_MARKER = "…[truncated]"

# OTel BatchSpanProcessor stock defaults (explicit here so env overrides have
# a documented baseline to fall back to).
_DEFAULT_SCHEDULE_DELAY_MILLIS = 5000
_DEFAULT_MAX_QUEUE_SIZE = 2048
_DEFAULT_MAX_EXPORT_BATCH_SIZE = 512


# ---------------------------------------------------------------------------
# Byte-cap helpers (applied at attribute-set time — MORSE_MAX_FIELD_BYTES)
# ---------------------------------------------------------------------------


def cap_bytes(value: str, max_bytes: int) -> str:
    """Truncate ``value`` to ``max_bytes`` (UTF-8), appending the truncation
    marker. No-op if already within the cap. Never splits a multi-byte
    UTF-8 codepoint."""
    if max_bytes <= 0:
        return value
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    marker_bytes = TRUNCATION_MARKER.encode("utf-8")
    budget = max(max_bytes - len(marker_bytes), 0)
    truncated = encoded[:budget]
    while truncated:
        try:
            return truncated.decode("utf-8") + TRUNCATION_MARKER
        except UnicodeDecodeError:
            truncated = truncated[:-1]
    return TRUNCATION_MARKER


def json_capped(value: Any, max_bytes: int) -> str | None:
    """JSON-encode ``value`` and byte-cap the result.

    ``None`` in -> ``None`` out (caller omits the attribute entirely).
    String values are capped BEFORE encoding so the result stays valid
    JSON. Structured (dict/list) values are encoded fully, then capped as
    a raw string if oversized — on the rare oversized-structured-payload
    path this can produce a value that no longer parses as JSON server
    side; the server's tolerant-JSON adapters already drop unparseable
    blobs as a whole (same behavior as a malformed ``morse.context_segments``
    attribute today), so this is a safety valve against unbounded payload
    growth, not a guarantee of partial preservation.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return json.dumps(cap_bytes(value, max_bytes))
    try:
        encoded = json.dumps(value, default=str)
    except (TypeError, ValueError):
        encoded = json.dumps(str(value))
    return cap_bytes(encoded, max_bytes)


def _decorrelate_trace_id_for_sampling(trace_id_int: int) -> int:
    """Return a value uniformly distributed over the full 128-bit range for
    sampling purposes, decorrelated from ``trace_id_int``'s UUIDv4 variant
    bits. See ``_OtelEngine.resolve_sampled`` for why this exists — feeding
    a raw uuid4 straight to ``TraceIdRatioBased`` produces a hard 0%/100%
    step function instead of a ratio.
    """
    digest = hashlib.sha256(trace_id_int.to_bytes(16, "big")).digest()
    return int.from_bytes(digest[:16], "big")


def _iso_to_ns(value: str | None) -> int | None:
    """Convert an ISO-8601 timestamp string to Unix nanoseconds."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp() * 1_000_000_000)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Redacting exporter — single redaction pass, off the hot path
# ---------------------------------------------------------------------------


def _redact_attr_value(value: Any, config: RedactionConfig) -> Any:
    """Redact an OTel attribute value (scalar or homogeneous sequence)."""
    if isinstance(value, str):
        return config.redact_string(value)
    if isinstance(value, (list, tuple)):
        return type(value)(
            config.redact_string(v) if isinstance(v, str) else v for v in value
        )
    return value


def _redact_events(span: Any, config: RedactionConfig) -> None:
    """Redact span-event attribute strings (MHQ-133).

    Morse's own spans carry no events, but third-party OTel emitters routed
    through this exporter (Strands Agents) put prompt and completion text in
    events (``gen_ai.user.message`` / ``gen_ai.choice``). Ended spans are
    frozen, so the event list is rebuilt rather than mutated.
    """
    events = getattr(span, "_events", None)  # noqa: SLF001
    if not events:
        return
    from opentelemetry.sdk.trace import Event  # noqa: PLC0415

    span._events = tuple(  # noqa: SLF001
        Event(
            event.name,
            {
                key: _redact_attr_value(value, config)
                for key, value in (event.attributes or {}).items()
            },
            event.timestamp,
        )
        for event in events
    )


class _RedactingSpanExporter(SpanExporter):
    """Delegating ``SpanExporter``: redacts span attribute string values in
    ``export()`` (called on the ``BatchSpanProcessor``'s background worker
    thread — never on the caller's hot path), then forwards to the real
    OTLP exporter. This is the SDK's single PII-redaction pass for the
    tracing pipeline; nothing upstream of this (``tracing.py``,
    ``otel_provider.emit_trace_entry``) redacts anything.
    """

    def __init__(
        self, inner: SpanExporter, redaction_config: RedactionConfig | None
    ) -> None:
        self._inner = inner
        self._redaction_config = redaction_config or DISABLED_CONFIG

    def export(self, spans):  # noqa: ANN001 - Sequence[ReadableSpan]
        config = self._redaction_config
        if config.enabled or config.extra_patterns:
            for span in spans:
                attrs = getattr(span, "_attributes", None)  # noqa: SLF001
                if not attrs:
                    continue
                # ReadableSpan.attributes is a MappingProxyType over
                # ``_attributes``, a ``BoundedAttributes`` — but by the time
                # a span reaches export() it has already been ``.end()``ed,
                # which freezes it (``BoundedAttributes.__setitem__`` raises
                # TypeError unconditionally once ``_immutable`` is set).
                # Mutating in place is therefore not an option; replace the
                # whole backing dict with a plain (mutable, unbounded — no
                # further attribute-count/length enforcement needed at this
                # point in the pipeline) dict holding the redacted values.
                # This is the one place in the pipeline allowed to reach
                # into the SDK-private attribute, precisely so redaction can
                # happen post-batch instead of per-attribute-set on the
                # calling thread.
                span._attributes = {  # noqa: SLF001
                    key: _redact_attr_value(value, config)
                    for key, value in attrs.items()
                }
            for span in spans:
                _redact_events(span, config)
        return self._inner.export(spans)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        fn = getattr(self._inner, "force_flush", None)
        if callable(fn):
            return bool(fn(timeout_millis))
        return True


def _drop_reason(status: int | None, attempts: int) -> str:
    """Name the failure the way ``transport.py`` names the same one.

    Mirrors the mapping in ``BatchTransport._send`` deliberately: MHQ-971
    established that a revoked key, a wrong URL and a rate limit must not all
    arrive looking alike, and a second, differently-worded mapping for the OTLP
    path would recreate exactly that.
    """
    from morse_ai._internal import diagnostics  # noqa: PLC0415 — lazy by design

    if status == 429:
        # MHQ-673: reaching here means the Retry-After retries were used up or
        # the budget ran out, not that the first 429 was discarded.
        return diagnostics.REASON_QUOTA_REJECTED
    if status in (401, 403):
        return diagnostics.REASON_AUTH_REJECTED
    if status is not None and 400 <= status < 500:
        return diagnostics.REASON_ENDPOINT_REJECTED
    # 5xx and network faults: the stock exporter retries these, so reaching a
    # terminal FAILURE means the retry budget was spent — unless it gave up on
    # the first attempt (timeout, shutdown), which is a different story.
    return (
        diagnostics.REASON_RETRY_EXHAUSTED
        if attempts > 1
        else diagnostics.REASON_SEND_FAILED
    )


# MHQ-673 — 429 retry budget. Kept small on purpose: this runs on the
# BatchSpanProcessor's single worker thread, so time spent waiting here is time
# the queue is not being drained. Morse's own ingest limiter answers
# ``Retry-After: 1``, which fits several times over inside the budget.
_RATE_LIMIT_MAX_RETRIES = 3
_RATE_LIMIT_MAX_SLEEP_SECONDS = 10.0
_RATE_LIMIT_RETRY_BUDGET_SECONDS = 30.0


def _parse_retry_after(raw: Any) -> float | None:
    """Seconds to wait from a ``Retry-After`` header, or None.

    ponytail: delta-seconds only. RFC 9110 also permits an HTTP-date, which no
    Morse endpoint sends (``apps/api/apps/core/ratelimit.py`` and the ingest
    middleware both write an integer). An unparseable value falls back to our
    own exponential backoff, so the date form degrades to "retried anyway,
    just not at the server's preferred moment" rather than to a drop. Parse it
    here if a proxy in front of a customer's deployment ever rewrites it.
    """
    if raw is None:
        return None
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    # A negative or absurd value is a broken server, not an instruction.
    return seconds if 0 <= seconds <= 3600 else None


# Bumped by `_OtelEngine.reset_for_testing()`. Exporters it abandons keep
# retrying on daemon threads; a stale generation stops them recording drops
# into the next test's process-global diagnostics counters.
_exporter_generation = 0


class _DropReportingOTLPSpanExporter(OTLPSpanExporter):
    """OTLP exporter that counts what it permanently discards (MHQ-1030).

    The stock exporter's terminal give-up is one log line and a ``FAILURE``
    return that ``BatchSpanProcessor`` drops on the floor without a word. From
    the app's side — and from ours — a rate-limited window and a healthy one
    are then indistinguishable: no exception, no counter, no dashboard signal.
    That is the failure mode Morse exists to prevent, so it does not get to
    live in Morse's own SDK.

    ``_export()`` is overridden purely to *observe* the HTTP status and
    ``Retry-After``. Neither is available from ``export()``'s bare
    SUCCESS/FAILURE return, and without the status every drop here would have to
    be reported under a single guessed reason. Signature is passed through
    untouched (``*args``), so a future version of the exporter that adds a
    parameter still works.

    It also retries 429 (MHQ-673), which the stock exporter does not: upstream
    treats only 408 and 5xx as retryable and ignores ``Retry-After`` entirely,
    so a rate-limited batch is discarded on its first response. The retry lives
    in ``export()`` around the stock loop rather than inside it — upstream's
    backoff/deadline logic is version-fragile internals, and wrapping it keeps
    this class working across the exporter versions pyproject allows.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._generation = _exporter_generation
        # Thread-local: force_flush() can drive an export concurrently with the
        # BatchSpanProcessor's own worker, and two exports must not read each
        # other's status.
        self._attempt_state = threading.local()

    def _export(self, *args: Any, **kwargs: Any) -> Any:
        state = self._attempt_state
        state.attempts = getattr(state, "attempts", 0) + 1
        state.status = None
        state.retry_after = None
        response = super()._export(*args, **kwargs)
        state.status = getattr(response, "status_code", None)
        if state.status == 429:
            state.retry_after = _parse_retry_after(
                (getattr(response, "headers", None) or {}).get("Retry-After")
            )
        return response

    def _wait_before_retry(self, delay: float) -> bool:
        """Sleep ``delay``. False if shutdown interrupted it.

        Prefers the exporter's own shutdown Event so a process exiting during a
        rate-limit window is not held up by the full backoff. ``getattr`` because
        that attribute is an upstream implementation detail and pyproject allows
        exporters back to 1.27.0.
        """
        event = getattr(self, "_shutdown_in_progress", None)
        if event is not None:
            return not event.wait(delay)
        time.sleep(delay)
        return not getattr(self, "_shutdown", False)

    def export(self, spans):  # noqa: ANN001 - Sequence[ReadableSpan]
        state = self._attempt_state
        deadline = time.monotonic() + _RATE_LIMIT_RETRY_BUDGET_SECONDS

        for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
            state.attempts = 0
            state.status = None
            state.retry_after = None
            result = super().export(spans)
            if result is SpanExportResult.SUCCESS:
                return result

            if getattr(state, "status", None) != 429 or attempt == (
                _RATE_LIMIT_MAX_RETRIES
            ):
                break

            # MHQ-673: the stock exporter retries 408 and 5xx only, and ignores
            # Retry-After entirely, so a 429 is dropped on the first response.
            # MHQ-660 then made the ingest limiter actually fire for Bearer SDK
            # traffic, which created exactly the condition — an over-quota
            # customer — under which the client silently discards telemetry.
            # A client held at its plan's ingest rate must degrade to delay,
            # not to loss.
            delay = getattr(state, "retry_after", None)
            if not delay:
                # No usable header — absent, unparseable, or ``Retry-After: 0``,
                # which means "retry now", NOT "stop retrying". Falling through
                # to ``delay <= 0`` below would have dropped the batch on its
                # first response: the exact failure this whole retry exists to
                # remove. Back off on our own rather than hammering.
                delay = float(2**attempt)
            delay = min(
                delay, _RATE_LIMIT_MAX_SLEEP_SECONDS, deadline - time.monotonic()
            )
            if delay <= 0:
                # Budget spent. Waiting longer would stall the
                # BatchSpanProcessor's only worker and lose spans upstream of
                # here instead — a bigger loss than the batch in hand.
                break
            logger.warning(
                "morse_ai: ingest rate-limited (429), retrying %d spans in %.1fs",
                len(spans),
                delay,
            )
            if not self._wait_before_retry(delay):
                break

        if self._generation != _exporter_generation:
            return result  # abandoned by reset_for_testing()
        try:
            from morse_ai._internal import diagnostics  # noqa: PLC0415

            diagnostics.record_drop(
                _drop_reason(
                    getattr(state, "status", None), getattr(state, "attempts", 0)
                ),
                "span",
                len(spans),
            )
        except Exception:  # noqa: BLE001 — counting must never break export
            logger.debug("morse_ai: export-drop accounting failed", exc_info=True)
        return result


class _RecordingSpanExporter(SpanExporter):
    """Test-only in-memory exporter: mirrors whatever the redacting exporter
    saw (i.e. POST-redaction data) into ``_last_export`` so
    ``tracing._get_last_trace()`` can be rewired off the real export path
    (per MHQ-668 design) without needing a live HTTP backend in unit tests.
    Wrapped BEHIND the redacting exporter so it observes the same redacted
    values the wire would carry.
    """

    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans):  # noqa: ANN001
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def clear(self) -> None:
        self.spans.clear()


# ---------------------------------------------------------------------------
# Provider singleton
# ---------------------------------------------------------------------------


class _StagedSpanIdGenerator(RandomIdGenerator):
    """MHQ-1880: hand OTel a pre-agreed span id for the next span.

    ``emit_trace_entry`` stages ``otel_span_id_for(span_id)`` right before
    ``tracer.start_span`` (which asks for the id synchronously on the same
    thread), so the exported id equals the one already placed in an outgoing
    ``traceparent`` header. Unstaged callers (third-party OTel spans on this
    provider) still get random ids.
    """

    def __init__(self) -> None:
        self._staged = threading.local()

    def stage(self, span_id_hex: str | None) -> None:
        self._staged.value = int(span_id_hex, 16) if span_id_hex else None

    def generate_span_id(self) -> int:
        staged = getattr(self._staged, "value", None)
        self._staged.value = None
        return staged if staged else super().generate_span_id()


class _OtelEngine:
    """Lazily-built TracerProvider + exporter chain. One per process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.provider: TracerProvider | None = None
        self.tracer: Any = None
        self.root_sampler: TraceIdRatioBased | None = None
        self.id_generator = _StagedSpanIdGenerator()
        self.recorder: _RecordingSpanExporter | None = None
        # MHQ-133: the redacting OTLP exporter, so third-party OTel spans
        # (Strands) emitted on a customer-owned provider can be forwarded to
        # Morse through the same redaction + delivery path.
        self.exporter: SpanExporter | None = None
        self.max_field_bytes: int = DEFAULT_MAX_FIELD_BYTES
        self._atexit_registered = False

    def configure(
        self,
        *,
        endpoint: str,
        api_key: str,
        sample_rate: float,
        schedule_delay_millis: float,
        max_queue_size: int,
        max_export_batch_size: int,
        max_field_bytes: int,
        timeout: float,
        redaction_config: RedactionConfig | None,
        capture_for_tests: bool = False,
    ) -> None:
        """(Re)build the provider. Mirrors ``_Client.__init__`` semantics:
        calling this again (e.g. a second ``morse_ai.init()``) tears down
        the previous provider cleanly first."""
        with self._lock:
            if self.provider is not None:
                try:
                    self.provider.shutdown()
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "morse_ai: prior TracerProvider shutdown swallowed: %s", exc
                    )

            self.max_field_bytes = max_field_bytes
            self.root_sampler = TraceIdRatioBased(sample_rate)
            sampler: Sampler = ParentBased(root=self.root_sampler)
            # MHQ-1100: an env-derived service.name must never become agent
            # identity. Pin the OTel default and keep the env value under
            # ``morse.process.service_name``; the agent travels as the
            # root-span ``morse.agent`` attribute instead.
            resource_attrs: dict[str, Any] = {"service.name": "unknown_service"}
            # MHQ-1740: framework spans routed through this provider (Strands)
            # carry no Morse scope, so the server reads the SDK release here.
            resource_attrs["morse.sdk.version"] = __version__
            proc_service = env_service_name()
            if proc_service:
                resource_attrs["morse.process.service_name"] = proc_service
            # Deploy markers: the JSON path already sends git.revision; the
            # OTel path (run()/span()/LangGraph) never did, so its traces had none.
            from morse_ai._internal.env import (
                get_git_revision,
                get_service_version,
            )

            git_revision = get_git_revision()
            if git_revision:
                resource_attrs["git.revision"] = git_revision
            # ATTR-06: service.version from OTEL_RESOURCE_ATTRIBUTES, set
            # explicitly so the JSON and OTLP paths agree on the value.
            service_version = get_service_version()
            if service_version:
                resource_attrs["service.version"] = service_version
            resource = Resource.create(resource_attrs)
            provider = TracerProvider(
                sampler=sampler, resource=resource, id_generator=self.id_generator
            )

            otlp_exporter = _DropReportingOTLPSpanExporter(
                endpoint=endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                compression=Compression.Gzip,
                timeout=timeout,
            )
            exporter: SpanExporter = _RedactingSpanExporter(
                otlp_exporter, redaction_config
            )

            self.exporter = exporter
            self.recorder = _RecordingSpanExporter() if capture_for_tests else None
            provider.add_span_processor(
                BatchSpanProcessor(
                    exporter,
                    max_queue_size=max_queue_size,
                    schedule_delay_millis=schedule_delay_millis,
                    max_export_batch_size=max_export_batch_size,
                )
            )
            if self.recorder is not None:
                # Second processor: independent in-memory capture so tests
                # can inspect exported spans without a live collector. Runs
                # its own BatchSpanProcessor so it doesn't share queue
                # pressure with the real exporter.
                provider.add_span_processor(BatchSpanProcessor(self.recorder))

            self.provider = provider
            # The scope version is how the server stamps traces.sdk_version on
            # the OTLP path (MHQ-1158); without it every trace read otlp/http.
            self.tracer = provider.get_tracer(_INSTRUMENTATION_NAME, __version__)

            if not self._atexit_registered:
                atexit.register(self.shutdown)
                self._atexit_registered = True

    def shutdown(self) -> None:
        with self._lock:
            if self.provider is not None:
                try:
                    self.provider.shutdown()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("morse_ai: TracerProvider.shutdown swallowed: %s", exc)
                    _report(exc, "morse_ai.otel_provider.00")
                self.provider = None
                self.tracer = None
                self.exporter = None

    def reset_for_testing(self) -> None:
        """Test-only fast reset: drop the provider reference WITHOUT
        blocking on ``TracerProvider.shutdown()``.

        Why this exists (and why plain ``shutdown()`` is unsafe to call
        from every test's teardown): the stock ``OTLPSpanExporter`` retries
        transient failures — including plain connection-refused, which is
        exactly what every SDK unit test's fake/unreachable endpoint
        produces — with exponential backoff across up to 6 attempts.
        Empirically that is 15-30+ seconds of REAL wall-clock blocking in
        ``TracerProvider.shutdown()``/``force_flush()`` per call once a
        span has been queued. Multiplied across dozens of test files each
        calling ``morse_ai._reset_for_testing()`` in teardown, that's not a
        slow suite, it's a hung one. The queued worker thread is a daemon
        (OTel SDK default) so abandoning it here is safe — it either
        finishes its own retries in the background or dies with the
        process; it never blocks test teardown or process exit. Real
        ``shutdown()`` (genuine app exit) is untouched and still blocks
        properly so production telemetry gets a fair flush attempt.
        """
        global _exporter_generation  # noqa: PLW0603
        with self._lock:
            _exporter_generation += 1
            self.provider = None
            self.tracer = None
            self.exporter = None
            self.recorder = None

    def flush(self, timeout_seconds: float = 30.0) -> None:
        """Block until in-flight exports finish (or the timeout elapses)."""
        provider = self.provider
        if provider is not None:
            try:
                provider.force_flush(timeout_millis=int(timeout_seconds * 1000))
            except Exception as exc:  # noqa: BLE001
                logger.debug("morse_ai: TracerProvider.force_flush swallowed: %s", exc)
                _report(exc, "morse_ai.otel_provider.01")

    def resolve_sampled(self, trace_id_int: int) -> bool:
        """Run the root sampler directly for a trace id with no real parent.

        The public ``Tracer.start_span()`` API cannot be handed an explicit
        trace id for a genuinely-new (non-continued) trace — it either
        mints a fresh random one or inherits one from a parent context. We
        need a *specific* pre-existing trace id (``state.trace_id``, shared
        with W3C propagation / ``_last_trace`` / incoming-traceparent
        continuation), so we invoke the configured ``TraceIdRatioBased``
        sampler directly, once, and feed its decision into a synthetic
        "remote parent" SpanContext (see ``emit_trace_entry``) that forces
        both the trace id AND the sampled flag for every span emitted.

        IMPORTANT — trace id is NOT fed to the sampler as-is. Morse's
        ``trace_id`` is ``uuid.uuid4().hex``: a real UUIDv4 has its variant
        bits fixed at the top of byte 8, which lands EXACTLY inside the low
        64 bits that ``TraceIdRatioBased`` inspects
        (``trace_id & ((1<<64)-1)``). That forces the low-64 value's top
        byte into ``0x80-0xBF`` on every single trace id — i.e. always
        ≥50% and <75% of the full 64-bit range — so a stock
        ``TraceIdRatioBased`` fed a raw uuid4 samples 0% of traces for any
        rate below 0.5 and 100% for any rate at/above 0.75, regardless of
        the configured rate (confirmed empirically while writing this
        module's tests — see ``tests/test_otel_provider.py::TestSampler``).
        A SHA-256 digest of the trace id decorrelates it from that fixed
        bit pattern before handing it to the sampler; the mapping is
        deterministic (same trace id always yields the same 8 hashed bytes)
        so repeated calls for the same trace — mid-trace via
        ``propagation.inject_traceparent()``, then again at emit time —
        agree with each other.
        """
        sampler = self.root_sampler
        if sampler is None:
            # Engine never configured (no init()/API key) — fall back to
            # reading MORSE_SAMPLE_RATE directly so callers like
            # propagation.inject_traceparent() still get a real decision
            # instead of assuming "no engine == always sampled" by fiat.
            from morse_ai._internal import env as _env  # noqa: PLC0415

            sampler = TraceIdRatioBased(_env.get_sample_rate())
        decorrelated = _decorrelate_trace_id_for_sampling(trace_id_int)
        result = sampler.should_sample(None, decorrelated, "")
        from opentelemetry.sdk.trace.sampling import Decision

        return result.decision != Decision.DROP


_engine = _OtelEngine()


def get_engine() -> _OtelEngine:
    return _engine


# ---------------------------------------------------------------------------
# Attribute mapping — SpanData dict -> OTel attributes
# ---------------------------------------------------------------------------

# See the module docstring + the final PR report for the full mapping
# table. This function is the single source of truth for it.


def _span_dict_to_otel_attributes(
    span_dict: dict[str, Any],
    trace_dict: dict[str, Any],
    resource_dict: dict[str, Any],
    *,
    is_root: bool,
    max_field_bytes: int,
    prompt_ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {}

    span_type = span_dict.get("type")
    if span_type:
        attrs["morse.span_type"] = span_type

    model = span_dict.get("model")
    if model is not None:
        attrs["gen_ai.request.model"] = model

    for src_key, otel_key in (
        ("input_tokens", "gen_ai.usage.input_tokens"),
        ("output_tokens", "gen_ai.usage.output_tokens"),
        ("cached_input_tokens", "gen_ai.usage.cache_read_input_tokens"),
        ("cache_write_tokens", "gen_ai.usage.cache_creation_input_tokens"),
        ("reasoning_tokens", "gen_ai.usage.reasoning_tokens"),
    ):
        value = span_dict.get(src_key)
        if value is not None:
            attrs[otel_key] = int(value)

    cost_usd = span_dict.get("cost_usd")
    if cost_usd is not None:
        # morse.cost_usd: direct SDK/adapter-computed cost (pricing-table based)
        # — distinct from gen_ai.usage.cost (provider-billing-reported,
        # feeds the separate `provider_reported` cost-engine tier server
        # side). Stringify to preserve precision across the wire.
        attrs["morse.cost_usd"] = str(cost_usd)

    input_data = span_dict.get("input_data")
    if input_data is not None:
        encoded = json_capped(input_data, max_field_bytes)
        if encoded is not None:
            attrs["morse.input_data"] = encoded

    output_data = span_dict.get("output_data")
    if output_data is not None:
        encoded = json_capped(output_data, max_field_bytes)
        if encoded is not None:
            attrs["morse.output_data"] = encoded

    context_segments = span_dict.get("context_segments")
    if context_segments is not None:
        encoded = json_capped(context_segments, max_field_bytes)
        if encoded is not None:
            attrs["morse.context_segments"] = encoded

    # Arbitrary custom metadata (SpanData.metadata / set_metadata()). Each
    # top-level key is namespaced under `morse.metadata.<key>` so the server's
    # OTLP translation layer can pop + un-prefix + JSON-decode them back
    # into a clean metadata dict (apps/api/apps/traces/adapters/otlp.py) —
    # without that unwrap step these would otherwise ALSO show up verbatim
    # under their prefixed key thanks to the (separate, intentional, tested)
    # "dump all OTLP attrs into Span.metadata" behavior for third-party
    # OTel emitters.
    metadata = span_dict.get("metadata")
    if metadata:
        for key, value in metadata.items():
            encoded = json_capped(value, max_field_bytes)
            if encoded is not None:
                attrs[f"morse.metadata.{key}"] = encoded

    error_message = span_dict.get("error_message")
    # Trace-level fields only travel on the root span.
    if is_root:
        agent = trace_dict.get("agent")
        if agent:
            attrs["morse.agent"] = str(agent)
        # AID-01/04: how the agent was named, and the run("app") label.
        trace_md = trace_dict.get("metadata") or {}
        if trace_md.get("identity_source"):
            attrs["morse.identity_source"] = str(trace_md["identity_source"])
        if trace_md.get("morse_app"):
            attrs["morse.app"] = str(trace_md["morse_app"])
        thread_id = trace_dict.get("thread_id")
        if thread_id is not None:
            attrs["morse.thread_id"] = str(thread_id)
        user_id = trace_dict.get("user_id")
        if user_id is not None:
            attrs["morse.user_id"] = str(user_id)
        user_metadata = trace_dict.get("user_metadata")
        if user_metadata:
            attrs["morse.user_metadata"] = json.dumps(
                {k: str(v) for k, v in user_metadata.items()}
            )
        success = trace_dict.get("success")
        if success is not None:
            attrs["morse.outcome.success"] = bool(success)
        outcome = trace_dict.get("outcome")
        if outcome is not None:
            attrs["morse.outcome.label"] = str(outcome)
        # ATTR-01/04: record keys and the declared agent version ride the root
        # span; the server reads the first value it sees.
        record_keys = trace_dict.get("record_keys")
        if record_keys:
            for key_type, key_value in record_keys.items():
                attrs[f"morse.record_key.{key_type}"] = str(key_value)
        agent_version = trace_dict.get("agent_version")
        if agent_version:
            attrs["morse.agent_version"] = str(agent_version)
        framework = resource_dict.get("morse_ai.framework")
        if framework:
            attrs["morse.framework"] = framework
        source = resource_dict.get("morse_ai.source")
        if source:
            attrs["morse.source"] = source

    # ATTR-04: a prompt reference lives on the span set_prompt() was called in
    # (the root span when it was called outside a span).
    if prompt_ref:
        attrs["morse.prompt.name"] = str(prompt_ref["name"])
        if prompt_ref.get("version") is not None:
            attrs["morse.prompt.version"] = int(prompt_ref["version"])
        if prompt_ref.get("link_source") == "fetch":
            attrs["morse.prompt.link_source"] = "fetch"

    exception_type = span_dict.get("exception_type")
    if exception_type and span_dict.get("status") == "failed":
        attrs["morse.exception_type"] = str(exception_type)

    if error_message and span_dict.get("status") == "failed":
        # Also carried as a plain attribute so status.message-agnostic
        # consumers (and the golden-corpus test) can read it without
        # round-tripping through OTel's Status object.
        attrs["morse.error_message"] = str(error_message)

    return attrs


def _prompt_refs_by_span(
    prompts: list[dict[str, Any]],
    known_span_ids: set[Any],
    root_span_id: str | None,
) -> dict[str, dict[str, Any]]:
    """Map span id -> the one prompt reference that span carries on OTLP.

    A span has a single ``morse.prompt.name`` / ``.version`` pair, so when
    several references were set on the same span the LAST one wins (the JSON
    path keeps them all). A reference with no ``span_id``, or whose span was
    never emitted, goes on the root span and competes there in call order
    (still last wins).
    """
    out: dict[str, dict[str, Any]] = {}
    for ref in prompts:
        if not isinstance(ref, dict) or not ref.get("name"):
            continue
        target = ref.get("span_id")
        if target not in known_span_ids:
            target = root_span_id
        if target:
            out[target] = ref
    return out


def emit_trace_entry(entry: dict[str, Any]) -> None:
    """Convert one already-built wire-format trace entry (the same shape
    ``tracing._build_trace_dict`` produces / ``_get_last_trace()`` returns)
    into real OTel spans and hand them to the configured pipeline.

    No-op (logged at debug) if the engine hasn't been configured yet —
    mirrors the "no transport configured -> silent" contract the rest of
    the SDK follows.
    """
    engine = _engine
    tracer = engine.tracer
    if tracer is None:
        return

    trace_dict = entry.get("trace") or {}
    span_dicts = entry.get("spans") or []
    resource_dict = entry.get("resource") or {}
    if not span_dicts:
        return

    trace_id_hex = trace_dict.get("trace_id")
    try:
        trace_id_int = int(trace_id_hex, 16) if trace_id_hex else None
    except (TypeError, ValueError):
        trace_id_int = None
    if not trace_id_int:
        return

    sampled = engine.resolve_sampled(trace_id_int)
    synthetic_parent_ctx = set_span_in_context(
        NonRecordingSpan(
            SpanContext(
                trace_id=trace_id_int,
                span_id=0x1,  # placeholder — only trace_id/flags matter here
                is_remote=True,
                trace_flags=TraceFlags(
                    TraceFlags.SAMPLED if sampled else TraceFlags.DEFAULT
                ),
            )
        )
    )

    root_span_id = None
    for s in span_dicts:
        if not s.get("parent_span_id"):
            root_span_id = s.get("span_id")
            break
    if root_span_id is None and span_dicts:
        root_span_id = span_dicts[0].get("span_id")

    # Parents before children: a child's parent SpanContext must exist by the
    # time we process it. MHQ-1412: completed spans arrive in COMPLETION order
    # (a child ends before its parent), so the former root-first sort left
    # nested spans without their parent context and OTel re-parented them onto
    # the synthetic span_id=0x1 placeholder (stored as 0000...0001).
    ordered = _parents_first(span_dicts, root_span_id)

    prompt_by_span = _prompt_refs_by_span(
        trace_dict.get("prompts") or [],
        {s.get("span_id") for s in span_dicts},
        root_span_id,
    )

    span_ctx_by_id: dict[str, SpanContext] = {}
    max_field_bytes = engine.max_field_bytes

    for span_dict in ordered:
        span_id_hex = span_dict.get("span_id")
        parent_id_hex = span_dict.get("parent_span_id")
        parent_ctx = synthetic_parent_ctx
        if parent_id_hex and parent_id_hex in span_ctx_by_id:
            parent_ctx = set_span_in_context(
                NonRecordingSpan(span_ctx_by_id[parent_id_hex])
            )

        attributes = _span_dict_to_otel_attributes(
            span_dict,
            trace_dict,
            resource_dict,
            is_root=(span_id_hex == root_span_id),
            max_field_bytes=max_field_bytes,
            prompt_ref=prompt_by_span.get(span_id_hex),
        )
        start_ns = _iso_to_ns(span_dict.get("started_at"))
        end_ns = _iso_to_ns(span_dict.get("ended_at")) or start_ns

        try:
            # MHQ-1880: export the id a traceparent header already carried.
            if span_id_hex:
                engine.id_generator.stage(otel_span_id_for(span_id_hex))
            otel_span: Span = tracer.start_span(
                name=span_dict.get("name") or "span",
                context=parent_ctx,
                kind=SpanKind.INTERNAL,
                attributes=attributes,
                start_time=start_ns,
            )
            if span_id_hex:
                span_ctx_by_id[span_id_hex] = otel_span.get_span_context()
            if span_dict.get("status") == "failed":
                # record_exception-shaped event so the server's OTLP adapter and any
                # third-party OTel backend see exception.type / exception.message.
                exc_type = span_dict.get("exception_type")
                exc_msg = span_dict.get("error_message")
                if exc_type or exc_msg:
                    event_attrs = {}
                    if exc_type:
                        event_attrs["exception.type"] = str(exc_type)
                    if exc_msg:
                        event_attrs["exception.message"] = str(exc_msg)
                    otel_span.add_event("exception", event_attrs)
                otel_span.set_status(
                    Status(StatusCode.ERROR, str(span_dict.get("error_message") or ""))
                )
            otel_span.end(end_time=end_ns)
        except Exception as exc:  # noqa: BLE001 — never break the host app
            logger.debug("morse_ai: OTel span emission swallowed error: %s", exc)
            _report(exc, "morse_ai.otel_provider.02")
        finally:
            engine.id_generator.stage(None)


def _parents_first(span_dicts: list[dict], root_span_id: str | None) -> list[dict]:
    """Order span dicts so every parent precedes its children (MHQ-1412).

    Stable apart from hoisting parents ahead of their children. A parent that
    is absent from the batch, or a cycle, falls through in input order.
    """
    by_id = {s.get("span_id"): s for s in span_dicts if s.get("span_id")}
    ordered: list[dict] = []
    placed: set[int] = set()

    def place(span_dict: dict) -> None:
        chain: list[dict] = []
        cur: dict | None = span_dict
        seen: set[int] = set()
        while cur is not None and id(cur) not in placed and id(cur) not in seen:
            seen.add(id(cur))
            chain.append(cur)
            cur = by_id.get(cur.get("parent_span_id"))
        for item in reversed(chain):
            placed.add(id(item))
            ordered.append(item)

    for s in span_dicts:
        if s.get("span_id") == root_span_id:
            place(s)
    for s in span_dicts:
        place(s)
    return ordered


def _report(exc: BaseException, fingerprint: str) -> None:
    """Route a swallowed internal error to the diagnostics channel (SDKDIAG-07).

    Control flow is unchanged — the site still swallows, still logs locally,
    still returns the same value. It is simply no longer invisible to us.
    """
    try:
        from morse_ai._internal import diagnostics

        diagnostics.report(exc, fingerprint=fingerprint)
    except Exception:  # noqa: BLE001 — reporting must never break the swallow
        pass
