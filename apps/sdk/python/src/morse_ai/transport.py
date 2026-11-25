"""Background HTTP transport for batched telemetry delivery."""

import atexit
import contextlib
import json
import logging
import random
import threading
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from morse_ai._internal import diagnostics
from morse_ai._internal.json_default import json_default
from morse_ai._internal.redaction import DISABLED_CONFIG, RedactionConfig
from morse_ai._version import __version__
from morse_ai.queue import SHUTDOWN, TelemetryQueue

logger = logging.getLogger("morse_ai")

# --- Retry/backoff parameters (TQE-04 / MHQ-676) -----------------------------
#
# Contract (ingestion-hotpath-hardening proposal, SDK section): "3 attempts,
# jittered exponential backoff, 4xx never retried except 429, Retry-After
# honored." Module-level so tests can shrink the delays without reaching into
# instance state.
_MAX_SEND_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 10.0
#: ``Retry-After`` values above this are clamped — a background flush thread
#: must never commit to sleeping minutes on one batch.
_RETRY_AFTER_CAP_SECONDS = 30.0
#: Granularity of the interruptible retry sleep — shutdown() aborts a pending
#: retry within one slice.
_SLEEP_SLICE_SECONDS = 0.05


def _backoff_delay(attempt: int) -> float:
    """Jittered exponential backoff delay for retry ``attempt`` (1-based)."""
    base = min(_BACKOFF_MAX_SECONDS, _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
    return base * random.uniform(0.5, 1.5)


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header (seconds form only), capped.

    HTTP-date form is not worth supporting here — Morse's own 429s always use
    the seconds form. Unparseable values return None (fall back to backoff).
    """
    if not value:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return min(seconds, _RETRY_AFTER_CAP_SECONDS)


def _derive_logs_endpoint(traces_endpoint: str) -> str:
    """Derive the logs ingest URL from the traces batch URL.

    Phase 12.5 Tier 1.1: the logging.Handler shim shares the traces transport
    instance, but logs land on a different path (``/api/v1/logs/ingest``).
    Keep the host/scheme; replace the path. Same shape as ``_Client.log``.
    """
    parts = urlsplit(traces_endpoint)
    return urlunsplit((parts.scheme, parts.netloc, "/api/v1/logs/ingest", "", ""))


# MHQ-1936: moved to ``_internal/json_default.py`` so ``otel_provider`` uses
# the same serializer; the old private name stays importable.
_json_default = json_default


class BatchTransport:
    """Sends telemetry batches to the Morse backend on a background thread.

    Architecture: daemon thread + queue.Queue + httpx.Client (sync).
    This pattern works regardless of whether the caller is sync or async,
    and queue.put_nowait() returns in <0.1ms (the performance requirement).

    Wire format routing:
    - Events with "traces" key → sent as {"traces": [...]} to the traces endpoint
    - Events with "logs" key → sent as {"logs": [...]} to ``/api/v1/logs/ingest``
      (Phase 12.5 Tier 1.1 — MorseHandler / structlog processor path).
    - Events without either key → sent as {"runs": [...]} to the traces endpoint.
      NOTE: the backend has no /runs ingest route; this path will 422. All SDK
      public APIs (record/trace_agent/run/span) emit the "traces" key. This
      branch is kept only for forward-compat with any direct transport.track()
      callers that pre-date the v2 wire format.

    PII redaction (Phase 6f Wave B G-02):
    - Every event passes through ``redaction_config.redact_value`` BEFORE it
      lands on the in-memory queue. This is the SDK's single canonical
      pre-transport hook point — span attributes, run records, trace
      payloads all funnel through ``track()``. Non-string leaves
      (numbers, booleans, ``None``) are passed through; the depth bound
      lives inside ``RedactionConfig``.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        *,
        flush_interval: float = 2.0,
        max_batch_size: int = 100,
        max_queue_size: int = 10_000,
        timeout: float = 30.0,
        redaction_config: RedactionConfig | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._flush_interval = flush_interval
        self._max_batch_size = max_batch_size
        self._timeout = timeout
        self._redaction_config = redaction_config or DISABLED_CONFIG

        self._queue = TelemetryQueue(max_size=max_queue_size)
        self._lock = threading.Lock()
        self._closed = False
        self._client: httpx.Client | None = None

        # TQE-04 (MHQ-676): latest quota/rate-limit state observed on a 429
        # response, exposed via ``quota_state()`` / ``morse_ai.quota_state()``.
        # ``_quota_warned_keys`` implements warn-once-per-billing-period.
        self._quota_state: dict[str, Any] | None = None
        self._quota_lock = threading.Lock()
        self._quota_warned_keys: set[str] = set()

        self._thread = threading.Thread(
            target=self._run,
            name="morse_ai-flush",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self.shutdown)

    @property
    def redaction_config(self) -> RedactionConfig:
        """The active redaction config (set at construction)."""
        return self._redaction_config

    def track(self, event: dict[str, Any]) -> bool:
        """Enqueue an event. Non-blocking, <0.1ms. Returns False if dropped.

        Applies SDK-side PII redaction to every string leaf in ``event``
        before it lands on the queue.
        """
        if self._closed:
            return False
        try:
            redacted = self._redaction_config.redact_value(event)
            return self._queue.put(redacted)
        except Exception as exc:  # noqa: BLE001
            logger.debug("morse_ai: track() swallowed error: %s", exc)
            return False

    def _run(self) -> None:
        """Background loop: drain queue and flush in batches."""
        self._client = httpx.Client(
            timeout=self._timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"morse_ai-python/{__version__}",
            },
        )
        try:
            while True:
                batch = self._queue.drain(self._max_batch_size)

                has_shutdown = SHUTDOWN in batch
                clean_batch = [e for e in batch if e is not SHUTDOWN]

                if clean_batch:
                    self._send_batch(clean_batch)

                if has_shutdown:
                    break

                # Sleep in small increments for responsive shutdown
                deadline = time.monotonic() + self._flush_interval
                while time.monotonic() < deadline:
                    if not self._queue.empty:
                        break
                    time.sleep(0.05)
        finally:
            if self._client:
                self._client.close()

    def _send_batch(self, batch: list[dict[str, Any]]) -> None:
        """POST a batch to the backend. Errors are logged, never raised.

        Routing logic:
        - Events that have a "traces" key are v2 trace payloads. They are
          collected and sent as {"traces": [<all trace entries>]} to the
          configured traces endpoint.
        - Events that have a "logs" key are Phase 12.5 Tier 1.1 log records
          (from ``MorseHandler`` / structlog processor). Collected and
          sent as {"logs": [<all log entries>]} to ``/api/v1/logs/ingest``
          on the same host.
        - Events without either key are v1 RunRecord dicts. They are
          collected and sent as {"runs": [<all run dicts>]}.

        This allows the transport to handle a mix of v1 and v2 events in
        the same drain cycle without data loss.
        """
        if not self._client:
            return

        # Partition events by type
        trace_entries: list[dict[str, Any]] = []
        log_entries: list[dict[str, Any]] = []
        run_entries: list[dict[str, Any]] = []

        for event in batch:
            if "traces" in event:
                # v2: event = {"traces": [{"trace": {...}, "spans": [...]}]}
                trace_entries.extend(event["traces"])
            elif "logs" in event:
                # Phase 12.5 Tier 1.1: event = {"logs": [<log record>, ...]}
                log_entries.extend(event["logs"])
            else:
                # v1 legacy: event is a flat RunRecord dict
                run_entries.append(event)

        if trace_entries:
            self._post({"traces": trace_entries})

        if log_entries:
            self._post(
                {"logs": log_entries},
                url=_derive_logs_endpoint(self._endpoint),
            )

        if run_entries:
            self._post({"runs": run_entries})

    def _post(self, payload: dict[str, Any], *, url: str | None = None) -> None:
        """POST a payload to the configured endpoint (or override ``url``).

        TQE-04 (MHQ-676) retry contract:
        - 429 → parse + store quota state, honor ``Retry-After`` (capped),
          retry with jittered exponential backoff on top, up to
          ``_MAX_SEND_ATTEMPTS`` total attempts. The final failure surfaces
          quota state (warn-once-per-billing-period) before the batch drops.
        - 5xx / network errors → same jittered-backoff retry.
        - Other 4xx → never retried; the payload is bad and won't get better.
        - Shutdown in progress (``self._closed``) → give up retrying so
          ``shutdown()``'s 30s join deadline is respected. Retry sleeps are
          interruptible for the same reason.

        Runs only on the background flush thread; errors are logged, never
        raised into caller code.
        """
        if not self._client:
            return
        target = url if url is not None else self._endpoint
        # SDKDIAG-09: what a drop on this path costs the customer, for the
        # delivery counters. Derived from the payload so every early return
        # below reports the same quantity.
        drop_category = next(
            (k for k in ("traces", "logs", "runs") if k in payload), "trace"
        ).rstrip("s")
        drop_count = sum(len(v) for v in payload.values() if isinstance(v, list))
        # AGE-240: pre-serialize with a tolerant ``default=`` so non-JSON
        # values (LangChain HumanMessage / AIMessage, datetime, Decimal,
        # UUID, custom classes leaking through SDK ``input_data`` /
        # ``output_data``) get coerced to their string / dict form rather
        # than dropping the entire batch with ``TypeError``.
        try:
            body = json.dumps(payload, default=_json_default).encode("utf-8")
        except Exception as exc:  # noqa: BLE001 — never let a payload kill the loop
            diagnostics.record_drop(
                diagnostics.REASON_ENCODE_FAILED, drop_category, drop_count
            )
            logger.warning("morse_ai: batch send payload-encode error: %s", exc)
            return

        for attempt in range(1, _MAX_SEND_ATTEMPTS + 1):
            if attempt > 1 and self._closed:
                # Shutdown began while we were backing off — the first
                # attempt already ran; don't fire more requests into a
                # closing process.
                logger.debug("morse_ai: shutdown in progress — abandoning retry")
                return
            try:
                resp = self._client.post(
                    target,
                    content=body,
                    headers={"Content-Type": "application/json"},
                )
            except (httpx.HTTPError, ConnectionError, OSError) as exc:
                # Transient network failure — retryable.
                if attempt >= _MAX_SEND_ATTEMPTS or self._closed:
                    diagnostics.record_drop(
                        diagnostics.REASON_SHUTDOWN_DROP
                        if self._closed
                        else diagnostics.REASON_RETRY_EXHAUSTED,
                        drop_category,
                        drop_count,
                    )
                    logger.warning(
                        "morse_ai: batch send error after %d attempt(s), "
                        "dropping batch: %s",
                        attempt,
                        exc,
                    )
                    return
                logger.debug(
                    "morse_ai: batch send error (attempt %d/%d), retrying: %s",
                    attempt,
                    _MAX_SEND_ATTEMPTS,
                    exc,
                )
                self._sleep_interruptible(_backoff_delay(attempt))
                continue
            except Exception as exc:  # noqa: BLE001 — never let a send kill the loop
                diagnostics.record_drop(
                    diagnostics.REASON_RETRY_EXHAUSTED, drop_category, drop_count
                )
                logger.warning("morse_ai: batch send unexpected error: %s", exc)
                return

            # SDKDIAG-05: the server pushes the SDK's self-reporting verbosity
            # on responses we are already receiving.
            with contextlib.suppress(Exception):  # advisory header
                diagnostics.set_level(resp.headers.get("X-Morse-Diagnostics"))

            status = resp.status_code
            if status < 400:
                return

            if status == 429:
                self._record_quota_state(resp)
                if attempt >= _MAX_SEND_ATTEMPTS or self._closed:
                    diagnostics.record_drop(
                        diagnostics.REASON_QUOTA_REJECTED,
                        drop_category,
                        drop_count,
                    )
                    self._surface_quota_drop(attempt)
                    return
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                delay = (retry_after or 0.0) + _backoff_delay(attempt)
                logger.debug(
                    "morse_ai: 429 from ingest (attempt %d/%d), retrying in %.2fs",
                    attempt,
                    _MAX_SEND_ATTEMPTS,
                    delay,
                )
                self._sleep_interruptible(delay)
                continue

            if status < 500:
                # Non-429 4xx: retrying can't help, but WHY it can't differs and
                # the remedies are unrelated. MHQ-971: this branch used to report
                # every one of these as `encode_failed`, so a revoked key and a
                # misconfigured endpoint both reached us looking like a payload
                # serialisation bug — pointing away from the actual fault.
                diagnostics.record_drop(
                    diagnostics.REASON_AUTH_REJECTED
                    if status in (401, 403)
                    else diagnostics.REASON_ENDPOINT_REJECTED,
                    drop_category,
                    drop_count,
                )
                logger.warning(
                    "morse_ai: batch send failed status=%d body=%s",
                    status,
                    resp.text[:200],
                )
                return

            # 5xx — retryable server error.
            if attempt >= _MAX_SEND_ATTEMPTS or self._closed:
                diagnostics.record_drop(
                    diagnostics.REASON_RETRY_EXHAUSTED, drop_category, drop_count
                )
                logger.warning(
                    "morse_ai: batch send failed after %d attempt(s) status=%d body=%s",
                    attempt,
                    status,
                    resp.text[:200],
                )
                return
            logger.debug(
                "morse_ai: batch send failed status=%d (attempt %d/%d), retrying",
                status,
                attempt,
                _MAX_SEND_ATTEMPTS,
            )
            self._sleep_interruptible(_backoff_delay(attempt))

    def _sleep_interruptible(self, seconds: float) -> None:
        """Sleep in small slices, returning early when shutdown begins.

        Keeps a pending retry from stalling ``shutdown()``'s thread join —
        the flush thread wakes within one slice of ``_closed`` flipping.
        """
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._closed:
                return
            time.sleep(_SLEEP_SLICE_SECONDS)

    # --- Quota state (TQE-04 / MHQ-676) -----------------------------------

    def _record_quota_state(self, resp: httpx.Response) -> None:
        """Parse a 429 response into ``self._quota_state``. Never raises.

        Two known body shapes, both defensively parsed:
        - ``event_limit_exceeded`` (traces controllers) — flat dict with
          ``current_count`` / ``limit_count`` / ``plan_id`` / ``period_start``
          / ``upgrade_url``.
        - ``RATE_LIMIT_INGEST`` (ingest rate-limit middleware) — nested
          ``{"error": {"code": ..., "details": {"limit": ..., "plan": ...}}}``.
        Anything else is stored as ``kind="unknown"`` so the state surface
        still reflects that ingest is throttled.
        """
        try:
            state: dict[str, Any] = {
                "kind": "unknown",
                "status": 429,
                "received_at": time.time(),
            }
            try:
                data = resp.json()
            except Exception:  # noqa: BLE001 — non-JSON 429 body
                data = None
            if isinstance(data, dict):
                err = data.get("error")
                if err == "event_limit_exceeded":
                    state.update(
                        kind="event_limit",
                        message=data.get("message"),
                        current_count=data.get("current_count"),
                        limit_count=data.get("limit_count"),
                        plan_id=data.get("plan_id"),
                        period_start=data.get("period_start"),
                        upgrade_url=data.get("upgrade_url"),
                        sample_rate_url=data.get("sample_rate_url"),
                    )
                elif isinstance(err, dict) and err.get("code") == "RATE_LIMIT_INGEST":
                    details = err.get("details") or {}
                    state.update(
                        kind="rate_limit",
                        message=err.get("message"),
                        limit_count=details.get("limit")
                        if isinstance(details, dict)
                        else None,
                        plan_id=details.get("plan")
                        if isinstance(details, dict)
                        else None,
                    )
            # TQE-05 headers (emitted by the server on every ingest response).
            for header, key in (
                ("X-Morse-Quota-Limit", "quota_limit"),
                ("X-Morse-Quota-Remaining", "quota_remaining"),
                ("X-Morse-Quota-Reset", "quota_reset"),
            ):
                raw = resp.headers.get(header)
                if raw is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        state[key] = int(raw)
            retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
            if retry_after is not None:
                state["retry_after_seconds"] = retry_after
            with self._quota_lock:
                self._quota_state = state
        except Exception as exc:  # noqa: BLE001 — state capture must never break sends
            logger.debug("morse_ai: quota-state capture swallowed error: %s", exc)

    def _surface_quota_drop(self, attempts: int) -> None:
        """Surface a finally-failed 429 batch before it drops. Never raises.

        WARNING once per billing period (keyed on ``period_start``, falling
        back to the 429 kind when the body carried none); every subsequent
        occurrence in the same period logs at DEBUG so a quota-exhausted
        month doesn't flood the host app's logs.
        """
        try:
            with self._quota_lock:
                state = dict(self._quota_state) if self._quota_state else {}
                warn_key = "{}:{}".format(
                    state.get("kind", "unknown"), state.get("period_start") or ""
                )
                first_this_period = warn_key not in self._quota_warned_keys
                if first_this_period:
                    self._quota_warned_keys.add(warn_key)
            message = (
                "Morse ingest quota exceeded — telemetry is being dropped. "
                "current=%s limit=%s upgrade: %s (batch dropped after %d attempts)"
            )
            args = (
                state.get("current_count", "?"),
                state.get("limit_count", "?"),
                state.get("upgrade_url", "https://app.morseai.dev/settings/billing"),
                attempts,
            )
            if first_this_period:
                logger.warning(message, *args)
            else:
                logger.debug(message, *args)
        except Exception as exc:  # noqa: BLE001
            logger.debug("morse_ai: quota-drop surfacing swallowed error: %s", exc)

    def quota_state(self) -> dict[str, Any] | None:
        """Latest quota/rate-limit state observed on a 429, or None.

        Returns a copy — callers can't mutate transport state.
        """
        with self._quota_lock:
            return dict(self._quota_state) if self._quota_state else None

    def flush(self) -> None:
        """Block until the queue is drained."""
        deadline = time.monotonic() + 30.0
        while not self._queue.empty and time.monotonic() < deadline:
            time.sleep(0.05)

    def shutdown(self) -> None:
        """Graceful shutdown: flush remaining events, stop background thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True

        self._queue.signal_shutdown()
        self._thread.join(timeout=30.0)
