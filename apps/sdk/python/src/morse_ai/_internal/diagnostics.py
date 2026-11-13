"""SDK self-diagnostics — Morse-owned internal error reporting (MHQ-908).

Replaces the Sentry reporter. When the SDK absorbs an internal exception (per
the non-invasive guarantee in ``_internal.errors``), it reports here, and we
send it to Morse's own ``/api/v1/diagnostics/sdk`` rather than to a third party
the customer may not even have installed.

Two channels, both silent on a healthy install:

* **errors** — one absorbed exception, carrying a *Morse-scoped* stack. Frames
  inside this package go in full; every run of frames outside it collapses to a
  count. Frame locals are never captured, at any verbosity level.
* **counters** (SDKDIAG-09) — aggregate ``{reason, category, quantity}`` for
  telemetry we discarded. Adopted from Sentry's client reports, which is what
  that vendor chose to send *instead of* SDK stack traces. Counts carry no
  customer-data risk and answer "is the SDK delivering?" better than a
  traceback does.

**Transport is deliberately its own** (founder decision, 2026-08-03, Option B):
own bounded buffer, own daemon thread, own ``httpx.Client``, built lazily on the
first report and never before. Routing through the shared ``BatchTransport``
would have been a smaller diff, but that queue is bounded and tail-drops, so a
diagnostic storm would evict the customer's telemetry and log a warning per
eviction — our bug, narrated into their console. A healthy process emits nothing
here, so lazy construction means the common case pays nothing at all.

**Nothing here is private from the host.** Everything transmitted is also
emitted to the ``morse_ai`` logger, rate-limited, whether or not transmission
succeeds and even when the customer has opted out entirely (SDKDIAG-10). There
is no category of self-reported information Morse receives that the customer's
engineers cannot see.

**Never raises.** A failure anywhere in this module is logged at DEBUG and
dropped — never retried, never reported about itself (SDKDIAG-06). If reporting
an error could raise, the non-invasive guarantee would be a lie.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import re
import threading
import time
import traceback
from collections import defaultdict, deque
from typing import Any

from morse_ai._internal import env
from morse_ai._internal.redaction import _PATTERN_SOURCES
from morse_ai._version import __version__

logger = logging.getLogger("morse_ai._internal")

# --- Verbosity (SDKDIAG-05) -------------------------------------------------
#: Learned from the ``X-Morse-Diagnostics`` response header on traffic the SDK
#: is already sending. ``error`` until a header says otherwise — the same
#: default the server model carries, so behaviour before the first response
#: matches behaviour after it.
LEVEL_OFF = "off"
LEVEL_ERROR = "error"
LEVEL_DEBUG = "debug"
_VALID_LEVELS = frozenset({LEVEL_OFF, LEVEL_ERROR, LEVEL_DEBUG})

_level = LEVEL_ERROR
_level_lock = threading.Lock()

# --- Rate limits ------------------------------------------------------------
_RATE_WINDOW_SECONDS = 60.0
_RATE_LIMIT_PER_WINDOW = 100
#: ``debug`` relaxes the per-fingerprint budget; it never relaxes what may be
#: transmitted (SDKDIAG-02).
_RATE_LIMIT_PER_WINDOW_DEBUG = 1000

_rate_lock = threading.Lock()
_rate_buckets: dict[str, deque[float]] = defaultdict(deque)

#: Local emission is rate-limited separately so a looping internal error cannot
#: flood the host's log (SDKDIAG-10) even while transmission is being throttled.
_LOCAL_LOG_WINDOW_SECONDS = 60.0
_LOCAL_LOG_LIMIT_PER_WINDOW = 10
_local_log_buckets: dict[str, deque[float]] = defaultdict(deque)

# --- Payload caps (mirror apps/api/apps/diagnostics/schemas.py) -------------
MAX_STACK_ENTRIES = 64
MAX_MESSAGE_CHARS = 4096
MAX_BATCH_EVENTS = 32

#: Applied to every transmitted exception message regardless of the org's
#: redaction config — see ``_redacted_message`` (MHQ-973). Imported rather than
#: re-declared so there is exactly one definition to keep in parity with the
#: server and the TypeScript SDK.
_SECRET_ASSIGNMENT_RE = re.compile(_PATTERN_SOURCES["secret_assignment"], re.IGNORECASE)
_SECRET_ASSIGNMENT_REPLACEMENT = "[REDACTED:secret]"
_ALLOWED_EXTRA_KEYS = frozenset(
    {"adapter", "module", "function", "operation", "endpoint_host"}
)

# --- Sender -----------------------------------------------------------------
_DIAGNOSTICS_PATH = "/api/v1/diagnostics/sdk"
#: Bounded so a sustained outage cannot grow the SDK's footprint (SDKDIAG-08).
_MAX_PENDING = 100
_FLUSH_INTERVAL_SECONDS = 5.0
#: Circuit breaker: after this many consecutive failures we stop trying for a
#: cooldown, then allow a single probe. A dead endpoint therefore sees a
#: decaying request rate from each SDK instance rather than a constant one.
_BREAKER_THRESHOLD = 3
_BREAKER_COOLDOWN_SECONDS = 300.0
_HTTP_TIMEOUT_SECONDS = 5.0

_sender: _DiagnosticsSender | None = None
_sender_lock = threading.Lock()

#: Re-entrancy fence (SDKDIAG-06). Set while this module is on the stack, so an
#: exception raised *by* diagnostics can never produce another diagnostic.
_reporting = threading.local()


# ---------------------------------------------------------------------------
# Verbosity
# ---------------------------------------------------------------------------


def set_level(raw: str | None) -> None:
    """Record the level observed on an ``X-Morse-Diagnostics`` header.

    Unrecognised values are ignored rather than treated as ``off`` — a future
    server value must not silently mute an old SDK.
    """
    global _level
    if not raw:
        return
    value = str(raw).strip().lower()
    if value not in _VALID_LEVELS:
        return
    with _level_lock:
        _level = value


def get_level() -> str:
    with _level_lock:
        return _level


def _transmission_allowed() -> bool:
    """True when this process may put diagnostics on the wire.

    The customer's kill switch beats every server-side setting — that ordering
    is the point of having a client-side switch at all.
    """
    if env.is_telemetry_disabled() or env.is_disabled():
        return False
    return get_level() != LEVEL_OFF


# ---------------------------------------------------------------------------
# Morse-scoped stack (SDKDIAG-02)
# ---------------------------------------------------------------------------

#: Directory containing the installed ``morse_ai`` package. Frames whose file
#: lives under it are ours; everything else is the host's and gets elided.
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _is_morse_frame(filename: str) -> bool:
    try:
        return os.path.abspath(filename).startswith(_PACKAGE_ROOT)
    except Exception:  # noqa: BLE001 — a weird filename is not ours
        return False


def build_scoped_stack(error: BaseException) -> list[dict[str, Any]]:
    """Morse-scoped frames for ``error``.

    In-package frames carry ``{file, line, function}``; each consecutive run of
    host frames collapses to ``{"elided": <count>}``. Locals are never read —
    we use ``traceback.extract_tb``, which does not capture them, rather than
    walking frame objects and choosing to skip ``f_locals``. Not reading them
    is a stronger guarantee than reading and discarding.
    """
    frames: list[dict[str, Any]] = []
    try:
        elided = 0
        for frame in traceback.extract_tb(error.__traceback__):
            if _is_morse_frame(frame.filename):
                if elided:
                    frames.append({"elided": elided})
                    elided = 0
                frames.append(
                    {
                        # Relative to the package root: an absolute path leaks
                        # the host's directory layout (usernames, deploy paths).
                        "file": os.path.relpath(frame.filename, _PACKAGE_ROOT),
                        "line": frame.lineno or 0,
                        "function": frame.name,
                    }
                )
            else:
                elided += 1
        if elided:
            frames.append({"elided": elided})
    except Exception:  # noqa: BLE001 — a partial stack beats no diagnostic
        logger.debug("morse_ai: scoped-stack build failed", exc_info=True)

    if len(frames) > MAX_STACK_ENTRIES:
        # Keep the innermost frames — that is where the failure is.
        frames = frames[-MAX_STACK_ENTRIES:]
    return frames


def _redacted_message(error: BaseException) -> str:
    """Exception message, redacted and capped.

    An exception message is not automatically safe: a ``KeyError`` can name a
    customer key and a JSON error can echo content. It goes through the same
    ``RedactionConfig`` the telemetry path uses, and then through the
    secret-assignment pattern a second time, unconditionally.

    The second pass is the point (MHQ-973). ``secret_assignment`` is
    disableable, because an org may legitimately want raw ``password=`` values
    in *their own* spans. This channel is different: the published disclosure
    (SDKDIAG-02) promises Morse receives no customer data, and that promise must
    not be defeatable by someone's telemetry preference. Over-redacting our own
    error text costs us a little debuggability; under-redacting leaks a
    credential to us.
    """
    try:
        raw = str(error)
    except Exception:  # noqa: BLE001 — __str__ can itself raise
        return ""
    try:
        from morse_ai import tracing

        config = tracing._redaction_config
        if config is not None:
            redacted = config.redact_value(raw)
            raw = redacted if isinstance(redacted, str) else str(redacted)
        raw = _SECRET_ASSIGNMENT_RE.sub(_SECRET_ASSIGNMENT_REPLACEMENT, raw)
    except Exception:  # noqa: BLE001 — never send unredacted on a redactor fault
        return "<redaction unavailable>"
    return raw[:MAX_MESSAGE_CHARS]


def _scrub_extra(extra: dict[str, Any] | None) -> dict[str, str]:
    """Allow-list. Anything unrecognised is dropped, not sent and sorted later."""
    if not extra:
        return {}
    out: dict[str, str] = {}
    for key, value in extra.items():
        if key in _ALLOWED_EXTRA_KEYS and value is not None:
            out[key] = str(value)[:255]
    return out


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def _check_bucket(
    buckets: dict[str, deque[float]], key: str, limit: int, window: float
) -> bool:
    now = time.monotonic()
    with _rate_lock:
        bucket = buckets[key]
        cutoff = now - window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True


def _check_rate_limit(fingerprint: str) -> bool:
    limit = (
        _RATE_LIMIT_PER_WINDOW_DEBUG
        if get_level() == LEVEL_DEBUG
        else _RATE_LIMIT_PER_WINDOW
    )
    return _check_bucket(_rate_buckets, fingerprint, limit, _RATE_WINDOW_SECONDS)


# ---------------------------------------------------------------------------
# Delivery counters (SDKDIAG-09)
# ---------------------------------------------------------------------------

REASON_QUEUE_FULL = "queue_full"
#: Retryable failure, attempts made and used up. Do not use where no retry
#: happened — see REASON_SEND_FAILED.
REASON_RETRY_EXHAUSTED = "retry_exhausted"
REASON_QUOTA_REJECTED = "quota_rejected"
#: A genuine serialisation failure — json.dumps raised. NOT a stand-in for
#: "the server said no" (MHQ-971).
REASON_ENCODE_FAILED = "encode_failed"
REASON_SHUTDOWN_DROP = "shutdown_drop"
#: 401/403 — key revoked, expired, or missing a scope.
REASON_AUTH_REJECTED = "auth_rejected"
#: Other non-429 4xx — wrong URL, or a payload the server will never accept.
REASON_ENDPOINT_REJECTED = "endpoint_rejected"
#: A send failed and the batch was dropped without further attempts.
REASON_SEND_FAILED = "send_failed"
# No breaker-open reason: the customer-telemetry transport has no circuit
# breaker (it uses bounded retries -> REASON_RETRY_EXHAUSTED), and the breaker
# that does exist — this module's own — may not report on itself (SDKDIAG-06).
# A constant here would be unreachable by construction. See SDKDIAG-09.

_counters_lock = threading.Lock()
_counters: dict[tuple[str, str], int] = defaultdict(int)


#: Plain-language cause for the local warning, so a wrong or revoked key reads as
#: that and not as an HTTP/OpenTelemetry internal (offshore QA 2026-11-12).
#: Unknown reasons keep the generic text.
_REASON_HINTS = {
    REASON_AUTH_REJECTED: (
        "Morse rejected your API key (wrong, revoked or expired); check MORSE_API_KEY"
    ),
    REASON_QUOTA_REJECTED: (
        "Morse is rate limiting or over quota for this key; telemetry is being lost"
    ),
    REASON_ENDPOINT_REJECTED: (
        "Morse rejected the request; check the endpoint URL and SDK version"
    ),
    REASON_RETRY_EXHAUSTED: (
        "could not reach the Morse API after retrying; check your network and endpoint"
    ),
    REASON_SEND_FAILED: "could not reach the Morse API; telemetry is being lost",
}


def record_drop(reason: str, category: str, quantity: int = 1) -> None:
    """Count telemetry we discarded. Never raises, never sends immediately.

    Aggregated in memory and flushed with the sender's normal cadence — one
    report per interval regardless of how many drops it covers. A period with
    no drops produces no transmission at all.
    """
    try:
        if quantity <= 0:
            return
        with _counters_lock:
            _counters[(reason, category)] += quantity
        # SDKDIAG-10: delivery loss is what the customer's on-call most needs,
        # so it is WARNING locally even though it is only a counter on the wire.
        if _check_bucket(
            _local_log_buckets,
            f"drop:{reason}:{category}",
            _LOCAL_LOG_LIMIT_PER_WINDOW,
            _LOCAL_LOG_WINDOW_SECONDS,
        ):
            logger.warning(
                "morse_ai: dropped %d %s (%s) — %s",
                quantity,
                category,
                reason,
                _REASON_HINTS.get(reason, "telemetry is being lost"),
            )
        # MHQ-1030: arm the sender so the count goes out on the normal 5s
        # cadence. Counters alone used to build nothing, so a process that lost
        # telemetry but never raised reported it only at ``shutdown()`` — i.e.
        # never, for a long-lived server or one that dies on SIGKILL. A process
        # actively dropping data is not the healthy install SDKDIAG-01's
        # allocate-nothing rule is about.
        if _transmission_allowed():
            _get_sender()
    except Exception:  # noqa: BLE001
        logger.debug("morse_ai: drop-counter record failed", exc_info=True)


def _drain_counters() -> list[dict[str, Any]]:
    with _counters_lock:
        if not _counters:
            return []
        drained = [
            {"reason": reason, "category": category, "quantity": quantity}
            for (reason, category), quantity in _counters.items()
        ]
        _counters.clear()
    return drained


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(
    error: BaseException,
    *,
    fingerprint: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Report an absorbed internal SDK error. Never raises.

    Signature is unchanged from the Sentry reporter it replaces, so every
    ``absorb_errors`` call site keeps working untouched.
    """
    if getattr(_reporting, "active", False):
        # SDKDIAG-06: we are already inside a report. A failure here must not
        # spawn another one.
        return
    _reporting.active = True
    try:
        # SDKDIAG-10: local FIRST, and gated only by its own limiter. The
        # transmission rate limit must not decide what the host gets to see —
        # that would make our budget the customer's visibility ceiling.
        if _check_bucket(
            _local_log_buckets,
            fingerprint,
            _LOCAL_LOG_LIMIT_PER_WINDOW,
            _LOCAL_LOG_WINDOW_SECONDS,
        ):
            logger.log(
                logging.ERROR if env.is_debug() else logging.WARNING,
                "morse_ai internal error [%s]: %s: %s",
                fingerprint,
                type(error).__name__,
                error,
                exc_info=env.is_debug(),
            )

        if not _transmission_allowed():
            return

        if not _check_rate_limit(fingerprint):
            # Transmission budget only — the local log above already happened.
            return

        event = {
            "kind": "error",
            "fingerprint": fingerprint[:255],
            "error_type": type(error).__name__[:255],
            "message": _redacted_message(error),
            "stack": build_scoped_stack(error),
            "sdk_language": "python",
            "sdk_version": __version__,
            "runtime_version": _runtime_version(),
            "platform": _platform(),
            "extra": _scrub_extra(extra),
        }
        _enqueue(event)
    except Exception:  # noqa: BLE001 — absolute last-resort guard
        logger.debug("morse_ai: diagnostics report failed", exc_info=True)
    finally:
        _reporting.active = False


def _runtime_version() -> str:
    try:
        import platform as _p

        return _p.python_version()
    except Exception:  # noqa: BLE001
        return ""


def _platform() -> str:
    try:
        import platform as _p

        return f"{_p.system()}-{_p.machine()}"[:128]
    except Exception:  # noqa: BLE001
        return ""


def _enqueue(event: dict[str, Any]) -> None:
    sender = _get_sender()
    if sender is not None:
        sender.submit(event)


def _get_sender() -> _DiagnosticsSender | None:
    """Build the sender on first use. Returns None when there is nowhere to send.

    Option B's lazy half: a process that never reports an internal error never
    creates a thread, a socket, or a queue.
    """
    global _sender
    if _sender is not None:
        return _sender
    with _sender_lock:
        if _sender is not None:
            return _sender
        target = _resolve_target()
        if target is None:
            return None
        api_key, url = target
        try:
            _sender = _DiagnosticsSender(url, api_key)
        except Exception:  # noqa: BLE001
            logger.debug("morse_ai: diagnostics sender init failed", exc_info=True)
            return None
        return _sender


def _resolve_target() -> tuple[str, str] | None:
    """``(api_key, url)`` for the diagnostics endpoint, or None.

    Prefers the live client's configuration so we follow ``init(endpoint=...)``;
    falls back to the environment. Without an API key there is no authenticated
    destination, and an unattributable diagnostic is one we do not send —
    those stay local-log-only, which is the accepted loss recorded in the spec.
    """
    api_key = None
    base = None
    try:
        import morse_ai

        client = getattr(morse_ai, "_client", None)
        if client is not None:
            api_key = getattr(client, "_api_key", None)
            base = getattr(client, "endpoint", None)
    except Exception:  # noqa: BLE001
        pass

    api_key = api_key or env.get_api_key()
    if not api_key:
        return None

    base = base or env.get_endpoint() or ""
    url = _derive_diagnostics_url(base)
    if not url:
        return None
    return api_key, url


def _derive_diagnostics_url(traces_endpoint: str) -> str | None:
    """Swap the path of a known ingest URL for the diagnostics path."""
    try:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(traces_endpoint)
        if not parts.scheme or not parts.netloc:
            return None
        return urlunsplit((parts.scheme, parts.netloc, _DIAGNOSTICS_PATH, "", ""))
    except Exception:  # noqa: BLE001
        return None


class _DiagnosticsSender:
    """Bounded buffer + daemon thread + client, exclusively for diagnostics.

    Separate from ``BatchTransport`` on purpose: diagnostics must never compete
    with customer telemetry for queue slots, and this component's circuit
    breaker must be able to open without affecting trace delivery at all.
    """

    def __init__(self, url: str, api_key: str) -> None:
        import httpx

        self._url = url
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=_MAX_PENDING)
        self._closed = False
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        #: Interruptible sleep. Without it ``shutdown()`` cannot reap the
        #: worker — it would sit out the rest of its 5s nap first, so a
        #: shut-down sender kept a live thread behind it.
        self._wake = threading.Event()
        self._client = httpx.Client(
            timeout=_HTTP_TIMEOUT_SECONDS,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"morse_ai-python/{__version__}",
            },
        )
        self._thread = threading.Thread(
            target=self._run, name="morse_ai-diagnostics", daemon=True
        )
        self._thread.start()

    def submit(self, event: dict[str, Any]) -> None:
        """Enqueue, dropping the OLDEST on overflow.

        Oldest-first because during an incident the newest diagnostics describe
        what is happening now; a full buffer of stale ones is worth less.
        """
        try:
            while True:
                try:
                    self._queue.put_nowait(event)
                    return
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        return
        except Exception:  # noqa: BLE001
            logger.debug("morse_ai: diagnostics submit failed", exc_info=True)

    def _breaker_blocks(self) -> bool:
        return bool(
            self._breaker_open_until and time.monotonic() < self._breaker_open_until
        )

    def _record_result(self, ok: bool) -> None:
        if ok:
            self._consecutive_failures = 0
            self._breaker_open_until = 0.0
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            # Open the breaker and let exactly one probe through when it lapses
            # (the counter is not reset, so a failed probe re-opens it).
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECONDS

    def _run(self) -> None:
        while not self._closed:
            self._wake.wait(_FLUSH_INTERVAL_SECONDS)
            if self._closed:
                # shutdown() drains; a second flush here would race it.
                return
            try:
                self._flush_once()
            except Exception:  # noqa: BLE001 — the loop must outlive any batch
                logger.debug("morse_ai: diagnostics flush failed", exc_info=True)

    def _flush_once(self) -> None:
        batch: list[dict[str, Any]] = []
        while len(batch) < MAX_BATCH_EVENTS:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break

        counters = _drain_counters()
        if counters:
            batch.append(
                {
                    "kind": "counters",
                    "fingerprint": "morse_ai.transport.delivery",
                    "counters": counters,
                    "sdk_language": "python",
                    "sdk_version": __version__,
                }
            )

        if not batch:
            # SDKDIAG-01/09: a healthy interval transmits nothing.
            return

        if self._breaker_blocks() or not _transmission_allowed():
            # Deliberately not re-queued: during an outage, holding events for a
            # 5-minute cooldown would fill the buffer with stale reports and
            # push out the fresh ones that describe the ongoing incident.
            return

        self._post(batch)

    def _post(self, batch: list[dict[str, Any]]) -> None:
        """One attempt. No retry — see SDKDIAG-06.

        A retry loop here would amplify exactly the situation we are reporting
        about, and a lost diagnostic is a cost we accept; a lost customer span
        is not, which is why the two paths have different policies.
        """
        import json

        try:
            body = json.dumps({"events": batch}).encode("utf-8")
        except Exception:  # noqa: BLE001
            logger.debug("morse_ai: diagnostics encode failed", exc_info=True)
            return
        try:
            response = self._client.post(self._url, content=body)
        except Exception:  # noqa: BLE001 — network fault; breaker handles it
            self._record_result(False)
            logger.debug("morse_ai: diagnostics send failed", exc_info=True)
            return
        with contextlib.suppress(Exception):
            set_level(response.headers.get("X-Morse-Diagnostics"))
        self._record_result(response.status_code < 400)

    def shutdown(self) -> None:
        # MHQ-946: flush BEFORE closing the client, or the drain has no
        # transport to drain through. `_closed` is set first so the daemon
        # thread stops looping; a concurrent wake is harmless because the
        # queue is thread-safe and would only split the batch.
        self._closed = True
        self._wake.set()
        with contextlib.suppress(Exception):
            # Bounded: a join that waits out an in-flight POST is worth a
            # second at exit; one that waits out a hung socket is not.
            self._thread.join(timeout=1.0)
        with contextlib.suppress(Exception):
            self._flush_once()
        with contextlib.suppress(Exception):
            self._client.close()


_exit_drain_registered = False
_exit_drain_lock = threading.Lock()


def register_exit_drain() -> None:
    """Drain diagnostics at interpreter exit. Idempotent. (MHQ-1054)

    ``drain_for_shutdown`` was written for exactly the process shape that loses
    diagnostics — one that finishes inside the sender's 5-second interval — and
    then wired only to ``morse_ai.shutdown()``, which nothing calls. The SDK's
    whole setup story is zero-code-change, so customer code does not call it;
    ``transport.py`` and ``otel_provider.py`` each register their OWN shutdown,
    so traces flushed at exit while diagnostics did not. Live prod runs hit two
    export timeouts and landed 0 rows in ``sdk_diagnostic_events``.

    **Called at init, not at first report, and the timing is the point.**
    ``atexit`` runs LIFO, and the single most likely moment to record a drop is
    the exporter's final flush during ITS atexit hook. Registering here — before
    the OTel engine configures itself and registers — puts this drain last,
    which is the same ordering ``morse_ai.shutdown()`` establishes deliberately.
    Registering lazily on the first drop would invert it and lose precisely the
    drops that matter most.

    This costs a healthy process one function reference: with no sender and no
    counters, ``drain_for_shutdown`` returns before allocating anything, so
    SDKDIAG-01's "a healthy install allocates nothing" still holds in substance.
    """
    global _exit_drain_registered
    with _exit_drain_lock:
        if _exit_drain_registered:
            return
        _exit_drain_registered = True
    try:
        import atexit  # noqa: PLC0415

        atexit.register(drain_for_shutdown)
    except Exception:  # noqa: BLE001 — never break init
        logger.debug("morse_ai: diagnostics exit-drain registration failed")


def drain_for_shutdown() -> None:
    """Flush buffered diagnostics synchronously. Never raises. (MHQ-946)

    The sender's thread is a daemon on a 5-second interval, so a process that
    finishes inside that window dies with its buffer unsent and nothing else
    drains it — every script, CLI, serverless invocation, short-lived Celery
    task, and every driver in the adapter matrix. This is the Python half of
    MHQ-927, which fixed only TypeScript; the symptom is identical and equally
    invisible, because the events are still written to the local logger.

    Called from ``morse_ai.shutdown()`` AFTER the telemetry transport closes:
    that close can itself report an internal error or record delivery-loss
    counters, and this is the last chance to send them.
    """
    global _sender
    try:
        sender = _sender
        if sender is None:
            # Nothing was ever reported. Counters alone never build a sender
            # (``record_drop`` only touches the counter map), so a process that
            # dropped telemetry without an internal error still has something
            # to say — build one now rather than lose the only signal that
            # delivery failed. A genuinely healthy process has neither and
            # still allocates nothing, which is Option B's whole point.
            with _counters_lock:
                if not _counters:
                    return
            sender = _get_sender()
            if sender is None:
                return
        sender.shutdown()
        _sender = None
    except Exception:  # noqa: BLE001 — shutdown must never raise at the host
        logger.debug("morse_ai: diagnostics drain failed", exc_info=True)


def _reset_for_testing() -> None:
    """Reset module state for tests. Private."""
    global _sender, _level
    with _rate_lock:
        _rate_buckets.clear()
        _local_log_buckets.clear()
    with _counters_lock:
        _counters.clear()
    with _sender_lock:
        if _sender is not None:
            _sender.shutdown()
        _sender = None
    with _level_lock:
        _level = LEVEL_ERROR
    _reporting.active = False
