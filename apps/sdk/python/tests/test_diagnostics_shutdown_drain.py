"""MHQ-946: ``morse_ai.shutdown()`` must drain the diagnostics buffer.

Diagnostics keeps its own buffer and its own sender thread, deliberately
separate from the telemetry transport. That thread is a **daemon** on a
5-second interval, so any process that finishes inside that window dies with
its buffer unsent — and nothing else drains it.

That is every short-lived process: a script, a CLI, a serverless invocation, a
Celery task that exits promptly, and every driver in the Python adapter matrix.
It is why prod ``sdk_diagnostic_events`` had effectively no Python rows: the
request was never issued before exit. The events ARE written to the local
logger either way, which is what made it invisible.

This is the Python half of MHQ-927, which fixed TypeScript only. The TS twin of
this file is ``apps/sdk/typescript/tests/diagnostics-shutdown-flush.test.ts``.

Asserted with **no sleep anywhere** — if the flush only happens on the
interval, these fail.
"""

import pytest

from morse_ai._internal import diagnostics


@pytest.fixture(autouse=True)
def _reset():
    diagnostics._reset_for_testing()
    yield
    diagnostics._reset_for_testing()


class _RecordingClient:
    """Stands in for the sender's httpx client, capturing posts."""

    def __init__(self):
        self.posts: list[bytes] = []
        self.closed = False

    def post(self, url, content=None, **kwargs):
        self.posts.append(content)

        class _Resp:
            status_code = 200
            headers: dict[str, str] = {}

        return _Resp()

    def close(self):
        self.closed = True


@pytest.fixture
def _sender_with_recording_client(monkeypatch):
    """Build a real sender, then swap in a recording client.

    The sender is otherwise untouched, so this exercises the real queue,
    the real batching, and the real drain path.
    """
    monkeypatch.setenv("MORSE_API_KEY", "mhq_test")  # pragma: allowlist secret
    monkeypatch.setenv("MORSE_ENDPOINT", "https://api.example.test")
    sender = diagnostics._get_sender()
    assert sender is not None, "sender should build once a target resolves"
    client = _RecordingClient()
    sender._client = client
    return sender, client


@pytest.mark.req("SDKDIAG-01")
def test_shutdown_drains_a_buffered_error_without_waiting_for_the_interval(
    _sender_with_recording_client,
):
    sender, client = _sender_with_recording_client

    try:
        raise RuntimeError("internal boom")
    except RuntimeError as exc:
        diagnostics.report(exc, fingerprint="shutdown-drain-probe")

    assert client.posts == [], "nothing should have been sent before shutdown"

    # No sleep: _FLUSH_INTERVAL_SECONDS is 5, so anything sent here came from
    # the drain and not from the sender thread.
    diagnostics.drain_for_shutdown()

    assert len(client.posts) == 1, "the buffered diagnostic was not drained"
    assert b"shutdown-drain-probe" in client.posts[0]
    assert client.closed, "the drain should close the client behind it"


@pytest.mark.req("SDKDIAG-09")
def test_shutdown_drains_delivery_counters_with_no_error_reported(monkeypatch):
    """A process can drop telemetry without ever raising internally.

    The counters are then the ONLY signal that delivery failed, which is
    exactly what an on-call needs. Losing them is the same bug wearing a hat.

    MHQ-1030 moved the sender's construction forward: a drop now arms it, so
    the count goes out on the normal cadence rather than waiting for an exit
    that a long-lived server never reaches. The drain still has to deliver
    whatever is left, which is what this pins.
    """
    monkeypatch.setenv("MORSE_API_KEY", "mhq_test")  # pragma: allowlist secret
    monkeypatch.setenv("MORSE_ENDPOINT", "https://api.example.test")

    posted: list[bytes] = []
    real_init = diagnostics._DiagnosticsSender.__init__

    def _init(self, url, api_key):
        real_init(self, url, api_key)
        self._client = _Recorder(posted)

    monkeypatch.setattr(diagnostics._DiagnosticsSender, "__init__", _init)

    diagnostics.record_drop("retry_exhausted", "trace", 3)
    assert diagnostics._sender is not None, "a drop must arm the sender (MHQ-1030)"
    assert posted == [], "nothing is sent inline — the flush cadence carries it"

    diagnostics.drain_for_shutdown()

    assert len(posted) == 1, "delivery counters were lost at shutdown"
    assert b"retry_exhausted" in posted[0]


class _Recorder(_RecordingClient):
    def __init__(self, sink):
        super().__init__()
        self._sink = sink

    def post(self, url, content=None, **kwargs):
        self._sink.append(content)
        return super().post(url, content=content, **kwargs)


@pytest.mark.req("SDKDIAG-06")
def test_shutdown_is_safe_when_nothing_was_reported():
    """A healthy process allocates nothing and the drain is a no-op."""
    assert diagnostics._sender is None

    diagnostics.drain_for_shutdown()  # must not raise, must not build a sender

    assert diagnostics._sender is None


@pytest.mark.req("SDKDIAG-06")
def test_drain_never_raises_even_when_the_send_explodes(
    _sender_with_recording_client,
):
    """Shutdown must never become the thing that breaks the host."""
    sender, client = _sender_with_recording_client

    def _boom(*args, **kwargs):
        raise OSError("socket is gone")

    client.post = _boom

    try:
        raise RuntimeError("internal boom")
    except RuntimeError as exc:
        diagnostics.report(exc, fingerprint="drain-explodes")

    diagnostics.drain_for_shutdown()  # must swallow


@pytest.mark.req("SDKDIAG-01")
def test_public_shutdown_drains_diagnostics(monkeypatch):
    """The wiring itself: morse_ai.shutdown(), not the internal helper."""
    import morse_ai

    monkeypatch.setenv("MORSE_API_KEY", "mhq_test")  # pragma: allowlist secret
    monkeypatch.setenv("MORSE_ENDPOINT", "https://api.example.test")

    sender = diagnostics._get_sender()
    assert sender is not None
    client = _RecordingClient()
    sender._client = client

    try:
        raise RuntimeError("internal boom")
    except RuntimeError as exc:
        diagnostics.report(exc, fingerprint="public-shutdown-probe")

    morse_ai.shutdown()

    assert len(client.posts) == 1, "morse_ai.shutdown() did not drain diagnostics"
    assert b"public-shutdown-probe" in client.posts[0]
