"""MHQ-1054 — diagnostics must survive a process that just ends.

Live OSS runs against prod hit two `Failed to export span batch … Read timed
out` errors and left `sdk_diagnostic_events` with 0 rows. The ticket named two
possibilities and warned against assuming the first:

1. transmission is gated on the `X-Morse-Diagnostics` handshake and the server
   never sent it, so the SDK correctly stayed quiet; or
2. the path is broken.

**(1) is false**, and the code says so without needing an experiment:
`diagnostics._level` starts at `LEVEL_ERROR`, and `_transmission_allowed()`
returns True for anything that is not `off`. No handshake is required to
transmit; the header can only ever *change* the level, never enable it.

(2) is the answer, in a specific place. The diagnostics sender is a daemon
thread on a 5-second flush interval. `drain_for_shutdown()` exists precisely so
a shorter-lived process does not die with its buffer unsent (MHQ-946, and
MHQ-927 for the TypeScript half) — but it is reachable only from
`morse_ai.shutdown()`, and nothing registers that with `atexit`.
`transport.py` and `otel_provider.py` each register their own shutdown, so
traces flush at exit while diagnostics do not. The SDK's whole setup story is
zero-code-change, so customer code never calls `shutdown()` either.

These tests drive a real subprocess against a real socket. Nothing else would
do: the defect IS interpreter exit, and any in-process test would have to fake
the one event it is about.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

_SRC = str(Path(__file__).parent.parent / "src")

# Records a drop, then exits immediately — the shape of every CLI, script,
# serverless invocation and short-lived Celery task, and of the OSS test run
# that filed this ticket. Deliberately never calls morse_ai.shutdown().
_CHILD = """
import sys
sys.path.insert(0, {src!r})
import morse_ai
from morse_ai._internal import diagnostics

morse_ai.init(api_key="test-key", endpoint={endpoint!r})  # pragma: allowlist secret
# The exact call the OTLP exporter makes when a batch is permanently lost.
diagnostics.record_drop(diagnostics.REASON_SEND_FAILED, "span", 2)
{extra}
"""


class _Collector:
    """Captures diagnostics POSTs made by the child process."""

    def __init__(self):
        self.batches: list[dict] = []
        collector = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                if self.path.endswith("/diagnostics/sdk"):
                    with contextlib.suppress(ValueError):
                        collector.batches.append(json.loads(body))
                self.send_response(200)
                payload = b'{"accepted": 1, "dropped": 0}'
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        self._server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._server.shutdown()
        self._server.server_close()

    @property
    def counter_events(self) -> list[dict]:
        return [
            event
            for batch in self.batches
            for event in batch.get("events", [])
            if event.get("kind") == "counters"
        ]


def _run_child(endpoint: str, extra: str = "") -> subprocess.CompletedProcess:
    script = _CHILD.format(src=_SRC, endpoint=endpoint, extra=extra)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.req("SDKDIAG-09")
def test_a_drop_reaches_the_server_when_the_process_merely_exits():
    """The regression. Pre-fix this collects nothing: the daemon thread's first
    flush is 5s away and the interpreter exits long before it."""
    with _Collector() as collector:
        result = _run_child(collector.endpoint)

        assert result.returncode == 0, result.stderr

    events = collector.counter_events
    assert events, (
        "no diagnostics reached the server — a process that drops telemetry and "
        "then exits reports nothing, which is MHQ-1054"
    )
    counters = [c for event in events for c in event["counters"]]
    assert {"reason": "send_failed", "category": "span", "quantity": 2} in counters


@pytest.mark.req("SDKDIAG-09")
def test_an_explicit_shutdown_still_works():
    """`morse_ai.shutdown()` was always the supported path; the atexit hook must
    not break it, and must not double-send when both run."""
    with _Collector() as collector:
        result = _run_child(collector.endpoint, extra="morse_ai.shutdown()")

        assert result.returncode == 0, result.stderr

    counters = [c for e in collector.counter_events for c in e["counters"]]
    assert (
        counters.count({"reason": "send_failed", "category": "span", "quantity": 2})
        == 1
    ), "the drop was sent twice — atexit re-drained an already-drained sender"


@pytest.mark.req("SDKDIAG-01")
def test_a_healthy_process_transmits_nothing_at_exit():
    """SDKDIAG-01's allocate-nothing rule. A process with no drops and no
    internal errors must still put nothing on the wire, or the fix has turned
    every clean exit into a request."""
    with _Collector() as collector:
        script = f"""
import sys
sys.path.insert(0, {_SRC!r})
import morse_ai
_ep = {collector.endpoint!r}
morse_ai.init(api_key="test-key", endpoint=_ep)  # pragma: allowlist secret
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    assert collector.batches == []


@pytest.mark.req("SDKDIAG-02")
def test_the_kill_switch_still_wins_at_exit():
    """MORSE_SDK_DISABLE_TELEMETRY must silence the exit path too — a customer
    switch that every path honours except the last one is not a switch."""
    import os

    with _Collector() as collector:
        script = _CHILD.format(src=_SRC, endpoint=collector.endpoint, extra="")
        env = {**os.environ, "MORSE_SDK_DISABLE_TELEMETRY": "1"}
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    assert collector.batches == []


@pytest.mark.req("SDKDIAG-09")
def test_the_env_var_auto_init_path_also_drains_at_exit():
    """The zero-code-change install never calls `init()`.

    `_ensure_initialized()` builds the client on first use from MORSE_API_KEY,
    on a code path of its own. Hooking only `init()` would leave the most
    common install exactly as broken as it was — and it is the shape the OSS
    run that filed this ticket used.
    """
    import os

    with _Collector() as collector:
        script = f"""
import sys
sys.path.insert(0, {_SRC!r})
import morse_ai
from morse_ai._internal import diagnostics

# No init() call anywhere — first use auto-initialises from the env.
with morse_ai.run("agent"):
    pass
diagnostics.record_drop(diagnostics.REASON_SEND_FAILED, "span", 2)
"""
        env = {
            **os.environ,
            "MORSE_API_KEY": "test-key",  # pragma: allowlist secret
            "MORSE_ENDPOINT": collector.endpoint,
        }
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    counters = [c for e in collector.counter_events for c in e["counters"]]
    assert {"reason": "send_failed", "category": "span", "quantity": 2} in counters


@pytest.mark.req("SDKDIAG-09")
def test_a_drop_recorded_during_the_exporters_own_exit_flush_is_still_reported():
    """The ordering claim, tested rather than asserted.

    This is the exact prod scenario: the spans that fail are the ones the
    BatchSpanProcessor flushes as the process winds down, so the drop is
    recorded from inside `otel_provider`'s atexit hook. `atexit` runs LIFO, so
    the diagnostics drain only sees that drop if it registered EARLIER — which
    is why `register_exit_drain()` is called at init and not on first report.
    Register it lazily and this test fails while every other one here passes.
    """
    import os

    collector = _Collector()

    # Same host for both, so the SDK derives the diagnostics URL from it as
    # usual; the OTLP path rejects, the diagnostics path accepts.
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            if "otlp" in self.path:
                # Non-retryable, so the batch is dropped immediately rather
                # than sitting through the stock exporter's backoff.
                self.send_response(401)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path.endswith("/diagnostics/sdk"):
                with contextlib.suppress(ValueError):
                    collector.batches.append(json.loads(body))
            payload = b'{"accepted": 1, "dropped": 0}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        script = f"""
import sys
sys.path.insert(0, {_SRC!r})
import morse_ai

# No explicit drop and no shutdown() — the only failure is the trace export,
# and it happens while the process is already on its way out.
with morse_ai.run("agent"):
    pass
"""
        env = {
            **os.environ,
            "MORSE_API_KEY": "test-key",  # pragma: allowlist secret
            "MORSE_ENDPOINT": endpoint,
        }
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            check=False,
        )
        assert result.returncode == 0, result.stderr
    finally:
        server.shutdown()
        server.server_close()

    counters = [c for e in collector.counter_events for c in e["counters"]]
    assert counters, (
        "the export failed at exit and nothing reported it — the diagnostics "
        "drain ran before the exporter that recorded the drop"
    )
    assert any(c["reason"] == "auth_rejected" for c in counters), counters
