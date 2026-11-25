# ruff: noqa: E501
"""INFRACAP-10 — overhead of default-on infra capture (Python SDK).

    cd apps/sdk/python && .venv/bin/python benchmarks/infra_capture_overhead.py

Two questions, per integration (sqlite3, httpx, requests, SQS via botocore):

1. No trace active (every untraced code path pays this): what does a call cost
   with capture on versus off?  Pass if the overhead is at most 2% of the call
   OR at most 5 microseconds.
2. Trace active: what does one captured call ADD?  p50 and p99 of the per-call
   latency with capture on minus the same percentile with capture off.  Pass if
   p50 <= 50 microseconds and p99 <= 150 microseconds.

Each question is measured two ways, because they answer different things:

  * isolated   The wrapped library call is replaced by a no-op stub, so the
               number is the capture's own cost and nothing else. The only
               measurement here that is not drowned in the library's variance.
               GRADED for the per-span thresholds (and for the 5 us rule).
  * end-to-end A real httpx/requests/sqlite/botocore call into a mock transport.
               Realistic for the 2% rule. For per-span cost it is printed but
               labelled noisy: a 190 us call with a 40 us standard deviation
               cannot resolve a 20 us effect from percentile differences.

Thresholds are fixed. A miss prints FAIL and the exit code is 1; nothing is tuned
to the result. Off/on blocks are interleaved so drift hits both equally, and
trace-active blocks stay under the per-trace cap so the cheaper capped path is
never what is measured.
"""

from __future__ import annotations

import gc
import sqlite3
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable

import boto3
import httpx
import requests
from botocore.client import BaseClient
from botocore.stub import Stubber

from amund import infra_capture as ic
from amund import queue_capture as qc
from amund.tracing import run

NO_TRACE_MAX_PCT = 2.0
NO_TRACE_MAX_US = 5.0
SPAN_P50_MAX_US = 50.0
SPAN_P99_MAX_US = 150.0
SPAN_MEMORY_MAX_MB_PER_1K = 1.0

DB_BLOCK = 150  # calls per trace; the caps are 200 db / 100 http / 100 queue
HTTP_BLOCK = 90
NO_TRACE_ROUNDS = 30
NO_TRACE_CALLS_E2E = 300
NO_TRACE_CALLS_ISOLATED = 3000
TRACE_ROUNDS = 200

URL = "https://bench.example.test/x"
SQS_URL = "https://sqs.us-east-1.amazonaws.com/123456789012/bench"
OK = httpx.Response(200, text="ok")


class _OkAdapter(requests.adapters.BaseAdapter):
    def send(self, request, **kwargs):  # type: ignore[no-untyped-def]
        resp = requests.Response()
        resp.status_code = 200
        resp._content = b"ok"
        resp.request = request
        return resp

    def close(self) -> None:
        return None


Factory = Callable[[], Callable[[], object]]


def _pct(values: list[int], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


# --- scenarios -------------------------------------------------------------


def _sqlite() -> Factory:
    def factory() -> Callable[[], object]:
        conn = sqlite3.connect(":memory:")  # patched connect() when capture is on
        conn.execute("CREATE TABLE IF NOT EXISTS t (a INTEGER)")
        return lambda: conn.execute("SELECT a FROM t WHERE a = 1")

    return factory


def _httpx(real: bool) -> Factory:
    def factory() -> Callable[[], object]:
        if real:
            client = httpx.Client(
                transport=httpx.MockTransport(lambda r: httpx.Response(200, text="ok"))
            )
            return lambda: client.get(URL)
        client = httpx.Client()
        request = client.build_request("GET", URL)
        return lambda: client.send(request)

    return factory


def _requests(real: bool) -> Factory:
    def factory() -> Callable[[], object]:
        session = requests.Session()
        session.mount("https://", _OkAdapter())
        if real:
            return lambda: session.get(URL)
        prepared = session.prepare_request(requests.Request("GET", URL))
        return lambda: session.send(prepared)

    return factory


def _sqs(real: bool) -> Factory:
    def factory() -> Callable[[], object]:
        client = boto3.client(
            "sqs",
            region_name="us-east-1",
            aws_access_key_id="bench",
            aws_secret_access_key="bench",  # pragma: allowlist secret
        )
        if real:
            stub = Stubber(client)
            stub.activate()

            def call() -> object:
                stub.add_response(
                    "send_message", {"MessageId": "m", "MD5OfMessageBody": "x"}
                )
                return client.send_message(QueueUrl=SQS_URL, MessageBody="b")

            return call
        return lambda: client.send_message(QueueUrl=SQS_URL, MessageBody="b")

    return factory


class _Stubs:
    """Swap the library's send for a no-op BEFORE capture wraps it, so an
    'isolated' measurement times only the wrapper."""

    def __enter__(self) -> _Stubs:
        _set_capture(
            False
        )  # never stub a wrapper left over from a previous measurement
        self._httpx = httpx.Client.send
        self._requests = requests.Session.send
        self._botocore = BaseClient._make_api_call
        httpx.Client.send = lambda self, request, *a, **k: OK  # type: ignore[method-assign,assignment]
        requests.Session.send = lambda self, request, **k: OK  # type: ignore[method-assign,assignment]
        BaseClient._make_api_call = lambda self, op, params: {"MessageId": "m"}  # type: ignore[method-assign,assignment]
        return self

    def __exit__(self, *exc: object) -> None:
        _set_capture(False)
        httpx.Client.send = self._httpx  # type: ignore[method-assign]
        requests.Session.send = self._requests  # type: ignore[method-assign]
        BaseClient._make_api_call = self._botocore  # type: ignore[method-assign]


def _set_capture(on: bool) -> None:
    ic.disable_db_capture()
    ic.disable_http_capture()
    qc.disable_queue_capture()
    if on:
        ic.enable_db_capture()
        ic.enable_http_capture()
        qc.enable_queue_capture()


# --- measurement -----------------------------------------------------------


def _time_calls(call: Callable[[], object], n: int) -> list[int]:
    out = []
    clock = time.perf_counter_ns
    for _ in range(n):
        t0 = clock()
        call()
        out.append(clock() - t0)
    return out


def measure_no_trace(factory: Factory, calls: int) -> tuple[float, float]:
    """(mean ns per call off, mean ns per call on), no trace active.

    Off and on run back to back each round with the order alternating, and the
    reported difference is the median of the per-round differences, so drift and
    order effects cancel instead of landing in the result.
    """
    off_means: list[float] = []
    deltas: list[float] = []
    for r in range(NO_TRACE_ROUNDS):
        round_means: dict[bool, float] = {}
        for on in (False, True) if r % 2 == 0 else (True, False):
            _set_capture(on)
            call = factory()
            call()  # warm
            gc.collect()
            round_means[on] = statistics.fmean(_time_calls(call, calls))
        off_means.append(round_means[False])
        deltas.append(round_means[True] - round_means[False])
    _set_capture(False)
    off = statistics.median(off_means)
    return off, off + statistics.median(deltas)


def measure_in_trace(factory: Factory, block: int) -> tuple[list[int], list[int]]:
    """Per-call latencies (ns) inside a trace: (capture off, capture on)."""
    off: list[int] = []
    on: list[int] = []
    for _ in range(TRACE_ROUNDS):
        for capture_on in (False, True):
            _set_capture(capture_on)
            call = factory()
            call()
            gc.collect()
            with run("bench-agent"):
                (on if capture_on else off).extend(_time_calls(call, block))
    _set_capture(False)
    return off, on


def measure_memory_per_1k_spans() -> float:
    """Python memory held by recorded db spans before flush, per 1000 spans (MB)."""
    _set_capture(True)
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (a INTEGER)")
    gc.collect()
    tracemalloc.start()
    with run("bench-memory"):
        before, _ = tracemalloc.get_traced_memory()
        for _ in range(200):
            conn.execute("SELECT a FROM t WHERE a = 1")
        after, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (after - before) / 200 * 1000 / (1024 * 1024)


# --- report ----------------------------------------------------------------


def main() -> int:
    failures: list[str] = []
    print(
        f"Python {sys.version.split()[0]}  httpx {httpx.__version__}  "
        f"requests {requests.__version__}  boto3 {boto3.__version__}"
    )
    print(
        f"Thresholds: no-trace <= {NO_TRACE_MAX_PCT}% or <= {NO_TRACE_MAX_US} us per call; "
        f"per span p50 <= {SPAN_P50_MAX_US} us, p99 <= {SPAN_P99_MAX_US} us\n"
    )

    sets = {
        # name: (isolated factory, end-to-end factory, calls per trace)
        "sqlite3 execute": (_sqlite(), _sqlite(), DB_BLOCK),
        "httpx send": (_httpx(False), _httpx(True), HTTP_BLOCK),
        "requests send": (_requests(False), _requests(True), HTTP_BLOCK),
        "sqs send_message": (_sqs(False), _sqs(True), HTTP_BLOCK),
    }

    print("1. NO TRACE ACTIVE: capture on vs off")
    print(
        f"{'integration':22} {'mode':11} {'off us':>9} {'on us':>9} {'delta us':>9} {'delta %':>8}  verdict"
    )
    for name, (iso, e2e, _block) in sets.items():
        with _Stubs():
            off_ns, on_ns = measure_no_trace(iso, NO_TRACE_CALLS_ISOLATED)
        d_us = (on_ns - off_ns) / 1000
        ok = d_us <= NO_TRACE_MAX_US  # % is meaningless against a no-op stub
        if not ok:
            failures.append(f"no-trace isolated {name}: {d_us:.2f} us")
        print(
            f"{name:22} {'isolated':11} {off_ns / 1000:9.2f} {on_ns / 1000:9.2f} {d_us:9.2f} "
            f"{'n/a':>8}  {'PASS' if ok else 'FAIL'}"
        )
        off_ns, on_ns = measure_no_trace(e2e, NO_TRACE_CALLS_E2E)
        d_us = (on_ns - off_ns) / 1000
        pct = (on_ns - off_ns) / off_ns * 100
        ok = pct <= NO_TRACE_MAX_PCT or d_us <= NO_TRACE_MAX_US
        if not ok:
            failures.append(f"no-trace end-to-end {name}: {d_us:.2f} us / {pct:.2f}%")
        print(
            f"{'':22} {'end-to-end':11} {off_ns / 1000:9.2f} {on_ns / 1000:9.2f} {d_us:9.2f} "
            f"{pct:8.2f}  {'PASS' if ok else 'FAIL'}"
        )

    print("\n2. TRACE ACTIVE: cost one captured call adds")
    print(f"{'integration':22} {'mode':11} {'p50 us':>9} {'p99 us':>9}  verdict")
    for name, (iso, e2e, block) in sets.items():
        with _Stubs():
            off, on = measure_in_trace(iso, block)
        p50 = (_pct(on, 0.50) - _pct(off, 0.50)) / 1000
        p99 = (_pct(on, 0.99) - _pct(off, 0.99)) / 1000
        ok = p50 <= SPAN_P50_MAX_US and p99 <= SPAN_P99_MAX_US
        if not ok:
            failures.append(
                f"per-span isolated {name}: p50 {p50:.2f} us, p99 {p99:.2f} us"
            )
        print(
            f"{name:22} {'isolated':11} {p50:9.2f} {p99:9.2f}  {'PASS' if ok else 'FAIL'}"
        )
        off, on = measure_in_trace(e2e, block)
        p50 = (_pct(on, 0.50) - _pct(off, 0.50)) / 1000
        p99 = (_pct(on, 0.99) - _pct(off, 0.99)) / 1000
        print(
            f"{'':22} {'end-to-end':11} {p50:9.2f} {p99:9.2f}  (informational, noisy)"
        )

    mem = measure_memory_per_1k_spans()
    mem_ok = mem <= SPAN_MEMORY_MAX_MB_PER_1K
    if not mem_ok:
        failures.append(f"memory per 1k spans: {mem:.2f} MB")
    print(
        f"\n3. MEMORY  {mem:.2f} MB per 1000 db spans held before flush "
        f"(limit {SPAN_MEMORY_MAX_MB_PER_1K} MB)  {'PASS' if mem_ok else 'FAIL'}"
    )

    _set_capture(False)
    print()
    if failures:
        print("FAILED thresholds:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("All thresholds met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
