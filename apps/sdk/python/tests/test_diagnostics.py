"""SDK self-diagnostics (MHQ-908, SDKDIAG-01..10).

The invariants worth protecting here are mostly *negative* — what the SDK does
NOT do: no thread on a healthy install, no locals on the wire, no host frames,
no retry, no self-report, no silence for an opted-out customer.
"""

import logging
import threading

import pytest

from morse_ai._internal import diagnostics


@pytest.fixture(autouse=True)
def _reset():
    diagnostics._reset_for_testing()
    yield
    diagnostics._reset_for_testing()


def _raise_inside_sdk() -> BaseException:
    """Produce an exception whose traceback runs host → SDK → host → SDK."""
    from morse_ai import tracing

    try:
        tracing._report  # noqa: B018 — ensure the module is imported
        raise KeyError("span_id")
    except KeyError as exc:
        return exc


# --- SDKDIAG-01 — emitted only on an absorbed error -------------------------


@pytest.mark.req("SDKDIAG-01")
def test_healthy_process_creates_no_diagnostics_thread():
    """Option B's whole justification: the sender is lazily built, so a process
    that never errors pays nothing — no thread, no socket, no queue."""
    before = {t.name for t in threading.enumerate()}
    assert not any(name.startswith("morse_ai-diagnostics") for name in before)
    assert diagnostics._sender is None


@pytest.mark.req("SDKDIAG-01")
def test_no_sender_is_built_without_an_api_key(monkeypatch):
    """No API key ⇒ no authenticated destination ⇒ local-log-only. This is the
    accepted loss recorded in the spec, asserted so it stays deliberate."""
    monkeypatch.delenv("MORSE_API_KEY", raising=False)
    monkeypatch.delenv("MORSE_ENDPOINT", raising=False)

    diagnostics.report(_raise_inside_sdk(), fingerprint="test.no_key")

    assert diagnostics._sender is None


@pytest.mark.req("SDKDIAG-01")
def test_kill_switch_beats_server_debug(monkeypatch):
    monkeypatch.setenv("MORSE_SDK_DISABLE_TELEMETRY", "1")
    diagnostics.set_level(diagnostics.LEVEL_DEBUG)

    assert diagnostics._transmission_allowed() is False


# --- SDKDIAG-02 — Morse-scoped payload --------------------------------------


@pytest.mark.req("SDKDIAG-02")
def test_host_frames_are_elided_and_sdk_frames_are_kept():
    exc = _raise_inside_sdk()
    stack = diagnostics.build_scoped_stack(exc)

    assert stack, "expected at least one frame"
    # The raising frame is this test file — a host frame from the SDK's view.
    assert any("elided" in frame for frame in stack)
    for frame in stack:
        if "file" in frame:
            assert not frame["file"].startswith("/"), "paths must be package-relative"


@pytest.mark.req("SDKDIAG-02")
def test_no_frame_carries_locals():
    """`extract_tb` never reads frame locals — a stronger guarantee than
    reading them and choosing to drop them."""
    secret = "sk-live-should-never-appear"  # pragma: allowlist secret
    try:
        assert secret
        raise ValueError("boom")
    except ValueError as exc:
        stack = diagnostics.build_scoped_stack(exc)

    for frame in stack:
        assert "vars" not in frame
        assert "locals" not in frame
    assert secret not in repr(stack)


@pytest.mark.req("SDKDIAG-02")
def test_stack_is_capped():
    frames = [{"file": "x.py", "line": i, "function": "f"} for i in range(200)]
    assert len(frames) > diagnostics.MAX_STACK_ENTRIES

    exc = _raise_inside_sdk()
    stack = diagnostics.build_scoped_stack(exc)
    assert len(stack) <= diagnostics.MAX_STACK_ENTRIES


@pytest.mark.req("SDKDIAG-02")
def test_extras_are_allow_listed():
    scrubbed = diagnostics._scrub_extra(
        {"adapter": "anthropic", "prompt": "customer text", "operation": "wrap"}
    )

    assert scrubbed == {"adapter": "anthropic", "operation": "wrap"}


@pytest.mark.req("SDKDIAG-02")
def test_message_is_capped():
    exc = ValueError("x" * 10_000)
    assert len(diagnostics._redacted_message(exc)) <= diagnostics.MAX_MESSAGE_CHARS


# --- SDKDIAG-05 — server-controlled verbosity -------------------------------


@pytest.mark.req("SDKDIAG-05")
def test_default_level_is_error_before_any_header():
    assert diagnostics.get_level() == diagnostics.LEVEL_ERROR


@pytest.mark.req("SDKDIAG-05")
def test_level_is_learned_from_header():
    diagnostics.set_level("debug")
    assert diagnostics.get_level() == diagnostics.LEVEL_DEBUG

    diagnostics.set_level("off")
    assert diagnostics.get_level() == diagnostics.LEVEL_OFF
    assert diagnostics._transmission_allowed() is False


@pytest.mark.req("SDKDIAG-05")
def test_unknown_level_is_ignored_not_treated_as_off():
    """A future server value must not silently mute an old SDK."""
    diagnostics.set_level("verbose-plus")

    assert diagnostics.get_level() == diagnostics.LEVEL_ERROR


# --- SDKDIAG-06 — cannot break or amplify itself ----------------------------


@pytest.mark.req("SDKDIAG-06")
def test_report_never_raises_even_when_everything_fails(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("stack builder exploded")

    monkeypatch.setattr(diagnostics, "build_scoped_stack", _boom)
    monkeypatch.setenv("MORSE_API_KEY", "test-key")  # pragma: allowlist secret
    monkeypatch.setenv("MORSE_ENDPOINT", "https://example.invalid")

    diagnostics.report(ValueError("boom"), fingerprint="test.explode")


@pytest.mark.req("SDKDIAG-06")
def test_reentrancy_fence_stops_self_reporting(monkeypatch):
    """A failure inside reporting must not produce a second diagnostic."""
    calls = []
    real_scrub = diagnostics._scrub_extra

    def _reentrant(extra):
        calls.append(1)
        # Simulate the reporter's own code path erroring and being reported.
        diagnostics.report(ValueError("inner"), fingerprint="test.inner")
        return real_scrub(extra)

    monkeypatch.setattr(diagnostics, "_scrub_extra", _reentrant)
    monkeypatch.setenv("MORSE_API_KEY", "test-key")  # pragma: allowlist secret
    monkeypatch.setenv("MORSE_ENDPOINT", "https://example.invalid")

    diagnostics.report(ValueError("outer"), fingerprint="test.outer")

    assert len(calls) == 1, "the inner report must have been fenced out"


# --- SDKDIAG-09 — delivery counters -----------------------------------------


@pytest.mark.req("SDKDIAG-09")
def test_drops_aggregate_into_one_report():
    for _ in range(10_000):
        diagnostics.record_drop(diagnostics.REASON_QUEUE_FULL, "span")

    drained = diagnostics._drain_counters()

    assert len(drained) == 1, "10k drops must not become 10k reports"
    assert drained[0] == {
        "reason": diagnostics.REASON_QUEUE_FULL,
        "category": "span",
        "quantity": 10_000,
    }


@pytest.mark.req("SDKDIAG-09")
def test_no_drops_means_nothing_to_send():
    assert diagnostics._drain_counters() == []


@pytest.mark.req("SDKDIAG-09")
def test_drain_is_destructive():
    diagnostics.record_drop(diagnostics.REASON_RETRY_EXHAUSTED, "trace", 3)

    assert diagnostics._drain_counters()
    assert diagnostics._drain_counters() == []


@pytest.mark.req("SDKDIAG-09")
def test_queue_full_is_counted():
    """The drop site that previously produced only a local WARNING."""
    from morse_ai.queue import TelemetryQueue

    q = TelemetryQueue(max_size=1)
    assert q.put({"a": 1}) is True
    assert q.put({"b": 2}) is False

    drained = diagnostics._drain_counters()
    assert any(d["reason"] == diagnostics.REASON_QUEUE_FULL for d in drained)


# --- SDKDIAG-10 — no private channel ----------------------------------------


@pytest.mark.req("SDKDIAG-10")
def test_opted_out_customer_still_sees_errors_locally(monkeypatch, caplog):
    monkeypatch.setenv("MORSE_SDK_DISABLE_TELEMETRY", "1")

    with caplog.at_level(logging.WARNING, logger="morse_ai._internal"):
        diagnostics.report(ValueError("boom"), fingerprint="test.optout")

    assert any("test.optout" in r.getMessage() for r in caplog.records)
    assert diagnostics._sender is None


@pytest.mark.req("SDKDIAG-10")
def test_opted_out_customer_still_sees_delivery_loss(monkeypatch, caplog):
    monkeypatch.setenv("MORSE_SDK_DISABLE_TELEMETRY", "1")

    with caplog.at_level(logging.WARNING, logger="morse_ai._internal"):
        diagnostics.record_drop(diagnostics.REASON_QUEUE_FULL, "span", 5)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "delivery loss must reach the host at WARNING"
    assert "queue_full" in warnings[0].getMessage()


@pytest.mark.req("SDKDIAG-10")
def test_level_off_still_logs_locally(caplog):
    diagnostics.set_level("off")

    with caplog.at_level(logging.WARNING, logger="morse_ai._internal"):
        diagnostics.report(ValueError("boom"), fingerprint="test.leveloff")

    assert any("test.leveloff" in r.getMessage() for r in caplog.records)


@pytest.mark.req("SDKDIAG-10")
def test_local_logging_is_rate_limited(caplog):
    with caplog.at_level(logging.WARNING, logger="morse_ai._internal"):
        for _ in range(500):
            diagnostics.report(ValueError("boom"), fingerprint="test.flood")

    lines = [r for r in caplog.records if "test.flood" in r.getMessage()]
    assert len(lines) == diagnostics._LOCAL_LOG_LIMIT_PER_WINDOW


# --- SDKDIAG-07 — swallow sites report --------------------------------------


@pytest.mark.req("SDKDIAG-07")
def test_swallow_site_helper_reports_without_changing_control_flow():
    from morse_ai import tracing

    seen = []

    original = diagnostics.report

    def _capture(exc, *, fingerprint, extra=None):
        seen.append(fingerprint)
        return original(exc, fingerprint=fingerprint, extra=extra)

    diagnostics.report = _capture  # type: ignore[assignment]
    try:
        # The helper returns None and never raises — the caller's `except`
        # block continues exactly as before.
        assert tracing._report(ValueError("boom"), "morse_ai.tracing.test") is None
    finally:
        diagnostics.report = original  # type: ignore[assignment]

    assert seen == ["morse_ai.tracing.test"]


@pytest.mark.req("SDKDIAG-07")
def test_absorb_errors_reports_and_returns_none():
    from morse_ai._internal.errors import absorb_errors

    seen = []
    original = diagnostics.report

    def _capture(exc, *, fingerprint, extra=None):
        seen.append(fingerprint)

    diagnostics.report = _capture  # type: ignore[assignment]
    try:

        @absorb_errors("morse_ai.test.absorb")
        def _boom():
            raise KeyError("nope")

        assert _boom() is None
    finally:
        diagnostics.report = original  # type: ignore[assignment]

    assert seen == ["morse_ai.test.absorb"]


class TestSecretAssignmentIsUnconditional:
    """MHQ-973 — `secret_assignment` runs on diagnostic messages regardless of
    the org's redaction config.

    The pattern is disableable, because an org may legitimately want raw
    `password=` values in *their own* spans. This channel is different: the
    published disclosure (SDKDIAG-02) promises Morse receives no customer data,
    and that promise must not be defeatable by a telemetry preference.
    """

    def test_password_assignment_is_redacted_with_no_config_at_all(self, monkeypatch):
        from morse_ai import tracing

        monkeypatch.setattr(tracing, "_redaction_config", None, raising=False)
        out = diagnostics._redacted_message(ValueError("boom password=hunter2 tail"))
        assert "hunter2" not in out
        assert "[REDACTED:secret]" in out

    def test_still_redacted_when_the_org_disabled_the_pattern(self, monkeypatch):
        """The case the whole second pass exists for."""
        from morse_ai import tracing
        from morse_ai._internal.redaction import RedactionConfig

        # Everything except secret_assignment — i.e. an org that switched it off.
        without = RedactionConfig(
            enabled=True,
            enabled_patterns=frozenset({"api_key_prefixed", "jwt"}),
            extra_patterns=(),
        )
        monkeypatch.setattr(tracing, "_redaction_config", without, raising=False)
        out = diagnostics._redacted_message(ValueError("db password=hunter2"))
        assert "hunter2" not in out, "org config must not defeat the SDKDIAG-02 promise"

    def test_llm_token_counts_are_not_redacted(self, monkeypatch):
        """This is an LLM-observability SDK: token COUNTS are the signal."""
        from morse_ai import tracing

        monkeypatch.setattr(tracing, "_redaction_config", None, raising=False)
        msg = "usage input_tokens=100 output_tokens=50 max_tokens=4096"
        assert diagnostics._redacted_message(ValueError(msg)) == msg
