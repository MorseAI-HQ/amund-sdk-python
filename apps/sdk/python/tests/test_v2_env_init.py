"""Phase 6c Wave 1.3 — env-based init + transparency log + non-invasive isolation tests.

Verifies:
- Env-based auto-init from MORSE_API_KEY
- MORSE_ENDPOINT / MORSE_FLUSH_INTERVAL_SECONDS overrides
- MORSE_DISABLED makes track()/record()/etc. silent no-ops
- MORSE_SDK_DEBUG raises internal-error log level
- Destination-URL transparency log fires once per (process, endpoint)
- Re-init with same endpoint does NOT re-log; different endpoint DOES
- @absorb_errors catches all internal exceptions; customer code keeps running
- Sentry reporter rate limits at 100 events / fingerprint / 60s
"""

# ruff: noqa: E501, SIM117
from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

import morse_ai
from morse_ai._internal import diagnostics


def setup_function() -> None:
    morse_ai._reset_for_testing()


def teardown_function() -> None:
    morse_ai._reset_for_testing()


# ---------------------------------------------------------------------------
# Env-based auto-init
# ---------------------------------------------------------------------------


def test_track_auto_inits_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting MORSE_API_KEY in env triggers auto-init on first use."""
    monkeypatch.setenv("MORSE_API_KEY", "av_env_test")
    monkeypatch.setenv("MORSE_ENDPOINT", "http://localhost:9999/api/v1/traces/batch")

    assert morse_ai._client is None
    morse_ai.track(agent_id="my-agent", run_data={"success": True})
    # After first track(), the singleton is up.
    assert morse_ai._client is not None
    assert "localhost:9999" in morse_ai._client.endpoint


def test_no_api_key_keeps_sdk_dormant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without MORSE_API_KEY, the SDK stays dormant and track() is no-op."""
    monkeypatch.delenv("MORSE_API_KEY", raising=False)
    morse_ai.track(agent_id="my-agent", run_data={"success": True})
    assert morse_ai._client is None


def test_no_api_key_emits_one_time_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("MORSE_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING, logger="morse_ai"):
        morse_ai.track(agent_id="a", run_data={})
        morse_ai.track(agent_id="a", run_data={})
        morse_ai.track(agent_id="a", run_data={})
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, "warning must fire exactly once per process"
    assert "MORSE_API_KEY" in warnings[0].getMessage()


def test_explicit_init_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MORSE_API_KEY", "av_env_value")
    morse_ai.init(api_key="av_explicit_value")
    assert morse_ai._client is not None
    assert morse_ai._client._api_key == "av_explicit_value"


# ---------------------------------------------------------------------------
# MORSE_DISABLED short-circuit
# ---------------------------------------------------------------------------


def test_disabled_makes_init_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MORSE_DISABLED", "1")
    morse_ai.init(api_key="av_test")
    assert morse_ai._client is None  # init bailed without constructing transport


def test_disabled_makes_track_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MORSE_DISABLED", "1")
    monkeypatch.setenv("MORSE_API_KEY", "av_test")
    morse_ai.track(agent_id="a", run_data={"x": 1})
    assert morse_ai._client is None


# ---------------------------------------------------------------------------
# Destination-URL transparency log
# ---------------------------------------------------------------------------


def test_init_logs_destination_url_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="morse_ai"):
        morse_ai.init(api_key="av_log_test")
        # Re-init with same endpoint — should NOT re-log.
        morse_ai.init(api_key="av_log_test")
    log_lines = [
        r.getMessage()
        for r in caplog.records
        if "Morse SDK initialized" in r.getMessage()
    ]
    assert len(log_lines) == 1, f"expected 1 transparency log, got {len(log_lines)}"
    assert "api.morsehq.dev" in log_lines[0]


def test_init_re_logs_when_endpoint_changes(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="morse_ai"):
        morse_ai.init(api_key="av_test")
        morse_ai.init(
            api_key="av_test",
            endpoint="https://other.example.com/api/v1/traces/batch",
        )
    log_lines = [
        r.getMessage()
        for r in caplog.records
        if "Morse SDK initialized" in r.getMessage()
    ]
    assert len(log_lines) == 2, "endpoint change should re-log"


# ---------------------------------------------------------------------------
# Non-invasive error isolation (@absorb_errors)
# ---------------------------------------------------------------------------


def test_track_absorbs_internal_exceptions() -> None:
    """If the underlying _Client.track raises, morse_ai.track() does NOT propagate."""
    morse_ai.init(api_key="av_test", endpoint="http://localhost:0/api/v1/traces/batch")
    assert morse_ai._client is not None

    # Inject a bug into the underlying client.
    def _broken_track(agent_id: str, run_data: dict) -> None:
        raise RuntimeError("simulated SDK bug")

    with patch.object(morse_ai._client, "track", side_effect=_broken_track):
        # MUST NOT raise:
        result = morse_ai.track(agent_id="my-agent", run_data={"success": True})
        assert result is None


def test_record_absorbs_internal_exceptions() -> None:
    morse_ai.init(api_key="av_test", endpoint="http://localhost:0/api/v1/traces/batch")
    assert morse_ai._client is not None

    with patch.object(morse_ai._client, "record", side_effect=ValueError("boom")):
        result = morse_ai.record(agent="a", success=True)
        assert result is None


def test_init_absorbs_constructor_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """If _Client construction raises, init() returns silently and SDK stays dormant."""

    def _broken_ctor(*args: object, **kwargs: object) -> None:
        raise RuntimeError("transport failed to start")

    monkeypatch.setattr("morse_ai._Client.__init__", _broken_ctor)
    # Must NOT raise:
    morse_ai.init(api_key="av_test")
    assert morse_ai._client is None


def test_keyboard_interrupt_still_propagates() -> None:
    """KeyboardInterrupt is intentionally NOT absorbed — Ctrl-C must work."""
    morse_ai.init(api_key="av_test", endpoint="http://localhost:0/api/v1/traces/batch")
    assert morse_ai._client is not None

    def _ki(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt()

    with patch.object(morse_ai._client, "track", side_effect=_ki):
        with pytest.raises(KeyboardInterrupt):
            morse_ai.track(agent_id="a", run_data={})


# ---------------------------------------------------------------------------
# Diagnostics reporter — rate limiting + scrubbing (MHQ-908)
# ---------------------------------------------------------------------------


def test_local_logging_is_rate_limited(caplog: pytest.LogCaptureFixture) -> None:
    """SDKDIAG-10: local emission is capped so a looping internal error cannot
    flood the host's log.

    Changed by MHQ-908: this used to assert 100 lines, because local logging
    was gated by the *transmission* limiter. The two are now separate — the
    host's visibility must not be bounded by our send budget — so the local cap
    is its own, smaller number.
    """
    diagnostics._reset_for_testing()
    with caplog.at_level(logging.WARNING, logger="morse_ai._internal"):
        for _ in range(150):
            diagnostics.report(ValueError("boom"), fingerprint="test.rate_limit")
    log_lines = [r for r in caplog.records if "test.rate_limit" in r.getMessage()]
    assert len(log_lines) == diagnostics._LOCAL_LOG_LIMIT_PER_WINDOW


def test_transmission_is_rate_limited_per_fingerprint() -> None:
    """100 events per fingerprint per minute reach the wire; the rest are dropped."""
    diagnostics._reset_for_testing()
    allowed = sum(
        1 for _ in range(150) if diagnostics._check_rate_limit("test.transmit_limit")
    )
    assert allowed == diagnostics._RATE_LIMIT_PER_WINDOW


def test_reporter_scrubs_unsafe_extras() -> None:
    """Extras outside the safe key allow-list are dropped before any send."""
    extra = {
        "adapter": "anthropic",  # safe
        "customer_prompt": "hi please don't leak this",  # unsafe → dropped
        "api_key": "av_secret",  # pragma: allowlist secret — unsafe → dropped
    }
    scrubbed = diagnostics._scrub_extra(extra)
    assert "adapter" in scrubbed
    assert "customer_prompt" not in scrubbed
    assert "api_key" not in scrubbed


def test_reporter_never_raises() -> None:
    """Even if everything inside report() goes wrong, it must not raise."""
    diagnostics.report(
        ValueError("ok"), fingerprint="test.no_raise", extra={"adapter": "x"}
    )


def test_telemetry_opt_out_still_logs_locally(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """SDKDIAG-10: MORSE_SDK_DISABLE_TELEMETRY stops TRANSMISSION, not visibility.

    Behaviour change from MHQ-908 — this test previously asserted the opposite
    (opting out silenced the local log too). That made opting out cost the
    customer their own diagnostics, which is backwards: the switch exists to
    stop data leaving their process, not to blind their engineers.
    """
    diagnostics._reset_for_testing()
    monkeypatch.setenv("MORSE_SDK_DISABLE_TELEMETRY", "1")
    with caplog.at_level(logging.WARNING, logger="morse_ai._internal"):
        diagnostics.report(ValueError("boom"), fingerprint="test.disabled")
    assert any("test.disabled" in r.getMessage() for r in caplog.records), (
        "opting out of transmission must not suppress the local log"
    )
    assert diagnostics._transmission_allowed() is False


def test_sdk_debug_raises_log_level(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """MORSE_SDK_DEBUG=1 logs internal errors at ERROR (default WARNING)."""
    diagnostics._reset_for_testing()
    monkeypatch.setenv("MORSE_SDK_DEBUG", "1")
    with caplog.at_level(logging.WARNING, logger="morse_ai._internal"):
        diagnostics.report(ValueError("boom"), fingerprint="test.debug_level")
    error_lines = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and "test.debug_level" in r.getMessage()
    ]
    assert error_lines, "MORSE_SDK_DEBUG should escalate to ERROR level"
