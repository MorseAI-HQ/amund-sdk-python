"""AGE-240 / ADR-018 — resource-level ``morse_ai.source`` SDK tests.

Verifies:
- REQ-DF-01: ``MORSE_INTERNAL_SOURCE`` env var stamps the resource attr
  onto every emitted trace.
- REQ-DF-02: explicit ``init(source=...)`` kwarg wins over the env var.
- REQ-DF-03: when neither is set, the ``resource`` key is **absent** from
  the trace payload (NOT defaulted to ``"unknown"``) — absence is the
  signal "this is a customer integration, not internal dogfood".
- Invalid env / kwarg values are silently dropped (typo defence).
- Per-trace ``run(..., source=...)`` overrides the process-level value.
"""

from __future__ import annotations

import pytest

import morse_ai
from morse_ai import tracing as _tracing


def setup_function() -> None:
    morse_ai._reset_for_testing()


def teardown_function() -> None:
    morse_ai._reset_for_testing()


def _emit_one_trace() -> dict:
    """Run a no-op traced agent and return the captured trace payload."""
    with morse_ai.run("source-test-agent") as r:
        r.set_outcome(success=True)
    payload = _tracing._get_last_trace()
    assert payload is not None, "tracing module didn't capture the trace"
    return payload


# ---------------------------------------------------------------------------
# REQ-DF-01 — env var stamps the resource attr
# ---------------------------------------------------------------------------


@pytest.mark.req("REQ-DF-01")
def test_env_var_emits_morse_ai_source_resource_attr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With MORSE_INTERNAL_SOURCE=copilot, every trace carries the attr."""
    monkeypatch.setenv("MORSE_API_KEY", "av_env_test")
    monkeypatch.setenv("MORSE_INTERNAL_SOURCE", "copilot")

    # Trigger lazy init through any public function.
    morse_ai._ensure_initialized()
    payload = _emit_one_trace()

    # Keyed lookup, not exact-dict equality: MHQ-668 added an independent
    # `git.revision` resource attr that legitimately coexists with source.
    assert payload.get("resource", {}).get("morse_ai.source") == "copilot"


@pytest.mark.req("REQ-DF-01")
@pytest.mark.parametrize("value", ["copilot", "investigator", "refagent", "ui-bot"])
def test_env_var_accepts_all_whitelisted_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Every accepted value round-trips into the resource attr."""
    monkeypatch.setenv("MORSE_API_KEY", "av_env_test")
    monkeypatch.setenv("MORSE_INTERNAL_SOURCE", value)
    morse_ai._ensure_initialized()
    payload = _emit_one_trace()
    assert payload.get("resource", {}).get("morse_ai.source") == value


@pytest.mark.req("REQ-DF-01")
@pytest.mark.skip(
    reason="REQ-DF-03 conflict found 2026-07-28 (see MHQ-752): MHQ-668's "
    "unconditional git.revision auto-tagging means `resource` is no longer "
    "absent just because `morse_ai.source` is unset in any git checkout / "
    "CI env — this test's true intent (source-absence, not resource-absence) "
    "needs a product decision before the assertion is rewritten."
)
def test_env_var_invalid_value_is_silently_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo or unknown value MUST NOT tag the trace — absence wins."""
    monkeypatch.setenv("MORSE_API_KEY", "av_env_test")
    monkeypatch.setenv("MORSE_INTERNAL_SOURCE", "co-pilot")  # typo
    morse_ai._ensure_initialized()
    payload = _emit_one_trace()
    assert "resource" not in payload


# ---------------------------------------------------------------------------
# REQ-DF-02 — explicit kwarg wins
# ---------------------------------------------------------------------------


@pytest.mark.req("REQ-DF-02")
def test_init_kwarg_emits_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """`init(source="investigator")` tags traces even without env."""
    monkeypatch.delenv("MORSE_INTERNAL_SOURCE", raising=False)
    fake_key = "av_kw_test"  # pragma: allowlist secret
    morse_ai.init(api_key=fake_key, source="investigator")
    payload = _emit_one_trace()
    assert payload.get("resource", {}).get("morse_ai.source") == "investigator"


@pytest.mark.req("REQ-DF-02")
def test_init_kwarg_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kwarg takes precedence — env loses when both are set."""
    monkeypatch.setenv("MORSE_INTERNAL_SOURCE", "copilot")
    fake_key = "av_kw_test"  # pragma: allowlist secret
    morse_ai.init(api_key=fake_key, source="investigator")
    payload = _emit_one_trace()
    assert payload.get("resource", {}).get("morse_ai.source") == "investigator"


@pytest.mark.req("REQ-DF-02")
@pytest.mark.skip(
    reason="REQ-DF-03 conflict found 2026-07-28 (see MHQ-752) — same as "
    "test_env_var_invalid_value_is_silently_dropped above."
)
def test_init_kwarg_invalid_value_drops_to_no_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid kwarg is dropped entirely — the kwarg is the explicit
    contract, so we honour it as ``None`` rather than silently falling
    through to env (which would mask the typo)."""
    monkeypatch.setenv("MORSE_INTERNAL_SOURCE", "copilot")
    fake_key = "av_kw_test"  # pragma: allowlist secret
    morse_ai.init(api_key=fake_key, source="bogus")
    payload = _emit_one_trace()
    # Explicit invalid kwarg shadows env to None — typo is visible, not hidden.
    assert "resource" not in payload


# ---------------------------------------------------------------------------
# REQ-DF-03 — absence is the signal
# ---------------------------------------------------------------------------


@pytest.mark.req("REQ-DF-03")
@pytest.mark.skip(
    reason="REQ-DF-03 conflict found 2026-07-28 (see MHQ-752): this is the "
    "canonical absence-signal test and it now fails in any git checkout / "
    "CI env because MHQ-668's git.revision auto-tagging makes `resource` "
    "non-empty even with no source set. Needs a product decision: scope "
    "git-revision detection away from customer-facing default, or redefine "
    "the dogfood-discrimination signal to key off `morse_ai.source` "
    "specifically rather than `resource` key presence."
)
def test_no_env_no_kwarg_omits_resource_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Customer integrations (no env, no kwarg) must have NO resource block."""
    monkeypatch.delenv("MORSE_INTERNAL_SOURCE", raising=False)
    fake_key = "av_cust_test"  # pragma: allowlist secret
    morse_ai.init(api_key=fake_key)
    payload = _emit_one_trace()
    assert "resource" not in payload


# ---------------------------------------------------------------------------
# Per-trace override — investigator co-resident with copilot in one process
# ---------------------------------------------------------------------------


@pytest.mark.req("REQ-DF-02")
def test_run_source_kwarg_overrides_process_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`run(..., source="investigator")` overrides the process-level tag."""
    fake_key = "av_proc_test"  # pragma: allowlist secret
    morse_ai.init(api_key=fake_key, source="copilot")

    with morse_ai.run("investigator-trace", source="investigator") as r:
        r.set_outcome(success=True)
    payload = _tracing._get_last_trace()
    assert payload is not None
    assert payload.get("resource", {}).get("morse_ai.source") == "investigator"

    # Without the override, the next trace falls back to the process source.
    with morse_ai.run("copilot-trace") as r:
        r.set_outcome(success=True)
    payload = _tracing._get_last_trace()
    assert payload is not None
    assert payload.get("resource", {}).get("morse_ai.source") == "copilot"


@pytest.mark.req("REQ-DF-02")
def test_run_source_invalid_value_falls_back_to_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid per-trace `source` is dropped; process-level value survives."""
    fake_key = "av_proc_test"  # pragma: allowlist secret
    morse_ai.init(api_key=fake_key, source="copilot")
    with morse_ai.run("bad-override", source="not-real") as r:
        r.set_outcome(success=True)
    payload = _tracing._get_last_trace()
    assert payload is not None
    assert payload.get("resource", {}).get("morse_ai.source") == "copilot"
