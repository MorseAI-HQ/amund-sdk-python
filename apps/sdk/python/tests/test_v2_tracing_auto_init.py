"""Regression: tracing primitives must trigger env-based auto-init.

Before this fix, only ``amund.track()`` / ``record()`` called
``_ensure_initialized()``.  A process that used ``amund.run()`` /
``amund.span()`` / ``@amund.trace_agent`` exclusively would never
build a transport — every trace silently fell through to ``_last_trace``
instead of being POSTed.  This was the gunicorn-sync-worker silent-drop
bug (AGE-240): refagents in apps/api/apps/refagents wrapped their runs in
``amund.run(source="refagent")`` and never called track() first.

These tests pin the behaviour: env-based auto-init must fire on first
tracing-API touch, exactly like it does for track().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import amund

if TYPE_CHECKING:
    import pytest


def setup_function() -> None:
    amund._reset_for_testing()


def teardown_function() -> None:
    amund._reset_for_testing()


def test_run_auto_inits_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MORSE_API_KEY", "av_run_auto_init")
    monkeypatch.setenv("MORSE_ENDPOINT", "http://localhost:9/v1")

    assert amund._client is None
    with amund.run("agent-x"):
        pass
    # First use of run() must have built the client.
    assert amund._client is not None
    assert "localhost:9" in amund._client.endpoint


def test_trace_agent_auto_inits_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MORSE_API_KEY", "av_trace_agent_auto_init")
    monkeypatch.setenv("MORSE_ENDPOINT", "http://localhost:9/v1")

    @amund.trace_agent("decorated")
    def work() -> int:
        return 42

    assert amund._client is None
    assert work() == 42
    assert amund._client is not None


def test_run_silent_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MORSE_API_KEY", raising=False)
    # Must not raise — SDK stays dormant.
    with amund.run("agent-y"):
        pass
    assert amund._client is None
