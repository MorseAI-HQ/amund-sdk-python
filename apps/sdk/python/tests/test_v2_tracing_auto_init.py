"""Regression: tracing primitives must trigger env-based auto-init.

Before this fix, only ``morse_ai.track()`` / ``record()`` called
``_ensure_initialized()``.  A process that used ``morse_ai.run()`` /
``morse_ai.span()`` / ``@morse_ai.trace_agent`` exclusively would never
build a transport — every trace silently fell through to ``_last_trace``
instead of being POSTed.  This was the gunicorn-sync-worker silent-drop
bug (AGE-240): refagents in apps/api/apps/refagents wrapped their runs in
``morse_ai.run(source="refagent")`` and never called track() first.

These tests pin the behaviour: env-based auto-init must fire on first
tracing-API touch, exactly like it does for track().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import morse_ai

if TYPE_CHECKING:
    import pytest


def setup_function() -> None:
    morse_ai._reset_for_testing()


def teardown_function() -> None:
    morse_ai._reset_for_testing()


def test_run_auto_inits_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MORSE_API_KEY", "av_run_auto_init")
    monkeypatch.setenv("MORSE_ENDPOINT", "http://localhost:9/v1")

    assert morse_ai._client is None
    with morse_ai.run("agent-x"):
        pass
    # First use of run() must have built the client.
    assert morse_ai._client is not None
    assert "localhost:9" in morse_ai._client.endpoint


def test_trace_agent_auto_inits_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MORSE_API_KEY", "av_trace_agent_auto_init")
    monkeypatch.setenv("MORSE_ENDPOINT", "http://localhost:9/v1")

    @morse_ai.trace_agent("decorated")
    def work() -> int:
        return 42

    assert morse_ai._client is None
    assert work() == 42
    assert morse_ai._client is not None


def test_run_silent_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MORSE_API_KEY", raising=False)
    # Must not raise — SDK stays dormant.
    with morse_ai.run("agent-y"):
        pass
    assert morse_ai._client is None
