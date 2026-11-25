"""AMUND_* env vars with MORSE_* fallback (Amund rebrand, 2026-11-25).

Every SDK setting is read as ``AMUND_<NAME>`` first and ``MORSE_<NAME>``
second. Existing deployments that only set ``MORSE_*`` must keep working
unchanged, and when both are set the new name wins.
"""

from __future__ import annotations

import logging

import pytest

from amund import infra_capture, propagation, queue_capture
from amund._internal import agent_name, env

# Every setting the SDK reads, by suffix. Kept in sync with env.py's docstring.
_ALL_SUFFIXES = [
    "API_KEY",
    "ENDPOINT",
    "FLUSH_INTERVAL_SECONDS",
    "DISABLED",
    "SDK_DEBUG",
    "SDK_DISABLE_TELEMETRY",
    "DISABLE_AUTO_DETECT",
    "INTERNAL_SOURCE",
    "GIT_REVISION",
    "SAMPLE_RATE",
    "MAX_FIELD_BYTES",
    "MAX_QUEUE_SIZE",
    "MAX_BATCH_SIZE",
    "LEGACY_TRANSPORT",
    "AGENT_IDENTITY",
    "PROPAGATE_TRACE_CONTEXT",
    "CAPTURE_INFRA",
    "CAPTURE_DB",
    "CAPTURE_HTTP",
    "CAPTURE_HTTP_PATHS",
    "CAPTURE_HTTP_URLS",
    "CAPTURE_HTTP_EXCLUDE_HOSTS",
    "CAPTURE_QUEUE",
    "CAPTURE_QUEUE_CONSUMER",
]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for suffix in _ALL_SUFFIXES:
        monkeypatch.delenv("AMUND_" + suffix, raising=False)
        monkeypatch.delenv("MORSE_" + suffix, raising=False)
    env._conflicts_logged.clear()


@pytest.mark.parametrize("suffix", _ALL_SUFFIXES)
def test_legacy_morse_name_is_read_when_amund_is_unset(monkeypatch, suffix):
    monkeypatch.setenv("MORSE_" + suffix, "legacy-value")
    assert env.get(suffix) == "legacy-value"


@pytest.mark.parametrize("suffix", _ALL_SUFFIXES)
def test_amund_name_is_read(monkeypatch, suffix):
    monkeypatch.setenv("AMUND_" + suffix, "new-value")
    assert env.get(suffix) == "new-value"


@pytest.mark.parametrize("suffix", _ALL_SUFFIXES)
def test_amund_wins_when_both_are_set(monkeypatch, suffix):
    monkeypatch.setenv("AMUND_" + suffix, "new-value")
    monkeypatch.setenv("MORSE_" + suffix, "legacy-value")
    assert env.get(suffix) == "new-value"


def test_name_may_be_given_with_either_prefix_or_bare(monkeypatch):
    monkeypatch.setenv("MORSE_API_KEY", "legacy")
    assert env.get("API_KEY") == env.get("MORSE_API_KEY") == "legacy"
    assert env.get("AMUND_API_KEY") == "legacy"
    monkeypatch.setenv("AMUND_API_KEY", "new")
    assert env.get("API_KEY") == env.get("MORSE_API_KEY") == "new"


def test_blank_amund_value_falls_back_to_morse(monkeypatch):
    """Empty/whitespace counts as unset, as it always did for MORSE_*."""
    monkeypatch.setenv("AMUND_API_KEY", "   ")
    monkeypatch.setenv("MORSE_API_KEY", "legacy")
    assert env.get_api_key() == "legacy"


def test_unset_everywhere_returns_default():
    assert env.get("API_KEY") == ""
    assert env.get("API_KEY", "fallback") == "fallback"
    assert env.get_api_key() is None


def test_conflict_is_logged_once_at_debug(monkeypatch, caplog):
    monkeypatch.setenv("AMUND_SAMPLE_RATE", "0.5")
    monkeypatch.setenv("MORSE_SAMPLE_RATE", "0.25")
    with caplog.at_level(logging.DEBUG, logger="amund._internal"):
        assert env.get_sample_rate() == 0.5
        assert env.get_sample_rate() == 0.5
    conflicts = [r for r in caplog.records if "both set and differ" in r.message]
    assert len(conflicts) == 1
    assert conflicts[0].levelno == logging.DEBUG
    # The values themselves never reach the log (one of them may be a key).
    assert "0.25" not in conflicts[0].getMessage()


def test_identical_values_are_not_a_conflict(monkeypatch, caplog):
    monkeypatch.setenv("AMUND_API_KEY", "same")  # pragma: allowlist secret
    monkeypatch.setenv("MORSE_API_KEY", "same")  # pragma: allowlist secret
    with caplog.at_level(logging.DEBUG, logger="amund._internal"):
        assert env.get_api_key() == "same"
    assert not [r for r in caplog.records if "both set" in r.message]


# --- the typed helpers in env.py -------------------------------------------


def test_typed_helpers_follow_precedence(monkeypatch):
    monkeypatch.setenv("MORSE_API_KEY", "legacy-key")  # pragma: allowlist secret
    monkeypatch.setenv("MORSE_ENDPOINT", "https://legacy.example")
    monkeypatch.setenv("MORSE_FLUSH_INTERVAL_SECONDS", "5")
    monkeypatch.setenv("MORSE_MAX_FIELD_BYTES", "100")
    assert env.get_api_key() == "legacy-key"
    assert env.get_endpoint() == "https://legacy.example/api/v1/traces/batch"
    assert env.get_flush_interval() == 5.0
    assert env.get_max_field_bytes() == 100

    monkeypatch.setenv("AMUND_API_KEY", "new-key")  # pragma: allowlist secret
    monkeypatch.setenv("AMUND_ENDPOINT", "https://new.example")
    monkeypatch.setenv("AMUND_FLUSH_INTERVAL_SECONDS", "7")
    monkeypatch.setenv("AMUND_MAX_FIELD_BYTES", "200")
    assert env.get_api_key() == "new-key"
    assert env.get_endpoint() == "https://new.example/api/v1/traces/batch"
    assert env.get_flush_interval() == 7.0
    assert env.get_max_field_bytes() == 200


def test_boolean_flags_follow_precedence(monkeypatch):
    monkeypatch.setenv("MORSE_DISABLED", "1")
    assert env.is_disabled() is True
    monkeypatch.setenv("AMUND_DISABLED", "0")
    assert env.is_disabled() is False


def test_internal_source_follows_precedence(monkeypatch):
    monkeypatch.setenv("MORSE_INTERNAL_SOURCE", "copilot")
    assert env.get_internal_source() == "copilot"
    monkeypatch.setenv("AMUND_INTERNAL_SOURCE", "investigator")
    assert env.get_internal_source() == "investigator"


def test_git_revision_follows_precedence(monkeypatch):
    for var in env._GIT_REVISION_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GITHUB_SHA", "from-ci")
    monkeypatch.setenv("MORSE_GIT_REVISION", "legacy-sha")
    assert env.get_git_revision() == "legacy-sha"
    monkeypatch.setenv("AMUND_GIT_REVISION", "new-sha")
    assert env.get_git_revision() == "new-sha"


# --- reads that live outside env.py go through env.get too -----------------


def test_agent_identity_reads_amund(monkeypatch):
    monkeypatch.setenv("MORSE_AGENT_IDENTITY", "legacy")
    assert agent_name.identity_legacy() is True
    monkeypatch.setenv("AMUND_AGENT_IDENTITY", "v2")
    assert agent_name.identity_legacy() is False


def test_propagation_switch_reads_amund(monkeypatch):
    monkeypatch.setenv("MORSE_PROPAGATE_TRACE_CONTEXT", "false")
    assert propagation.propagation_enabled() is False
    monkeypatch.setenv("AMUND_PROPAGATE_TRACE_CONTEXT", "true")
    assert propagation.propagation_enabled() is True


def test_capture_switches_read_amund(monkeypatch):
    monkeypatch.setenv("MORSE_CAPTURE_INFRA", "0")
    assert infra_capture.capture_enabled("AMUND_CAPTURE_DB") is False
    monkeypatch.setenv("AMUND_CAPTURE_DB", "1")
    assert infra_capture.capture_enabled("MORSE_CAPTURE_DB") is True


def test_queue_consumer_switch_reads_amund(monkeypatch):
    monkeypatch.setenv("MORSE_CAPTURE_QUEUE_CONSUMER", "1")
    assert queue_capture._consumer_enabled() is True
    monkeypatch.setenv("AMUND_CAPTURE_QUEUE_CONSUMER", "0")
    assert queue_capture._consumer_enabled() is False
