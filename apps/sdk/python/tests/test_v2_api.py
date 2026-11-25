"""Phase 6c Wave 1.2 — V2 module-level singleton API tests.

Verifies:
- morse_ai.init() / track() / batch_track() / record() / flush() / shutdown()
  are exposed as module-level functions
- morse_ai.Morse (V1 class import) is gone from the public surface
- _Client is the private class backing the singleton
- _reset_for_testing() clears the singleton cleanly
- track()/batch_track()/record() silently no-op before init()
- Repeated init() replaces the singleton without leaking transports
"""

# ruff: noqa: E501

import pytest

import morse_ai
from morse_ai.client import _Client


def setup_function() -> None:
    """Reset the module-level singleton before every test."""
    morse_ai._reset_for_testing()


def teardown_function() -> None:
    """Reset after every test as well."""
    morse_ai._reset_for_testing()


def test_v2_public_api_is_module_level():
    """The V2 public surface is module-level functions, not class methods."""
    assert callable(morse_ai.init)
    assert callable(morse_ai.track)
    assert callable(morse_ai.batch_track)
    assert callable(morse_ai.record)
    assert callable(morse_ai.flush)
    assert callable(morse_ai.shutdown)


def test_v1_class_name_is_not_in_all():
    """V1 `MorseClient` / `Morse` must NOT be in the documented public surface."""
    assert "MorseClient" not in morse_ai.__all__
    assert "Morse" not in morse_ai.__all__
    # _Client is private (no leading-public name) and also not exported
    assert "_Client" not in morse_ai.__all__


def test_track_before_init_is_silent_noop():
    """Non-invasive guarantee: track() before init() returns silently."""
    # No assertion needed — the contract is "doesn't raise".
    morse_ai.track(agent_id="my-agent", run_data={"success": True})


def test_batch_track_before_init_is_silent_noop():
    morse_ai.batch_track(
        [{"agent": "a", "success": True}, {"agent": "b", "success": False}]
    )


def test_record_before_init_is_silent_noop():
    morse_ai.record(agent="my-agent", success=True, cost=0.01)


def test_flush_before_init_is_silent_noop():
    morse_ai.flush()


def test_init_creates_singleton():
    morse_ai.init(api_key="av_test_1234567890")
    assert morse_ai._client is not None
    assert isinstance(morse_ai._client, _Client)


def test_init_replaces_existing_singleton_cleanly():
    """Calling init() twice replaces the singleton without leaking transports."""
    morse_ai.init(api_key="av_first_key")
    first_client = morse_ai._client
    assert first_client is not None

    morse_ai.init(
        api_key="av_second_key", endpoint="https://example.com/api/v1/traces/batch"
    )
    second_client = morse_ai._client
    assert second_client is not None
    assert second_client is not first_client
    # The new client targets the new endpoint
    assert "example.com" in second_client.endpoint


def test_reset_for_testing_clears_singleton():
    morse_ai.init(api_key="av_reset_test")
    assert morse_ai._client is not None
    morse_ai._reset_for_testing()
    assert morse_ai._client is None


def test_track_after_init_does_not_raise():
    morse_ai.init(
        api_key="av_after_init", endpoint="http://localhost:9999/api/v1/traces/batch"
    )
    # The endpoint is unreachable but track() must NOT raise — async batching
    # absorbs network failures silently.
    morse_ai.track(agent_id="my-agent", run_data={"success": True, "cost_usd": 0.01})


def test_record_emits_v2_trace_not_runs():
    """record() must enqueue a {"traces":[...]} v2 payload, not the defunct {"runs":[...]}."""
    from morse_ai.tracing import _get_last_trace

    # init with an unreachable endpoint — transport queues but can't flush; _last_trace
    # is set synchronously inside _send_trace() before the background POST fires.
    morse_ai.init(
        api_key="av_test_record_v2",  # pragma: allowlist secret
        endpoint="http://localhost:19999/api/v1/traces/batch",
    )

    morse_ai.record(
        agent="test-agent",
        success=True,
        outcome="completed",
        cost=0.04,
        model="claude-sonnet-4-6",
        input_tokens=10,
        output_tokens=5,
    )

    last = _get_last_trace()
    assert last is not None, "record() did not emit a trace"

    # Must be a v2 trace entry: {"trace": {...}, "spans": [...]}
    assert "trace" in last, f"Expected v2 trace entry, got: {list(last.keys())}"
    assert "spans" in last, "Expected spans in v2 trace entry"

    trace = last["trace"]
    assert trace["agent"] == "test-agent"
    assert trace.get("success") is True
    assert trace.get("outcome") == "completed"

    spans = last["spans"]
    assert len(spans) == 1
    span = spans[0]
    assert span["type"] == "agent"
    assert span["name"] == "test-agent"
    assert span.get("cost_usd") == 0.04
    assert span.get("model") == "claude-sonnet-4-6"
    assert span.get("input_tokens") == 10
    assert span.get("output_tokens") == 5


@pytest.mark.real_default_endpoint
def test_default_endpoint_is_railway_prod():
    """The default endpoint is the prod API on api.amundai.com."""
    morse_ai.init(api_key="av_default_endpoint_test")
    assert morse_ai._client is not None
    assert morse_ai._client.endpoint == "https://api.amundai.com/api/v1/traces/batch"


@pytest.mark.real_default_endpoint  # no dependency-scan POST to the prod host
def test_morseai_dev_endpoint_still_accepted(monkeypatch):
    """api.morseai.dev stays a supported alias: pinning it is used as given."""
    monkeypatch.setenv("MORSE_ENDPOINT", "https://api.morseai.dev")
    morse_ai.init(api_key="av_alias_endpoint_test")  # pragma: allowlist secret
    assert morse_ai._client is not None
    assert morse_ai._client.endpoint == "https://api.morseai.dev/api/v1/traces/batch"


def test_explicit_endpoint_overrides_default():
    morse_ai.init(
        api_key="av_explicit_endpoint",
        endpoint="https://my-self-hosted.example.com/api/v1/traces/batch",
    )
    assert morse_ai._client is not None
    assert "my-self-hosted.example.com" in morse_ai._client.endpoint
