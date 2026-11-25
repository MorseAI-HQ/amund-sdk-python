"""Tests for set_context() and user_metadata propagation (AGE-320 Wave B)."""

from amund.tracing import _active_trace, _TraceState, set_context


def test_set_context_noop_outside_trace():
    # Should not raise; just silently no-ops
    set_context(customer_id="acme")


def test_set_context_sets_user_metadata():
    state = _TraceState(trace_id="test123", agent_name="test-agent")
    token = _active_trace.set(state)
    try:
        set_context(customer_id="acme", feature="doc-qa")
        assert state.user_metadata == {"customer_id": "acme", "feature": "doc-qa"}
    finally:
        _active_trace.reset(token)


def test_set_context_coerces_non_string():
    state = _TraceState(trace_id="test123", agent_name="test-agent")
    token = _active_trace.set(state)
    try:
        set_context(count=42)
        assert state.user_metadata["count"] == "42"
    finally:
        _active_trace.reset(token)


def test_set_context_merges_with_existing():
    state = _TraceState(trace_id="test123", agent_name="test-agent")
    state.user_metadata = {"customer_id": "acme"}
    token = _active_trace.set(state)
    try:
        set_context(feature="doc-qa")
        assert state.user_metadata == {"customer_id": "acme", "feature": "doc-qa"}
    finally:
        _active_trace.reset(token)


def test_user_metadata_emitted_in_trace_payload():
    """user_metadata set via set_context() appears in the emitted trace dict."""
    import amund
    from amund import tracing

    # Capture via _last_trace (no transport needed)
    tracing._transport = None
    tracing._last_trace = None

    with amund.run("test-agent"):
        set_context(customer_id="acme", feature="doc-qa")

    payload = tracing._get_last_trace()
    assert payload is not None
    assert payload["trace"]["user_metadata"] == {
        "customer_id": "acme",
        "feature": "doc-qa",
    }


def test_user_metadata_via_run_kwarg():
    """user_metadata passed directly to run() appears in the emitted trace dict."""
    import amund
    from amund import tracing

    tracing._transport = None
    tracing._last_trace = None

    with amund.run("test-agent", user_metadata={"customer_id": "beta", "env": "prod"}):
        pass

    payload = tracing._get_last_trace()
    assert payload is not None
    assert payload["trace"]["user_metadata"] == {"customer_id": "beta", "env": "prod"}


def test_set_context_exported_from_package():
    """set_context is accessible as amund.set_context."""
    import amund

    assert callable(amund.set_context)
    assert "set_context" in dir(amund)
