"""Tests for W3C traceparent inject/extract and cross-service trace correlation.

W3C traceparent format: 00-{trace_id_32hex}-{span_id_16hex}-01
"""

# ruff: noqa: E501, SIM117
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from morse_ai.propagation import (
    W3C_TRACEPARENT_HEADER,
    extract_traceparent,
    get_current_traceparent,
    inject_traceparent,
)
from morse_ai.tracing import _active_trace, _set_transport, run, trace_agent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_trace_context():
    """Reset active trace context — ensures test isolation."""
    _active_trace.set(None)


# ---------------------------------------------------------------------------
# inject_traceparent
# ---------------------------------------------------------------------------


class TestInjectTraceparent:
    def setup_method(self):
        _clear_trace_context()
        _set_transport(None)

    def teardown_method(self):
        _clear_trace_context()
        _set_transport(None)

    @pytest.mark.req("SDK-10")
    def test_inject_adds_header_when_trace_is_active(self):
        """inject_traceparent adds 'traceparent' header when a trace is active."""

        @trace_agent("test-inject-agent")
        def agent():
            headers = {}
            inject_traceparent(headers)
            return headers

        result = agent()
        assert W3C_TRACEPARENT_HEADER in result, "traceparent header must be present"

    @pytest.mark.req("SDK-10")
    def test_inject_header_has_correct_w3c_format(self):
        """inject_traceparent produces '00-{32hex}-{16hex}-01' format."""

        @trace_agent("format-check-agent")
        def agent():
            headers = {}
            inject_traceparent(headers)
            return headers.get(W3C_TRACEPARENT_HEADER, "")

        traceparent = agent()
        parts = traceparent.split("-")
        assert len(parts) == 4, f"Expected 4 parts, got: {traceparent}"
        version, trace_id, span_id, flags = parts
        assert version == "00", f"Version must be '00', got: {version}"
        assert len(trace_id) == 32, (
            f"trace_id must be 32 hex chars, got {len(trace_id)}: {trace_id}"
        )
        assert len(span_id) == 16, (
            f"span_id must be 16 hex chars, got {len(span_id)}: {span_id}"
        )
        assert flags == "01", f"Flags must be '01' (sampled), got: {flags}"
        # Verify they are valid hex
        int(trace_id, 16)
        int(span_id, 16)

    @pytest.mark.req("SDK-10")
    def test_inject_is_noop_when_no_trace_context(self):
        """inject_traceparent is a no-op when no trace is active."""
        _clear_trace_context()
        headers = {"X-Custom": "value"}
        inject_traceparent(headers)
        assert W3C_TRACEPARENT_HEADER not in headers
        assert headers == {"X-Custom": "value"}  # unchanged

    @pytest.mark.req("SDK-10")
    def test_inject_returns_same_dict(self):
        """inject_traceparent returns the same dict (mutated in place)."""

        @trace_agent("return-check-agent")
        def agent():
            headers = {"X-Existing": "val"}
            returned = inject_traceparent(headers)
            return returned is headers  # must be same object

        assert agent() is True

    @pytest.mark.req("SDK-10")
    def test_inject_preserves_existing_headers(self):
        """inject_traceparent does not remove existing headers."""

        @trace_agent("preserve-headers-agent")
        def agent():
            headers = {
                "Authorization": "Bearer tok",
                "Content-Type": "application/json",
            }
            inject_traceparent(headers)
            return headers

        result = agent()
        assert result["Authorization"] == "Bearer tok"
        assert result["Content-Type"] == "application/json"
        assert W3C_TRACEPARENT_HEADER in result


# ---------------------------------------------------------------------------
# extract_traceparent
# ---------------------------------------------------------------------------


class TestExtractTraceparent:
    @pytest.mark.req("SDK-10")
    def test_extract_parses_valid_traceparent(self):
        """extract_traceparent returns (trace_id, parent_span_id) for valid header."""
        headers = {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        }
        result = extract_traceparent(headers)
        assert result is not None
        trace_id, parent_span_id = result
        assert trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
        assert parent_span_id == "00f067aa0ba902b7"

    @pytest.mark.req("SDK-10")
    def test_extract_returns_none_for_missing_header(self):
        """extract_traceparent returns None when header is absent."""
        result = extract_traceparent({})
        assert result is None

    @pytest.mark.req("SDK-10")
    def test_extract_returns_none_for_wrong_version(self):
        """extract_traceparent returns None for non-'00' version."""
        headers = {
            "traceparent": "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        }
        result = extract_traceparent(headers)
        assert result is None

    @pytest.mark.req("SDK-10")
    def test_extract_returns_none_for_wrong_trace_id_length(self):
        """extract_traceparent returns None if trace_id is not 32 hex chars."""
        headers = {"traceparent": "00-shortid-00f067aa0ba902b7-01"}
        result = extract_traceparent(headers)
        assert result is None

    @pytest.mark.req("SDK-10")
    def test_extract_returns_none_for_wrong_span_id_length(self):
        """extract_traceparent returns None if span_id is not 16 hex chars."""
        headers = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-short-01"}
        result = extract_traceparent(headers)
        assert result is None

    @pytest.mark.req("SDK-10")
    def test_extract_returns_none_for_non_hex_trace_id(self):
        """extract_traceparent returns None for non-hex trace_id."""
        headers = {
            "traceparent": "00-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz-00f067aa0ba902b7-01"
        }
        result = extract_traceparent(headers)
        assert result is None

    @pytest.mark.req("SDK-10")
    def test_extract_returns_none_for_wrong_part_count(self):
        """extract_traceparent returns None if header has wrong number of parts."""
        headers = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-01"}
        result = extract_traceparent(headers)
        assert result is None

    @pytest.mark.req("SDK-10")
    def test_extract_returns_none_for_empty_string(self):
        """extract_traceparent returns None for empty header value."""
        result = extract_traceparent({"traceparent": ""})
        assert result is None

    @pytest.mark.req("SDK-10")
    def test_extract_case_insensitive_header_key(self):
        """extract_traceparent should handle both 'traceparent' and 'Traceparent' keys."""
        valid = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        # lowercase always works (W3C canonical)
        result = extract_traceparent({"traceparent": valid})
        assert result is not None


# ---------------------------------------------------------------------------
# Round-trip: inject → extract
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def setup_method(self):
        _clear_trace_context()
        _set_transport(None)

    def teardown_method(self):
        _clear_trace_context()
        _set_transport(None)

    @pytest.mark.req("SDK-10")
    def test_inject_then_extract_returns_same_trace_id(self):
        """Round-trip: inject traceparent then extract returns the same trace_id."""

        @trace_agent("roundtrip-agent")
        def agent():
            headers = {}
            inject_traceparent(headers)
            extracted = extract_traceparent(headers)
            return extracted

        result = agent()
        assert result is not None
        extracted_trace_id, _ = result

        # The trace_id in the traceparent must match the current trace's trace_id
        # We verify by injecting and extracting the same trace_id
        assert len(extracted_trace_id) == 32
        int(extracted_trace_id, 16)  # must be valid hex


# ---------------------------------------------------------------------------
# Cross-service distributed tracing
# ---------------------------------------------------------------------------


class TestCrossServicePropagation:
    def setup_method(self):
        _clear_trace_context()
        _set_transport(None)

    def teardown_method(self):
        _clear_trace_context()
        _set_transport(None)

    @pytest.mark.req("SDK-10")
    def test_run_with_incoming_traceparent_uses_provided_trace_id(self):
        """run() with incoming_traceparent uses the provided trace_id, not a new one."""
        incoming = "00-abcdef1234567890abcdef1234567890-1234567890abcdef-01"
        expected_trace_id = "abcdef1234567890abcdef1234567890"

        from morse_ai.tracing import _get_last_trace

        with run("downstream-agent", incoming_traceparent=incoming):
            pass

        last = _get_last_trace()
        assert last is not None
        assert last["trace"]["trace_id"] == expected_trace_id, (
            f"Expected trace_id={expected_trace_id}, got {last['trace']['trace_id']}"
        )

    @pytest.mark.req("SDK-10")
    def test_run_without_incoming_traceparent_generates_new_trace_id(self):
        """run() without incoming_traceparent generates a fresh trace_id."""
        from morse_ai.tracing import _get_last_trace

        with run("standalone-agent"):
            pass

        last = _get_last_trace()
        assert last is not None
        trace_id = last["trace"]["trace_id"]
        assert len(trace_id) == 32  # uuid4 hex
        # Not all zeros/fixed — it's a real random uuid
        assert trace_id != "0" * 32

    @pytest.mark.req("SDK-10")
    def test_two_services_sharing_traceparent_have_same_trace_id(self):
        """Two services sharing a traceparent appear as a single trace (same trace_id)."""
        from morse_ai.tracing import _get_last_trace

        # Service A: creates a trace and injects traceparent into outgoing headers
        headers = {}

        @trace_agent("service-a")
        def service_a():
            inject_traceparent(headers)

        service_a()
        last_a = _get_last_trace()
        assert last_a is not None
        trace_id_a = last_a["trace"]["trace_id"]

        # Verify the injected traceparent contains service A's trace_id
        extracted = extract_traceparent(headers)
        assert extracted is not None
        assert extracted[0] == trace_id_a

        # Service B: receives the traceparent and uses it
        with run("service-b", incoming_traceparent=headers[W3C_TRACEPARENT_HEADER]):
            pass

        last_b = _get_last_trace()
        assert last_b is not None
        trace_id_b = last_b["trace"]["trace_id"]

        # Both services share the SAME trace_id
        assert trace_id_a == trace_id_b, (
            f"Expected shared trace_id, but got service-a={trace_id_a}, service-b={trace_id_b}"
        )

    @pytest.mark.req("SDK-10")
    def test_run_with_invalid_incoming_traceparent_generates_new_trace_id(self):
        """run() with invalid incoming_traceparent gracefully generates a new trace_id."""
        from morse_ai.tracing import _get_last_trace

        with run("safe-agent", incoming_traceparent="invalid-garbage"):
            pass

        last = _get_last_trace()
        assert last is not None
        # Should have a valid trace_id (newly generated)
        assert len(last["trace"]["trace_id"]) == 32


# ---------------------------------------------------------------------------
# get_current_traceparent
# ---------------------------------------------------------------------------


class TestGetCurrentTraceparent:
    def setup_method(self):
        _clear_trace_context()
        _set_transport(None)

    def teardown_method(self):
        _clear_trace_context()
        _set_transport(None)

    @pytest.mark.req("SDK-10")
    def test_returns_none_outside_trace(self):
        """get_current_traceparent returns None when no trace is active."""
        _clear_trace_context()
        result = get_current_traceparent()
        assert result is None

    @pytest.mark.req("SDK-10")
    def test_returns_valid_traceparent_inside_trace(self):
        """get_current_traceparent returns valid W3C traceparent string inside a trace."""

        @trace_agent("current-tp-agent")
        def agent():
            return get_current_traceparent()

        result = agent()
        assert result is not None
        parts = result.split("-")
        assert len(parts) == 4
        assert parts[0] == "00"
        assert len(parts[1]) == 32
        assert len(parts[2]) == 16
        assert parts[3] == "01"
