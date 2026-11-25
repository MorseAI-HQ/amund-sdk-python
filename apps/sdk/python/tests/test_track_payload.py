"""MHQ-1410 item 4 — ``track()`` / ``batch_track()`` send a payload the server accepts.

Before the fix both queued the bare run dict; ``POST /traces/batch`` answered
422 ``{"loc": ["body", "payload", "traces"], "msg": "Field required"}`` and
the SDK swallowed it, so nothing was stored. They now go through the same
single-span v2 trace as ``record()``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

import amund
from amund.tracing import _get_last_trace


@pytest.fixture
def queued() -> Any:
    amund._reset_for_testing()
    amund.init(
        api_key="av_test_track",  # pragma: allowlist secret
        endpoint="http://localhost:19999/api/v1/traces/batch",
    )
    events: list[dict[str, Any]] = []
    assert amund._client is not None
    with patch.object(
        amund._client.transport, "track", side_effect=lambda e: events.append(e)
    ):
        yield events
    amund._reset_for_testing()


@pytest.mark.req("MHQ-1410")
def test_track_queues_a_v2_trace(queued: list[dict[str, Any]]) -> None:
    amund.track(
        agent_id="lead-qualifier",
        run_data={
            "success": True,
            "cost_usd": 0.012,
            "model": "gpt-4.1-nano",
            "input_tokens": 100,
            "output_tokens": 20,
            "duration_ms": 900,
            "customer_tier": "pro",
        },
    )
    assert len(queued) == 1
    assert set(queued[0]) == {"traces"}
    entry = queued[0]["traces"][0]
    assert entry["trace"]["agent"] == "lead-qualifier"
    assert entry["trace"]["success"] is True
    (span,) = entry["spans"]
    assert span["type"] == "agent"
    assert span["cost_usd"] == 0.012
    assert span["model"] == "gpt-4.1-nano"
    assert span["input_tokens"] == 100
    assert span["metadata"] == {"customer_tier": "pro"}
    assert _get_last_trace()["trace"]["agent"] == "lead-qualifier"


@pytest.mark.req("MHQ-1410")
def test_batch_track_queues_one_trace_per_run(queued: list[dict[str, Any]]) -> None:
    amund.batch_track(
        [
            {"agent": "a", "success": True},
            {"agent": "b", "error_message": "boom"},
            {"success": True},  # no agent: skipped, not sent as an invalid payload
        ]
    )
    agents = [e["traces"][0]["trace"]["agent"] for e in queued]
    assert agents == ["a", "b"]
    assert queued[1]["traces"][0]["trace"]["status"] == "failed"
