"""
SDK integration test — requires a live Morse backend.

Skip in unit-only runs:
    pytest -m "not integration"

Run with:
    MORSE_API_KEY=av_... MORSE_API_URL=http://localhost:8000 \
    python -m pytest tests/test_integration.py -m integration -x -v
"""

import os
import time

import httpx
import pytest

import amund


@pytest.mark.req("US-01", "US-04")
@pytest.mark.integration
def test_sdk_record_appears_in_backend():
    """Integration: record() + flush() stores a run retrievable via the REST API."""
    api_url = os.environ.get("MORSE_API_URL", "http://localhost:8000")
    api_key = os.environ["MORSE_API_KEY"]  # KeyError = fast fail if not set

    agent_name = f"integration-test-agent-{int(time.time())}"

    amund.init(
        api_key=api_key,
        endpoint=f"{api_url}/api/v1/traces/batch",
    )
    amund.record(
        agent=agent_name,
        success=True,
        outcome="verified",
        cost=0.001,
        model="claude-sonnet-4-6",
        input_tokens=100,
        output_tokens=50,
    )
    amund.flush()

    # flush() waits for queue drain; background thread may still be mid-POST.
    # Give it a short grace period before polling.
    time.sleep(0.5)

    client = httpx.Client(
        headers={"Authorization": f"Bearer {api_key}"},
        base_url=api_url,
        timeout=10.0,
    )

    found = False
    for _attempt in range(10):
        resp = client.get("/api/v1/agents")
        resp.raise_for_status()
        data = resp.json()
        agents = data.get("results", data) if isinstance(data, dict) else data
        if any(a.get("name") == agent_name for a in agents):
            found = True
            break
        time.sleep(0.5)

    assert found, (
        f"Agent '{agent_name}' not found in backend after flush() + 5s polling. "
        f"Last response: {resp.text[:200]}"
    )
