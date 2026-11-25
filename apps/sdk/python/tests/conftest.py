"""Shared test fixtures for SDK tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure src is on path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(autouse=True)
def _no_ambient_amund_env(monkeypatch):
    """Drop any ``AMUND_*`` variables inherited from the developer's shell.

    ``AMUND_<NAME>`` outranks ``MORSE_<NAME>`` (see ``amund._internal.env``),
    so a stray ``AMUND_API_KEY`` or ``AMUND_ENDPOINT`` in the environment would
    silently override every test that sets or clears the ``MORSE_`` name.
    """
    for name in [k for k in os.environ if k.startswith("AMUND_")]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _dead_default_endpoint(request, monkeypatch):
    """Point every test at a dead local endpoint unless it sets its own.

    Without this, a bare `init()` with a fake key resolves to the production default
    and the abandoned OTLP exporter threads (see
    `_OtelEngine.reset_for_testing`) keep POSTing fake-key spans to prod.
    Tests that assert on the default endpoint opt out with
    `@pytest.mark.real_default_endpoint`; those keep the prod URL but lose
    `init()`'s dependency-scan POST, the one request they would otherwise make.
    """
    if request.node.get_closest_marker("real_default_endpoint") is None:
        monkeypatch.setenv("MORSE_ENDPOINT", "http://localhost:0")
    else:
        monkeypatch.delenv("MORSE_ENDPOINT", raising=False)
        monkeypatch.setattr("amund._flush_dependencies_async", lambda client: None)


@pytest.fixture
def anthropic_response():
    """Mock Anthropic Message response with usage data."""
    return SimpleNamespace(
        id="msg_test123",
        model="claude-sonnet-4-6-20260217",
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=1200,
            output_tokens=350,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
        ),
    )


@pytest.fixture
def anthropic_cached_response():
    """Mock Anthropic response with cache usage."""
    return SimpleNamespace(
        id="msg_test456",
        model="claude-sonnet-4-6-20260217",
        usage=SimpleNamespace(
            input_tokens=200,
            output_tokens=350,
            cache_creation_input_tokens=500,
            cache_read_input_tokens=300,
        ),
    )


@pytest.fixture
def openai_chat_response():
    """Mock OpenAI Chat Completions response."""
    return SimpleNamespace(
        id="chatcmpl-test789",
        model="gpt-4.1-2025-04-14",
        usage=SimpleNamespace(
            prompt_tokens=800,
            completion_tokens=200,
            total_tokens=1000,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )


@pytest.fixture
def openai_responses_api():
    """Mock OpenAI Responses API response (different field names)."""
    return SimpleNamespace(
        id="resp_test101",
        model="gpt-4.1",
        usage=SimpleNamespace(
            input_tokens=800,
            output_tokens=200,
            total_tokens=1000,
            input_tokens_details=SimpleNamespace(cached_tokens=100),
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )


@pytest.fixture
def legacy_identity(monkeypatch):
    """AID-09: pre-v2 naming and span shape (MORSE_AGENT_IDENTITY=legacy).

    Used by tests that pin the old "every chain is an agent span" shape.
    """
    monkeypatch.setenv("MORSE_AGENT_IDENTITY", "legacy")
