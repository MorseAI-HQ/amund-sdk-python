"""Shared test fixtures for SDK tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure src is on path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


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
