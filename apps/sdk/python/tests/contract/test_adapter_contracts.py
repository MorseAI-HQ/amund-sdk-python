"""Structural contract checks against the REAL agent SDKs — see ./README.md.

Every adapter's ``install()`` monkey-patches a method on someone else's class.
When upstream moves that method, ``install()`` swallows the failure, returns
``False``, and the customer gets no telemetry and no error. These tests assert
the patch targets still exist, so that becomes a red build instead.

Unlike the TypeScript contract check — which had to run outside vitest because
the adapters resolve through an eval'd dynamic import — Python's adapters use
ordinary imports, so plain pytest is the right home.

Offline only: no API keys, no network, no model calls. Nothing here constructs
a client that would talk to a provider.
"""

from __future__ import annotations

import importlib
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

import pytest

# MHQ-930: mirrors CONTRACT_ONLY in the TypeScript check. The nightly `latest`
# lanes bump one package at a time; a lane should report on its own package
# rather than re-reporting the others still at their declared floor.
_ONLY = (os.environ.get("CONTRACT_ONLY") or "").strip() or None

# Every package this file has assertions for. Used to reject a CONTRACT_ONLY
# value that matches nothing — otherwise a typo'd or renamed lane skips every
# test, reports "N skipped", and exits 0. A lane that asserted nothing must
# never read the same as a lane that passed.
_KNOWN_PACKAGES = frozenset(
    {
        "anthropic",
        "openai",
        "openai-agents",
        "claude-agent-sdk",
        "langgraph",
        "langchain-core",
    }
)


def _only(pkg: str) -> pytest.MarkDecorator:
    """Skip unless this package is the one under test (or no filter is set)."""
    return pytest.mark.skipif(
        _ONLY is not None and pkg != _ONLY,
        reason=f"filtered out by CONTRACT_ONLY={_ONLY}",
    )


def _installed(dist: str) -> str:
    try:
        return pkg_version(dist)
    except PackageNotFoundError:  # pragma: no cover - only when extra is absent
        return "not installed"


def test_contract_only_matches_a_real_package() -> None:
    """A filter that matches nothing is a broken lane, not a passing one."""
    if _ONLY is None:
        pytest.skip("no CONTRACT_ONLY filter set — full run")
    assert _ONLY in _KNOWN_PACKAGES, (
        f"CONTRACT_ONLY={_ONLY!r} matches no assertions in this file. Check the "
        f"spelling against {sorted(_KNOWN_PACKAGES)} — a lane that runs nothing "
        "must not report success."
    )


def test_report_installed_versions(capsys: pytest.CaptureFixture[str]) -> None:
    """Print what this run actually proved. A green run must say against what."""
    dists = [
        "anthropic",
        "openai",
        "openai-agents",
        "langgraph",
        "langchain-core",
        "claude-agent-sdk",
    ]
    with capsys.disabled():
        print("\n  agent-SDK contract check — installed versions")
        if _ONLY:
            print(f"  (filtered to {_ONLY})")
        for d in dists:
            print(f"    {d}=={_installed(d)}")


# ---------------------------------------------------------------------------
# anthropic — adapters/anthropic.py::install patches Messages.create
# ---------------------------------------------------------------------------


@_only("anthropic")
def test_anthropic_messages_create_is_patchable() -> None:
    from anthropic.resources.messages import Messages

    assert callable(getattr(Messages, "create", None)), (
        "anthropic.resources.messages.Messages.create is gone — "
        "adapters/anthropic.py::install() patches exactly this and would "
        "silently return False"
    )


# ---------------------------------------------------------------------------
# openai — adapters/openai.py::install patches Completions.create
# ---------------------------------------------------------------------------


@_only("openai")
def test_openai_completions_create_is_patchable() -> None:
    from openai.resources.chat.completions import Completions

    assert callable(getattr(Completions, "create", None)), (
        "openai.resources.chat.completions.Completions.create is gone — "
        "adapters/openai.py::install() patches exactly this"
    )


@_only("openai")
def test_openai_responses_create_is_patchable() -> None:
    from openai.resources.responses import AsyncResponses, Responses

    assert callable(getattr(Responses, "create", None)), (
        "openai.resources.responses.Responses.create is gone — "
        "adapters/openai.py::install() patches exactly this (MHQ-1198)"
    )
    assert callable(getattr(AsyncResponses, "create", None))


# ---------------------------------------------------------------------------
# openai-agents — adapters/openai_agents.py::install patches Runner.run
# ---------------------------------------------------------------------------


@_only("openai-agents")
def test_openai_agents_runner_methods_are_patchable() -> None:
    from agents.run import Runner

    # install() patches all three; a rename of any one silently drops that
    # entry point's telemetry while the other two keep working, which is the
    # hardest version of this bug to notice.
    for method in ("run", "run_sync", "run_streamed"):
        assert callable(getattr(Runner, method, None)), (
            f"agents.run.Runner.{method} is gone — adapters/openai_agents.py patches it"
        )


# ---------------------------------------------------------------------------
# claude-agent-sdk — adapters/claude_agent_sdk.py patches
# ClaudeSDKClient.receive_response
#
# Note this is the mirror image of TypeScript: the PyPI package really does
# export ClaudeSDKClient, which is why the Python adapter works and the TS one
# had to be rebuilt around query() (MHQ-936). Same product, different API.
# ---------------------------------------------------------------------------


@_only("claude-agent-sdk")
def test_claude_agent_sdk_client_receive_response_is_patchable() -> None:
    client_mod = importlib.import_module("claude_agent_sdk.client")
    client_cls = getattr(client_mod, "ClaudeSDKClient", None)

    assert client_cls is not None, (
        "claude_agent_sdk.client.ClaudeSDKClient is gone. Unlike the TS "
        "package (which never had it), the Python package's adapter depends "
        "on this class existing"
    )
    assert callable(getattr(client_cls, "receive_response", None)), (
        "ClaudeSDKClient.receive_response is gone — adapters/claude_agent_sdk.py "
        "patches exactly this"
    )


# ---------------------------------------------------------------------------
# langgraph — adapters/langgraph.py::install patches
# langchain_core BaseCallbackManager.__init__ to inject the handler globally
# ---------------------------------------------------------------------------


@_only("langgraph")
def test_langgraph_callback_manager_hook_is_patchable() -> None:
    from langchain_core.callbacks.manager import BaseCallbackManager

    assert callable(getattr(BaseCallbackManager, "__init__", None)), (
        "langchain_core.callbacks.manager.BaseCallbackManager.__init__ is gone "
        "— adapters/langgraph.py::install() wraps it to register the handler "
        "globally"
    )


# ---------------------------------------------------------------------------
# langchain-core — adapters/langchain.py subclasses BaseCallbackHandler
# ---------------------------------------------------------------------------


@_only("langchain-core")
def test_langchain_handler_subclasses_the_real_base() -> None:
    from langchain_core.callbacks import BaseCallbackHandler

    from amund.adapters.langchain import MorseCallbackHandler

    # The adapter falls back to a stub base class when langchain_core is
    # missing. With it installed the handler must be a REAL subclass, or
    # LangChain will not treat it as a callback handler at all.
    assert issubclass(MorseCallbackHandler, BaseCallbackHandler), (
        "MorseCallbackHandler is not a real BaseCallbackHandler subclass — the "
        "import-time fallback stub was used, so LangChain will ignore it"
    )

    handler = MorseCallbackHandler("contract-check")
    for hook in ("on_chain_start", "on_llm_start", "on_tool_start"):
        assert callable(getattr(handler, hook, None)), (
            f"MorseCallbackHandler.{hook} is missing — LangChain dispatches to it"
        )
