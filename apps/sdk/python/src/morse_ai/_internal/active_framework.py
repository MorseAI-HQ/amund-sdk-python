"""Thread-local active framework discovery (AGE-240 framework auto-detect).

Each Morse SDK adapter (anthropic, openai_agents, langgraph/langchain)
sets the active framework when it intercepts an LLM/agent call. The
``TracedRun.__enter__`` reads this thread-local at trace emission time and
stamps the framework as the ``morse_ai.framework`` resource attribute on
the trace envelope.

This is what makes the founder's "customer code is unchanged regardless of
SDK" promise load-bearing: the customer calls ``morse_ai.run("My Agent")``
once, the adapter populates this thread-local during ``messages.create`` /
``Runner.run`` / ``on_chain_start``, the SDK emits the framework as a
resource attribute, and the server uses that attribute to auto-disambiguate
agents with the same logical name but different implementations.

The thread-local sentinel value is ``None`` when no adapter is currently
active — the SDK omits the ``morse_ai.framework`` attribute in that case
(never invents a default; absence is the signal). All public functions are
no-ops if the thread-local isn't initialized.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar, Token

_STATE = threading.local()

# MHQ-1506: a framework adapter that records its own ``llm`` spans (the
# openai-agents runner hooks) marks the call tree it runs. Provider-level
# patches (the raw OpenAI client adapter) skip recording inside it so a model
# call is never counted twice. A ContextVar, not the thread-local above: it
# must follow the run into the asyncio tasks it creates and must not leak to
# unrelated coroutines on the same loop thread.
_LLM_OWNER: ContextVar[str | None] = ContextVar("morse_ai_llm_owner", default=None)


def set_llm_owner(name: str) -> Token[str | None]:
    """Mark ``name`` as recording the LLM calls made in this context."""
    return _LLM_OWNER.set(name)


def reset_llm_owner(token: Token[str | None]) -> None:
    """Undo ``set_llm_owner``. Never raises (a cross-context reset is ignored)."""
    try:
        _LLM_OWNER.reset(token)
    except (ValueError, RuntimeError):
        _LLM_OWNER.set(None)


def get_llm_owner() -> str | None:
    """The framework recording LLM calls in this context, or ``None``."""
    return _LLM_OWNER.get()


def set_active_framework(name: str) -> None:
    """Mark ``name`` as the active framework for this thread.

    Adapters call this at the start of their intercepted method (anthropic
    ``messages.create``, openai-agents ``Runner.run``, langchain
    ``on_chain_start``). Must be paired with ``clear_active_framework()``
    in a ``try/finally`` so a swallowed exception doesn't leak the marker
    across calls on the same worker thread.

    Side-effect: also writes the name onto the currently-active trace
    state's ``framework`` slot (if any). The thread-local alone is racy
    against the adapter's ``finally``: the trace's ``_build_trace_dict``
    runs AFTER the adapter cleared its marker, so the marker would be
    None at emit time. Stamping the trace state directly here makes
    framework discovery survive across the adapter's lifecycle.
    First framework wins per trace — if the customer is mixing two
    adapters in the same trace, the first one to fire defines the label.
    """
    _STATE.value = name
    # Local import to avoid a circular: tracing depends on adapters
    # indirectly via _ensure_sdk_initialized().
    try:
        from morse_ai.tracing import get_active_trace  # noqa: PLC0415

        state = get_active_trace()
        if state is not None and getattr(state, "framework", None) is None:
            state.framework = name
    except Exception:  # noqa: BLE001 — never break customer code
        pass


def get_active_framework() -> str | None:
    """Return the active framework for this thread, or ``None`` if unset."""
    return getattr(_STATE, "value", None)


def clear_active_framework() -> None:
    """Clear the thread-local marker. Idempotent; safe to call when unset."""
    if hasattr(_STATE, "value"):
        _STATE.value = None
