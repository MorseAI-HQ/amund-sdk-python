"""Auto-detect installed AI SDKs and install the matching adapter.

Pattern (auto-instrumentation, as popularised by Sentry): when the Morse
SDK initializes, scan
`sys.modules` for known AI SDK packages and call each detected adapter's
`install()` function. For modules not yet imported, register an import-time
hook so the adapter installs lazily on first import.

Customer flow becomes:

    pip install morse_ai
    export MORSE_API_KEY=av_...
    # ... existing anthropic / openai-agents / langgraph code is now traced ...

No `morse_ai.instrument_anthropic()` calls in customer code.

Priority rules (most specific wins):
- LangGraph beats generic LangChain (when both are importable, install only
  the LangGraph adapter to avoid duplicate spans).

Idempotency: each adapter's `install()` is itself idempotent (sets a flag
on its target module). Calling `install_adapters()` repeatedly is safe and
is the path the SDK takes on every `init()`.

Opt-out: customers who set `MORSE_DISABLE_AUTO_DETECT=1` skip this
step entirely. They can still call adapter `install()` functions directly.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import logging
import sys
import threading
import time
from typing import Any

from morse_ai._internal import env

logger = logging.getLogger("morse_ai")

# Module-level state for the import-hook lazy installer.
_install_lock = threading.Lock()
_lazy_targets: dict[str, str] = {}  # target module name -> adapter dotted path
_finder_installed = False


def _adapter_targets() -> dict[str, str]:
    """Return the resolved {target_module_name: adapter_dotted_path} mapping.

    Applies priority rules:
    - LangGraph wins over generic LangChain when both are importable.
    """
    targets: dict[str, str] = {
        "anthropic": "morse_ai.adapters.anthropic",
    }

    # LangGraph is preferred when available — registers under "langgraph" so
    # the lazy hook fires on `import langgraph` regardless of order.
    if importlib.util.find_spec("langgraph") is not None:
        targets["langgraph"] = "morse_ai.adapters.langgraph"

    # OpenAI Agents SDK — package is published as `openai-agents` on PyPI but
    # imports as the top-level `agents` module. Wave 4 adapter.
    if importlib.util.find_spec("agents") is not None:
        targets["agents"] = "morse_ai.adapters.openai_agents"

    # Anthropic claude-agent-sdk — shells out to the `claude` CLI binary, so
    # the generic Anthropic adapter never fires. This adapter intercepts the
    # message stream at ClaudeSDKClient.receive_response() instead.
    if importlib.util.find_spec("claude_agent_sdk") is not None:
        targets["claude_agent_sdk"] = "morse_ai.adapters.claude_agent_sdk"

    # The generic langchain.py adapter (callback handler) is consumed
    # explicitly by Copilot and other in-tree code; it's not auto-installed
    # because there's no global hook for it yet. When/if we add one, this
    # block adds it as a fallback when LangGraph is NOT present.
    # if "langgraph" not in targets and \
    #         importlib.util.find_spec("langchain") is not None:
    #     targets["langchain"] = "morse_ai.adapters.langchain"

    return targets


def _install_one(adapter_dotted_path: str) -> bool:
    """Import the adapter module and call its `install()`. Idempotent + safe.

    Returns True if install() returned truthy, False otherwise. Never raises —
    the adapter's `install()` is itself wrapped in `@absorb_errors`.
    """
    try:
        adapter_module = importlib.import_module(adapter_dotted_path)
    except Exception as exc:  # noqa: BLE001 — non-invasive
        logger.debug(
            "morse_ai: failed to import adapter %s: %s", adapter_dotted_path, exc
        )
        return False
    install_fn = getattr(adapter_module, "install", None)
    if install_fn is None:
        logger.debug(
            "morse_ai: adapter %s has no install(); skipping",
            adapter_dotted_path,
        )
        return False
    try:
        return bool(install_fn())
    except Exception as exc:  # noqa: BLE001 — defensive (install is already wrapped)
        logger.debug(
            "morse_ai: adapter %s install() raised: %s", adapter_dotted_path, exc
        )
        return False


class _LazyInstallFinder(importlib.abc.MetaPathFinder):
    """MetaPathFinder that fires adapter install() when a target module imports.

    Doesn't actually find or load anything — its only job is observing
    `find_spec` calls. Returns None so the standard import machinery
    proceeds; meanwhile, install() is scheduled to run after the target
    module finishes loading.
    """

    def __init__(self) -> None:
        # Track which modules we've already kicked off install() for so we
        # don't re-fire on every import statement that touches them.
        self._fired: set[str] = set()

    def find_spec(
        self,
        fullname: str,
        path: list[str] | None = None,
        target: Any | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in _lazy_targets or fullname in self._fired:
            return None
        # Mark as fired BEFORE returning None so the import doesn't recursively
        # trigger us via our own install path.
        self._fired.add(fullname)
        adapter_path = _lazy_targets[fullname]

        # The target module's body hasn't run yet — its attributes (like
        # anthropic.resources.messages.Messages) aren't accessible until
        # the import completes. Defer install() to a background thread that
        # waits for sys.modules registration.
        try:
            t = threading.Thread(
                target=self._install_when_ready,
                args=(fullname, adapter_path),
                daemon=True,
                name=f"morse_ai-lazy-install-{fullname}",
            )
            t.start()
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "morse_ai: lazy install thread for %s failed: %s", fullname, exc
            )
        return None

    @staticmethod
    def _install_when_ready(target_module: str, adapter_dotted_path: str) -> None:
        # Bounded wait — give the target module up to 5 seconds to finish
        # loading. In practice it's milliseconds.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if target_module in sys.modules:
                _install_one(adapter_dotted_path)
                return
            time.sleep(0.01)
        # Best-effort — try install anyway in case sys.modules entry slipped
        # past our polling window.
        _install_one(adapter_dotted_path)


_finder_singleton = _LazyInstallFinder()


def install_adapters() -> None:
    """Scan `sys.modules` and install adapters for any present AI SDKs.

    For modules not yet imported, register a lazy hook that fires install()
    when the import happens. Idempotent and safe to call repeatedly — every
    `morse_ai.init()` call goes through here.

    Honors `MORSE_DISABLE_AUTO_DETECT=1` and returns immediately.
    """
    global _finder_installed

    if env.is_auto_detect_disabled():
        return

    with _install_lock:
        targets = _adapter_targets()

        # Phase 1: install for already-imported modules.
        for module_name, adapter_path in targets.items():
            if module_name in sys.modules:
                _install_one(adapter_path)
            else:
                _lazy_targets[module_name] = adapter_path

        # Phase 2: install lazy import hook (once per process).
        if _lazy_targets and not _finder_installed:
            sys.meta_path.insert(0, _finder_singleton)
            _finder_installed = True


def _reset_for_testing() -> None:
    """Reset auto-detect state for tests.

    Removes the meta-path finder, clears the lazy targets dict, and resets
    each adapter's install flag.
    """
    global _finder_installed

    try:
        if _finder_installed and _finder_singleton in sys.meta_path:
            sys.meta_path.remove(_finder_singleton)
        _finder_installed = False
        _lazy_targets.clear()
        _finder_singleton._fired.clear()
    except Exception:  # noqa: BLE001
        pass

    # Best-effort uninstall on each adapter so flags are reset.
    for adapter_path in _adapter_targets().values():
        try:
            adapter_module = importlib.import_module(adapter_path)
        except Exception:  # noqa: BLE001
            continue
        uninstall_fn = getattr(adapter_module, "uninstall", None)
        if uninstall_fn is not None:
            with contextlib.suppress(Exception):
                uninstall_fn()
