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
_lazy_targets: dict[str, str] = {}  # trigger module name -> adapter dotted path
_finder_installed = False

# MHQ-1537: the module whose import installs an adapter, when it is not the
# package itself. ``langgraph`` is a namespace package: its spec has no loader
# at find time, so the hook could never wrap it and fell back to a background
# thread that called ``install()`` the moment ``langgraph`` appeared in
# ``sys.modules`` -- i.e. while the importing thread was still loading
# ``langgraph.graph`` and ``langchain_core``. Two threads importing the same
# circular ``langchain_core`` modules can deadlock (``_DeadlockError`` on
# ``langchain_core.messages.utils`` in the testbed), and even without a
# deadlock the patch landed after the import returned (the MHQ-1154 race).
# ``langgraph.pregel`` is a regular package that every graph entry point
# (``langgraph.graph``, ``langgraph.func``, ``langgraph.prebuilt``) imports and
# that itself imports ``langchain_core.callbacks`` and
# ``langgraph.checkpoint.base``, so installing at the end of its body runs in
# the importing thread and needs no new imports.
_TRIGGER_MODULES: dict[str, str] = {
    "langgraph": "langgraph.pregel",
}


def _trigger_module(target: str) -> str:
    return _TRIGGER_MODULES.get(target, target)


def _importable(name: str) -> bool:
    """``find_spec`` that never raises.

    It raises ``ValueError`` when ``sys.modules`` holds a module whose
    ``__spec__`` is None (a stub or a module built by hand), which must not
    stop the other adapters from installing.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return name in sys.modules


def _adapter_targets() -> dict[str, str]:
    """Return the resolved {target_module_name: adapter_dotted_path} mapping.

    Applies priority rules:
    - LangGraph wins over generic LangChain when both are importable.
    """
    targets: dict[str, str] = {
        "anthropic": "morse_ai.adapters.anthropic",
    }

    # MHQ-1506: raw OpenAI clients (and the libraries that drive one --
    # LiteLLM's OpenAI provider, Agno's OpenAIChat) were never traced by the
    # zero-code setup even though the adapter existed. The adapter skips calls
    # that a framework adapter already records (openai-agents runs, LangChain
    # llm spans), so auto-installing it does not double-count those.
    if _importable("openai"):
        targets["openai"] = "morse_ai.adapters.openai"

    # MHQ-1506: google-genai (Gemini API and Vertex AI). ``google`` is a
    # namespace package, so the lazy hook targets ``google.genai`` itself.
    if _importable("google.genai"):
        targets["google.genai"] = "morse_ai.adapters.google_genai"

    # LangGraph is preferred when available — registers under "langgraph" so
    # the lazy hook fires on `import langgraph` regardless of order.
    if _importable("langgraph"):
        targets["langgraph"] = "morse_ai.adapters.langgraph"

    # OpenAI Agents SDK — package is published as `openai-agents` on PyPI but
    # imports as the top-level `agents` module. Wave 4 adapter.
    if _importable("agents"):
        targets["agents"] = "morse_ai.adapters.openai_agents"

    # Anthropic claude-agent-sdk — shells out to the `claude` CLI binary, so
    # the generic Anthropic adapter never fires. This adapter intercepts the
    # message stream at ClaudeSDKClient.receive_response() instead.
    if _importable("claude_agent_sdk"):
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


class _InstallAfterExecLoader(importlib.abc.Loader):
    """Wrap the target's real loader; install the adapter once its body ran.

    MHQ-1154. Installing from a background thread that polled ``sys.modules``
    lost the race every time: the module is registered there before its body
    runs, so the thread blocked on the import lock and patched *after* the
    importing statement returned. ``morse_ai.init(); from agents import
    Runner; Runner.run_sync(...)`` therefore always called the unpatched
    ``Runner`` -- the trace carried only the caller's own ``run()`` span and
    every LLM/tool span was never created. Running ``install()`` at the end of
    ``exec_module`` makes the patch part of the import itself.
    """

    def __init__(self, inner: Any, adapter_dotted_path: str) -> None:
        self._inner = inner
        self._adapter_dotted_path = adapter_dotted_path

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        _install_one(self._adapter_dotted_path)

    def __getattr__(self, name: str) -> Any:
        # Resource readers, get_source, is_package, ... -- delegate the rest.
        return getattr(self._inner, name)


class _LazyInstallFinder(importlib.abc.MetaPathFinder):
    """MetaPathFinder that installs an adapter when its target module imports.

    It never loads anything itself: it asks the remaining finders for the
    real spec and wraps that spec's loader so ``install()`` runs right after
    the module body executes (synchronously, inside the import statement).
    Falls back to the old deferred background install when the real spec
    can't be wrapped.
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
        # Mark as fired BEFORE resolving so the import doesn't recursively
        # trigger us via our own install path.
        self._fired.add(fullname)
        adapter_path = _lazy_targets[fullname]

        spec = self._find_real_spec(fullname, path, target)
        loader = getattr(spec, "loader", None) if spec is not None else None
        if loader is not None and hasattr(loader, "exec_module"):
            spec.loader = _InstallAfterExecLoader(loader, adapter_path)
            return spec

        self._install_in_background(fullname, adapter_path)
        return None

    def _find_real_spec(
        self, fullname: str, path: list[str] | None, target: Any | None
    ) -> importlib.machinery.ModuleSpec | None:
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            try:
                spec = find(fullname, path, target)
            except Exception as exc:  # noqa: BLE001 — non-invasive
                logger.debug("morse_ai: find_spec via %r failed: %s", finder, exc)
                continue
            if spec is not None:
                return spec
        return None

    def _install_in_background(self, fullname: str, adapter_path: str) -> None:
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

    @staticmethod
    def _install_when_ready(target_module: str, adapter_dotted_path: str) -> None:
        # Bounded wait — give the target module up to 5 seconds to finish
        # loading. In practice it's milliseconds. MHQ-1537: wait until the
        # module's body has finished, not merely until it is registered in
        # ``sys.modules`` (which happens before the body runs); installing
        # while the importing thread is still mid-import is what deadlocked.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            module = sys.modules.get(target_module)
            if module is not None and not _is_initializing(module):
                _install_one(adapter_dotted_path)
                return
            time.sleep(0.01)
        # Best-effort — try install anyway in case sys.modules entry slipped
        # past our polling window.
        _install_one(adapter_dotted_path)


def _is_initializing(module: Any) -> bool:
    spec = getattr(module, "__spec__", None)
    return bool(getattr(spec, "_initializing", False))


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
            trigger = _trigger_module(module_name)
            if trigger in sys.modules:
                _install_one(adapter_path)
            else:
                _lazy_targets[trigger] = adapter_path

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
