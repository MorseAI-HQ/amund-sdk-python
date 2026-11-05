"""Import scanner and watcher for tracked AI/vector-store libraries.

Provides:
- scan_imported_libraries() — snapshot of currently-imported tracked libs + versions
- AVImportWatcher — MetaPathFinder that fires a callback on first tracked-lib import
- install_import_watcher() / uninstall_import_watcher() — idempotent lifecycle helpers
"""

from __future__ import annotations

import contextlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

TRACKED_LIBRARIES: list[str] = [
    "chromadb",
    "pinecone",
    "pinecone-client",
    "weaviate-client",
    "qdrant-client",
    "mem0ai",
    "mem0",
    "langchain",
    "langchain-core",
    "langchain-community",
    "llama-index",
    "llama-index-core",
    "anthropic",
    "openai",
    "openai-agents",
    "strands-agents",
    "google-genai",
    "langgraph",
    "claude-agent-sdk",
    "pgvector",
    "psycopg2",
    "asyncpg",
]

# Map package-install name → top-level import name (when they differ).
# Keys are the install names from TRACKED_LIBRARIES; values are the module
# names that appear in sys.modules when the package is imported.
_INSTALL_TO_MODULE: dict[str, str] = {
    "pinecone-client": "pinecone",
    "weaviate-client": "weaviate",
    "qdrant-client": "qdrant_client",
    "mem0ai": "mem0",
    "langchain-core": "langchain_core",
    "langchain-community": "langchain_community",
    "llama-index": "llama_index",
    "llama-index-core": "llama_index",
    "openai-agents": "agents",
    "strands-agents": "strands",
    "google-genai": "google.genai",
    "claude-agent-sdk": "claude_agent_sdk",
    "pgvector": "pgvector",
    "psycopg2": "psycopg2",
    "asyncpg": "asyncpg",
}


def _module_name(lib: str) -> str:
    """Return the top-level sys.modules key for an install-name."""
    return _INSTALL_TO_MODULE.get(lib, lib.replace("-", "_"))


def _get_version(lib: str) -> str:
    """Return the installed version of `lib`, or 'unknown' on any error."""
    try:
        return importlib.metadata.version(lib)
    except Exception:  # noqa: BLE001
        # Try alternate names (e.g. "pinecone" for "pinecone-client").
        alt = _module_name(lib)
        try:
            return importlib.metadata.version(alt)
        except Exception:  # noqa: BLE001
            return "unknown"


def scan_imported_libraries() -> dict[str, str]:
    """Return {library_name: version} for all tracked libs in sys.modules.

    Only returns libs whose top-level module is currently imported. The key
    is the canonical install name from TRACKED_LIBRARIES.
    """
    result: dict[str, str] = {}
    for lib in TRACKED_LIBRARIES:
        mod_name = _module_name(lib)
        if mod_name in sys.modules:
            result[lib] = _get_version(lib)
    return result


# ---------------------------------------------------------------------------
# AVImportWatcher — MetaPathFinder
# ---------------------------------------------------------------------------

# Singleton instance stored for idempotent install/uninstall.
_watcher_instance: AVImportWatcher | None = None


class AVImportWatcher(importlib.abc.MetaPathFinder):
    """MetaPathFinder that fires a callback when a tracked library is first imported.

    The watcher is a pass-through: find_spec always returns None, delegating
    actual import resolution to the next finder on sys.meta_path. The callback
    fires synchronously on the first import of each tracked top-level module.

    Args:
        callback: Called as callback(library_name, version) where library_name
            is the install-name from TRACKED_LIBRARIES and version is the
            importlib.metadata version string (or "unknown").
    """

    def __init__(self, callback: Callable[[str, str], None]) -> None:
        self._callback = callback
        # Build reverse map: module_name → install_name
        self._module_to_lib: dict[str, str] = {
            _module_name(lib): lib for lib in TRACKED_LIBRARIES
        }
        self._seen: set[str] = set()

    def find_spec(
        self,
        fullname: str,
        path: object,
        target: object = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Pass-through finder — never resolves modules, only observes imports."""
        # Only react to top-level module imports (no dots in fullname).
        is_tracked = "." not in fullname and fullname in self._module_to_lib
        if is_tracked and fullname not in self._seen:
            self._seen.add(fullname)
            lib = self._module_to_lib[fullname]
            try:
                version = _get_version(lib)
                self._callback(lib, version)
            except Exception:  # noqa: BLE001
                pass
        return None


def install_import_watcher(callback: Callable[[str, str], None]) -> None:
    """Install AVImportWatcher on sys.meta_path. Idempotent.

    If a watcher is already installed, it is replaced with a new one using
    the provided callback.
    """
    global _watcher_instance
    # Remove old instance first (idempotent).
    uninstall_import_watcher()
    _watcher_instance = AVImportWatcher(callback)
    sys.meta_path.insert(0, _watcher_instance)


def uninstall_import_watcher() -> None:
    """Remove the watcher from sys.meta_path. Idempotent."""
    global _watcher_instance
    if _watcher_instance is not None:
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(_watcher_instance)
        _watcher_instance = None
