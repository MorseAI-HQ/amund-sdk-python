"""Agent-name resolution (MHQ-1100): convention over configuration.

Ingest treats a real ``service.name`` as the agent name, so the SDK must NOT
let a process-level ``OTEL_SERVICE_NAME`` silently become identity. Instead
every trace carries ``morse.agent`` on its root span, resolved here.

First non-empty wins:
  1. explicit name from customer code (``run("x")``, ``agent_name=...``)
  2. framework-native name the adapter knows (``framework`` argument)
  3. env service name (``OTEL_SERVICE_NAME`` / ``service.name`` in
     ``OTEL_RESOURCE_ATTRIBUTES``)
  4. entry-script stem (``sys.argv[0]`` or ``__main__`` spec name)
  5. the literal ``"default-agent"``
"""

from __future__ import annotations

import os
import sys

DEFAULT_AGENT_NAME = "default-agent"

# Entry names that identify a launcher, not an agent.
_GENERIC_ENTRY_NAMES = frozenset(
    {
        "python",
        "python3",
        "-c",
        "-m",
        "-",
        "manage",
        "gunicorn",
        "uvicorn",
        "celery",
        "pytest",
        "py.test",
        "__main__",
        "__init__",
    }
)


def _clean(value: object) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def env_service_name() -> str | None:
    """OTEL_SERVICE_NAME, else ``service.name`` from OTEL_RESOURCE_ATTRIBUTES."""
    direct = _clean(os.environ.get("OTEL_SERVICE_NAME"))
    if direct:
        return direct
    raw = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    for pair in raw.split(","):
        key, sep, val = pair.partition("=")
        if sep and key.strip() == "service.name":
            return _clean(val)
    return None


def entry_script_stem() -> str | None:
    """Stem of the process entry script, or None when generic/unknown."""
    candidates: list[str] = []
    main = sys.modules.get("__main__")
    spec_name = getattr(getattr(main, "__spec__", None), "name", None)
    if isinstance(spec_name, str) and spec_name:
        # ``python -m pkg.mod`` -> "mod"; ``pkg.__main__`` -> "pkg"
        parts = [p for p in spec_name.split(".") if p != "__main__"]
        if parts:
            candidates.append(parts[-1])
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0:
        base = os.path.basename(argv0)
        stem = base[:-3] if base.endswith(".py") else base
        candidates.append(stem)
    for cand in candidates:
        cand = cand.strip()
        if cand and cand.lower() not in _GENERIC_ENTRY_NAMES:
            return cand
    return None


def resolve_agent_name(explicit: str | None, framework: str | None = None) -> str:
    """Resolve the agent name for a trace. See module docstring for tiers.

    ``framework`` is the framework-native name (e.g. an OpenAI Agents starting
    ``Agent.name`` or a LangGraph graph name), already filtered of generic
    defaults by the caller.
    """
    return (
        _clean(explicit)
        or _clean(framework)
        or env_service_name()
        or entry_script_stem()
        or DEFAULT_AGENT_NAME
    )
