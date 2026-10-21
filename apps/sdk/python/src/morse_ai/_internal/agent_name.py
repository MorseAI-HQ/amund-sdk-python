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

AID-01 (agent-identity-v2) keeps this ladder and records which tier won as
``identity_source`` so the server can flag guessed (entry-script) names.
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


def identity_legacy() -> bool:
    """AID-09: ``MORSE_AGENT_IDENTITY=legacy`` restores pre-v2 naming/shape.

    Read lazily so a test or a late ``os.environ`` edit takes effect.
    """
    return os.environ.get("MORSE_AGENT_IDENTITY", "").strip().lower() == "legacy"


# AID-01 identity_source values. Only ``entry_script`` (and ``default``) mean
# "the SDK guessed"; the Agents table prompts a rename for those.
SOURCE_RUN = "run"
SOURCE_FRAMEWORK = "framework"
SOURCE_SERVICE = "service_name"
SOURCE_ENTRY = "entry_script"
SOURCE_DEFAULT = "default"
# Sources a discovered framework root may replace (AID-01 ladder).
GUESSED_SOURCES = frozenset({SOURCE_SERVICE, SOURCE_ENTRY, SOURCE_DEFAULT})


def resolve_agent_identity(
    explicit: str | None, framework: str | None = None
) -> tuple[str, str]:
    """AID-01 ladder: ``(name, identity_source)``. See module docstring tiers.

    ``framework`` is the framework-native name (e.g. an OpenAI Agents starting
    ``Agent.name`` or a LangGraph graph name), already filtered of generic
    defaults by the caller.
    """
    if name := _clean(explicit):
        return name, SOURCE_RUN
    if name := _clean(framework):
        return name, SOURCE_FRAMEWORK
    if name := env_service_name():
        return name, SOURCE_SERVICE
    if name := entry_script_stem():
        return name, SOURCE_ENTRY
    return DEFAULT_AGENT_NAME, SOURCE_DEFAULT


def resolve_agent_name(explicit: str | None, framework: str | None = None) -> str:
    """Name half of :func:`resolve_agent_identity`."""
    return resolve_agent_identity(explicit, framework)[0]
