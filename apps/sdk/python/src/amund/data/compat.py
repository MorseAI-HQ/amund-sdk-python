"""Compatibility manifest loader."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    """Load and cache the compat manifest JSON."""
    manifest_path = Path(__file__).parent / "compat_manifest.json"
    with manifest_path.open() as f:
        return json.load(f)


def get_library_status(
    library: str, version: str | None = None
) -> dict[str, Any] | None:
    """Return the adapter entry for a library/version, or None if not in manifest.

    When version is None, returns the first adapter entry regardless of
    version_range. When version is given, tries to match against version_range
    using packaging.version (falls back to first entry if packaging not
    installed).
    """
    manifest = load_manifest()
    entry = manifest.get(library)
    if entry is None:
        return None
    adapters = entry.get("adapters", [])
    if not adapters:
        return None
    if version is None:
        return adapters[0]
    # Try packaging.version for range matching
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        v = Version(version)
        for adapter in adapters:
            vr = adapter.get("version_range", "")
            if not vr or v in SpecifierSet(vr, prereleases=True):
                return adapter
    except Exception:  # noqa: BLE001
        pass
    return adapters[0]
