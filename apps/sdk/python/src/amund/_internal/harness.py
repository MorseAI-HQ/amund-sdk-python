"""Harness telemetry v1 (H2-C5): validation, merging and the file-set hash.

The contract is frozen in ``harness-telemetry-v1.schema.json`` (vendored under
``tests/fixtures/harness2/``). Six optional fields describe the "harness" an
agent ran under (prompts, skills, guardrails, ...):

``harness_hash``, ``harness_version``, ``harness_channels_changed``,
``harness_channels_incomplete``, ``harness_parent_hash``, ``harness_run_role``.

Everything that touches a USER value here is total: invalid input is dropped
with one debug log line and never raises into host code. The only function that
may raise is :func:`compute_harness_hash`, which the developer calls
explicitly and which must not hand back a silently wrong identity.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import unicodedata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger("amund")

HARNESS_FIELDS: tuple[str, ...] = (
    "harness_hash",
    "harness_version",
    "harness_channels_changed",
    "harness_channels_incomplete",
    "harness_parent_hash",
    "harness_run_role",
)

# Observation enum (schema ``$defs.harness_channel``), in schema order.
HARNESS_CHANNELS: frozenset[str] = frozenset(
    {
        "prompt",
        "skill",
        "guardrail",
        "memory",
        "tool_guidance",
        "script",
        "subagent",
        "plugin",
    }
)
HARNESS_RUN_ROLES: frozenset[str] = frozenset({"production", "probe"})
MAX_HARNESS_VERSION_LEN = 128

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# Fields that need a valid ``harness_hash`` (schema ``dependentRequired``).
_NEEDS_HASH = (
    "harness_channels_changed",
    "harness_channels_incomplete",
    "harness_parent_hash",
)


def _drop(name: str, reason: str) -> None:
    logger.debug("amund: %s dropped: %s", name, reason)


def _clean_sha256(name: str, value: Any) -> str | None:
    if isinstance(value, str) and _SHA256.match(value):
        return value
    _drop(name, "not 64 lowercase hex characters")
    return None


def _clean_channels(value: Any) -> tuple[list[str], bool] | None:
    """``(valid unique channels in first-seen order, any item dropped)``.

    Unknown items are dropped one at a time; duplicates collapse silently (the
    schema requires unique items) and do not count as a drop. A non-collection
    drops the whole field.
    """
    if isinstance(value, (str, bytes)) or not isinstance(
        value, (list, tuple, set, frozenset)
    ):
        _drop("harness_channels_changed", "not a list of channel names")
        return None
    kept: list[str] = []
    dropped = False
    for item in value:
        if isinstance(item, str) and item in HARNESS_CHANNELS:
            if item not in kept:
                kept.append(item)
        else:
            dropped = True
            _drop("harness_channels_changed item", "unrecognised channel")
    return kept, dropped


def clean_harness_layer(
    *,
    harness_hash: Any = None,
    harness_version: Any = None,
    harness_channels_changed: Any = None,
    harness_channels_incomplete: Any = None,
    harness_parent_hash: Any = None,
    harness_run_role: Any = None,
) -> dict[str, Any]:
    """Validate ONE layer (client config or one run) field by field.

    Returns only the fields that survived, so an empty dict means "nothing
    declared". A ``"_channels_dropped"`` marker is set when an unknown channel
    was discarded; :func:`merge_harness` turns it into
    ``harness_channels_incomplete=True``. Dependent fields are NOT checked
    here: a run may legitimately supply only ``harness_channels_changed`` and
    inherit the hash from the client config, so that check runs on the merge.
    Never raises.
    """
    out: dict[str, Any] = {}
    try:
        if harness_hash is not None:
            cleaned = _clean_sha256("harness_hash", harness_hash)
            if cleaned is not None:
                out["harness_hash"] = cleaned
        if harness_version is not None:
            if (
                isinstance(harness_version, str)
                and 1 <= len(harness_version) <= MAX_HARNESS_VERSION_LEN
            ):
                out["harness_version"] = harness_version
            else:
                _drop("harness_version", "not a string of 1..128 characters")
        if harness_channels_changed is not None:
            channels = _clean_channels(harness_channels_changed)
            if channels is not None:
                out["harness_channels_changed"] = channels[0]
                if channels[1]:
                    out["_channels_dropped"] = True
        if harness_channels_incomplete is not None:
            if isinstance(harness_channels_incomplete, bool):
                out["harness_channels_incomplete"] = harness_channels_incomplete
            else:
                _drop("harness_channels_incomplete", "not a bool")
        if harness_parent_hash is not None:
            cleaned = _clean_sha256("harness_parent_hash", harness_parent_hash)
            if cleaned is not None:
                out["harness_parent_hash"] = cleaned
        if harness_run_role is not None:
            if isinstance(harness_run_role, str) and (
                harness_run_role in HARNESS_RUN_ROLES
            ):
                out["harness_run_role"] = harness_run_role
            else:
                _drop("harness_run_role", "not 'production' or 'probe'")
    except Exception as exc:  # noqa: BLE001 — host data must never escape
        logger.debug("amund: harness fields swallowed error: %s", exc)
        return {}
    return out


def merge_harness(
    client: dict[str, Any] | None, run: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge cleaned layers (run wins field by field) and apply dependencies.

    Returns the wire fields (schema names) to put on the trace; ``{}`` when
    nothing was declared. Without a valid ``harness_hash`` on the merged view,
    ``harness_channels_changed`` / ``_incomplete`` / ``harness_parent_hash``
    are dropped. A dropped channel item forces
    ``harness_channels_incomplete=True``.
    """
    if not client and not run:
        return {}
    try:
        merged: dict[str, Any] = {}
        channels_dropped = False
        for layer in (client or {}, run or {}):
            for key, value in layer.items():
                if key == "_channels_dropped":
                    continue
                merged[key] = value
            if "harness_channels_changed" in layer:
                # The drop marker belongs to whichever layer's list won.
                channels_dropped = bool(layer.get("_channels_dropped"))
        if channels_dropped and "harness_channels_changed" in merged:
            merged["harness_channels_incomplete"] = True
        if "harness_hash" not in merged:
            for dependent in _NEEDS_HASH:
                if dependent in merged:
                    _drop(dependent, "requires a valid harness_hash")
                    del merged[dependent]
        return merged
    except Exception as exc:  # noqa: BLE001
        logger.debug("amund: harness merge swallowed error: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# compute_harness_hash — x-amund-hash-rules.file_set_hash
# ---------------------------------------------------------------------------


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:  # noqa: PTH123
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hashable_file(path: str, real_root: str) -> bool:
    """Regular file; a symlink only when its target is a regular file in root."""
    if os.path.islink(path):
        target = os.path.realpath(path)
        inside = target == real_root or target.startswith(real_root + os.sep)
        return inside and os.path.isfile(target)
    return os.path.isfile(path)


def compute_harness_hash(
    paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    root: str | os.PathLike[str] | None = None,
) -> str:
    """SHA-256 identity of a set of harness files (``file_set_hash``).

    Each file contributes the line ``"<path>\\t<sha256 of its raw bytes>\\n"``
    where ``<path>`` is relative to ``root``, forward-slashed and Unicode NFC;
    lines are sorted by the UTF-8 bytes of the path and the result is the
    SHA-256 hex of their concatenation. No line-ending or encoding
    normalisation is applied. An empty set hashes the empty string.

    ``paths`` is a file, a directory (recursed, regular files only) or an
    iterable of them. Symlinks are never followed into directories; a symlink
    to a file is hashed only when its target lies inside ``root``, otherwise it
    is skipped. ``root`` defaults to the directory when one directory is given,
    else the common parent of the inputs.

    The result is a valid ``harness_hash``. Unlike the telemetry arguments this
    helper raises (``FileNotFoundError``, ``ValueError`` for a path outside
    ``root``): a guessed identity would be worse than an error.
    """
    if isinstance(paths, (str, os.PathLike)):
        items = [os.fspath(paths)]
    else:
        items = [os.fspath(p) for p in paths]
    absolute = [os.path.abspath(p) for p in items]
    for given, path in zip(items, absolute, strict=True):
        if not os.path.lexists(path):
            raise FileNotFoundError(given)

    if root is not None:
        base = os.path.abspath(os.fspath(root))
    elif absolute:
        bases = [p if os.path.isdir(p) else os.path.dirname(p) for p in absolute]
        base = os.path.commonpath(bases)
    else:
        base = os.getcwd()
    real_root = os.path.realpath(base)

    files: dict[str, str] = {}  # relative path -> absolute path

    def _add(path: str) -> None:
        rel = os.path.relpath(path, base)
        if rel == ".." or rel.startswith(".." + os.sep) or os.path.isabs(rel):
            raise ValueError(f"path {path!r} is outside root {base!r}")
        if _is_hashable_file(path, real_root):
            rel_posix = unicodedata.normalize("NFC", rel.replace(os.sep, "/"))
            files[rel_posix] = path

    for path in absolute:
        if os.path.isdir(path) and not os.path.islink(path):
            # Validate the directory itself is inside root before walking.
            rel_dir = os.path.relpath(path, base)
            if rel_dir == ".." or rel_dir.startswith(".." + os.sep):
                raise ValueError(f"path {path!r} is outside root {base!r}")
            for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
                # Symlinked directories show up in dirnames but are not walked.
                dirnames[:] = [
                    d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))
                ]
                for name in filenames:
                    _add(os.path.join(dirpath, name))
        else:
            _add(path)

    lines: list[tuple[bytes, str]] = []
    for rel, absolute_path in files.items():
        line = f"{rel}\t{_sha256_file(os.path.realpath(absolute_path))}\n"
        lines.append((rel.encode("utf-8"), line))
    lines.sort(key=lambda item: item[0])
    return hashlib.sha256(
        "".join(line for _, line in lines).encode("utf-8")
    ).hexdigest()
