"""Client-side normalisation for attribution fields (ATTR-01, ATTR-04).

Record keys, the declared ``agent_version`` and prompt references are USER
values that ride on the telemetry path, so every function here is total: bad
input is dropped with a debug log and ``None`` / the unchanged container is
returned. Nothing here raises into host code.

The limits mirror the server's (``apps/api`` validates again, per value, so a
bad value never fails a whole batch); validating here just avoids shipping
bytes the server would discard.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("amund")

KEY_TYPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
MAX_KEY_VALUE_LEN = 256
MAX_RECORD_KEYS = 20
MAX_PROMPT_REFS = 20
MAX_PROMPT_NAME_LEN = 255
MAX_AGENT_VERSION_LEN = 128

_PROMPT_VERSION_STR = re.compile(r"^v?(\d{1,9})$", re.IGNORECASE)
_prompt_version_warned = False


def coerce_prompt_version(version: Any) -> int | None:
    """An int >= 1, or a whole-number string ("3", "v3") read as one; else None.

    MHQ-1415: ``set_prompt("triage", "3")`` used to drop the version silently.
    """
    if isinstance(version, bool):
        return None
    if isinstance(version, int):
        return version if version >= 1 else None
    if isinstance(version, str):
        match = _PROMPT_VERSION_STR.match(version.strip())
        if match:
            number = int(match.group(1))
            return number if number >= 1 else None
    return None


def _warn_prompt_version_dropped() -> None:
    """Once per process; never includes the value (it may be an id or PII)."""
    global _prompt_version_warned
    if _prompt_version_warned:
        logger.debug("amund: prompt version dropped")
        return
    _prompt_version_warned = True
    logger.warning(
        "amund: a set_prompt() version that is not a whole number >= 1 was "
        "dropped and only the prompt name recorded; pass the registry version "
        'number (e.g. 3 or "3"). This warning prints once per process.'
    )


def clean_record_key(key_type: Any, key_value: Any) -> tuple[str, str] | None:
    """Return ``(key_type, key_value)`` normalised, or None when invalid.

    ``key_type`` is lower-cased and must match ``KEY_TYPE_PATTERN``.
    ``key_value`` is coerced with ``str()`` (integers are common: webhook
    templates often leave ids unquoted), stripped, and must be 1-256 chars.
    ``bool`` and ``None`` values are rejected: ``True`` is never a record id.
    """
    try:
        if not isinstance(key_type, str):
            logger.debug("amund: record key dropped: key_type is not a string")
            return None
        norm_type = key_type.strip().lower()
        if not KEY_TYPE_PATTERN.match(norm_type):
            logger.debug("amund: record key dropped: invalid key_type %r", key_type)
            return None
        if key_value is None or isinstance(key_value, bool):
            logger.debug("amund: record key %r dropped: unusable value", norm_type)
            return None
        norm_value = str(key_value).strip()
        if not norm_value or len(norm_value) > MAX_KEY_VALUE_LEN:
            logger.debug("amund: record key %r dropped: value length", norm_type)
            return None
        return norm_type, norm_value
    except Exception as exc:  # noqa: BLE001 - customer __str__ may raise
        logger.debug("amund: record key dropped: %s", exc)
        return None


def add_record_key(keys: dict[str, str], key_type: Any, key_value: Any) -> bool:
    """Validate and merge one key into ``keys`` in place. True if it was kept.

    One value per ``key_type`` (the wire shape is ``{type: value}``): a repeated
    type replaces the earlier value. A NEW type past the 20-key cap is dropped.
    """
    cleaned = clean_record_key(key_type, key_value)
    if cleaned is None:
        return False
    norm_type, norm_value = cleaned
    if norm_type not in keys and len(keys) >= MAX_RECORD_KEYS:
        logger.debug(
            "amund: record key %r dropped: %d-key cap reached",
            norm_type,
            MAX_RECORD_KEYS,
        )
        return False
    keys[norm_type] = norm_value
    return True


def clean_agent_version(value: Any) -> str | None:
    """Strip and truncate (to 128) a declared agent version; None if empty."""
    try:
        if value is None or isinstance(value, bool):
            return None
        text = str(value).strip()
        return text[:MAX_AGENT_VERSION_LEN] or None
    except Exception as exc:  # noqa: BLE001
        logger.debug("amund: agent_version dropped: %s", exc)
        return None


def clean_prompt_ref(
    name: Any,
    version: Any = None,
    span_id: str | None = None,
    link_source: str | None = None,
) -> dict[str, Any] | None:
    """Build one ``{name, version?, span_id?, link_source?}`` prompt reference, or None.

    ``link_source`` is serialized only when it is ``"fetch"`` (the reference
    came from ``get_prompt`` auto-linking); ``set_prompt`` refs omit it.

    An invalid ``version`` (not an int >= 1) is dropped but the name is kept,
    matching the server, which would drop the version the same way.
    """
    try:
        if not isinstance(name, str):
            logger.debug("amund: prompt ref dropped: name is not a string")
            return None
        norm_name = name.strip()
        if not norm_name or len(norm_name) > MAX_PROMPT_NAME_LEN:
            logger.debug("amund: prompt ref dropped: invalid name length")
            return None
        ref: dict[str, Any] = {"name": norm_name}
        if version is not None:
            coerced = coerce_prompt_version(version)
            if coerced is not None:
                ref["version"] = coerced
            else:
                _warn_prompt_version_dropped()
        if span_id:
            ref["span_id"] = span_id
        if link_source == "fetch":
            ref["link_source"] = "fetch"
        return ref
    except Exception as exc:  # noqa: BLE001
        logger.debug("amund: prompt ref dropped: %s", exc)
        return None


def add_prompt_ref(refs: list[dict[str, Any]], ref: dict[str, Any]) -> bool:
    """Append ``ref`` unless it duplicates one or the 20-ref cap is reached.

    Dedupe key is (name, version, span_id). When a ``set_prompt`` ref and a
    fetch-link ref share that key, ONE ref is kept and ``link_source="fetch"``
    wins: the fetch is the stronger evidence of which version actually ran,
    and it must not be lost to call order.
    """
    key = (ref.get("name"), ref.get("version"), ref.get("span_id"))
    for existing in refs:
        if (
            existing.get("name"),
            existing.get("version"),
            existing.get("span_id"),
        ) == key:
            if ref.get("link_source") == "fetch":
                existing["link_source"] = "fetch"
            return False
    if len(refs) >= MAX_PROMPT_REFS:
        logger.debug(
            "amund: prompt ref %r dropped: %d-ref cap reached",
            ref.get("name"),
            MAX_PROMPT_REFS,
        )
        return False
    refs.append(ref)
    return True
