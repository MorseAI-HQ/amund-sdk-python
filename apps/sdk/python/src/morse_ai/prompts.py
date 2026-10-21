"""Hosted prompt fetch (``morse_ai.get_prompt``).

Deliberately thin: one GET, typed result, typed errors. There is no retry, no
fallback. By default (``link=True``) a successful fetch inside an open run links
that run to the fetched prompt version (UP-09, see ``get_prompt``). An in-memory
cache exists but is OFF unless the caller passes ``cache_ttl`` (see
``get_prompt``'s docstring).
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from urllib.parse import quote, urlsplit

import httpx

#: Timeout for the single fetch request, in seconds.
_FETCH_TIMEOUT_SECONDS = 5.0


class PromptError(Exception):
    """Base class for hosted-prompt fetch errors."""


class PromptAccessError(PromptError):
    """403: the API key lacks the ``prompts:read`` scope."""


class PromptNotFoundError(PromptError):
    """404: no such prompt, label or version for this organization."""


class PromptRateLimitError(PromptError):
    """429: fetch rate limit hit. ``retry_after`` is in seconds (or None)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(frozen=True)
class Prompt:
    """A fetched prompt version.

    On a 304 (``if_none_match`` matched) ``not_modified`` is True, ``etag`` is
    the one you sent, and every other field is None.
    """

    name: str | None
    version: int | None
    template_text: str | None
    labels: list[str] | None
    etag: str | None
    not_modified: bool = False


def _parse_retry_after(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def fetch_prompt(
    *,
    endpoint: str,
    api_key: str,
    name: str,
    label: str | None,
    version: int | None,
    if_none_match: str | None,
) -> Prompt:
    """Perform the GET. ``endpoint`` is the SDK's configured ingest URL."""
    if label is not None and version is not None:
        raise ValueError("Pass either label or version, not both")
    params: dict[str, str] = (
        {"version": str(version)}
        if version is not None
        else {"label": label or "production"}
    )
    parts = urlsplit(endpoint)
    url = f"{parts.scheme}://{parts.netloc}/api/v1/prompts/{quote(name, safe='')}"
    headers = {"Authorization": f"Bearer {api_key}"}
    if if_none_match:
        headers["If-None-Match"] = if_none_match

    with httpx.Client(timeout=_FETCH_TIMEOUT_SECONDS) as client:
        resp = client.get(url, params=params, headers=headers)

    if resp.status_code == 304:
        return Prompt(
            None,
            None,
            None,
            None,
            resp.headers.get("ETag") or if_none_match,
            not_modified=True,
        )
    if resp.status_code == 403:
        raise PromptAccessError(
            "This API key cannot fetch hosted prompts: it is missing the "
            "'prompts:read' scope. Rotate the key with that scope enabled."
        )
    if resp.status_code == 404:
        raise PromptNotFoundError(f"Prompt not found: {name!r} ({params})")
    if resp.status_code == 429:
        retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
        raise PromptRateLimitError(
            "Prompt fetch rate limit exceeded"
            + (f"; retry after {retry_after:g}s" if retry_after is not None else ""),
            retry_after=retry_after,
        )
    resp.raise_for_status()
    data = resp.json()
    return Prompt(
        name=data["name"],
        version=data["version"],
        template_text=data["template_text"],
        labels=list(data.get("labels") or []),
        etag=data.get("etag") or resp.headers.get("ETag"),
    )


# --- Optional in-memory cache (SPF-04..06). OFF unless ``cache_ttl`` > 0. ---

_DEFAULT_MAX_ENTRIES = 256
#: Monotonic clock; tests patch this.
_clock = time.monotonic
_cache_lock = threading.Lock()
#: key -> (prompt, fetched_at). Most recently used last.
_cache: OrderedDict[tuple, tuple[Prompt, float]] = OrderedDict()
_max_entries = _DEFAULT_MAX_ENTRIES


def configure_prompt_cache(max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
    """Set the cache size limit (default 256) and evict down to it."""
    global _max_entries
    if max_entries < 1:
        raise ValueError("max_entries must be at least 1")
    with _cache_lock:
        _max_entries = max_entries
        while len(_cache) > _max_entries:
            _cache.popitem(last=False)


def clear_prompt_cache() -> None:
    """Empty the prompt cache."""
    with _cache_lock:
        _cache.clear()


def _cache_key(
    endpoint: str, api_key: str, name: str, label: str | None, version: int | None
) -> tuple:
    selector = (
        ("version", version)
        if version is not None
        else ("label", label or "production")
    )
    return (endpoint, hashlib.sha256(api_key.encode()).hexdigest()[:16], name, selector)


def peek_cached_prompt(
    *,
    endpoint: str,
    api_key: str,
    name: str,
    label: str | None,
    version: int | None,
) -> Prompt | None:
    """The SDK's own cached Prompt for this selector, ignoring TTL, or None.

    Used to link a run after a 304, whose body carries no name/version."""
    with _cache_lock:
        entry = _cache.get(_cache_key(endpoint, api_key, name, label, version))
    return entry[0] if entry is not None else None


def fetch_prompt_cached(
    *,
    endpoint: str,
    api_key: str,
    name: str,
    label: str | None,
    version: int | None,
    if_none_match: str | None,
    cache_ttl: float | None,
) -> Prompt:
    """``fetch_prompt`` with the opt-in cache. No ``cache_ttl`` (or a caller
    ``if_none_match``) means no cache read or write at all."""

    def fetch(inm: str | None) -> Prompt:
        return fetch_prompt(
            endpoint=endpoint,
            api_key=api_key,
            name=name,
            label=label,
            version=version,
            if_none_match=inm,
        )

    if not cache_ttl or cache_ttl <= 0 or if_none_match:
        return fetch(if_none_match)
    if label is not None and version is not None:
        raise ValueError("Pass either label or version, not both")

    key = _cache_key(endpoint, api_key, name, label, version)
    with _cache_lock:
        entry = _cache.get(key)
        if entry is not None:
            _cache.move_to_end(key)
    if entry is not None and _clock() - entry[1] < cache_ttl:
        return entry[0]

    result = fetch(entry[0].etag if entry is not None else None)
    if result.not_modified:
        if entry is None:  # cannot happen without a sent etag; do not cache it
            return result
        result = entry[0]
    with _cache_lock:
        _cache[key] = (result, _clock())
        _cache.move_to_end(key)
        while len(_cache) > _max_entries:
            _cache.popitem(last=False)
    return result
