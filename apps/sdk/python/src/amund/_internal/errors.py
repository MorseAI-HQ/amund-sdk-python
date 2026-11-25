"""Non-invasive error isolation — the foundational SDK guarantee.

Every public SDK function and every adapter-installed monkey-patch wrapper
goes through `@absorb_errors`. The decorator catches all exceptions
(except `KeyboardInterrupt`, `SystemExit`, `MemoryError` — the standard
"let these through" exceptions), reports them out-of-band to Amund's
own diagnostics endpoint via `_internal.diagnostics`, and returns silently.

This is the foundational SDK principle:

    Telemetry must never break the host application.

The customer's call returns its original result regardless of whether the
SDK's recording path succeeded. If a span construction bug raises
`KeyError` somewhere deep, the customer's `messages.create()` call still
returns the Anthropic response normally.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from amund._internal import diagnostics

F = TypeVar("F", bound=Callable[..., Any])


def absorb_errors(
    fingerprint: str, *, extra: dict[str, Any] | None = None
) -> Callable[[F], F]:
    """Decorator that absorbs all exceptions from the wrapped function.

    Args:
        fingerprint: Stable identifier for grouping. Used for server-side
            grouping and the rate limiter (max 100 events per fingerprint
            per minute). Recommended naming pattern:
            "morse_ai.<module>.<function>".
        extra: Optional metadata to attach. Scrubbed by the reporter
            against an allow-list before sending.

    Behavior:
        - On normal return: pass through the result unchanged.
        - On exception (other than KeyboardInterrupt/SystemExit/MemoryError):
            - Report via `diagnostics.report(...)` (Amund-owned; never a
              third-party service, and never customer data).
            - Return None (or the function's None-equivalent).
        - On KeyboardInterrupt/SystemExit/MemoryError: re-raise (standard
            Python convention for "let the program tear down").
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except (KeyboardInterrupt, SystemExit, MemoryError):
                # Re-raise — caller wants this to bubble up.
                raise
            except BaseException as exc:
                diagnostics.report(exc, fingerprint=fingerprint, extra=extra)
                return None

        return wrapper  # type: ignore[return-value]

    return decorator
