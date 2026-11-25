"""Environment variable resolution for SDK auto-init.

The V2 SDK supports zero-setup integration: customer sets `AMUND_API_KEY`
in their process environment, and the SDK auto-initializes on first use. This
module centralizes the env-var contract so the rules stay in one place.

Naming (Amund rebrand, 2026-11-25): every setting is read as ``AMUND_<NAME>``
first and falls back to the pre-rename ``MORSE_<NAME>``, so existing
deployments keep working unchanged. When both are set to non-empty values,
``AMUND_<NAME>`` wins (a debug log notes the conflict once per name). An empty
or whitespace-only value counts as unset, matching the rules below. Every read
in the SDK goes through :func:`get`; do not call ``os.environ`` for an SDK
setting anywhere else.

Env vars (all optional unless noted; each one is also read under its legacy
``MORSE_`` name):
- AMUND_API_KEY              — triggers auto-init when set; absent = SDK dormant
- AMUND_ENDPOINT             — base URL or full ingest URL
- AMUND_FLUSH_INTERVAL_SECONDS — override default 2.0s flush interval
- AMUND_DISABLED              — "1"/"true" makes all SDK calls no-op
- AMUND_SDK_DEBUG             — "1"/"true" raises internal-error log level to ERROR
- AMUND_SDK_DISABLE_TELEMETRY — "1"/"true" suppresses out-of-band SDK
                                 self-diagnostics (errors + delivery counters).
                                 Local logging is unaffected — see SDKDIAG-10.
- AMUND_DISABLE_AUTO_DETECT   — "1"/"true" skips Python sys.modules adapter scan
- AMUND_INTERNAL_SOURCE       — opt-in resource-level tag emitted as the
                                     `morse_ai.source` resource attribute on
                                     every trace. Used by Amund's own
                                     internal services (copilot, investigator,
                                     refagent, ui-bot) so dashboards can
                                     discriminate dogfood emitters and keep
                                     synthetic baselines from polluting the
                                     Anomaly Investigator's signal. UNSET on
                                     customer integrations — absence of the
                                     attribute means "real customer traffic".
- AMUND_GIT_REVISION          — (ATTR-06) explicit git revision for the deployed
                                     build; outranks CI/platform variables and
                                     ``git rev-parse``. For Docker images with
                                     no ``.git``: ``ARG GIT_SHA`` then
                                     ``ENV AMUND_GIT_REVISION=$GIT_SHA``.
- AMUND_SAMPLE_RATE            — (MHQ-668/671) deterministic trace-ID head
                                     sampling rate in [0.0, 1.0], default 1.0
                                     (sample everything). Whole-trace: child
                                     spans always follow the root's decision.
- AMUND_MAX_FIELD_BYTES        — (MHQ-668) byte cap applied to input_data /
                                     output_data / context_segments / custom
                                     metadata values at attribute-set time,
                                     default 65536. Oversized values are
                                     truncated with a "…[truncated]" marker.
- AMUND_MAX_QUEUE_SIZE          — (MHQ-668) OTel BatchSpanProcessor queue
                                     size, default 2048 (OTel SDK default).
- AMUND_MAX_BATCH_SIZE          — (MHQ-668) OTel BatchSpanProcessor max
                                     export batch size, default 512 (OTel
                                     SDK default).
- AMUND_LEGACY_TRANSPORT        — (MHQ-668) "1"/"true" reverts run()/span()/
                                     trace_agent to the pre-MHQ-668 JSON
                                     batch transport instead of OTLP. Escape
                                     hatch for the migration window only;
                                     record()/log() always use JSON.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("amund._internal")

#: Current prefix. Read first.
PREFIX = "AMUND_"
#: Pre-rename prefix. Still read as a fallback so existing deployments work.
LEGACY_PREFIX = "MORSE_"

_conflicts_logged: set[str] = set()


def _suffix(name: str) -> str:
    """``"MORSE_API_KEY"`` / ``"AMUND_API_KEY"`` / ``"API_KEY"`` -> ``"API_KEY"``."""
    for prefix in (PREFIX, LEGACY_PREFIX):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def get(name: str, default: str = "") -> str:
    """Raw value of an SDK setting, ``AMUND_`` first, ``MORSE_`` as fallback.

    ``name`` may be given with either prefix or bare (``"API_KEY"``); the
    lookup is the same. Returns the unmodified value of the winning variable,
    or ``default`` when neither is set to a non-blank value.
    """
    suffix = _suffix(name)
    current = os.environ.get(PREFIX + suffix, "")
    legacy = os.environ.get(LEGACY_PREFIX + suffix, "")
    if current.strip():
        if legacy.strip() and legacy != current and suffix not in _conflicts_logged:
            _conflicts_logged.add(suffix)
            logger.debug(
                "amund: %s%s and %s%s are both set and differ; using %s%s",
                PREFIX,
                suffix,
                LEGACY_PREFIX,
                suffix,
                PREFIX,
                suffix,
            )
        return current
    if legacy.strip():
        return legacy
    return default


# Whitelist of accepted internal source values. Anything outside this set is
# silently ignored at init time so a typo in a customer's env doesn't end up
# masquerading as an internal emitter on the server.
_VALID_INTERNAL_SOURCES = frozenset({"copilot", "investigator", "refagent", "ui-bot"})


def _truthy(value: str | None) -> bool:
    """Common parser for boolean-like env vars."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_api_key() -> str | None:
    """Return AMUND_API_KEY (or legacy MORSE_API_KEY), or None if unset/empty."""
    value = get("API_KEY").strip()
    return value or None


_INGEST_PATH = "/api/v1/traces/batch"

# Production API host. api.amundai.com is the default since 0.9.1;
# api.morseai.dev (the default from 0.8.1 to 0.9.0) is a permanent alias for the
# same service, so an explicit MORSE_ENDPOINT pinned to it keeps working.
DEFAULT_API_HOST = "api.amundai.com"
DEFAULT_ENDPOINT = f"https://{DEFAULT_API_HOST}{_INGEST_PATH}"
# Every host that serves the production API. Used where the SDK has to recognise
# its own backend (HTTP auto-capture never traces the SDK's own POSTs).
PROD_API_HOSTS: frozenset[str] = frozenset({"api.amundai.com", "api.morseai.dev"})


def get_endpoint() -> str | None:
    """Return the resolved ingest URL from AMUND_ENDPOINT / MORSE_ENDPOINT, or None.

    Accepts either a bare base URL (https://api.amundai.com) or the full
    ingest URL — both resolve to the same endpoint.
    """
    value = get("ENDPOINT").strip()
    if not value:
        return None
    value = value.rstrip("/")
    if not value.endswith(_INGEST_PATH):
        value = value + _INGEST_PATH
    return value


def get_flush_interval(default: float = 2.0) -> float:
    """Return AMUND_FLUSH_INTERVAL_SECONDS as float, or default."""
    raw = get("FLUSH_INTERVAL_SECONDS").strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
        if parsed <= 0:
            return default
        return parsed
    except ValueError:
        return default


def is_disabled() -> bool:
    """True when AMUND_DISABLED=1 — SDK becomes a complete no-op."""
    return _truthy(get("DISABLED"))


def is_debug() -> bool:
    """True when AMUND_SDK_DEBUG=1 — internal errors logged at ERROR level."""
    return _truthy(get("SDK_DEBUG"))


def is_telemetry_disabled() -> bool:
    """True when AMUND_SDK_DISABLE_TELEMETRY=1 — no out-of-band error reports."""
    return _truthy(get("SDK_DISABLE_TELEMETRY"))


def is_auto_detect_disabled() -> bool:
    """True when AMUND_DISABLE_AUTO_DETECT=1 — no sys.modules adapter scan."""
    return _truthy(get("DISABLE_AUTO_DETECT"))


def get_internal_source() -> str | None:
    """Return AMUND_INTERNAL_SOURCE if set to a recognised value, else None.

    Only ``copilot | investigator | refagent | ui-bot`` are accepted — any
    other value is treated as unset (returns None). This is intentional:
    absence of the resource attribute is the signal that a trace came from
    a real customer integration, not from Amund's own dogfood emitters.
    A typo in a customer's environment must NOT accidentally tag their data
    as internal.
    """
    raw = get("INTERNAL_SOURCE").strip().lower()
    if raw and raw in _VALID_INTERNAL_SOURCES:
        return raw
    return None


def is_valid_internal_source(value: str | None) -> bool:
    """True when ``value`` is one of the accepted internal-source labels."""
    return value is not None and value in _VALID_INTERNAL_SOURCES


def get_sample_rate(default: float = 1.0) -> float:
    """Return AMUND_SAMPLE_RATE as a float in [0.0, 1.0], or default.

    Deterministic trace-ID head sampling rate for the OTel export pipeline
    (MHQ-668/671). Out-of-range or unparseable values fall back to default.
    """
    raw = get("SAMPLE_RATE").strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
        if parsed < 0.0 or parsed > 1.0:
            return default
        return parsed
    except ValueError:
        return default


def get_max_field_bytes(default: int = 65536) -> int:
    """Return AMUND_MAX_FIELD_BYTES as an int, or default (64 KiB)."""
    raw = get("MAX_FIELD_BYTES").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
        return parsed if parsed > 0 else default
    except ValueError:
        return default


def get_max_queue_size(default: int = 2048) -> int:
    """Return AMUND_MAX_QUEUE_SIZE as an int, or default (OTel SDK default)."""
    raw = get("MAX_QUEUE_SIZE").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
        return parsed if parsed > 0 else default
    except ValueError:
        return default


def get_max_batch_size(default: int = 512) -> int:
    """Return AMUND_MAX_BATCH_SIZE as an int, or default (OTel SDK default)."""
    raw = get("MAX_BATCH_SIZE").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
        return parsed if parsed > 0 else default
    except ValueError:
        return default


def is_legacy_transport() -> bool:
    """True when AMUND_LEGACY_TRANSPORT=1 — escape hatch: run()/span()/
    trace_agent revert to the pre-MHQ-668 JSON batch transport instead of
    OTLP. Exists for emergency rollback during the OTLP migration window;
    ``record()``/``log()`` always use the JSON path regardless of this flag.
    """
    return _truthy(get("LEGACY_TRANSPORT"))


# Detection order for the deployed git revision (ATTR-06), after the explicit
# ``AMUND_GIT_REVISION`` (legacy ``MORSE_GIT_REVISION``) override a Docker build
# injects (``ARG GIT_SHA`` -> ``ENV AMUND_GIT_REVISION=$GIT_SHA``) when neither a
# CI variable nor a ``.git`` directory is present in the runtime image.
_GIT_REVISION_ENV_VARS = (
    "GITHUB_SHA",
    "CI_COMMIT_SHA",
    "RAILWAY_GIT_COMMIT_SHA",
    "VERCEL_GIT_COMMIT_SHA",
    "RENDER_GIT_COMMIT",
    "SOURCE_VERSION",
)


def get_git_revision() -> str | None:
    """Auto-detect git commit SHA. Never raises.

    Priority: ``AMUND_GIT_REVISION`` (or legacy ``MORSE_GIT_REVISION``) ->
    GitHub -> GitLab -> Railway -> Vercel -> Render -> Heroku
    (``SOURCE_VERSION``) -> ``git rev-parse HEAD``.
    Returns None silently if no source is available.
    """
    explicit = get("GIT_REVISION").strip()
    if explicit:
        return explicit
    for var in _GIT_REVISION_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    try:
        import subprocess  # noqa: PLC0415

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:  # noqa: BLE001
        pass
    return None


_MAX_SERVICE_VERSION = 128


def get_service_version() -> str | None:
    """``service.version`` from ``OTEL_RESOURCE_ATTRIBUTES``, or None. Never raises.

    The variable is a comma-separated list of ``key=value`` pairs whose values
    are URL-decoded (OpenTelemetry spec). There is deliberately no ``init()``
    parameter for this: a deploy that already sets the OTel variable needs no
    code change. Values longer than 128 characters are truncated, matching the
    server's column.
    """
    try:
        from urllib.parse import unquote  # noqa: PLC0415

        raw = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
        for pair in raw.split(","):
            key, sep, val = pair.partition("=")
            if sep and key.strip() == "service.version":
                value = unquote(val.strip()).strip()
                return value[:_MAX_SERVICE_VERSION] or None
    except Exception:  # noqa: BLE001
        pass
    return None
