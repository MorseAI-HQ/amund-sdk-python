"""SDK-side PII redaction (Phase 6f Wave B G-02).

Default-on, in-process scrub of high-risk values BEFORE they leave the
customer's process for the Morse ingest endpoint. Mirrors the canonical
pattern catalogue in ``openspec/changes/phase-6f-g02-pii-redaction/design.md``
§2 — the same set is implemented in the API server and the TS SDK; the
parity-corpus test asserts byte-equal output across all three.

Public surface (re-exported from ``morse_ai._internal``):

- ``RedactionConfig`` — immutable per-client config.
- ``redact_string(s, config)`` — pure-string redactor.
- ``PATTERN_KEYS`` — canonical list of all eight pattern keys, in stable
  iteration order (must match server + TS).
- ``MANDATORY_PATTERNS`` — keys the customer cannot disable (server enforces a
  floor; the SDK rejects attempts to disable these at construction time so the
  failure mode is loud rather than silent).
- ``DEFAULT_OFF_PATTERNS`` — keys that ship disabled by default
  (``ip_address`` only at present).

The hook point is in ``transport.BatchTransport.track`` (and equivalent
record/span sites) — string leaves of the wire payload are walked and
redacted via ``RedactionConfig.redact_value``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Canonical pattern catalogue
# ---------------------------------------------------------------------------
#
# Keep this in lockstep with:
#  - apps/api/apps/core/redaction/patterns.py
#  - apps/sdk/typescript/src/_internal/redaction.ts
#
# The regex *strings* may differ syntactically across languages, but the match
# set on the parity corpus must be identical.

_PATTERN_SOURCES: dict[str, str] = {
    "api_key_prefixed": (
        # WP-12 parity with the server: ``rk_`` restricted keys, ``whsec_`` webhook
        # secrets, ``github_pat_`` fine-grained PATs and Resend keys
        # (``re_<id>_<secret>``, which must carry a digit and an uppercase letter so
        # snake_case identifiers never match). ``sk-ant-`` / ``sk-proj-`` are
        # covered by ``sk-[A-Za-z0-9_-]{20,}``. This pattern is compiled
        # case-sensitively (see ``_COMPILED``) exactly like the server's.
        r"\b(?:sk-[A-Za-z0-9_-]{20,}|sk_(?:live|test)_[A-Za-z0-9]{20,}|"
        r"rk_(?:live|test)_[A-Za-z0-9]{20,}|whsec_[A-Za-z0-9]{24,}|"
        r"github_pat_[A-Za-z0-9_]{30,}|"
        r"re_(?=[A-Za-z0-9_]{24,})(?=[A-Za-z0-9_]*\d)(?=[A-Za-z0-9_]*[A-Z])"
        r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)?|"
        r"(?:mhq|av)_(?:pk_)?[\s\S]{0,48}_[A-Fa-f0-9]{64}|"
        r"mhq_[A-Za-z0-9]{20,}|av_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{36}|gho_[A-Za-z0-9]{36}|"
        r"ghu_[A-Za-z0-9]{36}|ghs_[A-Za-z0-9]{36}|ghr_[A-Za-z0-9]{36}|"
        r"xox[bopsa]-[0-9A-Za-z\-]{10,}|"
        r"(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASCA)[0-9A-Z]{16})\b"
    ),
    "jwt": r"\beyJ[A-Za-z0-9_=-]{16,}\.eyJ[A-Za-z0-9_=-]{16,}\.[A-Za-z0-9_=-]{16,}\b",
    # 13-19 digit groups, optional spaces / dashes; Luhn is enforced by
    # ``_luhn_valid`` post-match.
    "credit_card": r"\b(?:\d[ -]?){12,18}\d\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    # Anchored on the left with a negative lookbehind that excludes digits and
    # phone-separator characters so a 16-digit credit-card-shaped run doesn't
    # leak its tail as a "phone" match (parity-corpus regression).
    "phone_us": (
        r"(?<![\d.\-])(?:\+?1[ \-.])?\(?[2-9]\d{2}\)?[ \-.]?\d{3}[ \-.]?\d{4}\b"
    ),
    "phone_e164": r"(?<![\d.\-])\+(?:[0-9][ \-.]?){8,14}[0-9]\b",
    "ssn_us": (r"\b(?!000|666|9\d{2})\d{3}[ \-.](?!00)\d{2}[ \-.](?!0000)\d{4}\b"),
    "ip_address": (
        r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
        r"(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}"
        r"|(?:[A-F0-9]{1,4}:){2,7}[A-F0-9]{1,4})\b"
    ),
    # MHQ-973 — key-context secret assignment. Every pattern above is
    # VALUE-shaped: it recognises a credential by its own format. That cannot
    # catch `password=hunter2` — `hunter2` is an ordinary word, and only the KEY
    # marks it as a secret.
    #
    # Deliberately excludes a bare `token`: this is an LLM-observability SDK, and
    # `input_tokens=100` / `max_tokens=4096` are core telemetry. Only unambiguous
    # `*_token` compounds match. No leading `\b` either, so `PGPASSWORD=` and
    # `MYSQL_PWD=` are caught; the `[=:]` anchor is what prevents false positives.
    "secret_assignment": (
        # Compound keys accept `=` or `:`; bare `secret` requires `=`, because
        # with `:` it is more often a marker (`k8s-secret: mounted`) than an
        # assignment.
        r"(?:(?:password|passwd|pwd|client[-_]?secret|secret[-_]?key|"
        r"private[-_]?key|access[-_]?key|api[-_]?key|auth[-_]?token|"
        r"access[-_]?token|refresh[-_]?token|id[-_]?token|session[-_]?token|"
        r"authorization|credentials?)[\"']?\s*[=:]"
        r"|\bsecret[\"']?\s*=)"
        r"\s*(?:bearer\s+)?(?:\"[^\"]*\"|'[^']*'|\S+)"
    ),
}

# Replacement labels (short keys per design.md §2 footer).
_REPLACEMENTS: dict[str, str] = {
    "api_key_prefixed": "[REDACTED:api_key]",
    "jwt": "[REDACTED:jwt]",
    "credit_card": "[REDACTED:credit_card]",
    "email": "[REDACTED:email]",
    "phone_us": "[REDACTED:phone]",
    "phone_e164": "[REDACTED:phone]",
    "ssn_us": "[REDACTED:ssn]",
    "ip_address": "[REDACTED:ip]",
    "secret_assignment": "[REDACTED:secret]",
}

#: Stable iteration order. Matches the server and the TS SDK.
PATTERN_KEYS: tuple[str, ...] = (
    "api_key_prefixed",
    "jwt",
    "credit_card",
    "email",
    "phone_us",
    "phone_e164",
    "ssn_us",
    "ip_address",
    # Last: the value-shaped patterns above are more specific, so a JWT inside
    # `authorization: Bearer eyJ...` gets its own tag before this one runs.
    "secret_assignment",
)

#: Patterns the customer cannot disable. The server-side ingest layer applies
#: these as a floor regardless; the SDK refuses to construct a Client that
#: tries to disable them so the failure mode is loud at startup, not silent.
MANDATORY_PATTERNS: frozenset[str] = frozenset(
    {"api_key_prefixed", "jwt", "credit_card", "ssn_us"}
)

#: Patterns that ship disabled by default (opt-in).
DEFAULT_OFF_PATTERNS: frozenset[str] = frozenset({"ip_address"})

#: Cap on customer-supplied extra patterns (mirrors the server cap).
MAX_EXTRA_PATTERNS: int = 10
#: Cap on each extra pattern length in characters (mirrors the server cap).
MAX_EXTRA_PATTERN_LEN: int = 256
#: Replacement label used for any customer-supplied custom pattern.
CUSTOM_REPLACEMENT: str = "[REDACTED:custom]"

#: Maximum nesting depth walked in attribute structures. Matches the server
#: (``apps.core.redaction.services.MAX_JSON_DEPTH``) and the ingest limit: a
#: top-level value is depth 0 and a string inside 64 nested containers is depth
#: 64, which is still walked. Anything deeper is replaced by ``DEPTH_MARKER``
#: (fail closed) rather than passed through unredacted.
MAX_WALK_DEPTH: int = 64
#: Replaces a subtree nested deeper than ``MAX_WALK_DEPTH`` (same as the server).
DEPTH_MARKER: str = "[REDACTED:depth]"

# Pre-compile the canonical patterns once at module import.
# ``api_key_prefixed`` is case-sensitive, like the server's (the Resend branch
# needs a real uppercase letter; ``AKIA`` is a literal prefix).
_CASE_SENSITIVE_PATTERNS = frozenset({"api_key_prefixed"})
_COMPILED: dict[str, re.Pattern[str]] = {
    key: re.compile(
        _PATTERN_SOURCES[key],
        0 if key in _CASE_SENSITIVE_PATTERNS else re.IGNORECASE,
    )
    for key in PATTERN_KEYS
}


# ---------------------------------------------------------------------------
# Luhn validator (credit card)
# ---------------------------------------------------------------------------


def _luhn_valid(digits: str) -> bool:
    """Return True iff the all-digits string passes the Luhn checksum.

    Used to filter false-positive credit-card matches — random 16-digit
    sequences ("1234567890123456") fail Luhn, while real card numbers like
    "4111111111111111" pass. Length bounds (13-19) are enforced by the regex.
    """
    if not digits or not digits.isdigit():
        return False
    if len(digits) < 13 or len(digits) > 19:
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = ord(ch) - ord("0")
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ---------------------------------------------------------------------------
# RedactionConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RedactionConfig:
    """Per-client redaction configuration.

    Construct via ``RedactionConfig.build(...)``; the dataclass itself is the
    immutable result of validation.
    """

    enabled: bool
    enabled_patterns: frozenset[str]
    extra_patterns: tuple[re.Pattern[str], ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------ build

    @classmethod
    def build(
        cls,
        *,
        enabled: bool = True,
        disabled_patterns: list[str] | None = None,
        extra_patterns: list[str] | None = None,
    ) -> RedactionConfig:
        """Build a config from public-facing kwargs.

        Validation rules (raised as ``ValueError``):

        - Each entry in ``disabled_patterns`` must be a known pattern key.
        - No entry in ``disabled_patterns`` may be in ``MANDATORY_PATTERNS``.
        - ``extra_patterns`` is capped at ``MAX_EXTRA_PATTERNS`` entries, each
          ≤ ``MAX_EXTRA_PATTERN_LEN`` chars, and each must compile via
          ``re.compile``. Compilation failure names the offending pattern.
        """
        # Defensive copies so we never reflect caller mutation.
        disabled = list(disabled_patterns or [])
        extra = list(extra_patterns or [])

        # Validate disabled list.
        for key in disabled:
            if key not in _PATTERN_SOURCES:
                raise ValueError(
                    f"redaction_disabled_patterns: unknown pattern key {key!r} "
                    f"(known: {sorted(_PATTERN_SOURCES)})"
                )
            if key in MANDATORY_PATTERNS:
                raise ValueError(
                    f"redaction pattern {key!r} cannot be disabled "
                    f"(server-side mandatory)"
                )

        # Validate extra patterns count + length + compilability.
        if len(extra) > MAX_EXTRA_PATTERNS:
            raise ValueError(
                f"redaction_extra_patterns: too many entries "
                f"({len(extra)} > {MAX_EXTRA_PATTERNS})"
            )
        compiled_extra: list[re.Pattern[str]] = []
        for pattern in extra:
            if not isinstance(pattern, str):
                raise ValueError(
                    f"redaction_extra_patterns: each entry must be a string, "
                    f"got {type(pattern).__name__}"
                )
            if len(pattern) > MAX_EXTRA_PATTERN_LEN:
                raise ValueError(
                    f"redaction_extra_patterns: pattern exceeds "
                    f"{MAX_EXTRA_PATTERN_LEN}-char cap: {pattern[:32]!r}…"
                )
            try:
                compiled_extra.append(re.compile(pattern))
            except re.error as exc:
                raise ValueError(
                    f"redaction_extra_patterns: failed to compile {pattern!r}: {exc}"
                ) from exc

        # Resolve enabled set: defaults minus disabled.
        if not enabled:
            enabled_set: frozenset[str] = frozenset()
        else:
            defaults = set(PATTERN_KEYS) - set(DEFAULT_OFF_PATTERNS)
            enabled_set = frozenset(defaults - set(disabled))

        return cls(
            enabled=enabled,
            enabled_patterns=enabled_set,
            extra_patterns=tuple(compiled_extra),
        )

    # ------------------------------------------------------------------- run

    def redact_string(self, value: str) -> str:
        """Apply enabled patterns + extra patterns to a string."""
        return redact_string(value, self)

    def redact_value(self, value: Any, _depth: int = 0) -> Any:
        """Redact an attribute value of arbitrary shape.

        - ``str`` → ``redact_string``
        - ``dict`` / ``list`` / ``tuple`` → walk (depth-bounded)
        - everything else (numbers, booleans, ``None``, custom objects) → pass
          through unchanged
        - any node nested deeper than ``MAX_WALK_DEPTH`` → ``DEPTH_MARKER``
          (fail closed, same as the server)
        """
        if not self.enabled and not self.extra_patterns:
            return value
        if _depth > MAX_WALK_DEPTH:
            return DEPTH_MARKER
        if isinstance(value, str):
            return self.redact_string(value)
        if isinstance(value, dict):
            return {
                k: self.redact_value(v, _depth=_depth + 1) for k, v in value.items()
            }
        if isinstance(value, list):
            return [self.redact_value(v, _depth=_depth + 1) for v in value]
        if isinstance(value, tuple):
            return tuple(self.redact_value(v, _depth=_depth + 1) for v in value)
        return value


#: Module-level "redact nothing" config. Constructed once for cheap re-use by
#: clients that opt out entirely so we don't pay validation cost per call.
DISABLED_CONFIG: RedactionConfig = RedactionConfig(
    enabled=False, enabled_patterns=frozenset(), extra_patterns=()
)


# ---------------------------------------------------------------------------
# redact_string — pure utility, used by tests + the parity corpus
# ---------------------------------------------------------------------------


def redact_string(s: str, config: RedactionConfig) -> str:
    """Apply ``config``'s enabled patterns to ``s``, returning the redacted
    string.

    Iteration order is the canonical ``PATTERN_KEYS`` order. Built-in patterns
    run before customer-supplied extra patterns so the customer's regex sees
    text after the high-risk replacements have already happened.
    """
    if not s:
        return s

    out = s
    if config.enabled_patterns:
        for key in PATTERN_KEYS:
            if key not in config.enabled_patterns:
                continue
            compiled = _COMPILED[key]
            replacement = _REPLACEMENTS[key]
            # Only the digit-run patterns are kept off identifiers; key/token
            # patterns always see the full text.
            shielded, protected = (
                _shield_ids(out) if key in _ID_SHIELDED_PATTERNS else (out, [])
            )
            if key == "credit_card":
                shielded = compiled.sub(_credit_card_sub, shielded)
            else:
                shielded = compiled.sub(replacement, shielded)
            out = _unshield_ids(shielded, protected) if protected else shielded

    # Customer-supplied extras run last.
    for compiled in config.extra_patterns:
        out = compiled.sub(CUSTOM_REPLACEMENT, out)

    return out


# MHQ-1412: identifiers are not PII. A UUID or a long hex id (trace/span ids,
# vector-store document ids) often contains a 10+ digit run, which the phone
# and card patterns carved out of the middle (``bc41d39b-...-0c[REDACTED:phone]``),
# corrupting ids that exist to be joined on. Id-shaped tokens are set aside
# before the built-in patterns run and restored afterwards. A hex id must
# contain at least one letter, so an all-digit run (a card number) is never
# shielded. Only the digit-run patterns below are shielded: API-key, JWT and
# secret patterns always see the full text, so a key whose body is hex is
# still redacted.
_ID_SHIELDED_PATTERNS = frozenset({"credit_card", "phone_us", "phone_e164", "ssn_us"})
# Never shields part of an email address (local part or domain label), so the
# email pattern still sees the whole address.
_ID_TOKEN = re.compile(
    r"(?<![\w@.])(?:"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"|(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*[0-9])[0-9a-fA-F]{16,64}"
    r")(?![0-9A-Za-z])(?![\w.+\-]*@)(?!\.[A-Za-z]{2,})"
)
_ID_PLACEHOLDER = "\ue000morse-id-{}\ue000"
_ID_PLACEHOLDER_RE = re.compile("\ue000morse-id-(\\d+)\ue000")


def _shield_ids(s: str) -> tuple[str, list[str]]:
    protected: list[str] = []
    if "\ue000" in s:  # never collide with caller text using the sentinel
        return s, protected

    def _sub(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return _ID_PLACEHOLDER.format(len(protected) - 1)

    return _ID_TOKEN.sub(_sub, s), protected


def _unshield_ids(s: str, protected: list[str]) -> str:
    return _ID_PLACEHOLDER_RE.sub(lambda m: protected[int(m.group(1))], s)


def _credit_card_sub(match: re.Match[str]) -> str:
    """Replace the match only if its digit-only form passes Luhn."""
    raw = match.group(0)
    digits = "".join(ch for ch in raw if ch.isdigit())
    if _luhn_valid(digits):
        return _REPLACEMENTS["credit_card"]
    return raw


__all__ = [
    "CUSTOM_REPLACEMENT",
    "DEFAULT_OFF_PATTERNS",
    "DEPTH_MARKER",
    "DISABLED_CONFIG",
    "MANDATORY_PATTERNS",
    "MAX_EXTRA_PATTERNS",
    "MAX_EXTRA_PATTERN_LEN",
    "MAX_WALK_DEPTH",
    "PATTERN_KEYS",
    "RedactionConfig",
    "redact_string",
]
