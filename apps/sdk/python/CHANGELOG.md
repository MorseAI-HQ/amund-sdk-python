# Changelog

All notable changes to `morse-ai` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The Python and TypeScript SDKs version **independently** — they share `0.4.0` as
their first public release because they reached feature parity there, not
because they are released in lockstep. Expect the numbers to diverge.

## [0.6.0] — 2026-10-01

### Changed — BREAKING

- **Wire attributes moved from `av.*` to `morse.*`** (#1002). Every OTLP
  attribute the SDK emits now uses the `morse.*` namespace. A collector or
  dashboard keyed on `av.*` attributes must be updated.
- **A clean exit no longer claims success** (MHQ-1157). `run()` and
  `@trace_agent` used to send `success=true` when your code returned without error. The server stores
  any `success` as a developer-declared outcome — the top provenance tier,
  which no later eval, classifier or webhook signal could ever overwrite. Now
  only a declared outcome is sent; a raise/throw still records `success=false`.
  Declare outcomes with `morse_ai.set_outcome(success, outcome=None)`.

### Added

- **`morse_ai.set_outcome(success, outcome=None)`** — declares the outcome of
  whichever trace is open, inside `@trace_agent` as well as `run()`, without
  needing the run handle.

### Fixed

- **`sdk_version` is now recorded on OTLP traces.** The instrumentation scope
  carried no version, so every trace was stored as `otlp/http`.
- **OpenAI Agents spans were dropped when `morse_ai.init()` ran before
  `import agents`** (MHQ-1154). The lazy adapter hook patched `Runner` from a
  background thread after the import had returned, so a run started straight
  away raced it and stored only the outer agent span. Adapters now install
  inside the import itself (also covers anthropic, langgraph and
  claude_agent_sdk lazy installs).

- **`claude_agent_sdk.query()` now emits telemetry** (MHQ-1053). The adapter
  wrapped the module-level `query()` only to capture the prompt, so a public,
  documented entry point produced no trace and no `llm` span — while
  `install()` still logged `adapter installed`. You got an encouraging log line
  and silence. `ClaudeSDKClient` was unaffected and stays unchanged.

- **A rate-limited batch is retried instead of dropped** (MHQ-673). The OTLP
  exporter treats only 408 and 5xx as retryable and ignores `Retry-After`, so a
  429 discarded the batch on its first response. The SDK now waits out the
  server's `Retry-After` and redelivers, bounded at 3 retries / 30s. If you are
  over your ingest limit you lose throughput, not data. `Retry-After: 0` — a
  server saying "retry now" — is treated as such rather than as an exhausted
  budget; it briefly ended the retry loop instead of starting it, and never
  shipped in a release.

  `quota_rejected` in your delivery counters therefore means something stricter
  than before: you were rate-limited for long enough that redelivery could not
  catch up, not that one 429 arrived.

- **Spans are no longer attached to a trace that has already been sent**
  (MHQ-980). A finalized trace can survive in the context after its run ends,
  and every reader treated it as live; spans appended to it were silently
  discarded. This was found once in the `openai-agents` adapter (MHQ-672, 28 of
  33 conversations lost) and fixed there alone — the other 50 read sites are now
  fixed too. When it happens, the SDK logs a warning once per process instead of
  saying nothing.

- **Delivery-loss counters survive a process that just exits** (MHQ-1054). The
  diagnostics sender flushes every 5 seconds, so any script, CLI, serverless
  invocation or short-lived task that finished inside that window died with its
  buffer unsent — including runs that lost telemetry and had something to
  report. Draining now happens at interpreter exit, not only when an
  application calls `morse_ai.shutdown()` explicitly.

  A healthy process is unaffected and still transmits nothing at exit, and
  `MORSE_SDK_DISABLE_TELEMETRY=1` still wins.

## [0.5.0] — 2026-08-08

### Changed — BREAKING

- **Sentry is gone from the SDK, replaced by Morse-owned self-diagnostics**
  (MHQ-908). When the SDK absorbs one of its own internal errors it now reports
  to Morse's `/api/v1/diagnostics/sdk` instead of to a third party you may not
  have installed.

  The old reporter **hijacked your global Sentry client** (MHQ-729) — if you
  used Sentry, the SDK could reconfigure it underneath you. That is the defect
  this removes. If you were relying on Morse SDK errors reaching your Sentry,
  they no longer will; they reach your own `morse_ai` logger instead.

  **This is on by default, at level `error`.** Upgrading turns transmission on.
  Two channels, both silent on a healthy install: absorbed **errors** carrying
  a Morse-scoped stack (our frames only — yours collapse to a count, and frame
  locals are never read), and aggregate **delivery counters** telling you when
  telemetry is being dropped.

  Everything transmitted is also written to your own logger first, rate-limited,
  **even when you have opted out** — there is no category of self-reported
  information we receive that your engineers cannot see. Opt out with
  `MORSE_SDK_DISABLE_TELEMETRY=1`. Full field-by-field disclosure:
  https://docs.morsehq.dev/sdk/self-diagnostics

### Fixed

- **A context value that raises no longer breaks your call** (MHQ-947).
  `set_context()` and `run(user_metadata=...)` take YOUR values, and the SDK
  is the one that calls `str()` on them — so an object whose `__str__` or
  `__repr__` raises ran that code on the telemetry path and the exception
  escaped into your application. `run(user_metadata=...)` failed at trace
  start, taking the whole `with` block with it. A realistic trigger is an ORM
  object whose `__str__` lazily hits a dead connection.

  Coercion is now guarded per key: a value that raises is dropped and reported
  to us, and **every other dimension still arrives**. `set_context()` also
  gains `@absorb_errors`, which it was the only public export to be missing.

- **Self-diagnostics were lost at interpreter exit** (MHQ-946). `shutdown()`
  stopped the telemetry transport but never drained the diagnostics buffer,
  whose sender is a daemon thread on a 5-second interval — so any process that
  finished inside that window died with its reports unsent, and nothing else
  drained them. That is every script, CLI, serverless invocation, short-lived
  Celery task, and every driver in the adapter matrix; long-running services
  were unaffected, which is why it went unnoticed. Absorbed errors were still
  written to the local logger throughout, so the only visible symptom was an
  empty `sdk_diagnostic_events` for Python.

  This is the Python half of MHQ-927, which fixed TypeScript only. Delivery
  counters are now drained too: they are the sole signal that telemetry is
  being dropped, and they never built a sender on their own.

### Added

- **`claude_agent_sdk` optional extra** (`pip install "morse-ai[claude_agent_sdk]"`),
  floor `>=0.1.0`. The adapter has auto-installed since 2026-06-22 with no extra
  declaring it and the package never installed in CI — its tests ran against
  hand-written mock classes. Installing the real package moved the suite from
  638 passed / 11 skipped to 655 passed / 8 skipped; those package-gated tests
  had been silently skipping.

### Notes

- Framework version support is now governed by an explicit policy —
  `engineering/SDK_VERSION_SUPPORT.md`. The Python SDK's declared floors are
  unchanged in this release; the TypeScript SDK's were raised (breaking) in the
  same cycle. Python's real-package coverage remains partial: 5 test files bind
  to actual packages, the rest use fixtures, and there is no canary lane yet.

## [0.4.0] — 2026-07-31

First public release on PyPI.

### Added

- **Six framework adapters**, each dogfooded end-to-end against a real
  third-party application before release: Anthropic Messages API, Claude Agent
  SDK, OpenAI (raw client), OpenAI Agents SDK, LangChain (LCEL + `bind_tools`),
  and LangGraph.
- `input_data` / `output_data` capture on LLM spans for the Anthropic
  (MHQ-772) and OpenAI (MHQ-774) adapters — prompt and completion payloads are
  now visible in the trace, subject to the redaction config.
- OTLP export path alongside the REST transport, including W3C trace-context
  propagation (`extract_traceparent` / `inject_traceparent`).
- Retry/backoff on the telemetry transport: 3 attempts with jittered
  exponential backoff, and quota feedback read from the `X-Morse-Quota-*`
  response headers.

### Fixed

- Anthropic adapter LLM spans always reported ~0ms duration (MHQ-771).
- OpenAI adapter LLM spans had the same 0ms duration bug.
- Claude Agent SDK sub-agent spawns never finalized when the stream ended
  without a terminal task message — they now finalize via a `ResultMessage`
  fallback (MHQ-783), and in-flight spawn hooks are drained on shutdown
  (MHQ-762).
- API keys are redacted from telemetry under both the current `mhq_` prefix and
  the legacy `av_` prefix, so keys issued before the rename are still scrubbed.

### Changed

- **The SDK version is now declared in exactly one place** (`pyproject.toml`)
  and read back from package metadata at import time. It was previously
  copy-pasted into four locations; the copies could drift from the released
  version and would then misreport `sdk_version` on every ingested span and in
  the `User-Agent` header.
