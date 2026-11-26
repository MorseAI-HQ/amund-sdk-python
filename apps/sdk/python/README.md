# Amund Python SDK

Verdict-first performance intelligence for production AI agents.

> **Renamed from `morse-ai`.** Morse is now Amund. The package is `amund`
> (`pip install amund`, `import amund`). Existing installs keep working: the
> `morse-ai` package is now a thin shim that installs `amund` and keeps
> `import morse_ai` working as an alias (with a `DeprecationWarning`), and
> every `MORSE_*` environment variable is still read. See
> [Migrating from morse-ai](#migrating-from-morse-ai).

**Requires Python 3.11 or newer.** On Python 3.10 or older, `pip install amund` fails with
"No matching distribution found"; only `pip install -v amund` shows the real reason
(`requires-python >=3.11`).

## Set up with a coding agent

Using Claude Code, Cursor or Codex? Give it the Amund setup skill and it instruments your agent
for you: install, `init()` and `run()`, the client wrapper or framework adapter, outcomes only
where your code can check them, then a check that the first trace arrived.

```text
Read https://docs.amundai.com/skills/amund-setup/SKILL.md and follow it to instrument this agent with Amund.
```

How to install the skill for each editor: https://docs.amundai.com/start/coding-agent

## Install

```bash
pip install amund
```

With an adapter (installs the underlying provider SDK too):

```bash
pip install "amund[anthropic]"
pip install "amund[openai]"
pip install "amund[openai_agents]"
pip install "amund[langgraph]"     # also pulls langchain-core
pip install "amund[claude_agent_sdk]"
pip install "amund[all]"           # every adapter
```

The import name is `amund` (underscore) regardless of which extra
you install — `import amund`.

### Supported versions

Python 3.11+. We verify two versions of each framework — the declared floor
and the current latest. Versions between them are expected to work but are
not verified; below the floor is unsupported. The `latest` half is still
being built out (MHQ-930), so today only the floor is verified. Policy:
`engineering/SDK_VERSION_SUPPORT.md`.

The versions below are what CI installs and runs the adapter suite against
— not the widest version we believe works.

What is verified against the real installed package is the structural contract
— every adapter monkey-patches a method on someone else's class
(`Messages.create`, `Runner.run`, `ClaudeSDKClient.receive_response`, …), and
`tests/contract/` asserts those targets still exist on every PR. Without it an
upstream rename makes `install()` return `False` silently, and you get no
telemetry and no error.

Behavioural coverage is still mostly fixture-driven, so a green suite is weaker
evidence than it looks; cross-version coverage is MHQ-930.

| Framework          | Declared floor  | Tested in CI |
| ------------------ | --------------- | ------------ |
| `anthropic`        | `>=0.40.0`      | 0.86.0       |
| `openai`           | `>=1.50.0`      | 2.29.0       |
| `openai-agents`    | `>=0.15.0,<1.0` | 0.15.2       |
| `langgraph`        | `>=1.0`         | 1.2.2        |
| `langchain-core`   | `>=1.0`         | 1.4.0        |
| `claude-agent-sdk` | `>=0.1.0`       | 0.2.130      |

## Quick Start

```python
import amund

amund.init(api_key="mhq_xxxxx")  # pragma: allowlist secret

amund.record(
    agent="lead-qualifier",
    success=True,
    outcome="qualified",
    cost=0.043,
)
```

Or trace a multi-step run:

```python
with amund.run("lead-qualifier") as r:
    with amund.span("fetch-lead", "tool"):
        ...
    r.set_outcome(True, "qualified")
```

`record()`/`run()`/`span()`/`@trace_agent` all send real OpenTelemetry
spans over OTLP by default — see [Transport](#transport) below.

## Zero-config instrumentation

```bash
pip install "amund[anthropic]"
export AMUND_API_KEY=mhq_xxxxx   # MORSE_API_KEY still works
# your existing anthropic / openai / openai-agents / langgraph /
# claude-agent-sdk code is now traced — no amund.wrap() calls anywhere
```

When the SDK initializes, it scans `sys.modules` for installed
provider SDKs — and registers an import hook for ones imported later —
and calls each detected adapter's `install()`. Today that covers
`anthropic`, `openai` (raw clients, and libraries that drive one: LiteLLM's
OpenAI provider, Agno's `OpenAIChat`), `openai-agents` (imports as `agents`),
`langgraph`, and `claude-agent-sdk`. Priority rules: when both LangGraph and
generic LangChain are importable, only the LangGraph adapter installs; and the
raw OpenAI adapter does not record a call that a framework adapter already
records (an OpenAI Agents run, a LangChain `ChatOpenAI` call under the
callback handler), to avoid duplicate spans. `google-genai` and LiteLLM's
non-OpenAI providers are not covered: send those over OTLP. Opt out entirely with
`AMUND_DISABLE_AUTO_DETECT=1`; you can still call any adapter's
`install()`/`wrap()` directly.

The plain LangChain adapter (`MorseCallbackHandler`, for chains
outside a LangGraph graph) is **not** auto-installed — pass it via
`callbacks=[...]` explicitly (see below).

## Adapters

### Anthropic

```python
from amund.adapters import anthropic as av_anthropic
import anthropic

client = av_anthropic.wrap(anthropic.Anthropic())
# every messages.create call now emits an `llm` span automatically
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Qualify this lead..."}],
)
```

`wrap()` is idempotent and works whether or not auto-detect already
installed the adapter globally. Pass `summarize=True` (or use the
`summarize()` context manager per-call) to also emit a sibling
`memory` span.

### OpenAI

```python
from amund.adapters import openai as av_openai
import openai

client = av_openai.wrap(openai.OpenAI())
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Qualify this lead..."}],
)
```

Context segments (messages + tools) are captured automatically on
every call — no separate extraction step. `responses.create` is traced the
same way.

- **Streaming.** `stream=True` (and the `.stream()` helpers when the adapter
  is installed) records one `llm` span when the stream ends. Chat Completions
  only sends usage on a stream with `stream_options={"include_usage": True}`;
  without it the span has the model and output but no tokens
  (`metadata.usage_unavailable = true`).
- **Tools.** Your code runs the tools, so the adapter infers a `tool` span
  from the response that asked for a call and the next request that carries
  its result (`metadata.timing = "inferred"`: the span covers the gap between
  the two calls). A failed call records a failed `llm` span with the error.

### OpenAI Agents SDK, LangGraph, Claude Agent SDK

These three are auto-installed globally by zero-config detection (see
above) — no wrap call needed once the package is importable and
`AMUND_API_KEY` is set. To install manually (e.g. to control ordering,
or with auto-detect disabled):

```python
from amund.adapters import openai_agents, langgraph, claude_agent_sdk

openai_agents.install()      # patches agents.Runner.run / run_sync / run_streamed
langgraph.install()          # registers a callback handler globally with LangChain
claude_agent_sdk.install()   # patches ClaudeSDKClient.receive_response
```

Each emits an outer **agent** span, per-call **llm** + **tool** spans,
and adapter-specific spans (LangGraph: nested chain spans; OpenAI
Agents: **subagent.spawn**-shaped `handoff` spans; Claude Agent SDK: a
sub-agent launch is the `tool Agent` (or `Task`) span tagged
`metadata.askl_is_spawn = true`, with the sub-agent's own agent, tool and
llm spans nested under it. The Python adapter does not emit a span per
customer hook; failed tools carry the tool's error text).

### Plain LangChain (chains, LLMs, tools, retrievers)

Not auto-installed — attach `MorseCallbackHandler` via `callbacks`:

```python
from amund.adapters.langchain import MorseCallbackHandler

handler = MorseCallbackHandler("my-agent")
chain = SomeChain(..., callbacks=[handler])
```

Called inside an existing `amund.run()` context, the handler
appends child spans to that trace. Called standalone (no enclosing
`run()`), it auto-creates a trace that flushes when the outermost
chain completes — no `amund.run()` wrapper required.

## Linking runs to business records and versions

Tell Amund which business record a run handled and which version of your
agent produced it, so outcomes that arrive later (a ticket closed, a refund
issued) land on the right run and a change can be compared with what came
before.

```python
amund.init(api_key="mhq_xxxxx", agent_version="2.4.1")  # pragma: allowlist secret

with amund.run(
    "support-triage",
    record_keys={"zendesk_ticket": 4812},   # known up front...
    agent_version="2.5.0-rc1",              # ...overrides init() for this run
):
    amund.link_record("order_id", "A-1009")   # ...or discovered mid-run
    with amund.span("classify", "llm"):
        amund.set_prompt("triage", version=7)  # pinned to this span
    amund.set_prompt("summarise")              # no version: prompts kept in code
```

- **Record keys.** `link_record(key_type, key_value)` and `run(record_keys={...})`
  attach ids from your own systems. `key_type` is lower-cased and must match
  `^[a-z0-9][a-z0-9_.-]{0,63}$`; values are strings or ints, 1-256 characters.
  Up to 20 per run; a repeated `key_type` replaces its earlier value. Invalid
  keys are dropped and logged at debug level, never raised.
  **Use opaque ids, not personal data** such as email addresses.
- **`agent_version`.** Your own label (max 128 characters). `init(agent_version=...)`
  is the default for every run; `run(..., agent_version=...)` overrides it.
- **`set_prompt(name, version=None)`.** Records which named prompt a run used.
  Inside a `span()` the reference is pinned to that span. Up to 20 per run.
  Over OTLP a span carries one reference, so if several are set on the same
  span the last one wins there; the JSON path keeps them all.
- **Git revision.** Detected automatically from `AMUND_GIT_REVISION`,
  `GITHUB_SHA`, `CI_COMMIT_SHA`, `RAILWAY_GIT_COMMIT_SHA`,
  `VERCEL_GIT_COMMIT_SHA`, `RENDER_GIT_COMMIT`, `SOURCE_VERSION`, then
  `git rev-parse HEAD`, in that order. In a Docker image with no `.git`
  directory, inject it at build time:

  ```dockerfile
  ARG GIT_SHA
  ENV AMUND_GIT_REVISION=$GIT_SHA
  ```

  (`docker build --build-arg GIT_SHA=$(git rev-parse HEAD) .`)

- **`service.version`.** If `OTEL_RESOURCE_ATTRIBUTES` contains
  `service.version=...`, it is sent with every run. No code change needed.

Outside an open trace, `link_record()` and `set_prompt()` do nothing.

## Fetching hosted prompts

```python
import amund

prompt = amund.get_prompt("triage")                      # label="production"
prompt = amund.get_prompt("triage", version=3)           # or pin a version
same = amund.get_prompt("triage", if_none_match=prompt.etag)
if same.not_modified: ...                                   # 304, other fields are None
prompt = amund.get_prompt("triage", cache_ttl=60)        # optional in-memory cache
```

`get_prompt` is a thin GET: no retries, no fallback. Inside an open run it also links that run to
the fetched prompt version (`link_source="fetch"`), so the trace shows which version ran; pass
`link=False` to opt out. A cache hit links too, a 304 links via the SDK's own cached copy, and with
no open run nothing happens. If you also call `set_prompt` for the same name, version and span, one
reference is kept. It raises `PromptAccessError` (403, the key needs the
`prompts:read` scope), `PromptNotFoundError` (404) and `PromptRateLimitError` (429, `.retry_after`);
network failures raise `httpx`'s transport errors.

We recommend caching in some form so Amund is not on your request hot path. You have three options:

1. **Call the API directly** on every use (the default). Responses carry `Cache-Control: max-age=30`.
2. **Use your own cache**, with a bundled fallback prompt if you want one.
3. **Pass `cache_ttl`** (seconds) for a small process-local LRU. A fresh entry is returned without a
   request; a stale one is revalidated with `If-None-Match` (a 304 refreshes it). Errors are never
   masked by stale data, so keep your own fallback if you want one. `None` or `0` (the default)
   uses no cache, and passing your own `if_none_match` bypasses it.
   `configure_prompt_cache(max_entries=256)` sets the size and `clear_prompt_cache()` empties it.

## Capturing logs alongside agent traces

If you already use stdlib `logging` or `structlog`, you can ship every
log line to Amund with one line of config. Each record is
auto-correlated with the active trace (the one entered via
`amund.run()`, `@amund.trace_agent`, or any active
`amund.span(...)`) and appears under the trace's **Logs** tab on
the dashboard.

The shim re-uses the SDK's batched transport — no blocking I/O on your
hot path.

### Stdlib `logging`

```python
import logging
import amund
from amund.logging import MorseHandler

amund.init(api_key="mhq_xxxxx")  # pragma: allowlist secret
logging.basicConfig(level=logging.INFO, handlers=[MorseHandler()])
```

### `structlog`

```python
import structlog
import amund
from amund.logging import structlog_processor

amund.init(api_key="mhq_xxxxx")  # pragma: allowlist secret
structlog.configure(processors=[structlog_processor, structlog.processors.JSONRenderer()])
```

Log lines emitted outside any trace context still flow to Amund — they
just won't be correlated to a specific trace. Errors inside the
handler / processor are absorbed and never propagate into your
`logger.info(...)` call.

## Cost

`cost_usd` is computed automatically from the model + token counts on
every `llm` span the adapters above emit, using a bundled pricing
table — no separate call needed. `record()` doesn't auto-compute —
pass `cost` yourself if you have it.

## Transport

`run()`/`span()`/`trace_agent`/`record()` emit real OpenTelemetry spans
— `TracerProvider` / `BatchSpanProcessor` / `TraceIdRatioBased` sampler
— over OTLP (protobuf+gzip). This is a **core dependency**
(`opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`), not an
optional extra — every install speaks OTLP. `track()`/`batch_track()`
are a lower-level, non-trace-shaped escape hatch of their own — prefer
`record()`.

Spans go to the Amund API at `https://api.amundai.com` by default. `https://api.morseai.dev`
remains a supported alias for the same service. Set `MORSE_ENDPOINT` (or `init(endpoint=...)`)
to send elsewhere.

## PII Redaction

The SDK ships with PII redaction **on by default**. Span attributes and
event payloads are scrubbed inside your process before any data leaves
for the Amund ingest endpoint. You can opt out per-pattern, opt out
entirely, or extend the catalogue with your own regex.

### What gets redacted by default

| Pattern key         | Matches                                                                                                                                                                                                               | Replacement              |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `api_key_prefixed`  | `sk-…`, `sk_live_…`, `sk_test_…`, `mhq_…`, GitHub PATs (`ghp_…`/`gho_…`/…), Slack tokens (`xoxb-…`), AWS access keys (`AKIA…`/`ASIA…`/…)                                                                              | `[REDACTED:api_key]`     |
| `jwt`               | Three-segment base64url tokens (`eyJ…`)                                                                                                                                                                               | `[REDACTED:jwt]`         |
| `credit_card`       | 13-19 digit groups passing the Luhn checksum                                                                                                                                                                          | `[REDACTED:credit_card]` |
| `email`             | RFC-5322-ish address regex                                                                                                                                                                                            | `[REDACTED:email]`       |
| `phone_us`          | US-format phone numbers                                                                                                                                                                                               | `[REDACTED:phone]`       |
| `phone_e164`        | International E.164 phone numbers                                                                                                                                                                                     | `[REDACTED:phone]`       |
| `ssn_us`            | US Social Security numbers (`999-99-9999`)                                                                                                                                                                            | `[REDACTED:ssn]`         |
| `ip_address`        | IPv4 / IPv6 — **disabled by default**, opt-in                                                                                                                                                                         | `[REDACTED:ip]`          |
| `secret_assignment` | Key-context secret assignments: `password=`, `client_secret:`, `authorization: Bearer …`, `PGPASSWORD=` and similar. Excludes token _counts_ (`input_tokens=`, `max_tokens=`) — those are telemetry, not credentials. | `[REDACTED:secret]`      |

### Configuring redaction

```python
import amund

amund.init(
    api_key="mhq_xxxxx",  # pragma: allowlist secret
    # All four kwargs below are optional; defaults are shown.
    redaction_enabled=True,
    redaction_disabled_patterns=None,    # e.g. ["email", "phone_us"]
    redaction_extra_patterns=None,       # e.g. [r"INTERNAL-\w+"]
)
```

- `redaction_enabled=False` turns OFF every built-in pattern at the SDK
  layer. **The server-side mandatory floor still applies** —
  `api_key_prefixed`, `jwt`, `credit_card`, and `ssn_us` are rewritten
  on ingest regardless of SDK configuration. Opting out shifts _where_
  redaction happens, it does not remove it.
- `redaction_disabled_patterns` lets you skip individual non-mandatory
  keys (`email`, `phone_us`, `phone_e164`, `ip_address`). Trying to
  disable a mandatory pattern raises `ValueError` at `init()`.
- `redaction_extra_patterns` accepts up to 10 customer regex strings
  (each ≤ 256 chars). Each must compile via `re.compile()`; matches are
  replaced with `[REDACTED:custom]`. Built-in patterns run before extra
  patterns, so a custom regex sees the already-redacted text.

### What is NOT touched

Non-string leaves (numbers, booleans, `None`, opaque objects) pass
through untouched. Span IDs, timestamps, durations, token counts, and
other metadata are never redacted.

## Docs

Full reference — every adapter, all `init()` options, configuration,
troubleshooting — lives at **[docs.amundai.com](https://docs.amundai.com)**.
This README stays a quickstart.

## Naming

The PyPI project and the import name are both `amund`. The npm package for
the TypeScript SDK is `@morsehq-dev/sdk` — see that package's README for why
it doesn't match this one's naming.

## Migrating from morse-ai

Nothing breaks if you change nothing: `pip install morse-ai` now installs the
same version of `amund` plus a `morse_ai` alias package. `import morse_ai`,
`from morse_ai.adapters.openai import ...` and `mock.patch("morse_ai....")`
all resolve to the very same `amund` module objects, so singletons and
`isinstance` checks are shared. Importing `morse_ai` emits a
`DeprecationWarning`. To move over:

```bash
pip uninstall morse-ai
pip install amund              # same extras: "amund[anthropic]" etc.
```

- Replace `import morse_ai` / `from morse_ai...` with `import amund` /
  `from amund...`.
- Environment variables are now `AMUND_*` (`AMUND_API_KEY`,
  `AMUND_ENDPOINT`, `AMUND_DISABLED`, ...). Every `MORSE_*` name is still
  read; when both are set, the `AMUND_` value wins.
- The SDK's own logger is now `amund` (was `morse_ai`). If you configured it
  by name, e.g. `logging.getLogger("morse_ai").setLevel(...)` or a
  `dictConfig` `loggers` entry, rename it to `amund`.
- Nothing changes on the wire: span attributes, headers and the default
  endpoint are the same, so dashboards and alerts keep working.
