# Changelog

All notable changes to `morse-ai` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The Python and TypeScript SDKs version **independently** — they share `0.4.0` as
their first public release because they reached feature parity there, not
because they are released in lockstep. Expect the numbers to diverge.

## [Unreleased]

### Fixed — Azure OpenAI spans are priced on the canonical model (MHQ-1741)

- **Azure Responses API calls were unpriced.** Azure echoes the _deployment name_ as
  `response.model` on the Responses API (measured 2026-11-04 on api-version
  `2025-04-01-preview` and on `/openai/v1`, streamed and not), so an `AsyncAzureOpenAI`
  `responses.create(model="morse-hr-mini")` span recorded `model="morse-hr-mini"` and cost $0.
  Every Azure response, on both APIs, carries the canonical dated model in the
  `x-ms-served-model` header (`gpt-4.1-mini-2025-04-14`); the adapter now reads it and uses it
  as the span model and for pricing.
- **The deployment is no longer lost on Chat Completions.** Azure spans now carry
  `metadata.provider = "azure_openai"`, `metadata.azure_deployment` and
  `metadata.azure_model_source` (`served_model_header`, `response` or `deployment` — the last
  only when neither the header nor a differing body model is available, i.e. the old behaviour).
- Covers `chat.completions.create` and `responses.create`, sync, async and streaming,
  `with_raw_response` (LiteLLM's Azure path), `wrap()` and the zero-code `install()`.
  Detection: `AzureOpenAI` / `AsyncAzureOpenAI`, or any OpenAI client whose `base_url` is an
  Azure OpenAI host (`*.openai.azure.com`, `*.cognitiveservices.azure.com`,
  `*.services.ai.azure.com`). The header is read by a response hook appended (once) to the
  Azure client's own httpx client; the customer's hooks keep running and the returned objects
  are unchanged. Non-Azure clients are untouched.

## [0.8.0] — 2026-11-04

Everything below shipped first as `0.8.0rc1` (2026-11-04) and was acceptance-tested on the
testbed apps against Morse dev before this release (SDK_PUBLISHING.md §1.1).

### Added — zero-code AWS Strands Agents tracing (MHQ-133)

- **`strands` joins the auto-detect map.** With `MORSE_API_KEY` + `init()`, the spans Strands
  Agents emits through its own OpenTelemetry tracer (`invoke_agent`, `execute_event_loop_cycle`,
  `chat`, `execute_tool`, `invoke_swarm` / `invoke_graph`) are exported to Morse's OTLP endpoint,
  where the GenAI conventions are mapped to agent / llm / tool spans and priced server-side
  (Bedrock ids such as `us.amazon.nova-micro-v1:0` included). New optional extra:
  `morse-ai[strands]` (`strands-agents>=1.0.0`).
- **Nothing global changes.** When the app has no OTel SDK provider, Strands' spans are started on
  the Morse SDK's own provider (same sampling, redaction and delivery as `run()` traces). When the
  app configured its own provider (`StrandsTelemetry()`, another backend), its exporters keep
  receiving the spans unchanged and a copy of the Strands-scope spans is forwarded to Morse.
- **No double counting:** while Strands invokes a model it marks itself the LLM owner, so the
  zero-code OpenAI and Anthropic patches skip the same call (Strands' `OpenAIModel`,
  `AnthropicModel`, LiteLLM). The Anthropic patch now honours the LLM owner too.
- **Multi-agent spans are typed `agent`** (`morse.span_type`); the server otherwise types any span
  with a `gen_ai.*` attribute as `llm`.
- **Span-event attributes are redacted** in the export path, like span attributes. Strands puts
  prompts and completions in events by default.
- Known limits: a Strands trace is separate from an open `morse_ai.run()` trace; the server does
  not yet label these traces with a framework or an SDK version.

### Added — zero-code google-genai tracing (MHQ-1506)

- **`google-genai` joins the auto-detect map.** With `MORSE_API_KEY` + `init()`,
  `client.models.generate_content`, `generate_content_stream` and the async
  `client.aio.models` variants are traced for the Gemini API and Vertex AI
  (`vertexai=True`). New optional extra: `morse-ai[google_genai]` (`google-genai>=1.0.0`).
- **One `llm` span per billed request.** The adapter patches the SDK's per-request
  methods, so automatic function calling and continuation round trips each get their own
  span and cost (the public call only returns the last request's usage). Outside an open
  `run()`, one `generate_content` call (AFC loop included) is one trace; each streamed hop
  records its own.
- **Tokens and cost:** input = `prompt_token_count` + `tool_use_prompt_token_count`;
  output = `candidates_token_count` + `thoughts_token_count` (thinking is billed as
  output), also sent as `reasoning_tokens`; `cached_content_token_count` is sent as
  `cached_input_tokens` and priced at the cached rate. `metadata.provider` is `gemini`
  or `vertex_ai`, with `metadata.location` on Vertex.
- **Streams** record one span when the stream ends, with the final usage, the joined
  text and `metadata.time_to_first_chunk_ms`. **Failed calls** record a failed `llm`
  span. **Function calls** become inferred `tool` spans (manual loops and AFC).
- Calls a framework adapter already records (an LLM owner, or an open framework `llm`
  span such as LangChain under `MorseCallbackHandler`) are not recorded twice. Google ADK
  has no Morse adapter, so its model calls are recorded by this one.

## [0.7.0] — 2026-11-03

Everything below shipped first as `0.7.0rc1` (2026-10-24) and was soaked on the testbed apps
against Morse dev before this release (SDK_PUBLISHING.md §1.1).

### Fixed — client-side redaction matches the server (WP-12)

- **Provider keys the server already redacts are now redacted in the process too.** The default
  `api_key_prefixed` pattern gains GitHub fine-grained tokens (`github_pat_`), Stripe restricted
  keys (`rk_live_` / `rk_test_`), Stripe webhook secrets (`whsec_`) and Resend keys
  (`re_<id>_<secret>`, which must contain a digit and an uppercase letter so snake_case identifiers
  such as `re_compile_pattern_cache` are left alone). `sk-ant-` and `sk-proj-` keys were already
  covered by `sk-`.
- **The pattern is now case-sensitive, like the server's.** Before, the whole catalogue was
  compiled case-insensitively, so `SK-...` was redacted here but not on the server.
- **Deeply nested payloads fail closed.** The walk limit rises from 10 to 64 (the ingest limit),
  and a node nested deeper is replaced with `[REDACTED:depth]` instead of being sent unredacted.
- No public API change. A shared table of sample strings
  (`apps/sdk/shared-fixtures/redaction_provider_keys.json`) is asserted byte-for-byte by the server,
  this SDK and the TypeScript SDK.

### Fixed — launch verification (2026-10-24)

- **A failed Anthropic `messages.create` keeps an `llm` span** (MHQ-1415): status `failed`, the
  requested model, latency and the error; no usage, so it is never priced. It previously left no
  span at all.
- **LangChain-Anthropic under `init()` + `MorseCallbackHandler` records one span per call**
  (MHQ-1415). The zero-code Anthropic patch no longer records a second
  `anthropic.messages.create` span when the call runs inside an open framework `llm` span.
- **LangChain prompt-cache writes are reported** (MHQ-1414): `input_token_details.cache_creation`
  is sent as `cache_write_tokens`, so cache writes are no longer priced at the plain input rate.
- **A guardrail cancelled by a sibling's tripwire is not a failure** (MHQ-1406): it completes with
  `metadata.cancelled = true` instead of a `CancelledError` failure.
- **Redaction leaves identifiers alone** (MHQ-1412): the phone, card and SSN patterns no longer
  match digit runs inside UUIDs and hex ids (span ids, retriever document ids). API keys, JWTs,
  secrets and emails are still redacted wherever they appear. The server and TS SDK apply the
  same rule.

### Fixed — WP-3 Python SDK correctness (2026-10-24)

- **Zero-code setup traces raw OpenAI clients** (MHQ-1506, MHQ-1410): `openai` joins the
  auto-detect map, so `MORSE_API_KEY` + `init()` now traces `chat.completions.create` and
  `responses.create`, including through LiteLLM's OpenAI provider (`with_raw_response`) and Agno.
  It previously recorded nothing for these apps. The adapter skips calls a framework adapter
  already records (OpenAI Agents runs, LangChain `ChatOpenAI` under the callback handler).
  `google-genai` and LiteLLM's non-OpenAI providers are still not covered.
- **LangGraph's zero-code install no longer races the import** (MHQ-1537). `langgraph` is a
  namespace package the import hook could not wrap, so a background thread installed the adapter
  while the app was still importing, which could deadlock on a `langchain_core` module lock
  (`_DeadlockError`, LangGraph spans silently lost). The adapter now installs at the end of
  `langgraph.pregel`'s import, in the importing thread.
- **OpenAI streaming is traced** (MHQ-1407): `stream=True` on Chat Completions and Responses
  (sync, async, `next()`-style consumers, `.stream()` helpers, `with_raw_response`) records model,
  tokens, cost and output when the stream ends, instead of a span with all of them NULL and a $0
  trace. Without `stream_options={"include_usage": True}` the span has no tokens and says so
  (`metadata.usage_unavailable`).
- **Raw OpenAI: failed calls and tools** (MHQ-1410): a call that raises records a failed `llm`
  span (model, latency, error, exception type); tool calls become `tool` spans inferred from the
  response that requested them and the request carrying their result (`transfer_to_*` calls carry
  `metadata.handoff_target`); the input panel keeps assistant tool calls, tool results and the
  Responses API's `function_call` / `function_call_output` items.
- **`track()` / `batch_track()` store runs** (MHQ-1410): they sent a payload the server rejected
  with 422, which the SDK swallowed. They now build the same trace as `record()`; unknown keys go
  to `metadata`, and a `batch_track()` run without `agent` is skipped.
- **LangChain tool-calling turns have output** (MHQ-1410): the `llm` span lists the requested tool
  calls instead of `{"text": ""}`.
- **Cached tokens are reported** (MHQ-1236): OpenAI spans now carry `cached_input_tokens` (computed
  for pricing but never sent). OpenAI, Anthropic and Claude Agent SDK spans send `0` rather than
  NULL when usage is reported with no cache activity.
- **Bundled pricing has the WP-8 catalog rows** (2026-10-24): 50 models added to match the server
  catalog (OpenAI gpt-5.1 to gpt-6.x, gpt-5-pro, o3-pro, o1-pro; Gemini 3.x/4; Bedrock Nova, Llama,
  Mistral, gpt-oss; Claude Mythos 5/5.1). SDK-side estimates for these fell back to a cheaper
  prefix (`gpt-5-pro` priced as `gpt-5`) or to no price at all.
- **`set_prompt` reads string versions** (MHQ-1415): `"3"` / `"v3"` are recorded as version 3
  instead of being dropped silently; any other non-integer version is dropped with a one-time
  warning.
- **DB capture keeps psycopg 3 connections real** (MHQ-1357): `psycopg.connect()` returns the real
  `psycopg.Connection` (traced through its `cursor_factory`) instead of a proxy, so LangGraph's
  `PostgresSaver` no longer rejects it with "Invalid connection type" when capture is on.

### Fixed — Anthropic streaming is traced (MHQ-1415)

- **`messages.stream()` now records an `llm` span**, sync and async, under both zero-code
  instrumentation and `wrap()`. It previously recorded nothing: `stream()` sends its own request
  and never calls `create()`. The span is written when the `with` block ends, from the SDK's own
  accumulated message, so model, tokens, cache tokens, cost and output are all present.
- **`messages.create(stream=True)` records model, tokens and cost.** Usage is rebuilt from the
  `message_start` and `message_delta` events instead of producing a span with all of them empty.
- Streamed spans carry `metadata.streamed = true`. A stream the caller stops reading early is kept
  and marked `metadata.stream_incomplete = true` (output tokens are partial); an exception inside
  the stream records the span as `failed` with the error.

### Changed — behaviour change, default-on (agent-identity-v2, MHQ-1408)

Set `MORSE_AGENT_IDENTITY=legacy` to restore the previous naming and span shape.

- **One real agent per app.** The root agent is named in this order: `run("name")`, the framework's
  own root (LangGraph graph name, OpenAI Agents starting agent), `OTEL_SERVICE_NAME`, then the entry
  script. A name that came from the entry script is sent with `identity_source=entry_script` so the
  Agents page can ask you to name it.
- **`run("name")` no longer forks a second agent.** Around a framework app, the discovered root keeps
  the agent identity and the run name is sent as the `morse.app` label.
- **LangChain/LangGraph internals are spans, not agents.** Chains, prompt/parser wrappers and
  `__start__`/`tools`/`agent` nodes are `custom` spans (`metadata.node_kind` = `chain`/`node`). A
  graph node that makes an LLM call (an analyst, a supervisor) is still an agent.
- **Correct parents.** LangChain spans parent from the callback run tree, so parallel branches no
  longer nest under each other and LLM/tool spans no longer land on the root placeholder. Spans
  leave the shared stack by identity, not `pop()`.
- **OpenAI Agents:** a handoff target is a sub-agent of the agent that handed off to it.

### Added

- **`get_prompt(name, *, label=None, version=None, if_none_match=None, cache_ttl=None)`** fetches a hosted prompt
  from the registry (`GET /api/v1/prompts/{name}`, needs a key with the `prompts:read` scope).
  Returns `Prompt(name, version, template_text, labels, etag, not_modified)`; a 304 gives
  `not_modified=True` with the etag. Typed errors `PromptAccessError`, `PromptNotFoundError`,
  `PromptRateLimitError` (`.retry_after`), all subclassing `PromptError`. Deliberately thin: no
  retries, no fallback.
- **Optional prompt cache.** `get_prompt(..., cache_ttl=seconds)` serves a process-local LRU
  (default 256 entries, thread-safe); stale entries revalidate with `If-None-Match` and errors are
  never masked by stale data. Off by default (`None`/`0`): no cache read or write. Passing your own
  `if_none_match` bypasses it. New `configure_prompt_cache(max_entries=256)` and
  `clear_prompt_cache()`. We recommend caching in some form: call the API directly, use your own
  cache, or use `cache_ttl`.
- **OpenAI Responses thread facts.** `responses.create` LLM spans now carry `response_id`,
  `previous_response_id` and `conversation_id` (a string, or the `id` of a conversation object) in
  span metadata, only for the keys present. Streamed calls get `previous_response_id` and
  `conversation_id` only, since no final response object is available to the adapter.
  No new public function.
- **`init(capture_end_user=False)`: opt-in end-user capture.** When `True`, the OpenAI adapter
  reads `user` / `safety_identifier` and the Anthropic adapter reads `metadata.user_id`, and sets
  the trace `user_id` to `"h:" + sha256(value)[:32]`. The raw value is never sent, a `user_id` the
  app already set wins, and `customer_id` is never set. Off by default: behaviour is unchanged.

- **`get_prompt(..., link=True)` links the open run to the fetched prompt version** (UP-09). With a
  run open at call time, the fetched `(name, version)` is recorded with `link_source="fetch"` (JSON
  `prompts[].link_source`; OTLP `morse.prompt.link_source`), on the open span if any. Cache hits link;
  a 304 links via the SDK's own cached copy (none cached: no link); no open run: no-op; never raises.
  Opt out with `link=False`. A `set_prompt` for the same name, version and span is deduped into one
  reference marked `fetch`. One info log line per process on the first link.

### Changed — behaviour change, no API change

- **Database and HTTP capture are now ON by default.** Inside an open trace, queries from
  `sqlite3`, `psycopg2`, `psycopg` and Django, and calls from `httpx` and `requests`, are recorded
  as `db` / `http` child spans without any setup. Nothing is recorded when no trace is open.
  Before 0.7.0 this needed `MORSE_CAPTURE_DB=1`.
  **To opt out and keep the old behaviour, set `MORSE_CAPTURE_INFRA=0`** (or
  `MORSE_CAPTURE_DB=0` / `MORSE_CAPTURE_HTTP=0` individually; a specific flag wins).
  Expect a higher span count per trace.
- What is recorded is unchanged and unchanged in spirit: SQL statement text (never bound
  parameters), `METHOD host` for HTTP (never path, query, headers or bodies). LLM provider hosts
  and the Morse ingest host stay excluded.
- **Literal values written inline in SQL are now replaced with `?`** (strings, numbers, hex,
  dollar-quoted bodies, comments) before the statement is stored.
- **A failed db / http / queue span records the error class, not the message.** Driver and URL
  messages carry row values, request paths and broker credentials.
- **Per-trace caps:** at most 200 `db`, 100 `http` and 100 queue publish spans per trace; the rest
  fold into one summary span per kind (`infra.suppressed_count`).
- **Infra spans never fail a trace, and queue spans are not billable** (MHQ-1404). A failed
  `db`, `http` or `queue` span stays failed on its own row, but the server no longer marks the
  whole trace failed because of it. Queue publish spans now carry the new `queue` span type
  (was `custom`) and join `db`/`http` as non-billable (ADR-033 D1). Needs a server that knows the
  `queue` type; older servers reject it.

### Added

- **Queue spans.** Celery and SQS (`boto3`) publishes inside a trace record a `queue` span
  `publish <destination>` with OpenTelemetry messaging attributes, and inject a W3C `traceparent`
  into the message headers / message attributes. On by default; `MORSE_CAPTURE_QUEUE=0` disables.
- **Consumer-side trace continuation, opt-in** (`MORSE_CAPTURE_QUEUE_CONSUMER=1`): Celery workers
  join the producer's trace automatically; for SQS use
  `morse_ai.queue_capture.continue_from_sqs_message(msg, queue_url=...)`. A missing or malformed
  `traceparent` starts a fresh trace.
- `benchmarks/infra_capture_overhead.py` — measures no-trace overhead and per-span cost against
  fixed thresholds.

### Added — attribution capture (MHQ-1275)

All additive; old servers ignore the new fields and old SDKs send none.

- **`morse_ai.link_record(key_type, key_value)`** and
  **`run(..., record_keys={...})`** link a trace to business-record ids (a ticket,
  an order) so later outcomes can be matched to it. Keys accumulate; invalid
  ones are dropped with a debug log; max 20 per trace. Use opaque ids, not
  personal data.
- **`init(..., agent_version=)`** and **`run(..., agent_version=)`**: a declared
  version label (max 128 chars); the per-run value overrides the init default.
- **`morse_ai.set_prompt(name, version=None)`** records a prompt reference on the
  current trace, pinned to the open span when called inside one (max 20 per
  trace). OTLP carries one reference per span (last one wins on that span); the
  JSON path carries all.
- **Git revision detection** now also honours `MORSE_GIT_REVISION` (highest
  priority, for Docker builds with no `.git`) and `SOURCE_VERSION`.
- **`service.version`** is read from `OTEL_RESOURCE_ATTRIBUTES` and sent as a
  resource attribute on both the OTLP and JSON paths. No new `init()` parameter.
- Wire: trace fields `record_keys`, `agent_version`, `prompts`; OTLP root-span
  attributes `morse.record_key.<type>`, `morse.agent_version`; span attributes
  `morse.prompt.name` / `morse.prompt.version`.

### Fixed

- **OpenAI Agents adapter records failures and the served model** (MHQ-1406). A model-call
  exception under a bare `Runner.run/run_sync/run_streamed` now fails the trace (with
  `error_message` and `exception_type`) instead of storing it `completed`; a function tool that
  raises (which the Agents SDK swallows and feeds back to the model) now shows a `failed` tool
  span; and an `Agent` with no `model=` resolves `RunConfig.model` / the SDK default so the LLM
  span carries a model name and gets priced.

- **claude-agent-sdk adapter: tool error detail and sub-agent nesting (MHQ-1416).** A failed tool span now carries the tool's error text as `error_message` (first 500 chars) and a `<Tool>ToolError` exception type, one span per call (the CLI fires both `PostToolUseFailure` and `PostToolUse`). A sub-agent's tool and LLM spans now nest under its named agent span instead of the root, and spans are sent parents-first so the ingest no longer rewrites a forward-referenced parent to the root.

- **OpenAI Responses API calls are now traced** (MHQ-1198). `wrap()` and
  `install()` instrument `client.responses.create` for synchronous and asynchronous
  clients, alongside Chat Completions. Non-streaming Responses calls record model
  and token usage, with input/output capture and context segments. `install()` skips the Responses
  patch when the installed OpenAI package does not expose the resource classes.
- Untraced `httpx` / `requests` calls no longer parse the request URL just to find out there is
  no trace to attach to.
- **LangGraph adapter: async graphs, duplicate LLM spans, control flow** (MHQ-1402, MHQ-1403,
  MHQ-1405). `asyncio.run(graph.ainvoke(...))` without `morse_ai.run()` now records the same
  llm/tool/node spans, tokens and cost as the sync path (it stored only two spans: LangChain ran
  the handler in a copied context, so the auto-trace was invisible to later callbacks). Each real
  LLM/chain/tool call is recorded once however the handler was attached (auto-install plus an
  explicit `callbacks=[...]`, two instances, inherited child managers), so tokens and cost are no
  longer doubled or tripled. `ParentCommand`, `GraphInterrupt`, `NodeInterrupt` and other
  `GraphBubbleUp` control-flow exceptions no longer mark spans or traces failed.

- **LangChain/LCEL spans keep their nesting** (MHQ-1412). Spans reach the OTLP exporter in
  completion order (child before parent), and the replay only parented a span on parents it had
  already emitted, so prompt/LLM/parser spans under a chain were stored under the trace-root
  placeholder (`0000…0001`). Parents are now emitted before their children. Affects every adapter
  that nests spans, not only LangChain. Also: `framework` is now set from LLM/tool/retriever
  callbacks, not only `on_chain_start`.

### Added

- **Retriever `documents[]` (@beta).** The LangChain and LangGraph adapters now
  report each returned chunk as `{rank, id, content, score, source}` (content
  truncated to 2KB with `truncated: true`, max 10 chunks) so Morse can sample and
  judge retrieval quality. Additive: `docs_count`, `top_score`, `sources`,
  `content_preview` and the LangGraph `metadata` key are unchanged.

### Added

- Failed spans now send the raised exception's class name as `exception_type`, and the OTel
  exporter emits a `recordException`-shaped `exception` event (`exception.type`,
  `exception.message`) so repeated failures can be grouped by type and message.
  Additive; no behaviour change when nothing throws.

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
