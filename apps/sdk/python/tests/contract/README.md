# Contract checks — real agent SDKs, not fixtures (Python)

The counterpart to `apps/sdk/typescript/tests/contract/`. Same question, same
reason, different runtime.

Most of the pytest suite drives fixtures and mock classes. That is right for
behaviour, but it means almost nothing here binds to the packages the adapters
instrument. Python is in better shape than TypeScript was — five test files do
import real packages, and installing `claude-agent-sdk` un-skipped 17 tests
(MHQ-928) — but "better than a suite that passed with everything uninstalled"
is a low bar.

**These checks are the missing contract.** Each imports the REAL package and
asserts the single structural thing its adapter monkey-patches:

| Adapter            | What it patches                                                 |
| ------------------ | --------------------------------------------------------------- |
| `anthropic`        | `anthropic.resources.messages.Messages.create`                  |
| `openai`           | `openai.resources.chat.completions.Completions.create`          |
| `openai`           | `openai.resources.responses.Responses.create` (MHQ-1198)        |
| `openai_agents`    | `agents.run.Runner.run`                                         |
| `claude_agent_sdk` | `claude_agent_sdk.client.ClaudeSDKClient.receive_response`      |
| `langgraph`        | `langchain_core.callbacks.manager.BaseCallbackManager.__init__` |
| `langchain`        | subclasses `langchain_core.callbacks.BaseCallbackHandler`       |

Every one of those is a `setattr` onto someone else's class. When an upstream
release moves or renames the target, the adapter's `install()` catches the
`ImportError`/`AttributeError`, logs at debug, and returns `False` — the host
app keeps running and emits nothing. That is the failure this directory exists
to convert into a red build.

Note the asymmetry with TypeScript, which is real and not an oversight: the
Python packages expose patchable classes, so Python genuinely can monkey-patch
where TS cannot (`query` there is a read-only module export — MHQ-758,
MHQ-936). The two SDKs' adapters differ because the packages differ.

## Running

```sh
uv run pytest tests/contract                  # everything
CONTRACT_ONLY=anthropic uv run pytest tests/contract   # one package
```

`CONTRACT_ONLY` mirrors the TypeScript flag and exists for the same reason —
the nightly `latest` lanes (MHQ-930) bump one package at a time, and a lane
should report on its own package rather than re-reporting the others still at
their floor.

Keep them fast and offline: no API keys, no network, no model calls. Live
end-to-end coverage is the canary's job.

When one fails after a version bump, that is the signal working. Fix the
adapter or move the floor — do not delete the assertion
(`engineering/SDK_VERSION_SUPPORT.md`, rule 2).
