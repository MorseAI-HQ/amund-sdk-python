# Published-SDK E2E (`dogfood-published/`)

Verifies the **published** `morse-ai` from PyPI — not `pip install -e .` of this
directory. Same rationale as the TypeScript counterpart
(`apps/sdk/typescript/dogfood-published/README.md`): an editable install proves
the source tree works, not the wheel that customers actually get.

## Setup

Use a venv **outside** this repo, or at least one that has never had the local
package installed editable — an editable install shadows the PyPI one and the
test silently passes for the wrong reason.

```bash
python3 -m venv /tmp/morse-pub-e2e && cd /tmp/morse-pub-e2e
./bin/pip install "morse-ai[anthropic,openai,langgraph]==0.4.1" langchain-openai
./bin/python -c "import morse_ai; print(morse_ai.__version__)"   # must match the pin
```

Pin the exact version — the pin is the thing under test.

## Run

```bash
export MORSE_API_KEY=...           # a dogfood org key, see bizops/dogfood/ACCOUNTS.md
export MORSE_ENDPOINT=https://dev-api.morsehq.dev/api/v1/traces/batch
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...
python run.py all                  # or: python run.py anthropic | openai | langgraph
```

## Verifying delivery

Exit code 0 only means nothing raised. Confirm the traces actually landed with
the dev-DB query in `apps/sdk/typescript/dogfood-published/README.md`.
