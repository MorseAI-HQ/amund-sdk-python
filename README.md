# Amund Python SDK

[![PyPI](https://img.shields.io/pypi/v/amund)](https://pypi.org/project/amund/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

The Python SDK for [Amund](https://amundai.com): verdict-first performance intelligence for production AI agents. Record what your agents did, what it cost, and whether it worked. Amund tells you which agents are getting better and which are just getting more expensive.

```bash
pip install amund             # Python 3.11+
```

```python
import amund

amund.init(api_key="mhq_...")  # pragma: allowlist secret  (or set AMUND_API_KEY)

with amund.run("lead-qualifier") as r:
    ...
    r.set_outcome(True, "qualified")
```

> **Coming from `morse-ai`?** Morse is now Amund. `morse-ai` still installs and `import morse_ai`
> still works (it is now a compatibility shim over `amund`), and `MORSE_*` environment variables
> are still read. See [Migrating from morse-ai](apps/sdk/python/README.md#migrating-from-morse-ai).

Zero-code adapters for Anthropic, OpenAI, OpenAI Agents, LangGraph, Claude Agent SDK, Google GenAI and AWS Strands. It also works with any OpenTelemetry setup.

- **Full SDK guide:** [`apps/sdk/python/README.md`](apps/sdk/python/README.md)
- **Docs:** https://docs.amundai.com
- **Changelog:** [`apps/sdk/python/CHANGELOG.md`](apps/sdk/python/CHANGELOG.md)
- **Security issues:** https://amundai.com/security/disclosure (please don't open a public issue)

## License

[MIT](LICENSE)
