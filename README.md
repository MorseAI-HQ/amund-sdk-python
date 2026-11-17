# Morse Python SDK

[![PyPI](https://img.shields.io/pypi/v/morse-ai)](https://pypi.org/project/morse-ai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

The Python SDK for [Morse](https://morseai.dev): verdict-first performance intelligence for production AI agents. Record what your agents did, what it cost, and whether it worked. Morse tells you which agents are getting better and which are just getting more expensive.

```bash
pip install morse-ai          # Python 3.11+
```

```python
import morse_ai

morse_ai.init(api_key="mhq_...")  # pragma: allowlist secret

with morse_ai.run("lead-qualifier") as r:
    ...
    r.set_outcome(True, "qualified")
```

Zero-code adapters for Anthropic, OpenAI, OpenAI Agents, LangGraph, Claude Agent SDK, Google GenAI and AWS Strands. It also works with any OpenTelemetry setup.

- **Full SDK guide:** [`apps/sdk/python/README.md`](apps/sdk/python/README.md)
- **Docs:** https://docs.morseai.dev
- **Changelog:** [`apps/sdk/python/CHANGELOG.md`](apps/sdk/python/CHANGELOG.md)
- **Security issues:** https://morseai.dev/security/disclosure (please don't open a public issue)

## License

[MIT](LICENSE)
