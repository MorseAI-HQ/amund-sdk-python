"""Framework-specific adapters for extracting telemetry from LLM SDK responses.

The submodules in this package follow a common shape:

  * ``adapters.<sdk>`` exposes a passive extractor class (e.g.,
    ``AnthropicAdapter``) usable directly from the LangChain callback handler
    and other in-tree consumers.
  * Where the upstream SDK exposes a discrete client object the customer
    instantiates (Anthropic, OpenAI), the module ALSO exposes a
    ``wrap(client, ...)`` function that returns a drop-in instrumented
    replacement. This is the recommended public API and mirrors the
    TypeScript SDK (``wrapAnthropic``).
  * Auto-detect compatibility is provided by module-level ``install()`` /
    ``uninstall()`` helpers used by ``morse_ai._internal.auto_detect``
    when the customer relies on zero-config setup.

Customers can import the wrappers either as::

    from morse_ai.adapters.anthropic import wrap
    client = wrap(Anthropic())

or as::

    from morse_ai.adapters import anthropic as av_anthropic
    client = av_anthropic.wrap(Anthropic())
"""

from morse_ai.adapters.base import BaseAdapter

__all__ = ["BaseAdapter"]
