"""Launch-verification repro for MHQ-1412 item 1 (LCEL nesting lost on the wire).

The docs LangChain snippet verbatim (``MorseCallbackHandler(agent_name=...)``
via ``with_config``) on ``prompt | llm | parser``, captured from the real OTel
export path (in-memory exporter behind the redacting exporter). Before the fix
``ChatPromptTemplate``, the chat model and the parser were exported with the
synthetic ``0x1`` parent (stored as ``...0001``) instead of the span that
contains them.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from morse_ai import tracing  # noqa: E402
from morse_ai._internal.redaction import RedactionConfig  # noqa: E402
from morse_ai.adapters.langchain import MorseCallbackHandler  # noqa: E402
from morse_ai.otel_provider import get_engine  # noqa: E402


class _FakeChat(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        msg = AIMessage(
            content="Bonjour",
            usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
            response_metadata={"model_name": "gpt-4o-mini"},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def engine():
    eng = get_engine()
    eng.configure(
        endpoint="http://127.0.0.1:1/api/v1/otlp/v1/traces",  # never dialed
        api_key="av_test",  # pragma: allowlist secret
        sample_rate=1.0,
        schedule_delay_millis=50,
        max_queue_size=4096,
        max_export_batch_size=512,
        max_field_bytes=65536,
        timeout=1.0,
        redaction_config=RedactionConfig.build(enabled=True),
        capture_for_tests=True,
    )
    token = tracing._active_trace.set(None)
    try:
        yield eng
    finally:
        tracing._active_trace.reset(token)
        eng.reset_for_testing()


@pytest.mark.req("MHQ-1412")
def test_mhq1412_docs_lcel_chain_keeps_nesting_on_the_wire(engine):
    handler = MorseCallbackHandler(agent_name="my-agent")
    prompt = ChatPromptTemplate.from_messages([("user", "Translate {text} to French")])
    chain = (prompt | _FakeChat() | StrOutputParser()).with_config(callbacks=[handler])
    assert chain.invoke({"text": "Hello"}) == "Bonjour"

    entry = tracing._get_last_trace()
    assert entry is not None
    trace_hex = entry["trace"]["trace_id"].replace("-", "")
    engine.flush()  # the handler's auto-trace was already exported

    exported = [
        s for s in engine.recorder.spans if s.context.trace_id == int(trace_hex, 16)
    ]
    assert len(exported) == len(entry["spans"]) >= 4
    ids = {s.context.span_id for s in exported}
    # The OTel root carries the synthetic 0x1 placeholder parent by design;
    # every other span must point at a real exported span.
    roots = [s for s in exported if s.parent is None or s.parent.span_id not in ids]
    assert len(roots) == 1, [
        (s.name, hex(s.context.span_id), s.parent and hex(s.parent.span_id))
        for s in exported
    ]
    assert roots[0].name == "my-agent"
    children = {s.name for s in exported if s is not roots[0]}
    assert {"ChatPromptTemplate", "_FakeChat"} <= children


class _ToolCallChat(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake-tools"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1"}
            ],
            usage_metadata={"input_tokens": 9, "output_tokens": 5, "total_tokens": 14},
            response_metadata={"model_name": "gpt-4o-mini"},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.mark.req("MHQ-1410")
def test_mhq1410_tool_calling_turn_output_is_not_blank():
    """A tool-calling turn used to export ``output_data == {"text": ""}``."""
    token = tracing._active_trace.set(None)
    try:
        with tracing.run("lc-tools"):
            _ToolCallChat().invoke(
                "weather?",
                config={"callbacks": [MorseCallbackHandler(agent_name="lc-tools")]},
            )
    finally:
        tracing._active_trace.reset(token)
    trace = tracing._get_last_trace()
    (llm,) = [s for s in trace["spans"] if s["type"] == "llm"]
    assert llm["output_data"] == {"text": '[tool_call:get_weather] {"city": "Paris"}'}
