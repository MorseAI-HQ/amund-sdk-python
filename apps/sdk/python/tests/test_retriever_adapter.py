"""Tests for the LangChain retriever callback handler — AGE-342.

These tests do NOT require langchain installed — handler methods are called
directly with mock arguments.
"""

from __future__ import annotations

import uuid

import pytest

from morse_ai import tracing
from morse_ai.adapters.langchain import MorseCallbackHandler

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockDocument:
    """Minimal mock for a LangChain Document object."""

    def __init__(self, content: str, source: str = "", score: float | None = None):
        self.page_content = content
        self.metadata: dict = {"source": source}
        if score is not None:
            self.metadata["score"] = score


def _retriever_serialized(name: str = "VectorStoreRetriever") -> dict:
    return {"id": ["langchain", "retrievers", name]}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.req("AGE-342")
def test_retriever_start_end_creates_retriever_span():
    """on_retriever_start/end produces a span with type='retriever'."""
    handler = MorseCallbackHandler("rag-agent")
    run_id = uuid.uuid4()

    with tracing.run("rag-agent"):
        handler.on_retriever_start(
            _retriever_serialized(),
            "What is Morse?",
            run_id=run_id,
        )
        docs = [
            MockDocument("Morse is an observability platform.", "doc1.pdf", 0.95),
            MockDocument("It supports LangChain.", "doc2.pdf", 0.80),
        ]
        handler.on_retriever_end(docs, run_id=run_id)

    trace = tracing._get_last_trace()
    assert trace is not None

    retriever_spans = [s for s in trace["spans"] if s.get("type") == "retriever"]
    assert len(retriever_spans) == 1

    span = retriever_spans[0]
    assert span["name"] == "VectorStoreRetriever"
    assert span["input_data"]["query"] == "What is Morse?"

    out = span.get("output_data", {})
    assert out["docs_count"] == 2
    assert out["top_score"] == pytest.approx(0.95)
    assert "doc1.pdf" in out["sources"]


@pytest.mark.req("AGE-342")
def test_retriever_error_closes_span():
    """on_retriever_error closes the span without raising."""
    handler = MorseCallbackHandler("rag-agent-err")
    run_id = uuid.uuid4()

    with tracing.run("rag-agent-err"):
        handler.on_retriever_start(
            _retriever_serialized("FailingRetriever"),
            "query",
            run_id=run_id,
        )
        handler.on_retriever_error(
            RuntimeError("vector store unavailable"),
            run_id=run_id,
        )

    # Should not raise; span stack should be empty
    assert run_id not in handler._span_stack


@pytest.mark.req("AGE-342")
def test_retriever_start_swallows_bad_serialized():
    """on_retriever_start handles None/malformed serialized without raising."""
    handler = MorseCallbackHandler("rag-agent-safe")
    run_id = uuid.uuid4()

    with tracing.run("rag-agent-safe"):
        # None serialized — should not raise
        handler.on_retriever_start(None, "query", run_id=run_id)
        handler.on_retriever_end([], run_id=run_id)


@pytest.mark.req("AGE-342")
def test_retriever_end_handles_dict_documents():
    """on_retriever_end handles raw dict documents (not LangChain Document objects)."""
    handler = MorseCallbackHandler("rag-dict")
    run_id = uuid.uuid4()

    with tracing.run("rag-dict"):
        handler.on_retriever_start(
            _retriever_serialized(),
            "dict docs query",
            run_id=run_id,
        )
        handler.on_retriever_end(
            [{"content": "raw doc", "source": "manual"}],
            run_id=run_id,
        )

    trace = tracing._get_last_trace()
    retriever_spans = [s for s in trace["spans"] if s.get("type") == "retriever"]
    assert len(retriever_spans) == 1
    assert retriever_spans[0]["output_data"]["docs_count"] == 1
