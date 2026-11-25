"""RAGQ-01 — retriever ``documents[]`` wire field (@beta)."""

from __future__ import annotations

import uuid

import pytest

from amund import tracing
from amund.adapters._retrieval import build_documents
from amund.adapters.langchain import MorseCallbackHandler


class MockDocument:
    def __init__(self, content: str, source: str = "", score: float | None = None):
        self.page_content = content
        self.metadata: dict = {"source": source}
        if score is not None:
            self.metadata["score"] = score


def _serialized() -> dict:
    return {"id": ["langchain", "retrievers", "VectorStoreRetriever"]}


def _retriever_span() -> dict:
    spans = tracing._get_last_trace()["spans"]
    return [s for s in spans if s["type"] == "retriever"][0]


@pytest.mark.req("RAGQ-01")
def test_documents_carry_rank_id_content_score_source():
    handler = MorseCallbackHandler("rag-docs")
    run_id = uuid.uuid4()
    with tracing.run("rag-docs"):
        handler.on_retriever_start(_serialized(), "q", run_id=run_id)
        handler.on_retriever_end(
            [
                MockDocument("first chunk", "a.pdf", 0.9),
                MockDocument("second chunk", "b.pdf"),
            ],
            run_id=run_id,
        )
    docs = _retriever_span()["output_data"]["documents"]
    assert [d["rank"] for d in docs] == [1, 2]
    assert docs[0]["content"] == "first chunk"
    assert docs[0]["source"] == "a.pdf"
    assert docs[0]["score"] == pytest.approx(0.9)
    assert docs[1]["score"] is None
    assert docs[0]["id"] and docs[0]["id"] != docs[1]["id"]
    # legacy key retained for old readers
    assert docs[0]["content_preview"] == "first chunk"


@pytest.mark.req("RAGQ-01")
def test_documents_cap_chunks_and_truncate_content():
    handler = MorseCallbackHandler("rag-cap")
    run_id = uuid.uuid4()
    docs_in = [MockDocument("x" * 5000, "s")] + [
        MockDocument(f"d{i}") for i in range(14)
    ]
    with tracing.run("rag-cap"):
        handler.on_retriever_start(_serialized(), "q", run_id=run_id)
        handler.on_retriever_end(docs_in, run_id=run_id)
    out = _retriever_span()["output_data"]
    assert out["docs_count"] == 15
    assert len(out["documents"]) == 10
    assert len(out["documents"][0]["content"].encode()) <= 2048
    assert out["documents"][0]["truncated"] is True
    assert "truncated" not in out["documents"][1]


@pytest.mark.req("RAGQ-01")
def test_build_documents_accepts_dicts_and_uses_explicit_id():
    docs = build_documents(
        [{"id": "chunk-7", "content": "hello", "score": "0.5", "source": "s"}]
    )
    assert docs == [
        {"rank": 1, "id": "chunk-7", "content": "hello", "score": 0.5, "source": "s"}
    ]
