"""Tests for the compat manifest loader."""

from __future__ import annotations

import pytest

from amund.data.compat import get_library_status, load_manifest


@pytest.mark.req("MEM-20")
def test_load_manifest_returns_dict_with_known_keys() -> None:
    manifest = load_manifest()
    assert isinstance(manifest, dict)
    for key in (
        "chromadb",
        "pinecone",
        "weaviate-client",
        "qdrant-client",
        "mem0ai",
        "mem0",
        "langgraph",
        "langchain",
        "openai",
        "anthropic",
        "pgvector",
    ):
        assert key in manifest, f"Missing key: {key}"


@pytest.mark.req("MEM-20")
def test_get_chromadb_v5_adapter() -> None:
    adapter = get_library_status("chromadb", "0.5.3")
    assert adapter is not None
    assert adapter["id"] == "chroma_v5"
    assert adapter["status"] == "full"


@pytest.mark.req("MEM-20")
def test_get_unknown_library_returns_none() -> None:
    result = get_library_status("unknown_lib")
    assert result is None


@pytest.mark.req("MEM-20")
def test_get_openai_no_version_returns_first_adapter() -> None:
    adapter = get_library_status("openai")
    assert adapter is not None
    assert adapter["id"] == "openai_v1"


@pytest.mark.req("MEM-20")
def test_get_chromadb_v4_adapter() -> None:
    adapter = get_library_status("chromadb", "0.4.0")
    assert adapter is not None
    assert adapter["id"] == "chroma_v4"


@pytest.mark.req("MEM-20")
def test_get_pinecone_partial_adapter() -> None:
    adapter = get_library_status("pinecone", "2.5.0")
    assert adapter is not None
    assert adapter["id"] == "pinecone_v2"
    assert adapter["status"] == "partial"
