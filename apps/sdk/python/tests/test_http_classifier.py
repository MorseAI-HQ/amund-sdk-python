"""Tests for morse_ai.instrumentation.http_classifier."""

import pytest

from morse_ai.instrumentation.http_classifier import classify_http_span


@pytest.mark.req("MEM-17")
class TestHttpClassifier:
    def test_pinecone_query_is_retrieval(self):
        result = classify_http_span("myindex.svc.pinecone.io", "/query")
        assert result == "retrieval"

    def test_pinecone_upsert_is_store(self):
        result = classify_http_span("myindex.svc.pinecone.io", "/vectors/upsert")
        assert result == "store"

    def test_unknown_url_returns_none(self):
        result = classify_http_span("api.example.com", "/v1/something")
        assert result is None

    def test_unknown_host_known_path_returns_none(self):
        result = classify_http_span("api.example.com", "/query")
        assert result is None

    def test_chroma_v4_query_is_retrieval(self):
        result = classify_http_span(
            "localhost", "/api/v1/collections/my_collection/query"
        )
        assert result == "retrieval"

    def test_chroma_v5_query_is_retrieval(self):
        result = classify_http_span(
            "127.0.0.1",
            "/api/v2/tenants/default_tenant/collections/my_collection/query",
        )
        assert result == "retrieval"

    def test_qdrant_search_is_retrieval(self):
        result = classify_http_span(
            "localhost", "/collections/my_collection/points/search"
        )
        assert result == "retrieval"

    def test_qdrant_cloud_search_is_retrieval(self):
        result = classify_http_span(
            "abc123.us-east-1-0.aws.cloud.qdrant.io",
            "/collections/my_collection/points/search",
        )
        assert result == "retrieval"

    def test_qdrant_points_upsert_is_store(self):
        result = classify_http_span("localhost", "/collections/my_collection/points")
        assert result == "store"

    def test_weaviate_graphql_is_retrieval(self):
        result = classify_http_span("my-cluster.weaviate.network", "/v1/graphql")
        assert result == "retrieval"

    def test_weaviate_objects_is_store(self):
        result = classify_http_span("my-cluster.weaviate.network", "/v1/objects")
        assert result == "store"

    def test_never_raises_on_bad_input(self):
        # Malformed inputs should return None, not raise.
        assert classify_http_span("", "") is None
