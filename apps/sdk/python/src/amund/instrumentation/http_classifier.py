"""HTTP endpoint classifier — maps known vector-store URLs to memory_type.

Pure functions only. Never raises — returns None on any error.
"""

from __future__ import annotations

import re

# Each entry: (compiled_host_pattern, compiled_path_pattern, memory_type)
# "retrieval" = read/query operation; "store" = write/upsert operation.
_VECTOR_STORE_PATTERNS_RAW = [
    (r".*\.pinecone\.io", r"/query", "retrieval"),
    (r".*\.pinecone\.io", r"/vectors/upsert", "store"),
    (r".*\.weaviate\.network", r"/v1/graphql", "retrieval"),
    (r".*\.weaviate\.network", r"/v1/objects", "store"),
    # Chroma v4 (chromadb < 0.5)
    (r"localhost|127\.0\.0\.1", r"/api/v1/collections/.*/query", "retrieval"),
    # Chroma 0.5+
    (
        r"localhost|127\.0\.0\.1",
        r"/api/v2/tenants/.*/collections/.*/query",
        "retrieval",
    ),
    # Qdrant (cloud or local)
    (r".*\.qdrant\.io|localhost", r"/collections/.*/points/search", "retrieval"),
    (r".*\.qdrant\.io|localhost", r"/collections/.*/points", "store"),
]

# Compiled once at module load.
_COMPILED_PATTERNS: list[tuple[re.Pattern[str], re.Pattern[str], str]] = [
    (re.compile(host), re.compile(path), memory_type)
    for host, path, memory_type in _VECTOR_STORE_PATTERNS_RAW
]


def classify_http_span(host: str, path: str) -> str | None:
    """Return memory_type ("retrieval" or "store") if URL matches a known vector store.

    Args:
        host: Hostname (and optional port) of the HTTP request, e.g. "api.pinecone.io".
        path: URL path component, e.g. "/query".

    Returns:
        "retrieval", "store", or None when no pattern matches or an error occurs.
    """
    try:
        for host_pat, path_pat, memory_type in _COMPILED_PATTERNS:
            if host_pat.fullmatch(host) and path_pat.fullmatch(path):
                return memory_type
        return None
    except Exception:  # noqa: BLE001
        return None
