"""Read-only Knowledge MCP (walking skeleton: ``knowledge.search``).

It never runs pipelines and never writes. Results always carry provenance and state; only
``validated`` knowledge is returned unless the caller asks for drafts explicitly.
"""

from __future__ import annotations

import os
from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding
from mcp.server.mcpserver import MCPServer
from qdrant_client import QdrantClient, models

COLLECTION = os.environ.get("KNOWLEDGE_COLLECTION", "aether")
DENSE_MODEL = os.environ.get("DENSE_MODEL", "BAAI/bge-base-en-v1.5")
SPARSE_MODEL = os.environ.get("SPARSE_MODEL", "Qdrant/bm25")

server = MCPServer("aether-knowledge")
_state: dict[str, Any] = {}


def _clients() -> tuple[QdrantClient, TextEmbedding, SparseTextEmbedding]:
    if not _state:
        _state["qdrant"] = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
        _state["dense"] = TextEmbedding(model_name=DENSE_MODEL)
        _state["sparse"] = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _state["qdrant"], _state["dense"], _state["sparse"]


def build_filter(include_drafts: bool, section: str | None, key: str | None) -> models.Filter:
    states = ["validated", "draft"] if include_drafts else ["validated"]
    must: list[models.Condition] = [
        models.FieldCondition(key="lifecycle.state", match=models.MatchAny(any=states))
    ]
    if section:
        must.append(models.FieldCondition(key="section", match=models.MatchValue(value=section)))
    if key:
        must.append(models.FieldCondition(key="key", match=models.MatchValue(value=key)))
    return models.Filter(must=must)


def to_result(score: float, payload: dict[str, Any]) -> dict[str, Any]:
    """Shape of every result: the text plus provenance (source, ID, URL, version) and state."""
    return {
        "score": round(score, 4),
        "title": payload.get("title"),
        "heading_path": payload.get("heading_path", []),
        "text": payload.get("text"),
        "key": payload.get("key"),
        "section": payload.get("section"),
        "type": payload.get("artifact_type"),
        "state": payload.get("lifecycle", {}).get("state"),
        "source": payload.get("source"),
        "metadata_complete": payload.get("metadata_complete"),
        "missing_fields": payload.get("missing_fields", []),
    }


@server.tool(name="knowledge.search", description="Hybrid semantic search over the Aether knowledge base.")
def knowledge_search(
    query: str,
    limit: int = 5,
    include_drafts: bool = False,
    section: str | None = None,
    key: str | None = None,
) -> list[dict[str, Any]]:
    qdrant, dense, sparse = _clients()
    query_filter = build_filter(include_drafts, section, key)
    dense_vector = [float(x) for x in next(iter(dense.query_embed(query)))]
    sparse_raw = next(iter(sparse.query_embed(query)))
    sparse_vector = models.SparseVector(
        indices=[int(i) for i in sparse_raw.indices], values=[float(v) for v in sparse_raw.values]
    )
    result = qdrant.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_vector, using="dense", limit=limit * 4, filter=query_filter),
            models.Prefetch(query=sparse_vector, using="sparse", limit=limit * 4, filter=query_filter),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return [to_result(float(p.score), dict(p.payload or {})) for p in result.points]


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
