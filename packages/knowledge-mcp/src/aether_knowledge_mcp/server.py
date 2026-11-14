"""Knowledge MCP (AIEF ADR-010): query and sync tools in one server that never writes.

- Query: ``knowledge.search``. Results carry provenance and state; only ``validated`` knowledge is
  returned by default.
- Sync: ``sync.preview``, ``sync.run`` (the agent asks the user first), ``sync.status``.

It holds only the Qdrant read-only key and no source account: sync tools only ask the ``sync``
service, over the compose network, to act.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
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
        # QDRANT_API_KEY here is the read-only key: this server cannot write by construction.
        _state["qdrant"] = QdrantClient(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            api_key=os.environ.get("QDRANT_API_KEY") or None,
        )
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


SYNC_URL = os.environ.get("SYNC_URL", "http://sync:8081")


def _call(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    """Ask a platform service to act; the MCP itself holds no source or wiki credential."""
    try:
        response = httpx.request(method, url, timeout=60.0, **kwargs)
    except httpx.HTTPError as error:
        return {"error": f"service unreachable at {url}: {error}"}
    data = response.json() if response.content else {}
    if response.is_error:
        return {"error": data.get("error", response.text), "status": response.status_code}
    return data


@server.tool(
    name="sync.preview",
    description=(
        "Show what an incremental sync of the knowledge base would do: new, changed and deleted "
        "source items. Changes nothing."
    ),
)
def sync_preview() -> dict[str, Any]:
    return _call("POST", f"{SYNC_URL}/sync/preview")


@server.tool(
    name="sync.run",
    description=(
        "Start an incremental sync of the knowledge base from its sources. It changes the projection: "
        "ALWAYS show the user the result of sync.preview and ask for confirmation before calling it. "
        "Returns a run_id; follow it with sync.status."
    ),
)
def sync_run() -> dict[str, Any]:
    return _call("POST", f"{SYNC_URL}/sync/run", json={"trigger": "knowledge-mcp"})


@server.tool(
    name="sync.status",
    description=(
        "Status of a sync run (the latest when run_id is omitted): steps, counts, errors and the "
        "affected context (related items and their owners)."
    ),
)
def sync_status(run_id: str | None = None) -> dict[str, Any]:
    return _call("GET", f"{SYNC_URL}/sync/status", params={"run_id": run_id} if run_id else None)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
