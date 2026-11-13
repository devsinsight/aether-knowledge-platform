"""Qdrant projection: named dense vector + sparse vector with IDF, payload indexes, hybrid RRF search."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from aether_core.domain.contract import Chunk
from aether_core.ports import Embedding, SearchHit

DENSE, SPARSE = "dense", "sparse"
INDEXED_FIELDS = ("artifact_id", "key", "section", "artifact_type", "lifecycle.state", "source.version")


class QdrantStore:
    def __init__(self, url: str, client: QdrantClient | None = None) -> None:
        self._client = client or QdrantClient(url=url)

    def ensure_collection(self, name: str, model_info: dict[str, Any]) -> None:
        if self._client.collection_exists(name):
            return
        self._client.create_collection(
            name,
            vectors_config={
                DENSE: models.VectorParams(size=model_info["dense_size"], distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            metadata={"schema_version": "2", **model_info},
        )
        for field in INDEXED_FIELDS:
            self._client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)

    def upsert(self, collection: str, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        points = [
            models.PointStruct(
                id=chunk.point_id,
                vector={
                    DENSE: list(emb.dense),
                    SPARSE: models.SparseVector(
                        indices=list(emb.sparse.indices), values=list(emb.sparse.values)
                    ),
                },
                payload=chunk.payload(),
            )
            for chunk, emb in zip(chunks, embeddings, strict=True)
        ]
        if points:
            self._client.upsert(collection, points=points, wait=True)

    def delete_artifact(self, collection: str, artifact_id: str, keep_version: str | None = None) -> None:
        must_not: list[models.Condition] | None = (
            [models.FieldCondition(key="source.version", match=models.MatchValue(value=keep_version))]
            if keep_version is not None
            else None
        )
        selector = models.Filter(
            must=[models.FieldCondition(key="artifact_id", match=models.MatchValue(value=artifact_id))],
            must_not=must_not,
        )
        self._client.delete(collection, points_selector=models.FilterSelector(filter=selector), wait=True)

    def search(
        self, collection: str, query: Embedding, limit: int, filters: dict[str, Any] | None = None
    ) -> Sequence[SearchHit]:
        return search(self._client, collection, query, limit, filters)


def search(
    client: QdrantClient, collection: str, query: Embedding, limit: int, filters: dict[str, Any] | None
) -> list[SearchHit]:
    """Hybrid search: dense and sparse candidates fused with RRF, filtered by exact payload values."""
    query_filter = (
        models.Filter(
            must=[
                models.FieldCondition(key=k, match=models.MatchAny(any=list(v))) for k, v in filters.items()
            ]
        )
        if filters
        else None
    )
    sparse = models.SparseVector(indices=list(query.sparse.indices), values=list(query.sparse.values))
    result = client.query_points(
        collection,
        prefetch=[
            models.Prefetch(query=list(query.dense), using=DENSE, limit=limit * 4, filter=query_filter),
            models.Prefetch(query=sparse, using=SPARSE, limit=limit * 4, filter=query_filter),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return [SearchHit(str(p.id), float(p.score), dict(p.payload or {})) for p in result.points]
