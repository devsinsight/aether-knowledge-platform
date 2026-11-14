"""Qdrant projection loaded through LlamaIndex's QdrantVectorStore.

The collection is created here first (named dense vector, sparse vector with IDF, collection metadata
and keyword payload indexes), so LlamaIndex finds it and never applies its own defaults. Points are
written by QdrantVectorStore.add with hybrid enabled and the vector names "dense" and "sparse".

Contract guarantees kept on top of LlamaIndex:
- the node ID is the contract's deterministic point ID (idempotency and rebuild);
- the node metadata is the contract payload, nested (flat_metadata=False), so filters keep working;
- the sparse vectors are the ones the Embedder already computed, handed over through sparse_doc_fn.

Deleting old versions and hybrid search use the Qdrant client: LlamaIndex has no "delete every version
but this one", and the Knowledge MCP queries Qdrant directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llama_index.core.schema import BaseNode, TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from aether_core.domain.contract import Chunk
from aether_core.ports import Embedding, SearchHit, SparseVector

DENSE, SPARSE = "dense", "sparse"
INDEXED_FIELDS = ("artifact_id", "key", "section", "artifact_type", "lifecycle.state", "source.version")


class LlamaIndexQdrantStore:
    def __init__(self, url: str, client: QdrantClient | None = None) -> None:
        self._client = client or QdrantClient(url=url)
        self._pending: dict[str, SparseVector] = {}

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
        if not chunks:
            return
        nodes: list[BaseNode] = [_node(chunk, emb) for chunk, emb in zip(chunks, embeddings, strict=True)]
        self._pending = {node.get_content(): emb.sparse for node, emb in zip(nodes, embeddings, strict=True)}
        try:
            self._vector_store(collection).add(nodes)
        finally:
            self._pending = {}

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

    def _vector_store(self, collection: str) -> QdrantVectorStore:
        return QdrantVectorStore(
            collection_name=collection,
            client=self._client,
            enable_hybrid=True,
            dense_vector_name=DENSE,
            sparse_vector_name=SPARSE,
            sparse_doc_fn=self._precomputed_sparse,
            # Without a query function, LlamaIndex loads its default SPLADE model; queries never go
            # through LlamaIndex (see search below), so none is needed.
            sparse_query_fn=_no_sparse_queries,
            flat_metadata=False,
        )

    def _precomputed_sparse(self, texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
        vectors = [self._pending[text] for text in texts]
        return [list(v.indices) for v in vectors], [list(v.values) for v in vectors]


def _no_sparse_queries(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
    raise NotImplementedError("Search uses the Qdrant client directly, not LlamaIndex queries")


def _node(chunk: Chunk, embedding: Embedding) -> TextNode:
    payload = chunk.payload()
    return TextNode(
        id_=chunk.point_id,
        text=chunk.text,
        embedding=list(embedding.dense),
        metadata=payload,
        # The embedded and sparse-encoded text is the chunk text alone, never the metadata.
        excluded_embed_metadata_keys=list(payload),
        excluded_llm_metadata_keys=list(payload),
    )


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
