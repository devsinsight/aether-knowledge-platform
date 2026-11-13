"""Dense + sparse embeddings through FastEmbed (defaults: bge-base-en-v1.5 and Qdrant/bm25)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding

from aether_core.ports import Embedding, SparseVector


class FastEmbedEmbedder:
    def __init__(self, dense_model: str, sparse_model: str) -> None:
        self._dense_name, self._sparse_name = dense_model, sparse_model
        self._dense = TextEmbedding(model_name=dense_model)
        self._sparse = SparseTextEmbedding(model_name=sparse_model)
        self._dense_size = len(next(iter(self._dense.embed(["size probe"]))))

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "dense_model": self._dense_name,
            "dense_size": self._dense_size,
            "sparse_model": self._sparse_name,
        }

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Embedding]:
        dense = list(self._dense.embed(list(texts)))
        sparse = list(self._sparse.embed(list(texts)))
        return [_embedding(d, s) for d, s in zip(dense, sparse, strict=True)]

    def embed_query(self, text: str) -> Embedding:
        dense = next(iter(self._dense.query_embed(text)))
        sparse = next(iter(self._sparse.query_embed(text)))
        return _embedding(dense, sparse)


def _embedding(dense: Any, sparse: Any) -> Embedding:
    return Embedding(
        dense=tuple(float(x) for x in dense),
        sparse=SparseVector(tuple(int(i) for i in sparse.indices), tuple(float(v) for v in sparse.values)),
    )
