"""Dense + sparse embeddings (defaults: bge-base-en-v1.5 and Qdrant/bm25).

Dense vectors come from LlamaIndex's FastEmbedEmbedding. LlamaIndex has no sparse embedding abstraction
(its Qdrant integration wraps FastEmbed directly), so the BM25 vectors use FastEmbed's
SparseTextEmbedding, which also gives the query-side BM25 encoding the Knowledge MCP uses.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastembed import SparseTextEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from aether_core.ports import Embedding, SparseVector


class LlamaIndexFastEmbedEmbedder:
    def __init__(self, dense_model: str, sparse_model: str) -> None:
        self._dense_name, self._sparse_name = dense_model, sparse_model
        self._dense = FastEmbedEmbedding(model_name=dense_model)
        self._sparse = SparseTextEmbedding(model_name=sparse_model)
        self._dense_size = len(self._dense.get_text_embedding("size probe"))

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "dense_model": self._dense_name,
            "dense_size": self._dense_size,
            "sparse_model": self._sparse_name,
        }

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Embedding]:
        dense = self._dense.get_text_embedding_batch(list(texts))
        sparse = list(self._sparse.embed(list(texts)))
        return [_embedding(d, s) for d, s in zip(dense, sparse, strict=True)]

    def embed_query(self, text: str) -> Embedding:
        dense = self._dense.get_query_embedding(text)
        sparse = next(iter(self._sparse.query_embed(text)))
        return _embedding(dense, sparse)


def _embedding(dense: Sequence[float], sparse: Any) -> Embedding:
    return Embedding(
        dense=tuple(float(x) for x in dense),
        sparse=SparseVector(tuple(int(i) for i in sparse.indices), tuple(float(v) for v in sparse.values)),
    )
