"""LlamaIndexQdrantStore against an in-memory Qdrant: contract point IDs, payload and named vectors."""

from __future__ import annotations

from qdrant_client import QdrantClient

from aether_core.domain.contract import Authority, Chunk, Source, normalize
from aether_core.ports import Embedding, SparseVector
from aether_pipeline.adapters.llamaindex_qdrant_store import DENSE, SPARSE, LlamaIndexQdrantStore

MODEL_INFO = {"dense_model": "fake", "dense_size": 2, "sparse_model": "fake"}


def chunks(version: str = "1") -> list[Chunk]:
    artifact = normalize(
        artifact_id="confluence:1",
        title="Scoring & Chains",
        source=Source("confluence", "1", "https://example/wiki/pages/1", version),
        authority=Authority("textual"),
        text="unused",
        key="feature-scoring-and-chains",
        artifact_type="feature",
        section="product",
        state="validated",
    )
    return [
        Chunk(artifact, ("Scoring & Chains", "Rules"), 0, "A chain lasts 2.0 seconds."),
        Chunk(artifact, ("Scoring & Chains", "Display"), 1, "The HUD shows the multiplier."),
    ]


EMBEDDINGS = [
    Embedding((1.0, 0.0), SparseVector((1, 7), (0.5, 1.5))),
    Embedding((0.0, 1.0), SparseVector((3,), (2.0,))),
]


def store() -> tuple[LlamaIndexQdrantStore, QdrantClient]:
    client = QdrantClient(":memory:")
    adapter = LlamaIndexQdrantStore("unused", client=client)
    adapter.ensure_collection("aether", MODEL_INFO)
    return adapter, client


def test_points_use_contract_ids_payload_and_named_vectors() -> None:
    adapter, client = store()
    adapter.upsert("aether", chunks(), EMBEDDINGS)
    points = client.retrieve("aether", [c.point_id for c in chunks()], with_payload=True, with_vectors=True)
    assert {str(p.id) for p in points} == {c.point_id for c in chunks()}
    by_id = {str(p.id): p for p in points}
    first = by_id[chunks()[0].point_id]
    payload = first.payload or {}
    assert payload["key"] == "feature-scoring-and-chains"
    assert payload["lifecycle"]["state"] == "validated"
    assert payload["source"]["version"] == "1"
    vectors = first.vector
    assert isinstance(vectors, dict)
    assert list(vectors[DENSE]) == [1.0, 0.0]  # type: ignore[arg-type]
    sparse = vectors[SPARSE]
    assert list(sparse.indices) == [1, 7] and list(sparse.values) == [0.5, 1.5]  # type: ignore[union-attr]


def test_upsert_twice_is_idempotent_and_versions_replace() -> None:
    adapter, client = store()
    adapter.upsert("aether", chunks(), EMBEDDINGS)
    adapter.upsert("aether", chunks(), EMBEDDINGS)
    assert client.count("aether").count == 2
    adapter.upsert("aether", chunks("2"), EMBEDDINGS)
    adapter.delete_artifact("aether", "confluence:1", keep_version="2")
    points, _ = client.scroll("aether", with_payload=True)
    assert {(p.payload or {})["source"]["version"] for p in points} == {"2"}


def test_hybrid_search_filters_by_state() -> None:
    adapter, _ = store()
    adapter.upsert("aether", chunks(), EMBEDDINGS)
    hits = adapter.search("aether", EMBEDDINGS[0], 2, {"lifecycle.state": ["validated"]})
    assert hits and hits[0].point_id == chunks()[0].point_id
    assert adapter.search("aether", EMBEDDINGS[0], 2, {"lifecycle.state": ["draft"]}) == []
