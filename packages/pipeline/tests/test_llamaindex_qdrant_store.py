"""LlamaIndexQdrantStore against an in-memory Qdrant: contract point IDs, payload and named vectors."""

from __future__ import annotations

from dataclasses import replace

import pytest
from qdrant_client import QdrantClient

from aether_core.domain.contract import Authority, Chunk, Source, normalize
from aether_core.domain.links import Link, LinkReference, SourceRef
from aether_core.ports import Embedding, SparseVector
from aether_pipeline.adapters.llamaindex_qdrant_store import DENSE, SPARSE, LlamaIndexQdrantStore

# In-memory Qdrant ignores payload indexes; the server used in production applies them.
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")

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


def test_points_come_back_with_vectors_for_reuse() -> None:
    adapter, _ = store()
    adapter.upsert("aether", chunks(), EMBEDDINGS)
    (point,) = adapter.get_points("aether", [chunks()[1].point_id, "00000000-0000-0000-0000-000000000000"])
    assert point.embedding == EMBEDDINGS[1]
    assert point.payload["text"] == "The HUD shows the multiplier."


def test_find_mentions_and_delete_points() -> None:
    adapter, _ = store()
    adapter.upsert("aether", chunks(), EMBEDDINGS)
    assert [p["title"] for p in adapter.find_mentions("aether", "HUD")] == ["Scoring & Chains"]
    assert adapter.find_mentions("aether", "Leaderboard") == []
    adapter.delete_points("aether", [chunks()[0].point_id])
    assert [pid for pid, _ in adapter.list_points("aether")] == [chunks()[1].point_id]


def test_links_are_found_by_address_in_both_directions_and_updated_in_place() -> None:
    """Cross-source links: the payload indexes and lookups the resolver and the affected context use."""
    adapter, client = store()
    page, frame = chunks()
    link = Link(LinkReference("https://miro.com/app/board/b=/", "embed"), SourceRef("miro", "b="))
    page = replace(page, links=(link,), addresses=("confluence:1",))
    frame = replace(frame, addresses=("miro:b=:f1", "miro:b="), ancestors=("Ship Prototypes",))
    adapter.upsert("aether", [page, frame], EMBEDDINGS)
    adapter.ensure_collection("aether", MODEL_INFO)  # again: missing indexes only, nothing else changes

    (found,) = adapter.find_by_address("aether", ["miro:b="])
    assert found["addresses"] == ["miro:b=:f1", "miro:b="] and "text" not in found
    (linking,) = adapter.find_linking("aether", ["miro:b=:f1", "miro:b="])
    assert linking[0] == page.point_id and linking[1]["links"][0]["resolved"] is False
    assert adapter.find_linking("aether", []) == [] and adapter.find_by_address("aether", ["x"]) == []

    resolved = {**linking[1]["links"][0], "resolved": True}
    adapter.set_links("aether", page.point_id, [resolved])
    (point,) = client.retrieve("aether", [page.point_id], with_payload=True, with_vectors=True)
    assert (point.payload or {})["links"][0]["resolved"] is True
    assert (point.payload or {})["text"] == page.text and point.vector  # text and vectors stay
