"""EvaluateRetrieval with fakes: ranks by distinct item, matching by key or artifact_id."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from aether_core.domain.contract import Chunk
from aether_core.ports import Embedding, SearchHit, SparseVector
from aether_pipeline.application.evaluate_retrieval import EvaluateRetrieval, GoldenQuestion

QUERY = Embedding((1.0,), SparseVector((), ()))


class FakeEmbedder:
    model_info: dict[str, Any] = {}

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Embedding]:
        return [QUERY for _ in texts]

    def embed_query(self, text: str) -> Embedding:
        return QUERY


def hit(artifact_id: str, key: str = "") -> SearchHit:
    return SearchHit(artifact_id, 1.0, {"artifact_id": artifact_id, "key": key})


class FakeStore:
    """Three chunks of page 1 (keyed), then page 2 (no key), then page 3."""

    def ensure_collection(self, name: str, model_info: dict[str, Any]) -> None: ...

    def upsert(self, collection: str, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None: ...

    def delete_artifact(self, collection: str, artifact_id: str, keep_version: str | None = None) -> None: ...

    def search(
        self, collection: str, query: Embedding, limit: int, filters: dict[str, Any] | None = None
    ) -> Sequence[SearchHit]:
        return [
            hit("confluence:1", "feature-a"),
            hit("confluence:1", "feature-a"),
            hit("confluence:1", "feature-a"),
            hit("confluence:2"),
            hit("confluence:3", "decision-adr-001"),
        ]


def run(*questions: GoldenQuestion, k: int = 5):
    return EvaluateRetrieval(FakeEmbedder(), FakeStore()).execute("aether", questions, k)


def test_chunks_of_one_item_count_once_and_keyless_items_match_by_artifact_id() -> None:
    report = run(GoldenQuestion("q", ("confluence:2",)))
    assert report.results[0].returned == ("feature-a", "confluence:2", "decision-adr-001")
    assert report.results[0].rank == 2


def test_recall_and_mrr() -> None:
    report = run(
        GoldenQuestion("first", ("feature-a",)),
        GoldenQuestion("third", ("decision-adr-001",)),
        GoldenQuestion("missing", ("feature-z",)),
    )
    assert [r.rank for r in report.results] == [1, 3, None]
    assert report.recall_at_k == 2 / 3
    assert abs(report.mrr - (1 + 1 / 3) / 3) < 1e-9


def test_k_limits_the_items_considered() -> None:
    assert run(GoldenQuestion("third", ("decision-adr-001",)), k=2).results[0].rank is None
