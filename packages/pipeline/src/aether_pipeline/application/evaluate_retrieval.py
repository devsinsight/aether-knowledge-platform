"""Measure retrieval quality with a golden set: recall@k and MRR (verification.md).

Results are counted by distinct item, not by chunk: several chunks of the same page count once, in the
position of their best chunk. An item matches when its key or its artifact_id is in the expected list.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from aether_core.ports import Embedder, VectorStore


@dataclass(frozen=True)
class GoldenQuestion:
    question: str
    expected: tuple[str, ...]
    kind: str = "semantic"


@dataclass(frozen=True)
class QuestionResult:
    question: GoldenQuestion
    returned: tuple[str, ...]
    rank: int | None  # 1-based rank of the first expected item, None when missed


@dataclass(frozen=True)
class EvaluationReport:
    k: int
    results: tuple[QuestionResult, ...]

    @property
    def recall_at_k(self) -> float:
        return sum(r.rank is not None for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def mrr(self) -> float:
        return sum(1 / r.rank for r in self.results if r.rank) / len(self.results) if self.results else 0.0


def item_ids(payload: dict[str, Any]) -> tuple[str, str]:
    """(artifact_id, label): the label is the key, or the artifact_id when the item has no key."""
    artifact_id = str(payload.get("artifact_id") or "")
    return artifact_id, str(payload.get("key") or artifact_id)


class EvaluateRetrieval:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder, self._store = embedder, store

    def execute(
        self,
        collection: str,
        questions: Sequence[GoldenQuestion],
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        results: list[QuestionResult] = []
        for question in questions:
            hits = self._store.search(
                collection, self._embedder.embed_query(question.question), k * 4, filters
            )
            seen: set[str] = set()
            labels: list[str] = []
            rank: int | None = None
            for hit in hits:
                artifact_id, label = item_ids(hit.payload)
                if artifact_id in seen:
                    continue
                seen.add(artifact_id)
                labels.append(label)
                if rank is None and {artifact_id, label} & set(question.expected):
                    rank = len(labels)
                if len(labels) == k:
                    break
            results.append(QuestionResult(question, tuple(labels), rank))
        return EvaluationReport(k, tuple(results))
